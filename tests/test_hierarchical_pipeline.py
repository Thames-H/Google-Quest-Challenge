from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from quest.config import load_config
from quest.data import (
    GoogleQuestDataset,
    build_group_folds,
    build_group_keys,
    prepare_metadata_spec,
    quest_collate_fn,
)
from quest.losses import compute_mixed_loss
from quest.model import DualTransformerRegressor
from quest.postprocess import rank_based_distribution_matching


TARGET_COLUMNS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "answer_helpful",
    "answer_well_written",
]


class ChunkingDummyTokenizer:
    pad_token_id = 0
    cls_token_id = 101
    sep_token_id = 102

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        tokens = [len(token) + idx + 5 for idx, token in enumerate(text.split())]
        return tokens or [7]

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def prepare_for_model(
        self,
        ids,
        pair_ids=None,
        add_special_tokens=True,
        max_length=None,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    ):
        del add_special_tokens, truncation
        pair_ids = pair_ids or []
        tokens = [self.cls_token_id, *ids, self.sep_token_id, *pair_ids, self.sep_token_id]
        if max_length is not None:
            tokens = tokens[:max_length]
        attention = [1] * len(tokens)
        if padding == "max_length" and max_length is not None:
            pad = max_length - len(tokens)
            if pad > 0:
                tokens = tokens + [self.pad_token_id] * pad
                attention = attention + [0] * pad
        payload = {"input_ids": tokens}
        if return_attention_mask:
            payload["attention_mask"] = attention
        return payload


class HierarchicalDummyEncoder(torch.nn.Module):
    def __init__(self, hidden_size=12):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(4096, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, hidden_size)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = self.embedding(input_ids % 4096)
        hidden = self.projection(hidden)
        return type("Output", (), {"last_hidden_state": hidden})()


def build_dataframe(rows: int) -> pd.DataFrame:
    payload = []
    for idx in range(rows):
        payload.append(
            {
                "qa_id": idx + 1,
                "question_title": f"Title   {idx // 2}",
                "question_body": f"Body text {idx // 2} with repeated words and links http://example.com/{idx}",
                "answer": (
                    f"Answer text {idx} with code <code>print({idx})</code> "
                    f"and another link https://answers.example.com/{idx}"
                ),
                "url": f"https://stackexchange.example.com/questions/{idx}",
                "category": "SCIENCE" if idx % 2 == 0 else "TECHNOLOGY",
                "host": "stackoverflow.com" if idx % 2 == 0 else "superuser.com",
                **{
                    target: ((idx + target_idx) % 5) / 4.0
                    for target_idx, target in enumerate(TARGET_COLUMNS)
                },
            }
        )
    return pd.DataFrame(payload)


def write_config(tmp_path: Path, data_dir: Path) -> Path:
    config = {
        "data_dir": str(data_dir),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "backbone": "dummy-deberta",
        "folds": 2,
        "seeds": [7],
        "epochs": 1,
        "lr_encoder": 1e-3,
        "lr_head": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "max_len_question": 24,
        "max_len_answer": 24,
        "max_title_tokens": 6,
        "question_chunk_size": 5,
        "answer_chunk_size": 4,
        "question_chunk_overlap": 1,
        "answer_chunk_overlap": 1,
        "question_max_chunks": 3,
        "answer_max_chunks": 4,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "fp16": False,
        "gradient_checkpointing": True,
        "num_workers": 0,
        "dropout": 0.1,
        "device": "cpu",
        "use_metadata": True,
        "meta_embedding_dim": 8,
        "meta_hidden_dim": 12,
        "pointwise_loss": "smooth_l1",
        "pointwise_weight": 1.0,
        "ranking_weight": 0.2,
        "ranking_margin": 0.05,
        "distribution_matching": True,
    }
    config_path = tmp_path / "hierarchical.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_load_config_resolves_cli_then_env_for_data_dir(tmp_path, monkeypatch):
    config_path = write_config(tmp_path, tmp_path / "default-data")
    config = load_config(config_path)

    monkeypatch.setenv("QUEST_DATA_DIR", str(tmp_path / "env-data"))
    assert config.resolved_data_dir() == (tmp_path / "env-data")
    assert config.resolved_data_dir(tmp_path / "cli-data") == (tmp_path / "cli-data")


def test_build_group_keys_normalize_title_and_body():
    dataframe = pd.DataFrame(
        {
            "question_title": ["Hello   World", " hello world "],
            "question_body": ["Body\tText", "body text"],
        }
    )
    keys = build_group_keys(dataframe)
    assert keys.iloc[0] == keys.iloc[1]

    folds = build_group_folds(
        pd.DataFrame(
            {
                "question_title": ["A", "A", "B", "B"],
                "question_body": ["same", " same ", "other", " other "],
            }
        ),
        folds=2,
    )
    for train_idx, valid_idx in folds:
        assert set(train_idx).isdisjoint(set(valid_idx))


def test_dataset_and_collate_return_chunked_inputs_and_metadata(monkeypatch):
    train_df = build_dataframe(4)
    metadata_spec = prepare_metadata_spec(train_df, train_df)
    monkeypatch.setattr("quest.data.AutoTokenizer", ChunkingDummyTokenizer)

    dataset = GoogleQuestDataset(
        dataframe=train_df,
        tokenizer_name="dummy-deberta",
        max_len_question=24,
        max_len_answer=24,
        question_chunk_size=5,
        answer_chunk_size=4,
        question_chunk_overlap=1,
        answer_chunk_overlap=1,
        question_max_chunks=3,
        answer_max_chunks=4,
        max_title_tokens=6,
        target_columns=TARGET_COLUMNS,
        is_train=True,
        metadata_spec=metadata_spec,
    )
    batch = quest_collate_fn([dataset[0], dataset[1]])

    assert batch["q_input_ids"].shape[0] == 2
    assert batch["q_input_ids"].shape[2] == 24
    assert batch["q_input_ids"].shape[1] <= 3
    assert batch["a_input_ids"].shape[0] == 2
    assert batch["a_input_ids"].shape[2] == 24
    assert batch["a_input_ids"].shape[1] <= 4
    assert batch["q_chunk_mask"].shape == batch["q_input_ids"].shape[:2]
    assert batch["a_chunk_mask"].shape == batch["a_input_ids"].shape[:2]
    assert batch["meta_numeric"].shape[0] == 2
    assert batch["meta_category_id"].shape == (2,)
    assert batch["labels"].shape == (2, len(TARGET_COLUMNS))


def test_dataset_falls_back_to_slow_tokenizer_when_fast_loader_breaks(monkeypatch):
    calls = []
    slow_calls = []

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(_name, use_fast=True):
            calls.append(use_fast)
            raise AttributeError("'NoneType' object has no attribute 'endswith'")

    class SlowTokenizerLoader:
        @staticmethod
        def from_pretrained(_name):
            slow_calls.append("slow")
            return ChunkingDummyTokenizer()

    train_df = build_dataframe(2)
    metadata_spec = prepare_metadata_spec(train_df, train_df)
    monkeypatch.setattr("quest.data.AutoTokenizer", TokenizerLoader)
    monkeypatch.setattr("quest.data.DebertaV2Tokenizer", SlowTokenizerLoader)

    dataset = GoogleQuestDataset(
        dataframe=train_df,
        tokenizer_name="microsoft/deberta-v3-base",
        max_len_question=24,
        max_len_answer=24,
        question_chunk_size=5,
        answer_chunk_size=4,
        question_chunk_overlap=1,
        answer_chunk_overlap=1,
        question_max_chunks=3,
        answer_max_chunks=4,
        max_title_tokens=6,
        target_columns=TARGET_COLUMNS,
        is_train=True,
        metadata_spec=metadata_spec,
    )

    sample = dataset[0]
    assert calls == [True]
    assert slow_calls == ["slow"]
    assert sample["q_input_ids"].shape[-1] == 24
    assert sample["a_input_ids"].shape[-1] == 24


def test_model_forward_and_mixed_loss_support_chunked_batches(monkeypatch):
    monkeypatch.setattr("quest.model.AutoModel.from_pretrained", lambda *_args, **_kwargs: HierarchicalDummyEncoder())

    model = DualTransformerRegressor(
        backbone_name="dummy-deberta",
        target_columns=TARGET_COLUMNS,
        dropout=0.1,
        gradient_checkpointing=True,
        metadata_vocab_sizes={"category": 4, "host": 5, "domain": 6},
        meta_numeric_dim=6,
        meta_embedding_dim=8,
        meta_hidden_dim=12,
    )
    outputs = model(
        q_input_ids=torch.randint(0, 50, (2, 3, 24)),
        q_attention_mask=torch.ones(2, 3, 24, dtype=torch.long),
        q_chunk_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32),
        a_input_ids=torch.randint(0, 50, (2, 4, 24)),
        a_attention_mask=torch.ones(2, 4, 24, dtype=torch.long),
        a_chunk_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32),
        meta_numeric=torch.randn(2, 6),
        meta_category_id=torch.tensor([1, 2]),
        meta_host_id=torch.tensor([1, 3]),
        meta_domain_id=torch.tensor([2, 4]),
    )

    assert outputs.shape == (2, len(TARGET_COLUMNS))

    labels = torch.tensor([[0.1, 0.8, 0.5, 0.6], [0.9, 0.2, 0.3, 0.4]], dtype=torch.float32)
    loss_payload = compute_mixed_loss(
        logits=outputs,
        labels=labels,
        pointwise_kind="smooth_l1",
        pointwise_weight=1.0,
        ranking_weight=0.2,
        margin=0.05,
    )
    assert loss_payload["loss"].item() >= 0.0
    assert loss_payload["pointwise_loss"].item() >= 0.0
    assert loss_payload["ranking_loss"].item() >= 0.0


def test_rank_based_distribution_matching_preserves_order():
    predictions = np.array(
        [
            [0.2, 0.8],
            [0.7, 0.1],
            [0.4, 0.6],
        ],
        dtype=np.float32,
    )
    reference = np.array(
        [
            [0.1, 0.9],
            [0.5, 0.4],
            [0.8, 0.2],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    matched = rank_based_distribution_matching(predictions, reference)

    assert matched.shape == predictions.shape
    assert np.array_equal(np.argsort(predictions[:, 0]), np.argsort(matched[:, 0]))
    assert matched.min() >= reference.min() - 1e-6
    assert matched.max() <= reference.max() + 1e-6
