from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from quest.config import load_config
from quest.data import GoogleQuestDataset, build_group_folds, load_competition_frames
from quest.model import DualTransformerRegressor, masked_mean_pool
from quest.pipeline import predict_pipeline, train_pipeline


TARGET_COLUMNS = [f"target_{index}" for index in range(30)]


class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(
        self,
        text,
        text_pair,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"

        base = len(text.split()) + len(text_pair.split())
        tokens = torch.arange(max_length, dtype=torch.long) + base
        attention_mask = torch.ones(max_length, dtype=torch.long)
        return {
            "input_ids": tokens.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }


class DummyEncoder(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = torch.nn.Embedding(4096, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, hidden_size)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids % 4096)
        hidden = self.projection(hidden)
        return type("Output", (), {"last_hidden_state": hidden})()


def build_dummy_dataframe(rows):
    payload = []
    for idx in range(rows):
        row = {
            "qa_id": idx + 1,
            "question_title": f"title {idx}",
            "question_body": f"body group {idx // 2}",
            "question_user_name": f"asker_{idx}",
            "question_user_page": f"https://example.com/question/{idx}",
            "answer": f"answer {idx}",
            "answer_user_name": f"answerer_{idx}",
            "answer_user_page": f"https://example.com/answer/{idx}",
            "url": f"https://example.com/{idx}",
            "category": "TECHNOLOGY",
            "host": "stackoverflow.com",
        }
        for target_index, column in enumerate(TARGET_COLUMNS):
            row[column] = ((idx + target_index) % 7) / 6.0
        payload.append(row)
    return pd.DataFrame(payload)


def write_competition_files(base_dir: Path):
    train_df = build_dummy_dataframe(8)
    test_df = build_dummy_dataframe(4).drop(columns=TARGET_COLUMNS)
    sample_df = pd.DataFrame({"qa_id": test_df["qa_id"], **{column: 0.0 for column in TARGET_COLUMNS}})

    train_path = base_dir / "train.csv"
    test_path = base_dir / "test.csv"
    sample_path = base_dir / "sample_submission.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    sample_df.to_csv(sample_path, index=False)
    return train_path, test_path, sample_path


def write_config(tmp_path: Path, data_dir: Path):
    config = {
        "data_dir": str(data_dir),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "backbone": "dummy-backbone",
        "folds": 2,
        "seeds": [7],
        "epochs": 1,
        "lr_encoder": 1e-3,
        "lr_head": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.0,
        "max_len_question": 16,
        "max_len_answer": 16,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "fp16": False,
        "gradient_checkpointing": True,
        "num_workers": 0,
        "dropout": 0.1,
        "device": "cpu",
        "debug": False,
        "debug_folds": 1,
        "debug_epochs": 1,
        "debug_batches": 2,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_load_competition_frames_and_group_folds(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_competition_files(data_dir)

    train_df, test_df, target_columns, sample_df = load_competition_frames(data_dir)
    assert len(train_df) == 8
    assert len(test_df) == 4
    assert len(target_columns) == 30
    assert list(sample_df.columns) == ["qa_id", *TARGET_COLUMNS]

    folds = build_group_folds(train_df, folds=2, group_column="question_body")
    assert len(folds) == 2

    validation_indices = set()
    for train_indices, valid_indices in folds:
        assert set(train_indices).isdisjoint(set(valid_indices))
        validation_indices.update(valid_indices)

    assert validation_indices == set(range(len(train_df)))


def test_dataset_returns_dual_inputs_and_labels(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_competition_files(data_dir)
    train_df, _, target_columns, _ = load_competition_frames(data_dir)

    monkeypatch.setattr("quest.data.AutoTokenizer", DummyTokenizer)

    dataset = GoogleQuestDataset(
        dataframe=train_df,
        tokenizer_name="dummy-backbone",
        max_len_question=16,
        max_len_answer=12,
        target_columns=target_columns,
        is_train=True,
    )
    item = dataset[0]

    assert set(item.keys()) == {
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "labels",
        "qa_id",
    }
    assert item["q_input_ids"].shape == (16,)
    assert item["a_input_ids"].shape == (12,)
    assert item["labels"].shape == (30,)
    assert item["qa_id"].item() == 1


def test_masked_mean_pool_and_model_forward(monkeypatch):
    hidden_states = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = masked_mean_pool(hidden_states, attention_mask)
    assert torch.allclose(pooled[0], torch.tensor([3.0, 5.0]))
    assert torch.allclose(pooled[1], torch.tensor([2.0, 4.0]))

    monkeypatch.setattr("quest.model.AutoModel.from_pretrained", lambda *_args, **_kwargs: DummyEncoder())
    model = DualTransformerRegressor(
        backbone_name="dummy-backbone",
        num_targets=30,
        dropout=0.1,
        gradient_checkpointing=True,
    )

    batch = {
        "q_input_ids": torch.randint(0, 128, (2, 10)),
        "q_attention_mask": torch.ones(2, 10, dtype=torch.long),
        "a_input_ids": torch.randint(0, 128, (2, 10)),
        "a_attention_mask": torch.ones(2, 10, dtype=torch.long),
    }
    outputs = model(**batch)
    assert outputs.shape == (2, 30)
    assert model.question_encoder.gradient_checkpointing_enabled is True
    assert model.answer_encoder.gradient_checkpointing_enabled is True


def test_train_and_predict_pipeline_with_dummy_backbone(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_competition_files(data_dir)
    config_path = write_config(tmp_path, data_dir)

    monkeypatch.setattr("quest.data.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("quest.model.AutoModel.from_pretrained", lambda *_args, **_kwargs: DummyEncoder())

    config = load_config(config_path)
    train_summary = train_pipeline(config)

    assert train_summary["fold_count"] == 2
    assert len(train_summary["checkpoint_paths"]) == 2
    for checkpoint_path in train_summary["checkpoint_paths"]:
        assert Path(checkpoint_path).exists()

    output_path = tmp_path / "submission.csv"
    predict_pipeline(config, checkpoint_dir=Path(config.artifacts_dir) / "checkpoints", output_path=output_path)

    submission = pd.read_csv(output_path)
    assert submission.shape == (4, 31)
    assert list(submission.columns) == ["qa_id", *TARGET_COLUMNS]
    assert submission.iloc[:, 1:].ge(0.0).all().all()
    assert submission.iloc[:, 1:].le(1.0).all().all()


def test_train_pipeline_reuses_existing_checkpoints_without_retraining(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_competition_files(data_dir)
    config_path = write_config(tmp_path, data_dir)

    monkeypatch.setattr("quest.data.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("quest.model.AutoModel.from_pretrained", lambda *_args, **_kwargs: DummyEncoder())

    config = load_config(config_path)
    first_summary = train_pipeline(config)
    assert len(first_summary["checkpoint_paths"]) == 2

    artifacts_dir = Path(config.artifacts_dir)
    metrics_path = artifacts_dir / "metrics" / "cv_summary.json"
    oof_path = artifacts_dir / "oof" / "oof_seed7.csv"
    metrics_path.unlink()
    oof_path.unlink()

    def fail_if_retraining(*_args, **_kwargs):
        raise AssertionError("train_single_fold_with_fallback should not run when checkpoints already exist")

    monkeypatch.setattr("quest.pipeline.train_single_fold_with_fallback", fail_if_retraining)
    second_summary = train_pipeline(config)

    assert Path(second_summary["metrics_path"]).exists()
    assert oof_path.exists()
    assert second_summary["checkpoint_paths"] == first_summary["checkpoint_paths"]


def test_train_and_predict_respect_row_limits(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_competition_files(data_dir)
    config_path = write_config(tmp_path, data_dir)

    monkeypatch.setattr("quest.data.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("quest.model.AutoModel.from_pretrained", lambda *_args, **_kwargs: DummyEncoder())

    config = load_config(config_path)
    config.train_row_limit = 4
    config.test_row_limit = 3

    train_summary = train_pipeline(config)
    assert train_summary["fold_count"] == 2

    oof_path = Path(config.artifacts_dir) / "oof" / "oof_seed7.csv"
    oof_frame = pd.read_csv(oof_path)
    assert oof_frame.shape == (4, 31)

    output_path = tmp_path / "submission_limited.csv"
    predict_pipeline(config, checkpoint_dir=Path(config.artifacts_dir) / "checkpoints", output_path=output_path)
    submission = pd.read_csv(output_path)
    assert submission.shape == (3, 31)


def test_load_config_reads_expected_defaults(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_competition_files(data_dir)
    config_path = write_config(tmp_path, data_dir)

    config = load_config(config_path)
    assert config.backbone == "dummy-backbone"
    assert config.folds == 2
    assert config.seeds == [7]
    assert config.gradient_checkpointing is True


def test_resolved_model_source_prefers_cli_then_env_then_config(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_competition_files(data_dir)
    config_path = write_config(tmp_path, data_dir)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["model_dir"] = str(tmp_path / "config-model")
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_config(config_path)
    monkeypatch.setenv("QUEST_MODEL_DIR", str(tmp_path / "env-model"))

    assert config.resolved_model_source() == (tmp_path / "env-model")
    assert config.resolved_model_source(tmp_path / "cli-model") == (tmp_path / "cli-model")
