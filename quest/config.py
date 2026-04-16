from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


QUEST_DATA_DIR_ENV = "QUEST_DATA_DIR"


@dataclass
class TrainingConfig:
    data_dir: str | None = None
    artifacts_dir: str = "artifacts"
    backbone: str = "microsoft/deberta-v3-base"
    folds: int = 5
    seeds: list[int] | None = None
    epochs: int = 3
    lr_encoder: float = 3e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_len_question: int = 256
    max_len_answer: int = 256
    max_title_tokens: int = 32
    question_chunk_size: int = 160
    answer_chunk_size: int = 160
    question_chunk_overlap: int = 32
    answer_chunk_overlap: int = 32
    question_max_chunks: int = 3
    answer_max_chunks: int = 3
    batch_size: int = 2
    grad_accum_steps: int = 4
    fp16: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 0
    dropout: float = 0.1
    device: str = "auto"
    use_metadata: bool = True
    meta_embedding_dim: int = 16
    meta_hidden_dim: int = 64
    pointwise_loss: str = "smooth_l1"
    pointwise_weight: float = 1.0
    ranking_weight: float = 0.2
    ranking_margin: float = 0.05
    distribution_matching: bool = True
    debug: bool = False
    debug_folds: int = 1
    debug_epochs: int = 1
    debug_batches: int = 20
    train_row_limit: int | None = None
    test_row_limit: int | None = None

    def __post_init__(self) -> None:
        if self.seeds is None:
            self.seeds = [42, 2024]

    def resolved_data_dir(self, override_data_dir: str | Path | None = None) -> Path:
        candidate = override_data_dir or os.environ.get(QUEST_DATA_DIR_ENV) or self.data_dir
        if not candidate:
            raise ValueError(
                "A data directory is required. Set it in the config, pass --data-dir, "
                f"or export {QUEST_DATA_DIR_ENV}."
            )
        return Path(candidate)

    def resolved_artifacts_dir(self) -> Path:
        return Path(self.artifacts_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return TrainingConfig(**payload)
