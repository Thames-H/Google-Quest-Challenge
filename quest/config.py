from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainingConfig:
    data_dir: str
    artifacts_dir: str = "artifacts"
    backbone: str = "roberta-base"
    folds: int = 5
    seeds: list[int] | None = None
    epochs: int = 3
    lr_encoder: float = 3e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_len_question: int = 256
    max_len_answer: int = 256
    batch_size: int = 1
    grad_accum_steps: int = 8
    fp16: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 0
    dropout: float = 0.1
    device: str = "auto"
    debug: bool = False
    debug_folds: int = 1
    debug_epochs: int = 1
    debug_batches: int = 20
    train_row_limit: int | None = None
    test_row_limit: int | None = None

    def __post_init__(self) -> None:
        if self.seeds is None:
            self.seeds = [42, 2024]

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir)

    def resolved_artifacts_dir(self) -> Path:
        return Path(self.artifacts_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: str | Path) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return TrainingConfig(**payload)
