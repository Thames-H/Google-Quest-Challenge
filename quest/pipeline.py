from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from quest.config import TrainingConfig
from quest.data import GoogleQuestDataset, apply_row_limits, build_group_folds, load_competition_frames
from quest.metrics import mean_column_spearman
from quest.model import DualTransformerRegressor


MEMORY_FALLBACKS = [
    {},
    {"grad_accum_steps": 16},
    {"grad_accum_steps": 16, "max_len_answer": 224},
    {"grad_accum_steps": 16, "max_len_answer": 224, "max_len_question": 224},
]


def resolve_device(config: TrainingConfig) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_artifact_dirs(artifacts_dir: Path) -> dict[str, Path]:
    directories = {
        "root": artifacts_dir,
        "checkpoints": artifacts_dir / "checkpoints",
        "oof": artifacts_dir / "oof",
        "metrics": artifacts_dir / "metrics",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def apply_runtime_overrides(config: TrainingConfig) -> TrainingConfig:
    updated = deepcopy(config)
    if updated.debug:
        updated.epochs = updated.debug_epochs
    return updated


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _build_loader(
    config: TrainingConfig,
    dataset: GoogleQuestDataset,
    batch_size: int,
    shuffle: bool,
):
    pin_memory = resolve_device(config).type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )


def create_dataloaders(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_columns: list[str],
):
    train_dataset = GoogleQuestDataset(
        dataframe=train_df,
        tokenizer_name=config.backbone,
        max_len_question=config.max_len_question,
        max_len_answer=config.max_len_answer,
        target_columns=target_columns,
        is_train=True,
    )
    valid_dataset = GoogleQuestDataset(
        dataframe=valid_df,
        tokenizer_name=config.backbone,
        max_len_question=config.max_len_question,
        max_len_answer=config.max_len_answer,
        target_columns=target_columns,
        is_train=True,
    )
    train_loader = _build_loader(config, train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = _build_loader(config, valid_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, valid_loader


def create_test_loader(config: TrainingConfig, test_df: pd.DataFrame, target_columns: list[str]):
    dataset = GoogleQuestDataset(
        dataframe=test_df,
        tokenizer_name=config.backbone,
        max_len_question=config.max_len_question,
        max_len_answer=config.max_len_answer,
        target_columns=target_columns,
        is_train=False,
    )
    return _build_loader(config, dataset, batch_size=config.batch_size, shuffle=False)


def create_validation_loader(
    config: TrainingConfig,
    valid_df: pd.DataFrame,
    target_columns: list[str],
):
    dataset = GoogleQuestDataset(
        dataframe=valid_df,
        tokenizer_name=config.backbone,
        max_len_question=config.max_len_question,
        max_len_answer=config.max_len_answer,
        target_columns=target_columns,
        is_train=True,
    )
    return _build_loader(config, dataset, batch_size=config.batch_size, shuffle=False)


def create_optimizer(model: DualTransformerRegressor, config: TrainingConfig):
    encoder_parameters = list(model.question_encoder.parameters()) + list(model.answer_encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": config.lr_encoder,
                "weight_decay": config.weight_decay,
            },
            {
                "params": head_parameters,
                "lr": config.lr_head,
                "weight_decay": config.weight_decay,
            },
        ]
    )


def evaluate_model(
    model: DualTransformerRegressor,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
):
    model.eval()
    predictions = []
    labels = []
    qa_ids = []
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            logits = model(
                q_input_ids=batch["q_input_ids"],
                q_attention_mask=batch["q_attention_mask"],
                a_input_ids=batch["a_input_ids"],
                a_attention_mask=batch["a_attention_mask"],
            )
            loss = loss_fn(logits, batch["labels"])
            probs = torch.sigmoid(logits)

            losses.append(float(loss.item()))
            predictions.append(probs.detach().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
            qa_ids.append(batch["qa_id"].detach().cpu().numpy())

    predictions_array = np.concatenate(predictions, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    qa_ids_array = np.concatenate(qa_ids, axis=0)
    score = mean_column_spearman(labels_array, predictions_array)
    return {
        "loss": float(np.mean(losses)),
        "score": score,
        "predictions": predictions_array,
        "labels": labels_array,
        "qa_ids": qa_ids_array,
    }


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_single_fold(
    config: TrainingConfig,
    seed: int,
    fold_index: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_columns: list[str],
    checkpoint_path: Path,
):
    device = resolve_device(config)
    train_loader, valid_loader = create_dataloaders(config, train_df, valid_df, target_columns)
    model = DualTransformerRegressor(
        backbone_name=config.backbone,
        num_targets=len(target_columns),
        dropout=config.dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)
    optimizer = create_optimizer(model, config)
    loss_fn = nn.BCEWithLogitsLoss()
    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum_steps)
    total_steps = max(steps_per_epoch * config.epochs, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.fp16 and device.type == "cuda")

    best_score = float("-inf")
    best_predictions = None
    best_state = None
    best_qa_ids = None

    max_batches = config.debug_batches if config.debug else None
    progress_desc = f"seed={seed} fold={fold_index}"
    for _epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=progress_desc, leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            with _autocast_context(device, config.fp16):
                logits = model(
                    q_input_ids=batch["q_input_ids"],
                    q_attention_mask=batch["q_attention_mask"],
                    a_input_ids=batch["a_input_ids"],
                    a_attention_mask=batch["a_attention_mask"],
                )
                loss = loss_fn(logits, batch["labels"])
                loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()

            if batch_index % config.grad_accum_steps == 0 or batch_index == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if max_batches and batch_index >= max_batches:
                break

        evaluation = evaluate_model(model, valid_loader, device, loss_fn)
        if evaluation["score"] > best_score:
            best_score = evaluation["score"]
            best_predictions = evaluation["predictions"]
            best_qa_ids = evaluation["qa_ids"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "seed": seed,
                "fold": fold_index,
                "score": best_score,
                "target_columns": target_columns,
            }

    if best_state is None or best_predictions is None or best_qa_ids is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    torch.save(best_state, checkpoint_path)
    return {
        "checkpoint_path": str(checkpoint_path),
        "score": best_score,
        "predictions": best_predictions,
        "qa_ids": best_qa_ids,
    }


def train_single_fold_with_fallback(
    config: TrainingConfig,
    seed: int,
    fold_index: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_columns: list[str],
    checkpoint_path: Path,
):
    for fallback in MEMORY_FALLBACKS:
        effective_config = deepcopy(config)
        for key, value in fallback.items():
            setattr(effective_config, key, value)
        try:
            return train_single_fold(
                effective_config,
                seed=seed,
                fold_index=fold_index,
                train_df=train_df,
                valid_df=valid_df,
                target_columns=target_columns,
                checkpoint_path=checkpoint_path,
            )
        except RuntimeError as error:
            message = str(error).lower()
            if "out of memory" not in message:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    raise RuntimeError("All memory fallback configurations failed.")


def recover_fold_result_from_checkpoint(
    runtime_config: TrainingConfig,
    checkpoint_path: Path,
    valid_df: pd.DataFrame,
    target_columns: list[str],
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = TrainingConfig(**checkpoint["config"])
    device = resolve_device(runtime_config)
    loss_fn = nn.BCEWithLogitsLoss()
    valid_loader = create_validation_loader(checkpoint_config, valid_df, target_columns)
    model, checkpoint_targets = load_checkpoint_model(checkpoint_path, device)
    if checkpoint_targets != target_columns:
        raise ValueError("Checkpoint target columns do not match current target columns.")
    evaluation = evaluate_model(model, valid_loader, device, loss_fn)
    return {
        "checkpoint_path": str(checkpoint_path),
        "score": checkpoint.get("score", evaluation["score"]),
        "predictions": evaluation["predictions"],
        "qa_ids": evaluation["qa_ids"],
    }


def train_pipeline(config: TrainingConfig):
    config = apply_runtime_overrides(config)
    directories = make_artifact_dirs(config.resolved_artifacts_dir())
    train_df, _test_df, target_columns, _sample_df = load_competition_frames(config.resolved_data_dir())
    train_df, _test_df, _sample_df = apply_row_limits(
        train_df,
        _test_df,
        _sample_df,
        train_row_limit=config.train_row_limit,
        test_row_limit=config.test_row_limit,
    )
    folds = build_group_folds(train_df, config.folds, group_column="question_body")
    if config.debug:
        folds = folds[: config.debug_folds]
    checkpoint_paths = []
    metrics_summary = {"seeds": {}, "checkpoint_paths": checkpoint_paths}
    fold_count = 0

    for seed in config.seeds:
        set_seed(seed)
        oof_predictions = np.zeros((len(train_df), len(target_columns)), dtype=np.float32)
        per_seed_scores = []

        for fold_index, (train_indices, valid_indices) in enumerate(folds):
            fold_count += 1
            fold_train_df = train_df.iloc[train_indices].reset_index(drop=True)
            fold_valid_df = train_df.iloc[valid_indices].reset_index(drop=True)
            checkpoint_path = directories["checkpoints"] / f"model_seed{seed}_fold{fold_index}.pt"

            if checkpoint_path.exists():
                try:
                    result = recover_fold_result_from_checkpoint(
                        runtime_config=config,
                        checkpoint_path=checkpoint_path,
                        valid_df=fold_valid_df,
                        target_columns=target_columns,
                    )
                except Exception:
                    result = train_single_fold_with_fallback(
                        config,
                        seed=seed,
                        fold_index=fold_index,
                        train_df=fold_train_df,
                        valid_df=fold_valid_df,
                        target_columns=target_columns,
                        checkpoint_path=checkpoint_path,
                    )
            else:
                result = train_single_fold_with_fallback(
                    config,
                    seed=seed,
                    fold_index=fold_index,
                    train_df=fold_train_df,
                    valid_df=fold_valid_df,
                    target_columns=target_columns,
                    checkpoint_path=checkpoint_path,
                )
            checkpoint_paths.append(result["checkpoint_path"])
            per_seed_scores.append(result["score"])
            oof_predictions[valid_indices] = result["predictions"]

        oof_score = mean_column_spearman(
            train_df[target_columns].to_numpy(dtype=np.float32),
            oof_predictions,
        )
        oof_frame = pd.DataFrame(oof_predictions, columns=target_columns)
        oof_frame.insert(0, "qa_id", train_df["qa_id"].values)
        oof_path = directories["oof"] / f"oof_seed{seed}.csv"
        oof_frame.to_csv(oof_path, index=False)
        metrics_summary["seeds"][str(seed)] = {
            "fold_scores": per_seed_scores,
            "oof_score": oof_score,
            "oof_path": str(oof_path),
        }

    all_seed_predictions = []
    for seed in config.seeds:
        oof_path = directories["oof"] / f"oof_seed{seed}.csv"
        all_seed_predictions.append(
            pd.read_csv(oof_path)[target_columns].to_numpy(dtype=np.float32)
        )
    averaged_oof = np.mean(np.stack(all_seed_predictions, axis=0), axis=0)
    metrics_summary["ensemble_oof_score"] = mean_column_spearman(
        train_df[target_columns].to_numpy(dtype=np.float32),
        averaged_oof,
    )
    metrics_path = directories["metrics"] / "cv_summary.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    return {
        "fold_count": fold_count,
        "checkpoint_paths": checkpoint_paths,
        "metrics_path": str(metrics_path),
        "ensemble_oof_score": metrics_summary["ensemble_oof_score"],
    }


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_payload = checkpoint["config"]
    model = DualTransformerRegressor(
        backbone_name=config_payload["backbone"],
        num_targets=len(checkpoint["target_columns"]),
        dropout=config_payload.get("dropout", 0.1),
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["target_columns"]


def predict_pipeline(
    config: TrainingConfig,
    checkpoint_dir: str | Path,
    output_path: str | Path,
):
    device = resolve_device(config)
    train_df, test_df, target_columns, sample_df = load_competition_frames(config.resolved_data_dir())
    _train_df, test_df, sample_df = apply_row_limits(
        train_df,
        test_df,
        sample_df,
        train_row_limit=config.train_row_limit,
        test_row_limit=config.test_row_limit,
    )
    test_loader = create_test_loader(config, test_df, target_columns)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("model_seed*_fold*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    blended_predictions = []
    with torch.no_grad():
        for checkpoint_path in checkpoint_paths:
            model, checkpoint_targets = load_checkpoint_model(checkpoint_path, device)
            if checkpoint_targets != target_columns:
                raise ValueError("Checkpoint target columns do not match sample submission.")

            per_model_predictions = []
            for batch in test_loader:
                batch = move_batch_to_device(batch, device)
                logits = model(
                    q_input_ids=batch["q_input_ids"],
                    q_attention_mask=batch["q_attention_mask"],
                    a_input_ids=batch["a_input_ids"],
                    a_attention_mask=batch["a_attention_mask"],
                )
                probs = torch.sigmoid(logits)
                per_model_predictions.append(probs.cpu().numpy())
            blended_predictions.append(np.concatenate(per_model_predictions, axis=0))

    averaged_predictions = np.mean(np.stack(blended_predictions, axis=0), axis=0)
    submission = sample_df.copy()
    submission[target_columns] = averaged_predictions
    output_path = Path(output_path)
    submission.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "checkpoint_count": len(checkpoint_paths),
    }
