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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from quest.config import TrainingConfig
from quest.data import (
    GoogleQuestDataset,
    apply_metadata_spec,
    apply_row_limits,
    build_group_folds,
    load_competition_frames,
    prepare_metadata_spec,
    quest_collate_fn,
)
from quest.losses import compute_mixed_loss
from quest.metrics import mean_column_spearman
from quest.model import DualTransformerRegressor
from quest.postprocess import rank_based_distribution_matching


def build_memory_fallbacks(config: TrainingConfig) -> list[dict[str, int]]:
    return [
        {},
        {"grad_accum_steps": max(config.grad_accum_steps, 8)},
        {
            "grad_accum_steps": max(config.grad_accum_steps, 8),
            "answer_max_chunks": max(1, config.answer_max_chunks - 1),
        },
        {
            "grad_accum_steps": max(config.grad_accum_steps, 8),
            "answer_max_chunks": max(1, config.answer_max_chunks - 1),
            "question_max_chunks": max(1, config.question_max_chunks - 1),
        },
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
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def prepare_frames(
    config: TrainingConfig,
    data_dir: str | Path | None = None,
):
    train_df, test_df, target_columns, sample_df = load_competition_frames(config.resolved_data_dir(data_dir))
    train_df, test_df, sample_df = apply_row_limits(
        train_df,
        test_df,
        sample_df,
        train_row_limit=config.train_row_limit,
        test_row_limit=config.test_row_limit,
    )
    metadata_spec = prepare_metadata_spec(train_df, test_df) if config.use_metadata else None
    if metadata_spec is not None:
        train_df = apply_metadata_spec(train_df, metadata_spec)
        test_df = apply_metadata_spec(test_df, metadata_spec)
    return train_df, test_df, target_columns, sample_df, metadata_spec


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
        collate_fn=quest_collate_fn,
    )


def build_dataset(
    config: TrainingConfig,
    dataframe: pd.DataFrame,
    target_columns: list[str],
    is_train: bool,
    metadata_spec,
):
    return GoogleQuestDataset(
        dataframe=dataframe,
        tokenizer_name=config.backbone,
        max_len_question=config.max_len_question,
        max_len_answer=config.max_len_answer,
        question_chunk_size=config.question_chunk_size,
        answer_chunk_size=config.answer_chunk_size,
        question_chunk_overlap=config.question_chunk_overlap,
        answer_chunk_overlap=config.answer_chunk_overlap,
        question_max_chunks=config.question_max_chunks,
        answer_max_chunks=config.answer_max_chunks,
        max_title_tokens=config.max_title_tokens,
        target_columns=target_columns,
        is_train=is_train,
        metadata_spec=metadata_spec,
    )


def create_dataloaders(
    config: TrainingConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_columns: list[str],
    metadata_spec,
):
    train_loader = _build_loader(
        config,
        build_dataset(config, train_df, target_columns, is_train=True, metadata_spec=metadata_spec),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = _build_loader(
        config,
        build_dataset(config, valid_df, target_columns, is_train=True, metadata_spec=metadata_spec),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader


def create_validation_loader(
    config: TrainingConfig,
    valid_df: pd.DataFrame,
    target_columns: list[str],
    metadata_spec,
):
    return _build_loader(
        config,
        build_dataset(config, valid_df, target_columns, is_train=True, metadata_spec=metadata_spec),
        batch_size=config.batch_size,
        shuffle=False,
    )


def create_test_loader(
    config: TrainingConfig,
    test_df: pd.DataFrame,
    target_columns: list[str],
    metadata_spec,
):
    return _build_loader(
        config,
        build_dataset(config, test_df, target_columns, is_train=False, metadata_spec=metadata_spec),
        batch_size=config.batch_size,
        shuffle=False,
    )


def build_model(
    config: TrainingConfig,
    target_columns: list[str],
    metadata_spec,
) -> DualTransformerRegressor:
    metadata_vocab_sizes = metadata_spec.vocab_sizes() if metadata_spec is not None else None
    meta_numeric_dim = len(metadata_spec.numeric_columns) if metadata_spec is not None else 0
    return DualTransformerRegressor(
        backbone_name=config.backbone,
        target_columns=target_columns,
        dropout=config.dropout,
        gradient_checkpointing=config.gradient_checkpointing,
        metadata_vocab_sizes=metadata_vocab_sizes,
        meta_numeric_dim=meta_numeric_dim,
        meta_embedding_dim=config.meta_embedding_dim,
        meta_hidden_dim=config.meta_hidden_dim,
    )


def create_optimizer(model: DualTransformerRegressor, config: TrainingConfig):
    encoder_parameters = list(model.question_encoder.parameters()) + list(model.answer_encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids]
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


def _forward_model(model: DualTransformerRegressor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        q_input_ids=batch["q_input_ids"],
        q_attention_mask=batch["q_attention_mask"],
        q_chunk_mask=batch.get("q_chunk_mask"),
        a_input_ids=batch["a_input_ids"],
        a_attention_mask=batch["a_attention_mask"],
        a_chunk_mask=batch.get("a_chunk_mask"),
        meta_numeric=batch.get("meta_numeric"),
        meta_category_id=batch.get("meta_category_id"),
        meta_host_id=batch.get("meta_host_id"),
        meta_domain_id=batch.get("meta_domain_id"),
    )


def evaluate_model(
    model: DualTransformerRegressor,
    data_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
):
    model.eval()
    predictions = []
    labels = []
    qa_ids = []
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            logits = _forward_model(model, batch)
            loss_payload = compute_mixed_loss(
                logits=logits,
                labels=batch["labels"],
                pointwise_kind=config.pointwise_loss,
                pointwise_weight=config.pointwise_weight,
                ranking_weight=config.ranking_weight,
                margin=config.ranking_margin,
            )
            probabilities = torch.sigmoid(logits)
            losses.append(float(loss_payload["loss"].item()))
            predictions.append(probabilities.detach().cpu().numpy())
            labels.append(batch["labels"].detach().cpu().numpy())
            qa_ids.append(batch["qa_id"].detach().cpu().numpy())

    predictions_array = np.concatenate(predictions, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    qa_ids_array = np.concatenate(qa_ids, axis=0)
    score = mean_column_spearman(labels_array, predictions_array)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "score": score,
        "predictions": predictions_array,
        "labels": labels_array,
        "qa_ids": qa_ids_array,
    }


def build_checkpoint_payload(
    model: DualTransformerRegressor,
    config: TrainingConfig,
    target_columns: list[str],
    metadata_spec,
    seed: int,
    fold_index: int,
    score: float,
):
    return {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "seed": seed,
        "fold": fold_index,
        "score": score,
        "target_columns": target_columns,
        "model_kwargs": {
            "backbone_name": config.backbone,
            "target_columns": target_columns,
            "dropout": config.dropout,
            "gradient_checkpointing": False,
            "metadata_vocab_sizes": metadata_spec.vocab_sizes() if metadata_spec is not None else None,
            "meta_numeric_dim": len(metadata_spec.numeric_columns) if metadata_spec is not None else 0,
            "meta_embedding_dim": config.meta_embedding_dim,
            "meta_hidden_dim": config.meta_hidden_dim,
        },
    }


def train_single_fold(
    config: TrainingConfig,
    seed: int,
    fold_index: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target_columns: list[str],
    metadata_spec,
    checkpoint_path: Path,
):
    device = resolve_device(config)
    train_loader, valid_loader = create_dataloaders(config, train_df, valid_df, target_columns, metadata_spec)
    model = build_model(config, target_columns, metadata_spec).to(device)
    optimizer = create_optimizer(model, config)
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
    best_qa_ids = None
    best_state = None

    max_batches = config.debug_batches if config.debug else None
    for _epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"seed={seed} fold={fold_index}", leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            with _autocast_context(device, config.fp16):
                logits = _forward_model(model, batch)
                loss_payload = compute_mixed_loss(
                    logits=logits,
                    labels=batch["labels"],
                    pointwise_kind=config.pointwise_loss,
                    pointwise_weight=config.pointwise_weight,
                    ranking_weight=config.ranking_weight,
                    margin=config.ranking_margin,
                )
                loss = loss_payload["loss"] / config.grad_accum_steps

            scaler.scale(loss).backward()
            if batch_index % config.grad_accum_steps == 0 or batch_index == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if max_batches and batch_index >= max_batches:
                break

        evaluation = evaluate_model(model, valid_loader, device, config)
        if evaluation["score"] > best_score:
            best_score = evaluation["score"]
            best_predictions = evaluation["predictions"]
            best_qa_ids = evaluation["qa_ids"]
            best_state = build_checkpoint_payload(
                model=model,
                config=config,
                target_columns=target_columns,
                metadata_spec=metadata_spec,
                seed=seed,
                fold_index=fold_index,
                score=best_score,
            )

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
    metadata_spec,
    checkpoint_path: Path,
):
    for fallback in build_memory_fallbacks(config):
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
                metadata_spec=metadata_spec,
                checkpoint_path=checkpoint_path,
            )
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    raise RuntimeError("All memory fallback configurations failed.")


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DualTransformerRegressor(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["target_columns"], checkpoint.get("config", {})


def recover_fold_result_from_checkpoint(
    runtime_config: TrainingConfig,
    checkpoint_path: Path,
    valid_df: pd.DataFrame,
    metadata_spec,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = TrainingConfig(**checkpoint["config"])
    target_columns = checkpoint["target_columns"]
    device = resolve_device(runtime_config)
    valid_loader = create_validation_loader(checkpoint_config, valid_df, target_columns, metadata_spec)
    model, _, _ = load_checkpoint_model(checkpoint_path, device)
    evaluation = evaluate_model(model, valid_loader, device, checkpoint_config)
    return {
        "checkpoint_path": str(checkpoint_path),
        "score": checkpoint.get("score", evaluation["score"]),
        "predictions": evaluation["predictions"],
        "qa_ids": evaluation["qa_ids"],
    }


def train_pipeline(
    config: TrainingConfig,
    data_dir: str | Path | None = None,
):
    config = apply_runtime_overrides(config)
    directories = make_artifact_dirs(config.resolved_artifacts_dir())
    train_df, _test_df, target_columns, _sample_df, metadata_spec = prepare_frames(config, data_dir=data_dir)
    folds = build_group_folds(train_df, config.folds)
    if config.debug:
        folds = folds[: config.debug_folds]

    checkpoint_paths: list[str] = []
    metrics_summary: dict[str, object] = {"seeds": {}, "checkpoint_paths": checkpoint_paths}
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
                        metadata_spec=metadata_spec,
                    )
                except Exception:
                    result = train_single_fold_with_fallback(
                        config,
                        seed=seed,
                        fold_index=fold_index,
                        train_df=fold_train_df,
                        valid_df=fold_valid_df,
                        target_columns=target_columns,
                        metadata_spec=metadata_spec,
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
                    metadata_spec=metadata_spec,
                    checkpoint_path=checkpoint_path,
                )

            checkpoint_paths.append(result["checkpoint_path"])
            per_seed_scores.append(result["score"])
            oof_predictions[valid_indices] = result["predictions"]

        labels = train_df[target_columns].to_numpy(dtype=np.float32)
        oof_score = mean_column_spearman(labels, oof_predictions)
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
        all_seed_predictions.append(pd.read_csv(oof_path)[target_columns].to_numpy(dtype=np.float32))

    labels = train_df[target_columns].to_numpy(dtype=np.float32)
    averaged_oof = np.mean(np.stack(all_seed_predictions, axis=0), axis=0)
    metrics_summary["ensemble_oof_score"] = mean_column_spearman(labels, averaged_oof)
    if config.distribution_matching:
        matched_oof = rank_based_distribution_matching(averaged_oof, labels)
        metrics_summary["distribution_matched_oof_score"] = mean_column_spearman(labels, matched_oof)

    metrics_path = directories["metrics"] / "cv_summary.json"
    metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    return {
        "fold_count": fold_count,
        "checkpoint_paths": checkpoint_paths,
        "metrics_path": str(metrics_path),
        "ensemble_oof_score": metrics_summary["ensemble_oof_score"],
    }


def predict_pipeline(
    config: TrainingConfig,
    checkpoint_dir: str | Path,
    output_path: str | Path,
    data_dir: str | Path | None = None,
):
    device = resolve_device(config)
    train_df, test_df, _target_columns, sample_df, metadata_spec = prepare_frames(config, data_dir=data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("model_seed*_fold*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    blended_predictions = []
    target_columns = None
    with torch.no_grad():
        for checkpoint_path in checkpoint_paths:
            model, checkpoint_targets, checkpoint_config_payload = load_checkpoint_model(checkpoint_path, device)
            checkpoint_config = TrainingConfig(**checkpoint_config_payload) if checkpoint_config_payload else config
            if target_columns is None:
                target_columns = checkpoint_targets
            elif target_columns != checkpoint_targets:
                raise ValueError("Checkpoint target columns do not match across ensemble members.")

            test_loader = create_test_loader(checkpoint_config, test_df, checkpoint_targets, metadata_spec)
            per_model_predictions = []
            for batch in test_loader:
                batch = move_batch_to_device(batch, device)
                logits = _forward_model(model, batch)
                probabilities = torch.sigmoid(logits)
                per_model_predictions.append(probabilities.cpu().numpy())
            blended_predictions.append(np.concatenate(per_model_predictions, axis=0))

    averaged_predictions = np.mean(np.stack(blended_predictions, axis=0), axis=0)
    if config.distribution_matching:
        reference_labels = train_df[target_columns].to_numpy(dtype=np.float32)
        averaged_predictions = rank_based_distribution_matching(averaged_predictions, reference_labels)

    submission = sample_df.copy()
    submission[target_columns] = averaged_predictions
    output_path = Path(output_path)
    submission.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "checkpoint_count": len(checkpoint_paths),
    }
