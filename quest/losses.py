from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_pointwise_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pointwise_kind: str = "smooth_l1",
) -> torch.Tensor:
    kind = pointwise_kind.lower()
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels)
    probabilities = torch.sigmoid(logits)
    if kind == "mse":
        return F.mse_loss(probabilities, labels)
    return F.smooth_l1_loss(probabilities, labels)


def compute_margin_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    batch_size = probabilities.shape[0]
    if batch_size < 2:
        return probabilities.new_zeros(())

    pair_losses = []
    for target_index in range(labels.shape[1]):
        target_labels = labels[:, target_index]
        target_predictions = probabilities[:, target_index]
        for left_index in range(batch_size):
            for right_index in range(left_index + 1, batch_size):
                difference = target_labels[left_index] - target_labels[right_index]
                if torch.isclose(difference, difference.new_tensor(0.0)):
                    continue
                direction = difference.sign().unsqueeze(0)
                pair_loss = F.margin_ranking_loss(
                    target_predictions[left_index].unsqueeze(0),
                    target_predictions[right_index].unsqueeze(0),
                    direction,
                    margin=margin,
                )
                pair_losses.append(pair_loss)
    if not pair_losses:
        return probabilities.new_zeros(())
    return torch.stack(pair_losses).mean()


def compute_mixed_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pointwise_kind: str = "smooth_l1",
    pointwise_weight: float = 1.0,
    ranking_weight: float = 0.2,
    margin: float = 0.05,
) -> dict[str, torch.Tensor]:
    pointwise_loss = compute_pointwise_loss(logits, labels, pointwise_kind=pointwise_kind)
    ranking_loss = compute_margin_ranking_loss(logits, labels, margin=margin)
    total_loss = pointwise_weight * pointwise_loss + ranking_weight * ranking_loss
    return {
        "loss": total_loss,
        "pointwise_loss": pointwise_loss,
        "ranking_loss": ranking_loss,
    }
