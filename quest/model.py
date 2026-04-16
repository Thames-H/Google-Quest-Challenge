from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-6)
    return summed / counts


class TargetHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DualTransformerRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_targets: int,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.question_encoder = AutoModel.from_pretrained(backbone_name)
        self.answer_encoder = AutoModel.from_pretrained(backbone_name)
        if gradient_checkpointing:
            if hasattr(self.question_encoder, "gradient_checkpointing_enable"):
                self.question_encoder.gradient_checkpointing_enable()
            if hasattr(self.answer_encoder, "gradient_checkpointing_enable"):
                self.answer_encoder.gradient_checkpointing_enable()

        hidden_size = self.question_encoder.config.hidden_size
        merged_size = hidden_size * 2
        self.target_heads = nn.ModuleList(
            [TargetHead(merged_size, hidden_dim=64, dropout=dropout) for _ in range(num_targets)]
        )

    def forward(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        question_outputs = self.question_encoder(
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
        )
        answer_outputs = self.answer_encoder(
            input_ids=a_input_ids,
            attention_mask=a_attention_mask,
        )

        question_embedding = masked_mean_pool(
            question_outputs.last_hidden_state,
            q_attention_mask,
        )
        answer_embedding = masked_mean_pool(
            answer_outputs.last_hidden_state,
            a_attention_mask,
        )
        merged = torch.cat([question_embedding, answer_embedding], dim=-1)
        logits = [head(merged) for head in self.target_heads]
        return torch.cat(logits, dim=-1)
