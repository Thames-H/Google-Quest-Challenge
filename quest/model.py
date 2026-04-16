from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-6)
    return summed / counts


class HierarchicalAttentionPool(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        chunk_embeddings = chunk_embeddings.to(dtype=self.scorer[0].weight.dtype)
        scores = self.scorer(chunk_embeddings).squeeze(-1)
        scores = scores.masked_fill(chunk_mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=-1) * chunk_mask
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.sum(chunk_embeddings * weights.unsqueeze(-1), dim=1)


class MetadataEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int] | None,
        numeric_dim: int,
        embedding_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        vocab_sizes = vocab_sizes or {}
        self.numeric_dim = numeric_dim
        self.output_dim = output_dim
        self.category_embedding = nn.Embedding(max(vocab_sizes.get("category", 1), 1), embedding_dim)
        self.host_embedding = nn.Embedding(max(vocab_sizes.get("host", 1), 1), embedding_dim)
        self.domain_embedding = nn.Embedding(max(vocab_sizes.get("domain", 1), 1), embedding_dim)
        input_dim = numeric_dim + embedding_dim * 3
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(
        self,
        meta_numeric: torch.Tensor | None = None,
        meta_category_id: torch.Tensor | None = None,
        meta_host_id: torch.Tensor | None = None,
        meta_domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = 1
        device = None
        if meta_numeric is not None:
            batch_size = meta_numeric.shape[0]
            device = meta_numeric.device
        elif meta_category_id is not None:
            batch_size = meta_category_id.shape[0]
            device = meta_category_id.device
        else:
            device = self.category_embedding.weight.device

        if meta_numeric is None:
            meta_numeric = torch.zeros(batch_size, self.numeric_dim, device=device)
        if meta_category_id is None:
            meta_category_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if meta_host_id is None:
            meta_host_id = torch.zeros(batch_size, dtype=torch.long, device=device)
        if meta_domain_id is None:
            meta_domain_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        pieces = [
            meta_numeric,
            self.category_embedding(meta_category_id),
            self.host_embedding(meta_host_id),
            self.domain_embedding(meta_domain_id),
        ]
        return self.projection(torch.cat(pieces, dim=-1))


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
        num_targets: int | None = None,
        target_columns: list[str] | None = None,
        dropout: float = 0.1,
        gradient_checkpointing: bool = True,
        metadata_vocab_sizes: dict[str, int] | None = None,
        meta_numeric_dim: int = 0,
        meta_embedding_dim: int = 16,
        meta_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if target_columns is None:
            if num_targets is None:
                raise ValueError("Either target_columns or num_targets must be provided.")
            target_columns = [f"target_{index}" for index in range(num_targets)]
        self.target_columns = target_columns

        self.question_encoder = AutoModel.from_pretrained(backbone_name)
        self.answer_encoder = AutoModel.from_pretrained(backbone_name)
        if gradient_checkpointing:
            if hasattr(self.question_encoder, "gradient_checkpointing_enable"):
                self.question_encoder.gradient_checkpointing_enable()
            if hasattr(self.answer_encoder, "gradient_checkpointing_enable"):
                self.answer_encoder.gradient_checkpointing_enable()

        hidden_size = self.question_encoder.config.hidden_size
        self.question_pooler = HierarchicalAttentionPool(hidden_size, dropout=dropout)
        self.answer_pooler = HierarchicalAttentionPool(hidden_size, dropout=dropout)
        self.metadata_encoder = MetadataEncoder(
            vocab_sizes=metadata_vocab_sizes,
            numeric_dim=meta_numeric_dim,
            embedding_dim=meta_embedding_dim,
            output_dim=meta_hidden_dim,
        )

        self.question_target_indices = [
            index for index, name in enumerate(target_columns) if name.startswith("question_")
        ]
        self.answer_target_indices = [
            index for index, name in enumerate(target_columns) if name.startswith("answer_")
        ]
        assigned = set(self.question_target_indices) | set(self.answer_target_indices)
        self.shared_target_indices = [index for index in range(len(target_columns)) if index not in assigned]

        question_input_dim = hidden_size * 2 + meta_hidden_dim
        answer_input_dim = hidden_size * 2 + meta_hidden_dim
        shared_input_dim = hidden_size * 2 + meta_hidden_dim

        self.question_heads = nn.ModuleList(
            [TargetHead(question_input_dim, hidden_dim=64, dropout=dropout) for _ in self.question_target_indices]
        )
        self.answer_heads = nn.ModuleList(
            [TargetHead(answer_input_dim, hidden_dim=64, dropout=dropout) for _ in self.answer_target_indices]
        )
        self.shared_heads = nn.ModuleList(
            [TargetHead(shared_input_dim, hidden_dim=64, dropout=dropout) for _ in self.shared_target_indices]
        )

    def _encode_branch(
        self,
        encoder: nn.Module,
        pooler: HierarchicalAttentionPool,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
            if chunk_mask is None:
                chunk_mask = torch.ones(input_ids.shape[:2], dtype=torch.float32, device=input_ids.device)

        batch_size, chunk_count, token_count = input_ids.shape
        flat_input_ids = input_ids.reshape(batch_size * chunk_count, token_count)
        flat_attention_mask = attention_mask.reshape(batch_size * chunk_count, token_count)
        outputs = encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        chunk_embeddings = masked_mean_pool(outputs.last_hidden_state, flat_attention_mask)
        chunk_embeddings = chunk_embeddings.reshape(batch_size, chunk_count, -1)

        if chunk_mask is None:
            chunk_mask = attention_mask.any(dim=-1).to(dtype=torch.float32)
        return pooler(chunk_embeddings, chunk_mask.to(dtype=torch.float32))

    def forward(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
        q_chunk_mask: torch.Tensor | None = None,
        a_chunk_mask: torch.Tensor | None = None,
        meta_numeric: torch.Tensor | None = None,
        meta_category_id: torch.Tensor | None = None,
        meta_host_id: torch.Tensor | None = None,
        meta_domain_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        question_embedding = self._encode_branch(
            self.question_encoder,
            self.question_pooler,
            input_ids=q_input_ids,
            attention_mask=q_attention_mask,
            chunk_mask=q_chunk_mask,
        )
        answer_embedding = self._encode_branch(
            self.answer_encoder,
            self.answer_pooler,
            input_ids=a_input_ids,
            attention_mask=a_attention_mask,
            chunk_mask=a_chunk_mask,
        )
        if (
            meta_numeric is None
            and meta_category_id is None
            and meta_host_id is None
            and meta_domain_id is None
        ):
            metadata_embedding = question_embedding.new_zeros(
                question_embedding.shape[0],
                self.metadata_encoder.output_dim,
            )
        else:
            metadata_embedding = self.metadata_encoder(
                meta_numeric=meta_numeric,
                meta_category_id=meta_category_id,
                meta_host_id=meta_host_id,
                meta_domain_id=meta_domain_id,
            )

        question_features = torch.cat([question_embedding, answer_embedding, metadata_embedding], dim=-1)
        answer_features = torch.cat([answer_embedding, question_embedding, metadata_embedding], dim=-1)
        shared_features = torch.cat([question_embedding, answer_embedding, metadata_embedding], dim=-1)

        logits = torch.zeros(
            question_embedding.shape[0],
            len(self.target_columns),
            dtype=question_embedding.dtype,
            device=question_embedding.device,
        )

        for head, target_index in zip(self.question_heads, self.question_target_indices, strict=False):
            logits[:, target_index] = head(question_features).squeeze(-1)
        for head, target_index in zip(self.answer_heads, self.answer_target_indices, strict=False):
            logits[:, target_index] = head(answer_features).squeeze(-1)
        for head, target_index in zip(self.shared_heads, self.shared_target_indices, strict=False):
            logits[:, target_index] = head(shared_features).squeeze(-1)
        return logits
