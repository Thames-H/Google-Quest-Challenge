from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, DebertaV2Tokenizer


WHITESPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+")
CODE_RE = re.compile(r"<code>.*?</code>|```.*?```", re.IGNORECASE | re.DOTALL)
WORD_RE = re.compile(r"\w+")


def load_tokenizer(tokenizer_name: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception as exc:
        is_deberta_v3 = "deberta-v3" in tokenizer_name.lower()
        if not is_deberta_v3:
            raise
        warnings.warn(
            (
                f"Fast tokenizer load failed for {tokenizer_name}; "
                "falling back to the explicit DebertaV2 slow tokenizer. "
                f"Original error: {exc}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return DebertaV2Tokenizer.from_pretrained(tokenizer_name)


@dataclass
class MetadataSpec:
    category_vocab: dict[str, int]
    host_vocab: dict[str, int]
    domain_vocab: dict[str, int]
    numeric_columns: list[str]
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]

    def vocab_sizes(self) -> dict[str, int]:
        return {
            "category": max(self.category_vocab.values(), default=0) + 1,
            "host": max(self.host_vocab.values(), default=0) + 1,
            "domain": max(self.domain_vocab.values(), default=0) + 1,
        }


def load_competition_frames(data_dir: str | Path):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    target_columns = list(sample_df.columns[1:])
    return train_df, test_df, target_columns, sample_df


def normalize_text(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    return WHITESPACE_RE.sub(" ", text).strip().lower()


def build_group_keys(dataframe: pd.DataFrame) -> pd.Series:
    title = dataframe["question_title"].map(normalize_text)
    body = dataframe["question_body"].map(normalize_text)
    return title + " [sep] " + body


def build_group_folds(
    dataframe: pd.DataFrame,
    folds: int,
    group_column: str | None = None,
):
    splitter = GroupKFold(n_splits=folds)
    if group_column:
        groups = dataframe[group_column].fillna("").map(normalize_text)
    else:
        groups = build_group_keys(dataframe)

    splits = []
    for train_indices, valid_indices in splitter.split(dataframe, groups=groups):
        splits.append((train_indices, valid_indices))
    return splits


def apply_row_limits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    train_row_limit: int | None = None,
    test_row_limit: int | None = None,
):
    if train_row_limit is not None:
        train_df = train_df.head(train_row_limit).reset_index(drop=True)
    if test_row_limit is not None:
        test_df = test_df.head(test_row_limit).reset_index(drop=True)
        sample_df = sample_df.head(test_row_limit).reset_index(drop=True)
    return train_df, test_df, sample_df


def _safe_text(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _url_count(text: str) -> int:
    return len(URL_RE.findall(text))


def _code_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(len(match.group(0)) for match in CODE_RE.finditer(text))
    return float(code_chars) / max(len(text), 1)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(WORD_RE.findall(normalize_text(left)))
    right_tokens = set(WORD_RE.findall(normalize_text(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _extract_domain(url_value: object) -> str:
    parsed = urlparse(_safe_text(url_value))
    return parsed.netloc.lower() or "unknown"


def _build_numeric_feature_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    title = dataframe["question_title"].map(_safe_text)
    body = dataframe["question_body"].map(_safe_text)
    answer = dataframe["answer"].map(_safe_text) if "answer" in dataframe.columns else pd.Series([""] * len(dataframe))

    feature_frame = pd.DataFrame(index=dataframe.index)
    feature_frame["meta_title_char_len"] = title.map(len)
    feature_frame["meta_body_char_len"] = body.map(len)
    feature_frame["meta_answer_char_len"] = answer.map(len)
    feature_frame["meta_title_word_len"] = title.map(_word_count)
    feature_frame["meta_body_word_len"] = body.map(_word_count)
    feature_frame["meta_answer_word_len"] = answer.map(_word_count)
    feature_frame["meta_body_url_count"] = body.map(_url_count)
    feature_frame["meta_answer_url_count"] = answer.map(_url_count)
    feature_frame["meta_body_code_ratio"] = body.map(_code_ratio)
    feature_frame["meta_answer_code_ratio"] = answer.map(_code_ratio)
    feature_frame["meta_title_body_jaccard"] = [
        _token_jaccard(left, right) for left, right in zip(title, body, strict=False)
    ]
    feature_frame["meta_title_answer_jaccard"] = [
        _token_jaccard(left, right) for left, right in zip(title, answer, strict=False)
    ]
    return feature_frame


def prepare_metadata_spec(train_df: pd.DataFrame, test_df: pd.DataFrame) -> MetadataSpec:
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    def build_vocab(series: pd.Series) -> dict[str, int]:
        values = sorted({normalize_text(value) or "unknown" for value in series})
        return {value: index for index, value in enumerate(["unknown", *[item for item in values if item != "unknown"]])}

    category_vocab = build_vocab(combined["category"] if "category" in combined.columns else pd.Series(["unknown"]))
    host_vocab = build_vocab(combined["host"] if "host" in combined.columns else pd.Series(["unknown"]))
    domain_vocab = build_vocab(combined["url"].map(_extract_domain) if "url" in combined.columns else pd.Series(["unknown"]))

    train_numeric = _build_numeric_feature_frame(train_df)
    numeric_columns = list(train_numeric.columns)
    numeric_means = {column: float(train_numeric[column].mean()) for column in numeric_columns}
    numeric_stds = {
        column: float(train_numeric[column].std()) if float(train_numeric[column].std()) > 1e-6 else 1.0
        for column in numeric_columns
    }
    return MetadataSpec(
        category_vocab=category_vocab,
        host_vocab=host_vocab,
        domain_vocab=domain_vocab,
        numeric_columns=numeric_columns,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
    )


def apply_metadata_spec(dataframe: pd.DataFrame, metadata_spec: MetadataSpec) -> pd.DataFrame:
    transformed = dataframe.copy()
    numeric_frame = _build_numeric_feature_frame(transformed)
    for column in metadata_spec.numeric_columns:
        transformed[column] = (
            numeric_frame[column].astype("float32") - metadata_spec.numeric_means[column]
        ) / metadata_spec.numeric_stds[column]

    transformed["meta_category_id"] = transformed.get("category", pd.Series(["unknown"] * len(transformed))).map(
        lambda value: metadata_spec.category_vocab.get(normalize_text(value) or "unknown", 0)
    )
    transformed["meta_host_id"] = transformed.get("host", pd.Series(["unknown"] * len(transformed))).map(
        lambda value: metadata_spec.host_vocab.get(normalize_text(value) or "unknown", 0)
    )
    transformed["meta_domain_id"] = transformed.get("url", pd.Series(["unknown"] * len(transformed))).map(
        lambda value: metadata_spec.domain_vocab.get(_extract_domain(value), 0)
    )
    return transformed.reset_index(drop=True)


def _split_with_overlap(
    tokens: list[int],
    chunk_size: int,
    overlap: int,
    max_chunks: int,
) -> list[list[int]]:
    if not tokens:
        return [[]]

    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + chunk_size]
        if not chunk:
            continue
        chunks.append(chunk)
        if len(chunks) >= max_chunks or start + chunk_size >= len(tokens):
            break
    return chunks or [[]]


class GoogleQuestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer_name: str,
        max_len_question: int,
        max_len_answer: int,
        question_chunk_size: int | None = None,
        answer_chunk_size: int | None = None,
        question_chunk_overlap: int = 0,
        answer_chunk_overlap: int = 0,
        question_max_chunks: int = 1,
        answer_max_chunks: int = 1,
        max_title_tokens: int = 32,
        target_columns: list[str] | None = None,
        is_train: bool = True,
        tokenizer=None,
        metadata_spec: MetadataSpec | None = None,
    ) -> None:
        frame = dataframe.reset_index(drop=True)
        if metadata_spec is not None and "meta_category_id" not in frame.columns:
            frame = apply_metadata_spec(frame, metadata_spec)
        self.dataframe = frame
        self.target_columns = target_columns or []
        self.is_train = is_train
        self.max_len_question = max_len_question
        self.max_len_answer = max_len_answer
        self.question_chunk_size = question_chunk_size or max_len_question
        self.answer_chunk_size = answer_chunk_size or max_len_answer
        self.question_chunk_overlap = question_chunk_overlap
        self.answer_chunk_overlap = answer_chunk_overlap
        self.question_max_chunks = question_max_chunks
        self.answer_max_chunks = answer_max_chunks
        self.max_title_tokens = max_title_tokens
        self.metadata_spec = metadata_spec
        self.tokenizer = tokenizer or load_tokenizer(tokenizer_name)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _encode_chunk_pair_from_ids(
        self,
        title_ids: list[int],
        chunk_ids: list[int],
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer.prepare_for_model(
            title_ids,
            pair_ids=chunk_ids,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        }

    def _encode_chunk_pair_from_text(
        self,
        title: str,
        chunk_text: str,
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text=title,
            text_pair=chunk_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0).to(dtype=torch.long),
            "attention_mask": encoded["attention_mask"].squeeze(0).to(dtype=torch.long),
        }

    def _encode_document(
        self,
        title: str,
        document: str,
        max_length: int,
        chunk_size: int,
        overlap: int,
        max_chunks: int,
    ) -> dict[str, torch.Tensor]:
        has_token_api = hasattr(self.tokenizer, "encode") and hasattr(self.tokenizer, "prepare_for_model")
        if has_token_api:
            title_ids = self.tokenizer.encode(title, add_special_tokens=False)[: self.max_title_tokens]
            document_ids = self.tokenizer.encode(document, add_special_tokens=False)
            special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
            available_chunk = max(max_length - len(title_ids) - special_tokens, 1)
            effective_chunk_size = min(chunk_size, available_chunk)
            chunk_ids = _split_with_overlap(document_ids, effective_chunk_size, overlap, max_chunks)
            encoded_chunks = [
                self._encode_chunk_pair_from_ids(title_ids=title_ids, chunk_ids=ids, max_length=max_length)
                for ids in chunk_ids
            ]
        else:
            words = document.split()
            chunks = _split_with_overlap(list(range(len(words))), chunk_size, overlap, max_chunks)
            encoded_chunks = []
            for chunk in chunks:
                chunk_words = words[chunk[0] : chunk[-1] + 1] if chunk else []
                encoded_chunks.append(
                    self._encode_chunk_pair_from_text(
                        title=title,
                        chunk_text=" ".join(chunk_words),
                        max_length=max_length,
                    )
                )

        input_ids = torch.stack([chunk["input_ids"] for chunk in encoded_chunks])
        attention_mask = torch.stack([chunk["attention_mask"] for chunk in encoded_chunks])
        if input_ids.shape[0] == 1:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        title = _safe_text(row.get("question_title", ""))
        question_body = _safe_text(row.get("question_body", ""))
        answer = _safe_text(row.get("answer", ""))

        question_inputs = self._encode_document(
            title=title,
            document=question_body,
            max_length=self.max_len_question,
            chunk_size=self.question_chunk_size,
            overlap=self.question_chunk_overlap,
            max_chunks=self.question_max_chunks,
        )
        answer_inputs = self._encode_document(
            title=title,
            document=answer,
            max_length=self.max_len_answer,
            chunk_size=self.answer_chunk_size,
            overlap=self.answer_chunk_overlap,
            max_chunks=self.answer_max_chunks,
        )

        item = {
            "q_input_ids": question_inputs["input_ids"],
            "q_attention_mask": question_inputs["attention_mask"],
            "a_input_ids": answer_inputs["input_ids"],
            "a_attention_mask": answer_inputs["attention_mask"],
            "qa_id": torch.tensor(int(row["qa_id"]), dtype=torch.long),
        }
        if self.metadata_spec is not None:
            numeric_values = row[self.metadata_spec.numeric_columns].astype("float32").to_numpy()
            item["meta_numeric"] = torch.tensor(numeric_values, dtype=torch.float32)
            item["meta_category_id"] = torch.tensor(int(row["meta_category_id"]), dtype=torch.long)
            item["meta_host_id"] = torch.tensor(int(row["meta_host_id"]), dtype=torch.long)
            item["meta_domain_id"] = torch.tensor(int(row["meta_domain_id"]), dtype=torch.long)
        if self.is_train:
            labels = row[self.target_columns].astype("float32").to_numpy()
            item["labels"] = torch.tensor(labels, dtype=torch.float32)
        return item


def _pad_chunk_batch(values: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    normalized = [value.unsqueeze(0) if value.dim() == 1 else value for value in values]
    max_chunks = max(value.shape[0] for value in normalized)
    max_length = normalized[0].shape[-1]
    padded = torch.full((len(values), max_chunks, max_length), pad_value, dtype=values[0].dtype)
    for index, value in enumerate(normalized):
        padded[index, : value.shape[0]] = value
    return padded


def _pad_chunk_mask(values: list[torch.Tensor]) -> torch.Tensor:
    max_chunks = max(value.shape[0] for value in values)
    padded = torch.zeros((len(values), max_chunks), dtype=torch.float32)
    for index, value in enumerate(values):
        padded[index, : value.shape[0]] = value
    return padded


def quest_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    def chunk_count(value: torch.Tensor) -> int:
        return 1 if value.dim() == 1 else value.shape[0]

    q_chunk_mask_values = [torch.ones(chunk_count(item["q_input_ids"]), dtype=torch.float32) for item in batch]
    a_chunk_mask_values = [torch.ones(chunk_count(item["a_input_ids"]), dtype=torch.float32) for item in batch]
    collated = {
        "q_input_ids": _pad_chunk_batch([item["q_input_ids"] for item in batch], pad_value=0),
        "q_attention_mask": _pad_chunk_batch([item["q_attention_mask"] for item in batch], pad_value=0),
        "q_chunk_mask": _pad_chunk_mask(q_chunk_mask_values),
        "a_input_ids": _pad_chunk_batch([item["a_input_ids"] for item in batch], pad_value=0),
        "a_attention_mask": _pad_chunk_batch([item["a_attention_mask"] for item in batch], pad_value=0),
        "a_chunk_mask": _pad_chunk_mask(a_chunk_mask_values),
        "qa_id": torch.stack([item["qa_id"] for item in batch]),
    }
    if "labels" in batch[0]:
        collated["labels"] = torch.stack([item["labels"] for item in batch])
    if "meta_numeric" in batch[0]:
        collated["meta_numeric"] = torch.stack([item["meta_numeric"] for item in batch])
        collated["meta_category_id"] = torch.stack([item["meta_category_id"] for item in batch])
        collated["meta_host_id"] = torch.stack([item["meta_host_id"] for item in batch])
        collated["meta_domain_id"] = torch.stack([item["meta_domain_id"] for item in batch])
    return collated
