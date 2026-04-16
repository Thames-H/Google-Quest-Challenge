from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer


def load_competition_frames(data_dir: str | Path):
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    sample_df = pd.read_csv(data_dir / "sample_submission.csv")
    target_columns = list(sample_df.columns[1:])
    return train_df, test_df, target_columns, sample_df


def build_group_folds(
    dataframe: pd.DataFrame,
    folds: int,
    group_column: str = "question_body",
):
    splitter = GroupKFold(n_splits=folds)
    splits = []
    for train_indices, valid_indices in splitter.split(
        dataframe,
        groups=dataframe[group_column].fillna(""),
    ):
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


class GoogleQuestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer_name: str,
        max_len_question: int,
        max_len_answer: int,
        target_columns: list[str],
        is_train: bool,
        tokenizer=None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.target_columns = target_columns
        self.is_train = is_train
        self.max_len_question = max_len_question
        self.max_len_answer = max_len_answer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _encode_pair(self, text: str, text_pair: str, max_length: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text=text,
            text_pair=text_pair,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        title = str(row["question_title"] if pd.notna(row["question_title"]) else "")
        question_body = str(row["question_body"] if pd.notna(row["question_body"]) else "")
        answer = str(row["answer"] if pd.notna(row["answer"]) else "")

        question_inputs = self._encode_pair(
            text=title,
            text_pair=question_body,
            max_length=self.max_len_question,
        )
        answer_inputs = self._encode_pair(
            text=title,
            text_pair=answer,
            max_length=self.max_len_answer,
        )

        item = {
            "q_input_ids": question_inputs["input_ids"],
            "q_attention_mask": question_inputs["attention_mask"],
            "a_input_ids": answer_inputs["input_ids"],
            "a_attention_mask": answer_inputs["attention_mask"],
            "qa_id": torch.tensor(int(row["qa_id"]), dtype=torch.long),
        }
        if self.is_train:
            labels = row[self.target_columns].astype("float32").to_numpy()
            item["labels"] = torch.tensor(labels, dtype=torch.float32)
        return item
