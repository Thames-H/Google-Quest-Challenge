from __future__ import annotations

import numpy as np


def _quantiles(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    try:
        return np.quantile(values, quantiles, method="linear")
    except TypeError:
        return np.quantile(values, quantiles, interpolation="linear")


def match_column_distribution(predictions: np.ndarray, reference_values: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    reference_values = np.asarray(reference_values, dtype=np.float32)
    if predictions.size == 0:
        return predictions.copy()

    order = np.argsort(predictions, kind="mergesort")
    quantiles = (np.arange(predictions.shape[0], dtype=np.float32) + 0.5) / predictions.shape[0]
    mapped_values = _quantiles(np.sort(reference_values), quantiles).astype(np.float32)
    matched = np.empty_like(predictions, dtype=np.float32)
    matched[order] = mapped_values
    return matched


def rank_based_distribution_matching(
    predictions: np.ndarray,
    reference_matrix: np.ndarray,
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float32)
    reference_matrix = np.asarray(reference_matrix, dtype=np.float32)
    if predictions.ndim == 1:
        return match_column_distribution(predictions, reference_matrix)

    matched = np.zeros_like(predictions, dtype=np.float32)
    for column_index in range(predictions.shape[1]):
        matched[:, column_index] = match_column_distribution(
            predictions[:, column_index],
            reference_matrix[:, column_index],
        )
    return matched
