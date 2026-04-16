from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def mean_column_spearman(targets: np.ndarray, predictions: np.ndarray) -> float:
    scores = []
    for index in range(targets.shape[1]):
        if np.allclose(targets[:, index], targets[0, index]) or np.allclose(
            predictions[:, index], predictions[0, index]
        ):
            scores.append(0.0)
            continue
        score = spearmanr(targets[:, index], predictions[:, index]).correlation
        if np.isnan(score):
            score = 0.0
        scores.append(float(score))
    return float(np.mean(scores))
