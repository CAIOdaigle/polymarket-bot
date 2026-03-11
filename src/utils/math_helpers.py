from __future__ import annotations

import numpy as np


def log_sum_exp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp: log(sum(exp(x_i)))."""
    max_x = np.max(x)
    return float(max_x + np.log(np.sum(np.exp(x - max_x))))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
