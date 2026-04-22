import numpy as np
from typing import List, Dict, Optional

# Updated: 2026-04-22 09:37:34 UTC (raw_patch test)

def compute_loss(predictions: List[float], targets: List[float]) -> float:
    """MSE 손실 계산"""
    arr_pred = np.array(predictions)
    arr_tgt  = np.array(targets)
    return float(np.mean((arr_pred - arr_tgt) ** 2))

def normalize_features(data: np.ndarray, method: str = "minmax") -> np.ndarray:
    """특성 정규화 (minmax / zscore)"""
    if method == "zscore":
        return (data - data.mean()) / (data.std() + 1e-8)
    mn, mx = data.min(), data.max()
    return (data - mn) / (mx - mn + 1e-8)

class ExperimentTracker:
    def __init__(self, name: str):
        self.name = name
        self.metrics: Dict[str, List[float]] = {}

    def log(self, key: str, value: float) -> None:
        self.metrics.setdefault(key, []).append(value)

    def summary(self) -> Dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.metrics.items()}
