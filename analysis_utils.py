import numpy as np
from typing import List, Dict

def compute_loss(predictions: List[float], targets: List[float]) -> float:
    return float(np.mean((np.array(predictions) - np.array(targets)) ** 2))

def normalize_features(data: np.ndarray) -> np.ndarray:
    return (data - data.mean()) / (data.std() + 1e-8)

class ExperimentTracker:
    def __init__(self, name: str):
        self.name = name
        self.metrics: Dict[str, List[float]] = {}

    def log(self, key: str, value: float) -> None:
        self.metrics.setdefault(key, []).append(value)

    def best(self, key: str) -> float:
        return min(self.metrics.get(key, [float("inf")]))
