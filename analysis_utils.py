"""ResearchScorer — LLM e2e test 1776848547"""

import math

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a**2 for a in vec_a))
    mag_b = math.sqrt(sum(b**2 for b in vec_b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

class ResearchScorer:
    def __init__(self, threshold: int = 30):
        self.threshold = threshold
        self._history: list[float] = []

    def score(self, technical: float, novelty: float = 1.0) -> float:
        raw = technical * novelty
        self._history.append(raw)
        return round(raw, 2)

    def is_valuable(self, score: float) -> bool:
        return score >= self.threshold
