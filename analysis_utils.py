"""ResearchScorer — LLM pipeline e2e test 1776846510"""

import math

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """코사인 유사도 계산."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a**2 for a in vec_a))
    mag_b = math.sqrt(sum(b**2 for b in vec_b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def batch_normalize(scores: list[float], scale: float = 100.0) -> list[float]:
    """점수 리스트를 0~scale 범위로 정규화."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [scale / 2] * len(scores)
    return [(s - min_s) / (max_s - min_s) * scale for s in scores]

class ResearchScorer:
    """연구 커밋 가치 점수 계산기."""

    def __init__(self, threshold: int = 30, decay: float = 0.95):
        self.threshold = threshold
        self.decay = decay
        self._history: list[float] = []

    def score(self, technical: float, novelty: float = 1.0) -> float:
        raw = technical * novelty
        self._history.append(raw)
        return round(raw * self.decay, 2)

    def is_valuable(self, score: float) -> bool:
        return score >= self.threshold

    def recent_avg(self, n: int = 5) -> float:
        window = self._history[-n:]
        return sum(window) / len(window) if window else 0.0
