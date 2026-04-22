"""QuickScan pipeline test — 1776845725"""

def compute_similarity(vec_a: list, vec_b: list) -> float:
    """코사인 유사도 계산."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a**2 for a in vec_a) ** 0.5
    mag_b = sum(b**2 for b in vec_b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def normalize_score(raw: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """점수를 0~100 범위로 정규화."""
    return max(min_val, min(max_val, raw))

class ResearchScorer:
    """연구 커밋 점수 계산기."""
    def __init__(self, threshold: int = 30):
        self.threshold = threshold

    def is_valuable(self, score: float) -> bool:
        return score >= self.threshold
