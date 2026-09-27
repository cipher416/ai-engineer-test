import math
from collections.abc import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Vectors must have the same nonzero dimension")
    a, b = [float(x) for x in left], [float(x) for x in right]
    if not all(math.isfinite(x) for x in a + b):
        raise ValueError("Vector components must be finite")
    scale_a, scale_b = max(map(abs, a)), max(map(abs, b))
    if scale_a == 0 or scale_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    a, b = [x / scale_a for x in a], [x / scale_b for x in b]
    numerator = math.fsum(x * y for x, y in zip(a, b))
    denominator = math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b))
    return max(-1.0, min(1.0, numerator / denominator))
