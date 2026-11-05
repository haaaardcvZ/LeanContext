"""Centered linear CKA; no dimensionality reduction or per-example normalization.

The implementation computes the paper's complete-matrix statistic, rather than
averaging CKA values from minibatches. NumPy is only imported when computing CKA.
"""

from __future__ import annotations

import math
import random
import hashlib


def central_proof_positions(proof_length: int) -> list[int]:
    """Return zero-based indices for 1-based ceil(.25L) <= t <= floor(.75L).

    EOS, context, statement, and padding are not included in ``proof_length``.
    The paper does not specify a fallback when this window is empty, so we fail
    explicitly rather than changing the pooling region or dropping a theorem.
    """
    if proof_length < 1:
        raise ValueError("A nonempty reference proof is required")
    start = math.ceil(0.25 * proof_length)
    end = math.floor(0.75 * proof_length)
    positions = list(range(start - 1, end))
    if not positions:
        raise ValueError(
            f"Empty central proof window for length {proof_length}; "
            "the paper needs an explicit short-proof policy"
        )
    return positions


def fixed_mask_positions(proof_length: int, seed: int) -> list[int]:
    """Uniform fixed-size corruption, reused across checkpoints and input views.

    Exactly floor(L/2) proof positions are masked (the nearest lower integer for
    odd lengths). We do not force a central token to be masked: that would alter
    the sampling distribution. An empty central intersection is reported by
    the extraction routine and is never silently replaced or resampled.
    """
    if proof_length < 2:
        raise ValueError("A 50% corruption requires at least two proof tokens")
    return sorted(random.Random(seed).sample(range(proof_length), proof_length // 2))


def conditional_mask_positions(proof_length: int, seed: int) -> tuple[list[int], int]:
    """Resample a fixed-size mask until central pooling is nonempty.

    Each draw remains a uniform subset with exactly floor(L/2) masked proof
    tokens. Only the draw seed changes deterministically; token count, central
    region, theorem ID, and proof text never change. The second return value is
    the number of draws, including the accepted draw. Reusing the theorem seed
    across all 16 conditions reproduces exactly the same pattern and draw count.
    """
    central = set(central_proof_positions(proof_length))
    for attempt in range(100000):
        derived = seed if attempt == 0 else int.from_bytes(hashlib.sha256(
            f"leancontext-central-mask-v2:{seed}:{attempt}".encode()).digest()[:8], "big")
        positions = fixed_mask_positions(proof_length, derived)
        if central.intersection(positions):
            return positions, attempt + 1
    raise RuntimeError("Could not obtain a nonempty conditional central mask; no fallback is allowed")


def centered_linear_cka(x, y) -> float:
    """CKA(X,Y)=||Xc.T@Yc||_F^2/(||Xc.T@Xc||_F ||Yc.T@Yc||_F)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Inputs must be 2D matrices with the same >=2 theorem rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Representation matrices contain non-finite values")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = x.T @ y
    numerator = float(np.sum(xy * xy))
    xx = x.T @ x
    yy = y.T @ y
    denominator = float(np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro"))
    if denominator <= 0:
        raise ValueError("CKA is undefined for a constant centered representation")
    value = numerator / denominator
    if value < -1e-10 or value > 1 + 1e-10:
        raise ArithmeticError(f"CKA outside its numerical range: {value}")
    return min(1.0, max(0.0, value))
