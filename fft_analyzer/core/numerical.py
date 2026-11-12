"""Error measures, norms, condition numbers, and a cancellation demo.

Conditioning follows the usual product of a norm and the norm of the inverse.
The inverse comes from the pivoted LU solver in ``matrix.py``.
"""

from __future__ import annotations

import numpy as np

from .matrix import inverse, matmul_classical, power_method, solve


def vector_norm(x: np.ndarray, p: float = 2) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    if p == 1:
        return float(np.sum(np.abs(x)))
    if p == 2:
        return float(np.sqrt(np.dot(x, x)))
    if np.isinf(p):
        return float(np.max(np.abs(x)))
    raise ValueError("p must be 1, 2, or inf")


def matrix_norm(A: np.ndarray, p: float = 2) -> float:
    """Induced 1-norm, inf-norm, or 2-norm.

    The 2-norm is the largest singular value, estimated as the square root of
    the dominant eigenvalue of A^T A from the power method.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("matrix_norm expects a 2D array")
    if p == 1:
        return float(np.max(np.sum(np.abs(A), axis=0)))
    if np.isinf(p):
        return float(np.max(np.sum(np.abs(A), axis=1)))
    if p == 2:
        ata = matmul_classical(A.T, A)
        value, _vec = power_method(ata, v0=np.ones(A.shape[1]))
        return float(np.sqrt(abs(value)))
    raise ValueError("p must be 1, 2, or inf")


def absolute_error(approx: np.ndarray, exact: np.ndarray) -> float:
    diff = np.asarray(approx, dtype=float) - np.asarray(exact, dtype=float)
    return vector_norm(diff, p=2)


def relative_error(approx: np.ndarray, exact: np.ndarray) -> float:
    denom = vector_norm(exact, p=2)
    err = absolute_error(approx, exact)
    if denom == 0.0:
        return err
    return err / denom


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """||A x - b||_2 / ||b||_2. A relative residual for a linear solve."""
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    got = matmul_classical(A, x.reshape(-1, 1)).reshape(-1)
    return relative_error(got, b)


def condition_number(A: np.ndarray, p: float = 2) -> float:
    """kappa(A) = ||A|| ||A^{-1}||. Returns inf when A is singular."""
    A = np.asarray(A, dtype=float)
    try:
        Ainv = inverse(A)
    except ValueError:
        return float("inf")
    return matrix_norm(A, p=p) * matrix_norm(Ainv, p=p)


def is_ill_conditioned(A: np.ndarray, threshold: float = 1e6, p: float = 2) -> bool:
    """True when the condition number reaches ``threshold`` (default 1e6)."""
    return condition_number(A, p=p) >= threshold


def hilbert(n: int) -> np.ndarray:
    """Hilbert matrix, the chapter's standard ill-conditioned example."""
    idx = np.arange(1, n + 1, dtype=float)
    return 1.0 / (idx[:, None] + idx[None, :] - 1.0)


def kahan_sum(values: np.ndarray) -> float:
    """Kahan summation. Keeps a running correction for the low bits."""
    total = 0.0
    compensation = 0.0
    for value in np.asarray(values, dtype=float).reshape(-1):
        y = float(value) - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return total


def naive_sum(values: np.ndarray) -> float:
    total = 0.0
    for value in np.asarray(values, dtype=float).reshape(-1):
        total += float(value)
    return total


def welford_mean_variance(values: np.ndarray):
    """One-pass mean and sample variance."""
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in np.asarray(values, dtype=float).reshape(-1):
        n += 1
        delta = float(x) - mean
        mean += delta / n
        m2 += delta * (float(x) - mean)
    if n < 2:
        return mean, 0.0
    return mean, m2 / (n - 1)


def quadratic_roots_unstable(a: float, b: float, c: float):
    """Schoolbook quadratic formula. Cancels when b^2 is much larger than 4ac."""
    disc = np.sqrt(b * b - 4.0 * a * c)
    x1 = (-b + disc) / (2.0 * a)
    x2 = (-b - disc) / (2.0 * a)
    return float(x1), float(x2)


def quadratic_roots_stable(a: float, b: float, c: float):
    """Quadratic roots that compute the non-cancelling root first, then use Vieta."""
    disc = np.sqrt(b * b - 4.0 * a * c)
    if b >= 0.0:
        x_primary = (-b - disc) / (2.0 * a)
    else:
        x_primary = (-b + disc) / (2.0 * a)
    x_other = c / (a * x_primary)
    return float(x_primary), float(x_other)


def cancellation_demo(a: float = 1.0, b: float = 1e8, c: float = 1.0) -> dict:
    """Show a lost unit at 1e16, Kahan on a long sum, and the quadratic formula.

    (1e16 + 1) - 1e16 comes back as 0. Summing 1e16, 1, and -1e16 also comes
    back as 0 for both the plain sum and Kahan, because the correction bit is
    not representable next to 1e16. Kahan does help on 100,000 copies of 0.1,
    where the exact total is 10,000.

    The tiny root of x^2 + b x + c = 0 is about -c/b. The schoolbook formula
    subtracts two close quantities for that root. The stable formula does not.
    """
    big = 1e16
    lost_unit = (big + 1.0) - big
    tenths = [0.1] * 100000
    exact_tenths = 10000.0
    unstable = quadratic_roots_unstable(a, b, c)
    stable = quadratic_roots_stable(a, b, c)
    large = min(stable, key=lambda z: -abs(z))
    tiny_ref = c / (a * large)
    unstable_tiny = min(unstable, key=abs)
    stable_tiny = min(stable, key=abs)
    return {
        "lost_unit": lost_unit,
        "expected_lost_unit": 0.0,
        "unstable_roots": unstable,
        "stable_roots": stable,
        "tiny_reference": float(tiny_ref),
        "unstable_tiny_error": abs(unstable_tiny - tiny_ref),
        "stable_tiny_error": abs(stable_tiny - tiny_ref),
        "kahan_many_tenths": kahan_sum(tenths),
        "naive_many_tenths": naive_sum(tenths),
        "exact_many_tenths": exact_tenths,
        "kahan_large_cancel": kahan_sum([big, 1.0, -big]),
        "naive_large_cancel": naive_sum([big, 1.0, -big]),
    }


def perturbed_solution_change(A: np.ndarray, b: np.ndarray, noise: float = 1e-8, seed: int = 0):
    """Relative change in x when b is nudged by ``noise`` times a normal vector."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    x = solve(A, b)
    rng = np.random.default_rng(seed)
    bp = b + noise * rng.normal(size=b.shape[0])
    xp = solve(A, bp)
    return {
        "x": x,
        "x_perturbed": xp,
        "relative_change": relative_error(xp, x),
        "condition_number": condition_number(A, p=2),
    }


class NumericalStability:
    """Chapter-style wrapper around the stability helpers."""

    vector_norm = staticmethod(vector_norm)
    matrix_norm = staticmethod(matrix_norm)
    absolute_error = staticmethod(absolute_error)
    relative_error = staticmethod(relative_error)
    residual_norm = staticmethod(residual_norm)
    condition_number = staticmethod(condition_number)
    is_ill_conditioned = staticmethod(is_ill_conditioned)
    hilbert = staticmethod(hilbert)
    kahan_sum = staticmethod(kahan_sum)
    cancellation_demo = staticmethod(cancellation_demo)
    quadratic_roots_stable = staticmethod(quadratic_roots_stable)
    quadratic_roots_unstable = staticmethod(quadratic_roots_unstable)
