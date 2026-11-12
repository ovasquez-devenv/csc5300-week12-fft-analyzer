"""Matrix multiplication, LU with partial pivoting, QR, and the power method.

Classical multiplication is the direct triple loop. Strassen's seven-product
recurrence is optional and falls back to the triple loop on small blocks.
Householder QR is the factorization used for A = QR.
"""

from __future__ import annotations

import numpy as np


def matmul_classical(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Product C = A B with three nested loops. Square cost is O(n^3)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul_classical expects two 2D arrays")
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError("incompatible shapes %s and %s" % (A.shape, B.shape))
    C = np.zeros((m, n), dtype=float)
    # Walk B by rows of B.T so the inner product reads contiguous samples.
    BT = np.ascontiguousarray(B.T)
    A = np.ascontiguousarray(A)
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for j in range(n):
            Bj = BT[j]
            acc = 0.0
            for t in range(k):
                acc += Ai[t] * Bj[t]
            Ci[j] = acc
    return C


def matmul_blocked(A: np.ndarray, B: np.ndarray, block: int = 32) -> np.ndarray:
    """Same triple-loop product, tiled so each block stays smaller."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("matmul_blocked expects two 2D arrays")
    m, k = A.shape
    k2, n = B.shape
    if k != k2:
        raise ValueError("incompatible shapes %s and %s" % (A.shape, B.shape))
    if block < 1:
        raise ValueError("block must be positive")
    C = np.zeros((m, n), dtype=float)
    for i0 in range(0, m, block):
        i1 = min(i0 + block, m)
        for t0 in range(0, k, block):
            t1 = min(t0 + block, k)
            left = A[i0:i1, t0:t1]
            for j0 in range(0, n, block):
                j1 = min(j0 + block, n)
                C[i0:i1, j0:j1] += matmul_classical(left, B[t0:t1, j0:j1])
    return C


def _strassen_square(A: np.ndarray, B: np.ndarray, leaf: int) -> np.ndarray:
    n = A.shape[0]
    if n <= leaf:
        return matmul_classical(A, B)
    if n % 2 == 1:
        Ap = np.pad(A, ((0, 1), (0, 1)))
        Bp = np.pad(B, ((0, 1), (0, 1)))
        return _strassen_square(Ap, Bp, leaf)[:n, :n]
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    M1 = _strassen_square(A11 + A22, B11 + B22, leaf)
    M2 = _strassen_square(A21 + A22, B11, leaf)
    M3 = _strassen_square(A11, B12 - B22, leaf)
    M4 = _strassen_square(A22, B21 - B11, leaf)
    M5 = _strassen_square(A11 + A12, B22, leaf)
    M6 = _strassen_square(A21 - A11, B11 + B12, leaf)
    M7 = _strassen_square(A12 - A22, B21 + B22, leaf)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    top = np.concatenate((C11, C12), axis=1)
    bot = np.concatenate((C21, C22), axis=1)
    return np.concatenate((top, bot), axis=0)


def strassen_matmul(A: np.ndarray, B: np.ndarray, leaf: int = 32) -> np.ndarray:
    """Strassen multiplication for square matrices. Other shapes use the triple loop.

    The recurrence is O(n^log2(7)), about O(n^2.807). The leaf size keeps the
    extra additions from dominating tiny problems.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("strassen_matmul expects two 2D arrays")
    if A.shape[1] != B.shape[0]:
        raise ValueError("incompatible shapes %s and %s" % (A.shape, B.shape))
    if leaf < 1:
        raise ValueError("leaf must be positive")
    m, k = A.shape
    n = B.shape[1]
    if not (m == k == n):
        return matmul_classical(A, B)
    return _strassen_square(A, B, leaf)


def lu_decomposition(A: np.ndarray, tol: float = 1e-14):
    """PA = LU with partial pivoting.

    Returns P, L, U, and the number of row swaps. A zero pivot raises
    ``ValueError``.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("LU requires a square matrix")
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swaps = 0
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot, k]) <= tol:
            raise ValueError("singular matrix")
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]
            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]
            swaps += 1
        if k == n - 1:
            break
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    return P, L, U, swaps


def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = b.shape[0]
    y = np.zeros(n, dtype=float)
    for i in range(n):
        acc = b[i]
        for j in range(i):
            acc -= L[i, j] * y[j]
        y[i] = acc / L[i, i]
    return y


def back_substitution(U: np.ndarray, y: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.shape[0]
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) <= tol:
            raise ValueError("singular matrix")
        acc = y[i]
        for j in range(i + 1, n):
            acc -= U[i, j] * x[j]
        x[i] = acc / U[i, i]
    return x


def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b with the pivoted LU factors."""
    P, L, U, _swaps = lu_decomposition(A)
    b = np.asarray(b, dtype=float)
    if b.ndim == 1:
        y = forward_substitution(L, P @ b)
        return back_substitution(U, y)
    if b.ndim != 2 or b.shape[0] != A.shape[0]:
        raise ValueError("b has an incompatible shape")
    cols = []
    for j in range(b.shape[1]):
        y = forward_substitution(L, P @ b[:, j])
        cols.append(back_substitution(U, y))
    return np.column_stack(cols)


def determinant(A: np.ndarray) -> float:
    """Determinant from pivoted LU. Zero when a pivot vanishes."""
    try:
        _P, _L, U, swaps = lu_decomposition(A)
    except ValueError:
        return 0.0
    sign = -1.0 if swaps % 2 else 1.0
    return float(sign * np.prod(np.diag(U)))


def inverse(A: np.ndarray) -> np.ndarray:
    """Inverse by solving A X = I one column at a time."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    identity = np.eye(n)
    return solve(A, identity)


def qr_householder(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full Householder QR. Returns Q (m x m) and R (m x n) with A = Q R."""
    A = np.array(A, dtype=float, copy=True)
    if A.ndim != 2:
        raise ValueError("QR expects a 2D array")
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for k in range(min(m, n)):
        x = R[k:, k].copy()
        norm_x = float(np.sqrt(np.dot(x, x)))
        if norm_x == 0.0:
            continue
        sign = 1.0 if x[0] >= 0.0 else -1.0
        u = x.copy()
        u[0] += sign * norm_x
        u_norm = float(np.sqrt(np.dot(u, u)))
        if u_norm == 0.0:
            continue
        u /= u_norm
        R[k:, k:] -= 2.0 * np.outer(u, u @ R[k:, k:])
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ u, u)
    return Q, R


def power_method(A: np.ndarray, max_iter: int = 500, tol: float = 1e-12, v0: np.ndarray | None = None):
    """Dominant eigenvalue and a matching eigenvector.

    The matrix-vector product uses the classical triple loop. Iteration stops
    when the Rayleigh quotient settles. A negative dominant eigenvalue is fine:
    the quotient is what we watch, not a fixed vector sign.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("power method expects a square matrix")
    n = A.shape[0]
    if v0 is None:
        v = np.ones(n, dtype=float)
    else:
        v = np.asarray(v0, dtype=float).reshape(-1).copy()
        if v.shape[0] != n:
            raise ValueError("start vector has the wrong length")
    norm = float(np.sqrt(np.dot(v, v)))
    if norm == 0.0:
        raise ValueError("start vector is zero")
    v /= norm
    rayleigh = 0.0
    for _ in range(max_iter):
        w = matmul_classical(A, v.reshape(n, 1)).reshape(-1)
        denom = float(np.dot(v, v))
        new_rayleigh = float(np.dot(v, w) / denom)
        w_norm = float(np.sqrt(np.dot(w, w)))
        if w_norm == 0.0:
            return 0.0, v
        v = w / w_norm
        if abs(new_rayleigh - rayleigh) <= tol * max(1.0, abs(new_rayleigh)):
            rayleigh = new_rayleigh
            break
        rayleigh = new_rayleigh
    w = matmul_classical(A, v.reshape(n, 1)).reshape(-1)
    rayleigh = float(np.dot(v, w))
    return rayleigh, v


class MatrixOps:
    """Chapter-style wrapper around the matrix routines."""

    multiply = staticmethod(matmul_classical)
    blocked = staticmethod(matmul_blocked)
    strassen = staticmethod(strassen_matmul)
    lu = staticmethod(lu_decomposition)
    solve = staticmethod(solve)
    determinant = staticmethod(determinant)
    inverse = staticmethod(inverse)
    qr = staticmethod(qr_householder)
    power_method = staticmethod(power_method)
    forward_substitution = staticmethod(forward_substitution)
    back_substitution = staticmethod(back_substitution)

    @staticmethod
    def transpose(A: np.ndarray) -> np.ndarray:
        return np.asarray(A).T.copy()
