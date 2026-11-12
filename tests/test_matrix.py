"""Matrix products and factorizations."""

import numpy as np
import pytest

from fft_analyzer.core.matrix import (
    back_substitution,
    determinant,
    forward_substitution,
    inverse,
    lu_decomposition,
    matmul_blocked,
    matmul_classical,
    power_method,
    qr_householder,
    solve,
    strassen_matmul,
)


def test_classical_and_blocked_and_strassen():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 4))
    B = rng.normal(size=(4, 3))
    expected = A @ B
    assert np.allclose(matmul_classical(A, B), expected, atol=1e-8)
    assert np.allclose(matmul_blocked(A, B, block=2), expected, atol=1e-8)
    square_a = rng.normal(size=(5, 5))
    square_b = rng.normal(size=(5, 5))
    product = square_a @ square_b
    assert np.allclose(strassen_matmul(square_a, square_b, leaf=2), product, atol=1e-6)
    even_a = rng.normal(size=(8, 8))
    even_b = rng.normal(size=(8, 8))
    assert np.allclose(strassen_matmul(even_a, even_b, leaf=2), even_a @ even_b, atol=1e-6)


def test_lu_reconstructs_and_solves():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(6, 6))
    P, L, U, _swaps = lu_decomposition(A)
    assert np.allclose(P @ A, L @ U, atol=1e-8)
    assert np.allclose(np.tril(L, k=-1) + np.eye(6), L, atol=1e-8)
    b = rng.normal(size=6)
    x = solve(A, b)
    assert np.allclose(A @ x, b, atol=1e-8)
    assert np.allclose(x, np.linalg.solve(A, b), atol=1e-8)


def test_lu_partial_pivot_on_zero_diagonal():
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    P, L, U, swaps = lu_decomposition(A)
    assert swaps == 1
    assert np.allclose(P @ A, L @ U, atol=1e-12)
    assert np.isclose(determinant(A), -1.0)


def test_determinant_matches_numpy():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 5))
    assert np.isclose(determinant(A), np.linalg.det(A), rtol=1e-8, atol=1e-8)


def test_singular_determinant_and_solve():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    assert determinant(A) == 0.0
    with pytest.raises(ValueError):
        solve(A, np.array([1.0, 2.0]))


def test_forward_and_back_substitution():
    L = np.array([[2.0, 0.0], [1.0, 3.0]])
    b = np.array([2.0, 7.0])
    y = forward_substitution(L, b)
    assert np.allclose(L @ y, b)
    U = np.array([[4.0, 1.0], [0.0, 2.0]])
    c = np.array([6.0, 2.0])
    x = back_substitution(U, c)
    assert np.allclose(U @ x, c)


def test_inverse_and_qr():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(4, 4))
    Ainv = inverse(A)
    assert np.allclose(A @ Ainv, np.eye(4), atol=1e-8)
    Q, R = qr_householder(A)
    assert np.allclose(Q @ R, A, atol=1e-8)
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-8)
    assert np.allclose(R, np.triu(R), atol=1e-8)


def test_qr_rectangular():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 3))
    Q, R = qr_householder(A)
    assert Q.shape == (6, 6)
    assert R.shape == (6, 3)
    assert np.allclose(Q @ R, A, atol=1e-8)
    assert np.allclose(Q.T @ Q, np.eye(6), atol=1e-8)


def test_power_method_known_eigenvalue():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    value, vector = power_method(A)
    assert np.isclose(value, 3.0, atol=1e-8)
    assert np.allclose(A @ vector, value * vector, atol=1e-8)
