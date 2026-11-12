"""Norms, conditioning, and the cancellation example."""

import numpy as np
import pytest

from fft_analyzer.core.numerical import (
    absolute_error,
    cancellation_demo,
    condition_number,
    hilbert,
    is_ill_conditioned,
    matrix_norm,
    perturbed_solution_change,
    relative_error,
    residual_norm,
    vector_norm,
    welford_mean_variance,
)
from fft_analyzer.core.matrix import solve


def test_vector_and_matrix_norms():
    x = np.array([3.0, -4.0])
    assert vector_norm(x, 1) == 7.0
    assert np.isclose(vector_norm(x, 2), 5.0)
    assert vector_norm(x, np.inf) == 4.0
    A = np.array([[1.0, -2.0], [3.0, 4.0]])
    assert np.isclose(matrix_norm(A, 1), np.linalg.norm(A, 1))
    assert np.isclose(matrix_norm(A, np.inf), np.linalg.norm(A, np.inf))
    assert np.isclose(matrix_norm(A, 2), np.linalg.norm(A, 2), rtol=1e-8)


def test_errors_and_identity_condition():
    assert absolute_error([1.0, 2.0], [1.0, 0.0]) == pytest.approx(2.0)
    assert relative_error([2.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    eye = np.eye(4)
    assert np.isclose(condition_number(eye), 1.0, atol=1e-8)
    assert not is_ill_conditioned(eye)
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    b = rng.normal(size=4)
    x = solve(A, b)
    assert residual_norm(A, x, b) < 1e-8


def test_hilbert_is_ill_conditioned_and_sensitive():
    H = hilbert(8)
    cond = condition_number(H)
    assert cond > 1e6
    assert is_ill_conditioned(H)
    # numpy's 2-norm condition number is the reference for this small matrix
    assert np.isclose(cond, np.linalg.cond(H, 2), rtol=1e-5, atol=1.0)
    change = perturbed_solution_change(H, np.ones(8), noise=1e-8, seed=0)
    well = perturbed_solution_change(np.eye(8), np.ones(8), noise=1e-8, seed=0)
    assert change["relative_change"] > well["relative_change"] * 10


def test_cancellation_and_kahan():
    demo = cancellation_demo()
    assert demo["lost_unit"] == 0.0
    assert demo["stable_tiny_error"] < demo["unstable_tiny_error"]
    # 1e16 swamps a unit. Kahan cannot rescue that particular triple.
    assert demo["kahan_large_cancel"] == pytest.approx(0.0)
    assert demo["naive_large_cancel"] == pytest.approx(0.0)
    kahan_err = abs(demo["kahan_many_tenths"] - demo["exact_many_tenths"])
    naive_err = abs(demo["naive_many_tenths"] - demo["exact_many_tenths"])
    assert kahan_err < naive_err
    assert demo["kahan_many_tenths"] == pytest.approx(10000.0)


def test_welford_matches_numpy():
    rng = np.random.default_rng(2)
    data = rng.normal(size=50)
    mean, var = welford_mean_variance(data)
    assert np.isclose(mean, np.mean(data))
    assert np.isclose(var, np.var(data, ddof=1))
