"""FFT correctness, round trip, and convolution."""

import numpy as np
import pytest

from fft_analyzer.core.fft import (
    bit_reverse_indices,
    circular_convolve,
    dft_naive,
    fft2,
    fft_iterative,
    fft_recursive,
    fftfreq,
    fftshift,
    ifft2,
    ifft_iterative,
    ifft_recursive,
    linear_convolve,
    polynomial_multiply,
)


def test_bit_reversal_n8():
    order = bit_reverse_indices(8)
    assert list(order) == [0, 4, 2, 6, 1, 5, 3, 7]


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 64, 256, 1024])
def test_iterative_matches_numpy(n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    got = fft_iterative(x)
    assert np.allclose(got, np.fft.fft(x), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("n", [2, 8, 64, 512])
def test_recursive_matches_numpy(n):
    rng = np.random.default_rng(1000 + n)
    x = rng.normal(size=n)
    assert np.allclose(fft_recursive(x), np.fft.fft(x), rtol=1e-8, atol=1e-8)


def test_naive_dft_small():
    x = np.array([1.0, 2.0, 3.0, 4.0, 0.5])
    assert np.allclose(dft_naive(x), np.fft.fft(x), rtol=1e-8, atol=1e-8)


def test_impulse_is_flat():
    x = np.zeros(16)
    x[0] = 1.0
    spec = fft_iterative(x)
    assert np.allclose(spec, np.ones(16))


def test_roundtrip_iterative_and_recursive():
    rng = np.random.default_rng(3)
    x = rng.normal(size=128)
    y = ifft_iterative(fft_iterative(x))
    z = ifft_recursive(fft_recursive(x))
    assert np.allclose(y.real, x, atol=1e-8)
    assert np.max(np.abs(y.imag)) < 1e-8
    assert np.allclose(z.real, x, atol=1e-8)


def test_linearity():
    rng = np.random.default_rng(4)
    a = rng.normal(size=32)
    b = rng.normal(size=32)
    left = fft_iterative(a + 2.0 * b)
    right = fft_iterative(a) + 2.0 * fft_iterative(b)
    assert np.allclose(left, right)


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fft_iterative(np.zeros(6))


def test_circular_convolution():
    rng = np.random.default_rng(5)
    a = rng.normal(size=32)
    b = rng.normal(size=32)
    got = circular_convolve(a, b)
    exp = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))
    assert np.allclose(got, exp, atol=1e-8)


def test_linear_convolution_matches_numpy():
    rng = np.random.default_rng(6)
    a = rng.normal(size=20)
    b = rng.normal(size=13)
    got = np.real(linear_convolve(a, b))
    assert np.allclose(got, np.convolve(a, b), atol=1e-8)


def test_polynomial_multiply():
    # (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
    got = np.real(polynomial_multiply([1.0, 2.0], [3.0, 4.0]))
    assert np.allclose(got, [3.0, 10.0, 8.0], atol=1e-8)


def test_fftfreq_and_shift():
    freqs = fftfreq(8, d=0.5)
    assert np.allclose(freqs, np.fft.fftfreq(8, d=0.5))
    x = np.arange(8)
    assert np.array_equal(fftshift(x), np.fft.fftshift(x))
    image = np.arange(12).reshape(3, 4)
    assert np.array_equal(fftshift(image), np.fft.fftshift(image))


def test_fft2_roundtrip():
    rng = np.random.default_rng(7)
    image = rng.normal(size=(16, 32))
    spec = fft2(image)
    assert np.allclose(spec, np.fft.fft2(image), rtol=1e-8, atol=1e-8)
    back = ifft2(spec)
    assert np.allclose(back.real, image, atol=1e-8)


def test_million_point_iterative_fft():
    rng = np.random.default_rng(8)
    n = 2**20
    x = rng.normal(size=n)
    spec = fft_iterative(x)
    reference = np.fft.fft(x)
    assert spec.shape == (n,)
    assert np.allclose(spec, reference, rtol=1e-6, atol=1e-6)
    back = ifft_iterative(spec)
    assert np.allclose(back.real, x, rtol=1e-6, atol=1e-6)
