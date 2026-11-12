"""Cooley-Tukey FFT, inverse FFT, and convolution.

The iterative transform vectorizes each butterfly stage so a million-point
array (2^20) stays practical. The recursive form follows the even/odd split
in Amakobe, Chapter 11. Both require a power-of-two length. Convolution
zero-pads up to the next power of two when it needs a linear result.
"""

from __future__ import annotations

import cmath

import numpy as np

_PI = cmath.pi


def next_power_of_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _require_power_of_two(n: int) -> None:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("length must be a power of 2, got %d" % n)


def bit_reverse_indices(n: int) -> np.ndarray:
    """Indices that reorder a length-n buffer into bit-reversed order.

    For n = 8 the order is 0, 4, 2, 6, 1, 5, 3, 7.
    """
    _require_power_of_two(n)
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for bit in range(bits):
        rev |= ((idx >> bit) & 1) << (bits - 1 - bit)
    return rev


def _as_complex_batch(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("expected a 1D signal or a batch with shape (batch, n)")
    return arr


def fft_iterative_batch(batch: np.ndarray) -> np.ndarray:
    """Iterative radix-2 FFT along the last axis.

    ``batch`` has shape (batch, n) with n a power of two. Butterfly stages
    are NumPy operations on whole blocks, not a Python loop over samples.
    """
    data = np.array(_as_complex_batch(batch), dtype=np.complex128, copy=True)
    b, n = data.shape
    _require_power_of_two(n)
    if n == 1:
        return data
    data = data[:, bit_reverse_indices(n)]
    length = 2
    while length <= n:
        half = length // 2
        twiddle = np.exp(-2j * np.pi * np.arange(half) / length)
        view = data.reshape(b, n // length, length)
        upper = view[:, :, :half].copy()
        lower = view[:, :, half:] * twiddle
        view[:, :, :half] = upper + lower
        view[:, :, half:] = upper - lower
        length *= 2
    return data


def fft_iterative(x: np.ndarray) -> np.ndarray:
    """Iterative Cooley-Tukey FFT with a bit-reversal permutation."""
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    return fft_iterative_batch(x.reshape(1, -1))[0]


def iter_fft_stages(x: np.ndarray):
    """Yield the buffer after bit reversal and after each butterfly stage.

    Used by the FFT animation so the frames are the real algorithm, not a sketch.
    """
    data = np.array(np.asarray(x, dtype=np.complex128).reshape(1, -1), copy=True)
    b, n = data.shape
    _require_power_of_two(n)
    if n == 1:
        yield data[0].copy()
        return
    data = data[:, bit_reverse_indices(n)]
    yield data[0].copy()
    length = 2
    while length <= n:
        half = length // 2
        twiddle = np.exp(-2j * np.pi * np.arange(half) / length)
        view = data.reshape(b, n // length, length)
        upper = view[:, :, :half].copy()
        lower = view[:, :, half:] * twiddle
        view[:, :, :half] = upper + lower
        view[:, :, half:] = upper - lower
        yield data[0].copy()
        length *= 2


def _fft_recursive_core(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    if n <= 1:
        return x
    even = _fft_recursive_core(x[0::2])
    odd = _fft_recursive_core(x[1::2])
    twiddle = np.exp(-2j * np.pi * np.arange(n // 2) / n)
    mixed = twiddle * odd
    return np.concatenate((even + mixed, even - mixed))


def fft_recursive(x: np.ndarray) -> np.ndarray:
    """Recursive Cooley-Tukey FFT. Length must be a power of two."""
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    _require_power_of_two(x.shape[0])
    return _fft_recursive_core(x)


def fft(x: np.ndarray, method: str = "iterative", pad: bool = False) -> np.ndarray:
    """FFT of ``x``.

    With ``pad=False`` the length must already be a power of two.
    With ``pad=True`` the buffer is zero-padded and the result is longer.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if pad:
        n_pad = next_power_of_two(x.shape[0])
        if n_pad != x.shape[0]:
            x = np.pad(x, (0, n_pad - x.shape[0]))
    if method == "iterative":
        return fft_iterative(x)
    if method == "recursive":
        return fft_recursive(x)
    raise ValueError("method must be 'iterative' or 'recursive'")


def ifft_iterative(X: np.ndarray) -> np.ndarray:
    """Inverse FFT. Uses conj(FFT(conj(X))) / N."""
    X = np.asarray(X, dtype=np.complex128).reshape(-1)
    n = X.shape[0]
    return np.conj(fft_iterative(np.conj(X))) / n


def ifft_recursive(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.complex128).reshape(-1)
    n = X.shape[0]
    return np.conj(fft_recursive(np.conj(X))) / n


def ifft(X: np.ndarray, method: str = "iterative") -> np.ndarray:
    if method == "iterative":
        return ifft_iterative(X)
    if method == "recursive":
        return ifft_recursive(X)
    raise ValueError("method must be 'iterative' or 'recursive'")


def fft2(image: np.ndarray) -> np.ndarray:
    """2D FFT by a 1D iterative FFT on rows, then on columns."""
    image = np.asarray(image, dtype=np.complex128)
    if image.ndim != 2:
        raise ValueError("fft2 expects a 2D array")
    _require_power_of_two(image.shape[0])
    _require_power_of_two(image.shape[1])
    rows = fft_iterative_batch(image)
    cols = fft_iterative_batch(rows.T).T
    return cols


def ifft2(spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    n = spectrum.shape[0] * spectrum.shape[1]
    return np.conj(fft2(np.conj(spectrum))) / n


def rfft(x: np.ndarray) -> np.ndarray:
    """Positive-frequency half of a real FFT, including Nyquist."""
    spec = fft_iterative(x)
    return spec[: len(spec) // 2 + 1]


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """Frequency bins matching the usual FFT layout, in cycles per sample unit."""
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    val = 1.0 / (n * d)
    freqs = np.empty(n, dtype=float)
    half = (n - 1) // 2 + 1
    freqs[:half] = np.arange(0, half)
    freqs[half:] = np.arange(-(n // 2), 0)
    return freqs * val


def fftshift(x: np.ndarray) -> np.ndarray:
    """Move the zero-frequency sample to the center. Works for 1D and 2D."""
    x = np.asarray(x)
    if x.ndim == 1:
        return np.roll(x, x.shape[0] // 2)
    if x.ndim == 2:
        shifted = np.roll(x, x.shape[0] // 2, axis=0)
        return np.roll(shifted, x.shape[1] // 2, axis=1)
    raise ValueError("fftshift expects 1D or 2D data")


def ifftshift(x: np.ndarray) -> np.ndarray:
    """Undo ``fftshift``."""
    x = np.asarray(x)
    if x.ndim == 1:
        return np.roll(x, -(x.shape[0] // 2))
    if x.ndim == 2:
        shifted = np.roll(x, -(x.shape[0] // 2), axis=0)
        return np.roll(shifted, -(x.shape[1] // 2), axis=1)
    raise ValueError("ifftshift expects 1D or 2D data")


def dft_naive(x: np.ndarray) -> np.ndarray:
    """Direct DFT from the definition. Time grows like N squared.

    Kept for benchmarks on short arrays. The inner loop is a Python
    complex multiply, not a call to ``numpy.fft``.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    n = x.shape[0]
    out = np.zeros(n, dtype=np.complex128)
    for k in range(n):
        acc = 0j
        for t in range(n):
            acc += x[t] * cmath.exp(-2j * _PI * k * t / n)
        out[k] = acc
    return out


def circular_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution. Both arrays must share a power-of-two length."""
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("circular convolution needs equal lengths")
    return ifft_iterative(fft_iterative(a) * fft_iterative(b))


def linear_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Linear convolution by zero-padding to the next power of two."""
    a = np.asarray(a, dtype=np.complex128).reshape(-1)
    b = np.asarray(b, dtype=np.complex128).reshape(-1)
    n_out = a.shape[0] + b.shape[0] - 1
    if n_out < 1:
        return np.zeros(0, dtype=np.complex128)
    n_fft = next_power_of_two(n_out)
    fa = fft_iterative(np.pad(a, (0, n_fft - a.shape[0])))
    fb = fft_iterative(np.pad(b, (0, n_fft - b.shape[0])))
    return ifft_iterative(fa * fb)[:n_out]


def polynomial_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply polynomials by linear convolution of their coefficients."""
    return linear_convolve(a, b)


class FFT:
    """Namespace matching the chapter's FFT helper."""

    fft = staticmethod(fft)
    ifft = staticmethod(ifft)
    fft_iterative = staticmethod(fft_iterative)
    fft_recursive = staticmethod(fft_recursive)
    ifft_iterative = staticmethod(ifft_iterative)
    ifft_recursive = staticmethod(ifft_recursive)
    rfft = staticmethod(rfft)
    fftfreq = staticmethod(fftfreq)
    fftshift = staticmethod(fftshift)
    fft2 = staticmethod(fft2)
    ifft2 = staticmethod(ifft2)
    dft_naive = staticmethod(dft_naive)
    circular_convolve = staticmethod(circular_convolve)
    linear_convolve = staticmethod(linear_convolve)
    polynomial_multiply = staticmethod(polynomial_multiply)
