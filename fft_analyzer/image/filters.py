"""Gaussian blur, Sobel, Laplacian, and frequency-domain denoising.

Spatial kernels are applied by zero-padded FFT convolution. The Gaussian
blur is a multiply in the frequency domain.
"""

from __future__ import annotations

import numpy as np

from ..core.fft import fft2, fftfreq, ifft2, next_power_of_two


def _pad_pow2(image: np.ndarray, min_h: int | None = None, min_w: int | None = None):
    image = np.asarray(image, dtype=float)
    h, w = image.shape
    nh = next_power_of_two(h if min_h is None else max(h, min_h))
    nw = next_power_of_two(w if min_w is None else max(w, min_w))
    padded = np.zeros((nh, nw), dtype=float)
    padded[:h, :w] = image
    return padded, h, w


def fft_convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Linear convolution, cropped back to the image shape (same-size, centered)."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if image.ndim != 2 or kernel.ndim != 2:
        raise ValueError("image and kernel must be 2D")
    h, w = image.shape
    kh, kw = kernel.shape
    nh = next_power_of_two(h + kh - 1)
    nw = next_power_of_two(w + kw - 1)
    padded = np.zeros((nh, nw), dtype=float)
    padded[:h, :w] = image
    kern = np.zeros((nh, nw), dtype=float)
    cy, cx = kh // 2, kw // 2
    for i in range(kh):
        for j in range(kw):
            kern[(i - cy) % nh, (j - cx) % nw] = kernel[i, j]
    filtered = np.real(ifft2(fft2(padded) * fft2(kern)))
    return filtered[:h, :w]


def gaussian_blur(image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Blur by multiplying the 2D FFT by a Gaussian frequency response."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    padded, h, w = _pad_pow2(image)
    spec = fft2(padded)
    fy = fftfreq(padded.shape[0])
    fx = fftfreq(padded.shape[1])
    grid_y, grid_x = np.meshgrid(fy, fx, indexing="ij")
    response = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (grid_x ** 2 + grid_y ** 2))
    out = np.real(ifft2(spec * response))
    return out[:h, :w]


def sobel(image: np.ndarray):
    """Horizontal and vertical Sobel responses via FFT convolution."""
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    gx = fft_convolve2d(image, kx)
    gy = fft_convolve2d(image, ky)
    return gx, gy


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gx, gy = sobel(image)
    return np.hypot(gx, gy)


def laplacian(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return fft_convolve2d(image, kernel)


def denoise(image: np.ndarray, cutoff_ratio: float = 0.2) -> np.ndarray:
    """Drop high spatial frequencies.

    ``cutoff_ratio`` is a fraction of the Nyquist radius (0.5 cycles/pixel).
    A ratio of 0.2 keeps a disk out to 0.1 cycles/pixel.
    """
    if not 0.0 < cutoff_ratio <= 1.0:
        raise ValueError("cutoff_ratio must be in (0, 1]")
    padded, h, w = _pad_pow2(image)
    spec = fft2(padded)
    fy = fftfreq(padded.shape[0])
    fx = fftfreq(padded.shape[1])
    grid_y, grid_x = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(grid_x ** 2 + grid_y ** 2)
    keep = radius <= (cutoff_ratio * 0.5)
    out = np.real(ifft2(spec * keep))
    return out[:h, :w]


class ImageFilter:
    gaussian_blur = staticmethod(gaussian_blur)
    sobel = staticmethod(sobel)
    sobel_magnitude = staticmethod(sobel_magnitude)
    laplacian = staticmethod(laplacian)
    denoise = staticmethod(denoise)
    fft_convolve2d = staticmethod(fft_convolve2d)
