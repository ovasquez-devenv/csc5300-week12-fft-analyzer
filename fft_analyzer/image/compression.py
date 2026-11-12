"""8x8 DCT-II and JPEG-style luminance quantization.

The DCT is the orthonormal type-II transform, applied separably with a
cosine matrix we build ourselves. Quality uses the usual JPEG scale:
below 50 the scale is 5000/quality, otherwise 200 - 2*quality. Table
entries are clipped to 1..255, so even quality 100 still rounds.
"""

from __future__ import annotations

import numpy as np

# Standard JPEG luminance quantization table, quality 50.
JPEG_LUMA_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=float,
)

_DCT_CACHE: dict[int, np.ndarray] = {}


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix. Rows are the cosine basis."""
    n = int(n)
    cached = _DCT_CACHE.get(n)
    if cached is not None:
        return cached
    k = np.arange(n, dtype=float)[:, None]
    i = np.arange(n, dtype=float)[None, :]
    matrix = np.cos(np.pi * (i + 0.5) * k / n)
    matrix[0, :] *= np.sqrt(1.0 / n)
    if n > 1:
        matrix[1:, :] *= np.sqrt(2.0 / n)
    _DCT_CACHE[n] = matrix
    return matrix


def dct1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return dct_matrix(x.shape[0]) @ x


def idct1(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float).reshape(-1)
    return dct_matrix(X.shape[0]).T @ X


def dct2(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=float)
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError("dct2 expects a square block")
    matrix = dct_matrix(block.shape[0])
    return matrix @ block @ matrix.T


def idct2(coeff: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=float)
    if coeff.ndim != 2 or coeff.shape[0] != coeff.shape[1]:
        raise ValueError("idct2 expects a square block")
    matrix = dct_matrix(coeff.shape[0])
    return matrix.T @ coeff @ matrix


def quality_scale(quality: int) -> float:
    quality = int(quality)
    if quality < 1 or quality > 100:
        raise ValueError("quality must be from 1 to 100")
    if quality < 50:
        return 5000.0 / quality
    return 200.0 - 2.0 * quality


def quantization_matrix(quality: int, block: int = 8) -> np.ndarray:
    """JPEG luminance table resized to ``block`` and scaled by quality."""
    scale = quality_scale(quality)
    base = JPEG_LUMA_Q
    if block != 8:
        # Nearest-neighbor resize so other even sizes still have a table.
        ys = np.linspace(0, 7, block)
        xs = np.linspace(0, 7, block)
        yi = np.clip(np.rint(ys).astype(int), 0, 7)
        xi = np.clip(np.rint(xs).astype(int), 0, 7)
        base = JPEG_LUMA_Q[np.ix_(yi, xi)]
    q = np.floor((base * scale + 50.0) / 100.0)
    return np.clip(q, 1.0, 255.0)


def compress_image(image: np.ndarray, quality: int = 50, block: int = 8):
    """Quantize 8x8 DCT blocks and reconstruct.

    ``image`` is grayscale, about 0..255. The level shift is 128, matching
    the JPEG offset. Returns the reconstructed image in 0..255 and the
    integer coefficient blocks (useful if a caller wants a crude bit count).
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError("compress_image expects a grayscale image")
    h, w = img.shape
    pad_h = (block - h % block) % block
    pad_w = (block - w % block) % block
    shifted = np.pad(img - 128.0, ((0, pad_h), (0, pad_w)))
    q = quantization_matrix(quality, block=block)
    matrix = dct_matrix(block)
    recon = np.zeros_like(shifted)
    coeffs = np.zeros_like(shifted)
    hh, ww = shifted.shape
    for i in range(0, hh, block):
        for j in range(0, ww, block):
            patch = shifted[i : i + block, j : j + block]
            freq = matrix @ patch @ matrix.T
            quantized = np.round(freq / q)
            coeffs[i : i + block, j : j + block] = quantized
            restored = matrix.T @ (quantized * q) @ matrix
            recon[i : i + block, j : j + block] = restored
    recon = np.clip(recon[:h, :w] + 128.0, 0.0, 255.0)
    return recon, coeffs[:h, :w]


def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.mean(diff * diff))


class ImageCompressor:
    dct1 = staticmethod(dct1)
    idct1 = staticmethod(idct1)
    dct2 = staticmethod(dct2)
    idct2 = staticmethod(idct2)
    quantization_matrix = staticmethod(quantization_matrix)
    compress_image = staticmethod(compress_image)
    mean_squared_error = staticmethod(mean_squared_error)
