"""FFT filters, frequency-domain views, and DCT compression for grayscale images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..core.fft import fft2, fftshift, next_power_of_two
from ..image.compression import compress_image, mean_squared_error
from ..image.filters import denoise, gaussian_blur, laplacian, sobel_magnitude


def to_uint8(image: np.ndarray, mode: str = "clip") -> np.ndarray:
    img = np.asarray(image, dtype=float)
    if mode == "scale":
        lo = float(np.min(img))
        hi = float(np.max(img))
        if hi - lo < 1e-12:
            return np.zeros(img.shape, dtype=np.uint8)
        scaled = (img - lo) / (hi - lo) * 255.0
        return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def save_png(path: str | Path, image: np.ndarray, mode: str = "clip"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = to_uint8(image, mode=mode)
    Image.fromarray(arr, mode="L").save(path)


def load_png(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=float)
    return arr


def frequency_magnitude(image: np.ndarray) -> np.ndarray:
    """Log magnitude of the centered 2D FFT. The image is padded to a power of two."""
    img = np.asarray(image, dtype=float)
    nh = next_power_of_two(img.shape[0])
    nw = next_power_of_two(img.shape[1])
    padded = np.zeros((nh, nw), dtype=float)
    padded[: img.shape[0], : img.shape[1]] = img
    spec = fft2(padded)
    mag = np.abs(fftshift(spec))
    return np.log10(mag + 1e-12)


class ImageProcessor:
    def blur(self, image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        return gaussian_blur(image, sigma=sigma)

    def edges(self, image: np.ndarray, method: str = "sobel") -> np.ndarray:
        method = method.lower()
        if method == "sobel":
            return sobel_magnitude(image)
        if method == "laplacian":
            return laplacian(image)
        raise ValueError("method must be 'sobel' or 'laplacian'")

    def denoise(self, image: np.ndarray, cutoff_ratio: float = 0.2) -> np.ndarray:
        return denoise(image, cutoff_ratio=cutoff_ratio)

    def compress(self, image: np.ndarray, quality: int = 50):
        return compress_image(image, quality=quality)

    def frequency_magnitude(self, image: np.ndarray) -> np.ndarray:
        return frequency_magnitude(image)

    def save(self, path: str | Path, image: np.ndarray, mode: str = "clip"):
        save_png(path, image, mode=mode)

    def load(self, path: str | Path) -> np.ndarray:
        return load_png(path)

    @staticmethod
    def error(original: np.ndarray, reconstructed: np.ndarray) -> float:
        return mean_squared_error(original, reconstructed)
