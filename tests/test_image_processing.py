"""Image filters and DCT compression."""

import numpy as np

from fft_analyzer.applications.image_processor import ImageProcessor
from fft_analyzer.image.compression import compress_image, dct2, idct2, mean_squared_error
from fft_analyzer.image.filters import denoise, gaussian_blur, laplacian, sobel


def test_dct_roundtrip():
    rng = np.random.default_rng(1)
    block = rng.normal(size=(8, 8))
    assert np.allclose(idct2(dct2(block)), block, atol=1e-10)


def test_gaussian_blur_spreads_an_impulse():
    image = np.zeros((32, 32))
    image[16, 16] = 1.0
    blurred = gaussian_blur(image, sigma=2.0)
    assert blurred[16, 16] == np.max(blurred)
    assert blurred[16, 16] > blurred[16, 18] > blurred[16, 24]
    assert abs(blurred[14, 16] - blurred[18, 16]) < 1e-8


def test_sobel_sees_a_vertical_step():
    image = np.zeros((32, 32))
    image[:, 16:] = 1.0
    gx, gy = sobel(image)
    # Ignore the outer border, where zero-padding changes the stencil.
    assert np.sum(gx[2:-2, 2:-2] ** 2) > 10.0 * np.sum(gy[2:-2, 2:-2] ** 2)
    edges = laplacian(image)
    assert np.max(np.abs(edges)) > 0.5


def test_denoise_reduces_high_frequency_noise():
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:64, 0:64]
    clean = 128.0 + 30.0 * np.sin(2.0 * np.pi * xx / 64.0)
    noisy = clean + rng.normal(0.0, 12.0, clean.shape)
    cleaned = denoise(noisy, cutoff_ratio=0.15)
    assert mean_squared_error(cleaned, clean) < mean_squared_error(noisy, clean)


def test_compression_error_falls_as_quality_rises():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(32, 32)).astype(float)
    errors = []
    for quality in (10, 50, 90):
        recon, _coeffs = compress_image(image, quality=quality)
        errors.append(mean_squared_error(image, recon))
    assert errors[0] > errors[1] > errors[2]
    # Quality 90 should be a small fraction of the 0..255 range.
    assert errors[2] < 30.0


def test_image_processor_saves_png(tmp_path):
    rng = np.random.default_rng(4)
    image = rng.integers(0, 256, size=(24, 24)).astype(float)
    proc = ImageProcessor()
    blurred = proc.blur(image, sigma=1.0)
    edges = proc.edges(image)
    recon, _coeffs = proc.compress(image, quality=70)
    assert blurred.shape == image.shape
    assert edges.shape == image.shape
    assert mean_squared_error(image, recon) < mean_squared_error(
        image, proc.compress(image, quality=20)[0]
    )
    path = tmp_path / "img.png"
    proc.save(path, image)
    loaded = proc.load(path)
    assert loaded.shape == image.shape
    assert mean_squared_error(loaded, image) < 1.0
    heat = proc.frequency_magnitude(image)
    assert heat.shape[0] >= image.shape[0]
