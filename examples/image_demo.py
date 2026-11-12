"""Draw a small test image in code, then blur, edge, denoise, and compress it.

Run from the repo root:
    py -3.14 examples/image_demo.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fft_analyzer.applications.image_processor import ImageProcessor
from fft_analyzer.visualization.plots import plot_fft2_heatmap


def make_image(n=128, seed=4):
    yy, xx = np.mgrid[0:n, 0:n]
    image = 40.0 + 0.4 * xx + 30.0 * np.sin(2.0 * np.pi * yy / n)
    image[(yy - 40) ** 2 + (xx - 90) ** 2 < 18 ** 2] = 220.0
    image[70:100, 20:55] = 180.0
    rng = np.random.default_rng(seed)
    noisy = np.clip(image + rng.normal(0.0, 12.0, image.shape), 0.0, 255.0)
    return image, noisy


def main():
    out = ROOT / "outputs" / "image"
    out.mkdir(parents=True, exist_ok=True)
    clean, noisy = make_image()
    proc = ImageProcessor()
    proc.save(out / "original.png", clean)
    proc.save(out / "noisy.png", noisy)
    blurred = proc.blur(noisy, sigma=1.5)
    edges = proc.edges(clean, method="sobel")
    denoised = proc.denoise(noisy, cutoff_ratio=0.18)
    proc.save(out / "blur.png", blurred)
    proc.save(out / "edges.png", edges, mode="scale")
    proc.save(out / "denoised.png", denoised)

    qualities = [10, 50, 90]
    errors = []
    for quality in qualities:
        recon, _coeffs = proc.compress(noisy, quality=quality)
        proc.save(out / ("compressed_q%d.png" % quality), recon)
        errors.append(proc.error(noisy, recon))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(qualities, errors, marker="o", color="#b85c38")
    ax.set_xlabel("JPEG-style quality")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Compression error on the synthetic image")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "compression_error.png", dpi=120)
    plt.close(fig)

    heat = proc.frequency_magnitude(clean)
    plot_fft2_heatmap(heat, out / "fft_heatmap.png", title="Log magnitude, synthetic image")

    lines = ["image demo", "size %d" % clean.shape[0]]
    for quality, err in zip(qualities, errors):
        lines.append("mse_q%d %.6f" % (quality, err))
    lines.append("denoise_mse_to_clean %.6f" % proc.error(denoised, clean))
    lines.append("noisy_mse_to_clean %.6f" % proc.error(noisy, clean))
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
