"""GIF animations written with Pillow. No ffmpeg."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..core.fft import fft_iterative, fftfreq, iter_fft_stages, next_power_of_two
from ..signal.filters import lowpass


def _figure_frame(fig) -> Image.Image:
    # savefig goes through Agg and Pillow. We do not ask matplotlib for ffmpeg.
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    frame = image.copy()
    buf.close()
    return frame


def _save_gif(frames, path, fps: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames to save")
    duration = int(round(1000.0 / fps))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        format="GIF",
    )
    return path


def animate_fft_process(signal, path, fps: int = 2):
    """One frame per butterfly stage of the iterative FFT, plus bit reversal."""
    x = np.asarray(signal, dtype=float).reshape(-1)
    nfft = next_power_of_two(x.shape[0])
    padded = np.zeros(nfft, dtype=float)
    padded[: x.shape[0]] = x
    # Keep the GIF small. The algorithm is the same one used everywhere.
    if nfft > 256:
        padded = padded[:256].copy()
        # 256 is a power of two, so this slice is still a valid FFT length.
        if padded.shape[0] & (padded.shape[0] - 1):
            padded = padded[: next_power_of_two(padded.shape[0]) // 2]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    frames = []
    stages = list(iter_fft_stages(padded))
    for index, stage in enumerate(stages):
        ax.clear()
        ax.plot(np.abs(stage), color="#1f4e79", linewidth=1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("bin")
        ax.set_ylabel("magnitude")
        if index == 0:
            label = "after bit reversal"
        else:
            label = "after butterfly length %d" % (2 ** index)
        ax.set_title("Iterative FFT, %s" % label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        frames.append(_figure_frame(fig))
    plt.close(fig)
    return _save_gif(frames, path, fps)


def animate_filtering(signal, sample_rate: float, path, cutoffs=None, fps: int = 2):
    """Show the spectrum as a low-pass cutoff walks downward."""
    x = np.asarray(signal, dtype=float).reshape(-1)
    if cutoffs is None:
        nyquist = 0.5 * float(sample_rate)
        cutoffs = np.linspace(nyquist * 0.9, nyquist * 0.05, 8)
    nfft = next_power_of_two(x.shape[0])
    freqs = fftfreq(nfft, d=1.0 / float(sample_rate))
    half = nfft // 2

    fig, ax = plt.subplots(figsize=(7, 3.5))
    frames = []
    for cutoff in cutoffs:
        filtered = lowpass(x, sample_rate, float(cutoff))
        padded = np.zeros(nfft, dtype=float)
        padded[: filtered.shape[0]] = filtered
        mag = np.abs(fft_iterative(padded))
        ax.clear()
        ax.plot(freqs[:half], mag[:half], color="#b85c38", linewidth=1.0)
        ax.axvline(cutoff, color="#1f4e79", linestyle="--", label="cutoff")
        ax.set_xlim(0, 0.5 * float(sample_rate))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.set_title("Low-pass cutoff at %.0f Hz" % cutoff)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        frames.append(_figure_frame(fig))
    plt.close(fig)
    return _save_gif(frames, path, fps)
