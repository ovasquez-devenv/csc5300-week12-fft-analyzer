"""Matplotlib figures saved with the Agg backend. Nothing is shown on screen."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..core.fft import fft_iterative, fftfreq, next_power_of_two
from ..signal.analysis import spectrogram


def _prepare(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_time_and_frequency(signal, sample_rate: float, path, title: str = "Signal", max_freq: float | None = None):
    signal = np.asarray(signal, dtype=float).reshape(-1)
    n = signal.shape[0]
    nfft = next_power_of_two(n)
    padded = np.zeros(nfft, dtype=float)
    padded[:n] = signal
    spec = fft_iterative(padded)
    freqs = fftfreq(nfft, d=1.0 / float(sample_rate))
    half = nfft // 2
    t = np.arange(n) / float(sample_rate)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    show_n = min(n, max(1, int(sample_rate)))
    axes[0].plot(t[:show_n], signal[:show_n], color="#1f4e79", linewidth=1.0)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title + ", time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs[:half], np.abs(spec[:half]), color="#b85c38", linewidth=1.0)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title(title + ", frequency")
    if max_freq is not None:
        axes[1].set_xlim(0, max_freq)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    path = _prepare(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_spectrogram(signal, sample_rate: float, path, window_size: int = 256, hop_size: int = 64, title: str = "Spectrogram", max_freq: float | None = None):
    mag, times, freqs = spectrogram(
        signal,
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mesh = ax.pcolormesh(times, freqs, 10.0 * np.log10(mag + 1e-12), shading="auto", cmap="magma")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    if max_freq is not None:
        ax.set_ylim(0, max_freq)
    fig.tight_layout()
    path = _prepare(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_fft2_heatmap(magnitude, path, title: str = "Centered log magnitude of the 2D FFT"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(magnitude, cmap="viridis", origin="lower")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    fig.tight_layout()
    path = _prepare(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_condition_numbers(sizes, conds, errors, path, title: str = "Hilbert matrices"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].semilogy(sizes, conds, marker="o", color="#1f4e79")
    axes[0].set_xlabel("matrix size")
    axes[0].set_ylabel("2-norm condition number")
    axes[0].set_title(title)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].semilogy(sizes, errors, marker="o", color="#b85c38")
    axes[1].set_xlabel("matrix size")
    axes[1].set_ylabel("relative change in x")
    axes[1].set_title("1e-8 nudge in b")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = _prepare(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_series(xs, ys_list, labels, path, xlabel: str, ylabel: str, title: str, loglog: bool = False):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ys, label in zip(ys_list, labels):
        pairs = [(x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y)]
        if not pairs:
            continue
        px, py = zip(*pairs)
        if loglog:
            ax.loglog(px, py, marker="o", label=label)
        else:
            ax.plot(px, py, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = _prepare(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
