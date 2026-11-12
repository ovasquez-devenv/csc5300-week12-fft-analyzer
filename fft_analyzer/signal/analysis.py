"""Periodogram, power spectral density, STFT spectrogram, and peak picking."""

from __future__ import annotations

import numpy as np

from ..core.fft import fft_iterative, ifft_iterative, next_power_of_two, rfft
from .filters import get_window


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Linear autocorrelation, lags 0 through n-1, via ifft(|FFT|^2)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    nfft = next_power_of_two(max(2 * n - 1, 1))
    spec = fft_iterative(np.pad(x, (0, nfft - n)))
    ac = np.real(ifft_iterative(spec * np.conj(spec)))
    return ac[:n]


def _positive_frequencies(n: int, sample_rate: float) -> np.ndarray:
    return np.arange(n // 2 + 1, dtype=float) * (float(sample_rate) / n)


def periodogram(x: np.ndarray, sample_rate: float = 1.0, window: str = "hann"):
    """One-sided periodogram.

    P[k] = |FFT(x w)|^2 / (fs * sum(w^2)), with positive bins doubled
    except for DC and Nyquist. The FFT length is the next power of two
    when ``len(x)`` is not already one, and the window is applied first.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    w = get_window(window, n)
    nfft = next_power_of_two(n)
    windowed = np.zeros(nfft, dtype=float)
    windowed[:n] = x * w
    spec = fft_iterative(windowed)
    power = np.abs(spec) ** 2
    scale = float(sample_rate) * float(np.sum(w * w))
    if scale == 0.0:
        raise ValueError("window has zero energy")
    pxx = power / scale
    n_pos = nfft // 2 + 1
    one_sided = pxx[:n_pos].copy()
    if n_pos > 2:
        one_sided[1:-1] *= 2.0
    freqs = _positive_frequencies(nfft, sample_rate)
    return freqs, one_sided


def power_spectral_density(x: np.ndarray, sample_rate: float = 1.0, window: str = "hann"):
    """Alias for the one-sided periodogram used as a PSD estimate."""
    return periodogram(x, sample_rate=sample_rate, window=window)


def stft(x: np.ndarray, window_size: int = 256, hop_size: int = 128, window: str = "hann"):
    """Short-time FFT. Returns complex bins with shape (freq, time)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    window_size = int(window_size)
    hop_size = int(hop_size)
    if window_size < 2 or hop_size < 1:
        raise ValueError("bad STFT size")
    if window_size & (window_size - 1):
        raise ValueError("window_size must be a power of 2")
    if x.shape[0] < window_size:
        x = np.pad(x, (0, window_size - x.shape[0]))
    n_frames = 1 + (x.shape[0] - window_size) // hop_size
    w = get_window(window, window_size)
    spec = np.zeros((window_size // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop_size
        frame = x[start : start + window_size] * w
        spec[:, i] = rfft(frame)
    return spec


def spectrogram(x: np.ndarray, sample_rate: float = 1.0, window_size: int = 256, hop_size: int = 128, window: str = "hann"):
    """Magnitude spectrogram plus time and frequency axes."""
    spec = stft(x, window_size=window_size, hop_size=hop_size, window=window)
    n_frames = spec.shape[1]
    times = np.arange(n_frames, dtype=float) * hop_size / float(sample_rate)
    freqs = _positive_frequencies(window_size, sample_rate)
    return np.abs(spec), times, freqs


def find_peaks(magnitude: np.ndarray, freqs: np.ndarray | None = None, k: int = 3, min_height: float | None = None):
    """Return the ``k`` tallest local maxima.

    A sample is a peak when it is strictly taller than its left neighbor
    and at least as tall as its right neighbor.
    """
    mag = np.asarray(magnitude, dtype=float).reshape(-1)
    if freqs is None:
        freqs_arr = np.arange(mag.shape[0], dtype=float)
    else:
        freqs_arr = np.asarray(freqs, dtype=float).reshape(-1)
        if freqs_arr.shape[0] != mag.shape[0]:
            raise ValueError("freqs and magnitude must have the same length")
    hits = []
    for i in range(1, mag.shape[0] - 1):
        if mag[i] > mag[i - 1] and mag[i] >= mag[i + 1]:
            if min_height is None or mag[i] >= min_height:
                hits.append(i)
    hits.sort(key=lambda i: mag[i], reverse=True)
    chosen = hits[:k]
    return freqs_arr[chosen], mag[chosen]


def dominant_frequency(x: np.ndarray, sample_rate: float = 1.0, window: str = "hann") -> float:
    """Frequency of the tallest positive-frequency periodogram bin, ignoring DC."""
    freqs, pxx = periodogram(x, sample_rate=sample_rate, window=window)
    if freqs.shape[0] < 2:
        return 0.0
    idx = 1 + int(np.argmax(pxx[1:]))
    return float(freqs[idx])


class SpectralAnalyzer:
    periodogram = staticmethod(periodogram)
    power_spectral_density = staticmethod(power_spectral_density)
    stft = staticmethod(stft)
    spectrogram = staticmethod(spectrogram)
    find_peaks = staticmethod(find_peaks)
    dominant_frequency = staticmethod(dominant_frequency)
    autocorrelation = staticmethod(autocorrelation)
