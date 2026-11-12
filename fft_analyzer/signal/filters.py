"""FFT brick-wall filters and Hann, Hamming, and Blackman windows."""

from __future__ import annotations

import numpy as np

from ..core.fft import fft_iterative, fftfreq, ifft_iterative, next_power_of_two


def hann(n: int) -> np.ndarray:
    """Hann window (often called Hanning). Length 1 returns [1]."""
    n = int(n)
    if n < 1:
        raise ValueError("window length must be positive")
    if n == 1:
        return np.ones(1, dtype=float)
    i = np.arange(n, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))


def hamming(n: int) -> np.ndarray:
    n = int(n)
    if n < 1:
        raise ValueError("window length must be positive")
    if n == 1:
        return np.ones(1, dtype=float)
    i = np.arange(n, dtype=float)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1))


def blackman(n: int) -> np.ndarray:
    n = int(n)
    if n < 1:
        raise ValueError("window length must be positive")
    if n == 1:
        return np.ones(1, dtype=float)
    i = np.arange(n, dtype=float)
    return (
        0.42
        - 0.5 * np.cos(2.0 * np.pi * i / (n - 1))
        + 0.08 * np.cos(4.0 * np.pi * i / (n - 1))
    )


def hanning(n: int) -> np.ndarray:
    """Alias for ``hann``. The assignment uses the Hanning name."""
    return hann(n)


WINDOWS = {
    "hann": hann,
    "hanning": hann,
    "hamming": hamming,
    "blackman": blackman,
    "rect": lambda n: np.ones(int(n), dtype=float),
    "rectangular": lambda n: np.ones(int(n), dtype=float),
}


def get_window(name: str, n: int) -> np.ndarray:
    key = name.lower()
    if key not in WINDOWS:
        raise ValueError("unknown window %r" % name)
    return WINDOWS[key](n)


def _apply_mask(signal: np.ndarray, sample_rate: float, keep) -> np.ndarray:
    x = np.asarray(signal, dtype=float).reshape(-1)
    n = x.shape[0]
    nfft = next_power_of_two(n)
    spec = fft_iterative(np.pad(x, (0, nfft - n)))
    freqs = fftfreq(nfft, d=1.0 / float(sample_rate))
    spec = spec * keep(freqs).astype(float)
    y = np.real(ifft_iterative(spec))
    return y[:n]


def lowpass(signal: np.ndarray, sample_rate: float, cutoff: float) -> np.ndarray:
    """Keep frequencies at or below ``cutoff`` Hz. Both sides of the spectrum."""
    return _apply_mask(signal, sample_rate, lambda freqs: np.abs(freqs) <= cutoff)


def highpass(signal: np.ndarray, sample_rate: float, cutoff: float) -> np.ndarray:
    """Drop frequencies below ``cutoff`` Hz."""
    return _apply_mask(signal, sample_rate, lambda freqs: np.abs(freqs) >= cutoff)


def bandpass(signal: np.ndarray, sample_rate: float, low: float, high: float) -> np.ndarray:
    """Keep frequencies from ``low`` through ``high`` Hz."""
    if low > high:
        raise ValueError("low cutoff is above high cutoff")

    def keep(freqs):
        mag = np.abs(freqs)
        return (mag >= low) & (mag <= high)

    return _apply_mask(signal, sample_rate, keep)


class AudioFilter:
    hann = staticmethod(hann)
    hanning = staticmethod(hanning)
    hamming = staticmethod(hamming)
    blackman = staticmethod(blackman)
    get_window = staticmethod(get_window)
    lowpass = staticmethod(lowpass)
    highpass = staticmethod(highpass)
    bandpass = staticmethod(bandpass)
