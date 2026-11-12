"""Sine, multi-tone, chirp, and noise generators."""

from __future__ import annotations

import numpy as np

from ..core.fft import fft_iterative, ifft_iterative, next_power_of_two


def _time_axis(duration: float, sample_rate: int) -> np.ndarray:
    n = int(round(duration * sample_rate))
    if n < 1:
        raise ValueError("duration and sample_rate produce an empty signal")
    return np.arange(n, dtype=float) / float(sample_rate)


def sine_wave(frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 1.0, phase: float = 0.0):
    t = _time_axis(duration, sample_rate)
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
    return signal, t


def multi_tone(frequencies, duration: float, sample_rate: int = 44100, amplitudes=None, phases=None):
    frequencies = list(frequencies)
    t = _time_axis(duration, sample_rate)
    if amplitudes is None:
        amplitudes = np.ones(len(frequencies), dtype=float)
    if phases is None:
        phases = np.zeros(len(frequencies), dtype=float)
    signal = np.zeros(t.shape[0], dtype=float)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += float(amp) * np.sin(2.0 * np.pi * float(freq) * t + float(phase))
    return signal, t


def chirp(f0: float, f1: float, duration: float, sample_rate: int = 44100, method: str = "linear"):
    """Sweep from f0 to f1. ``method`` is 'linear' or 'exponential'."""
    t = _time_axis(duration, sample_rate)
    if method == "linear":
        rate = (f1 - f0) / duration
        phase = 2.0 * np.pi * (f0 * t + 0.5 * rate * t * t)
    elif method == "exponential":
        if f0 <= 0.0 or f1 <= 0.0:
            raise ValueError("exponential chirp needs positive frequencies")
        growth = (f1 / f0) ** (1.0 / duration)
        phase = 2.0 * np.pi * f0 * (growth ** t - 1.0) / np.log(growth)
    else:
        raise ValueError("method must be 'linear' or 'exponential'")
    return np.sin(phase), t


def square_wave(frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 1.0):
    t = _time_axis(duration, sample_rate)
    return amplitude * np.sign(np.sin(2.0 * np.pi * frequency * t)), t


def sawtooth_wave(frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 1.0):
    t = _time_axis(duration, sample_rate)
    return amplitude * 2.0 * (t * frequency - np.floor(t * frequency + 0.5)), t


def white_noise(duration: float, sample_rate: int = 44100, amplitude: float = 1.0, seed: int = 0):
    n = int(round(duration * sample_rate))
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(n)


def pink_noise(duration: float, sample_rate: int = 44100, amplitude: float = 1.0, seed: int = 0):
    """Pink noise by shaping white noise with a 1/sqrt(f) spectrum.

    The filter is our FFT, not ``numpy.fft``. DC is removed, then the
    result is scaled to the requested amplitude (standard deviation).
    """
    n = int(round(duration * sample_rate))
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    nfft = next_power_of_two(n)
    spec = fft_iterative(np.pad(white, (0, nfft - n)))
    k = np.arange(nfft)
    folded = np.minimum(k, nfft - k).astype(float)
    folded[0] = 1.0
    spec = spec / np.sqrt(folded)
    spec[0] = 0.0
    pink = np.real(ifft_iterative(spec))[:n]
    scale = float(np.std(pink))
    if scale == 0.0:
        return pink
    return amplitude * pink / scale


def impulse(n: int, delay: int = 0, amplitude: float = 1.0):
    if n < 1:
        raise ValueError("n must be positive")
    out = np.zeros(n, dtype=float)
    out[int(delay) % n] = amplitude
    return out


class SignalGenerator:
    sine_wave = staticmethod(sine_wave)
    multi_tone = staticmethod(multi_tone)
    chirp = staticmethod(chirp)
    square_wave = staticmethod(square_wave)
    sawtooth_wave = staticmethod(sawtooth_wave)
    white_noise = staticmethod(white_noise)
    pink_noise = staticmethod(pink_noise)
    impulse = staticmethod(impulse)
