"""Period finding, convolution smoothing, and frequency-domain autocorrelation.

The series in the demos are synthetic. Nothing here reads a market file.
"""

from __future__ import annotations

import numpy as np

from ..core.fft import linear_convolve
from ..signal.analysis import autocorrelation, find_peaks, periodogram


def smooth(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving average through linear FFT convolution, cropped to the input length."""
    x = np.asarray(x, dtype=float).reshape(-1)
    window = int(window)
    if window < 1:
        raise ValueError("window must be positive")
    if window == 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    full = np.real(linear_convolve(x, kernel))
    start = (window - 1) // 2
    return full[start : start + x.shape[0]]


def dominant_period(x: np.ndarray, min_lag: int = 2, max_lag: int | None = None) -> int:
    """Lag of the strongest autocorrelation peak, skipping lag 0."""
    x = np.asarray(x, dtype=float).reshape(-1)
    ac = autocorrelation(x - np.mean(x))
    if max_lag is None:
        max_lag = max(min_lag + 1, x.shape[0] // 2)
    max_lag = min(int(max_lag), ac.shape[0] - 1)
    min_lag = max(1, int(min_lag))
    if max_lag <= min_lag:
        raise ValueError("lag range is empty")
    segment = ac[min_lag : max_lag + 1]
    return int(min_lag + np.argmax(segment))


def periodicities(x: np.ndarray, sample_rate: float = 1.0, top_k: int = 3, window: str = "hann"):
    """Strongest positive frequencies in the periodogram, DC excluded."""
    freqs, power = periodogram(x, sample_rate=sample_rate, window=window)
    if freqs.shape[0] < 3:
        return freqs[:0], power[:0]
    return find_peaks(power[1:], freqs=freqs[1:], k=top_k)


def seasonality(x: np.ndarray, sample_rate: float = 1.0):
    """Dominant period in samples, and the matching frequency if a sample rate is set."""
    lag = dominant_period(x)
    frequency = float(sample_rate) / lag if lag else 0.0
    return {"period_samples": lag, "frequency": frequency}


class TimeSeriesAnalyzer:
    smooth = staticmethod(smooth)
    dominant_period = staticmethod(dominant_period)
    periodicities = staticmethod(periodicities)
    seasonality = staticmethod(seasonality)
    autocorrelation = staticmethod(autocorrelation)
