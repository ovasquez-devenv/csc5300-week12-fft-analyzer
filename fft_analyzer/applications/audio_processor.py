"""Load and save WAV files, filter them, and build a spectrogram."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..signal.analysis import autocorrelation, spectrogram
from ..signal.filters import bandpass, highpass, lowpass


def load_wav(path: str | Path):
    """Read a PCM WAV file. Mono is returned as 1D, stereo is averaged.

    Samples come back as float64 in about [-1, 1].
    """
    path = Path(path)
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        nframes = handle.getnframes()
        raw = handle.readframes(nframes)
    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(float)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(float) / 2147483648.0
    else:
        raise ValueError("unsupported sample width: %d bytes" % width)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def save_wav(path: str | Path, signal: np.ndarray, sample_rate: int):
    """Write a mono 16-bit PCM WAV file. Values are clipped to [-1, 1]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sig = np.clip(np.asarray(signal, dtype=float).reshape(-1), -1.0, 1.0)
    pcm = np.round(sig * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def apply_filter(signal: np.ndarray, sample_rate: float, kind: str, **kwargs) -> np.ndarray:
    kind = kind.lower()
    if kind in ("lowpass", "low"):
        return lowpass(signal, sample_rate, kwargs["cutoff"])
    if kind in ("highpass", "high"):
        return highpass(signal, sample_rate, kwargs["cutoff"])
    if kind in ("bandpass", "band"):
        return bandpass(signal, sample_rate, kwargs["low"], kwargs["high"])
    raise ValueError("unknown filter %r" % kind)


class AudioProcessor:
    def __init__(self):
        self.signal = None
        self.sample_rate = None

    def load(self, path: str | Path):
        self.signal, self.sample_rate = load_wav(path)
        return self.signal, self.sample_rate

    def save(self, path: str | Path, signal: np.ndarray | None = None, sample_rate: int | None = None):
        if signal is None:
            signal = self.signal
        if sample_rate is None:
            sample_rate = self.sample_rate
        if signal is None or sample_rate is None:
            raise ValueError("nothing to save")
        save_wav(path, signal, sample_rate)

    def filter(self, kind: str, signal: np.ndarray | None = None, sample_rate: float | None = None, **kwargs):
        if signal is None:
            signal = self.signal
        if sample_rate is None:
            sample_rate = self.sample_rate
        return apply_filter(signal, sample_rate, kind, **kwargs)

    def spectrogram(self, signal: np.ndarray | None = None, sample_rate: float | None = None, window_size: int = 1024, hop_size: int = 256):
        if signal is None:
            signal = self.signal
        if sample_rate is None:
            sample_rate = self.sample_rate
        return spectrogram(signal, sample_rate=sample_rate, window_size=window_size, hop_size=hop_size)

    def autocorrelation(self, signal: np.ndarray | None = None):
        if signal is None:
            signal = self.signal
        return autocorrelation(signal)
