"""Windows, FFT filters, generators, spectra, and the time-series helpers."""

import numpy as np
import pytest

from fft_analyzer.applications.audio_processor import autocorrelation, load_wav, save_wav
from fft_analyzer.applications.data_analyzer import TimeSeriesAnalyzer
from fft_analyzer.core.fft import fft_iterative, fftfreq
from fft_analyzer.signal.analysis import dominant_frequency, find_peaks, periodogram, spectrogram
from fft_analyzer.signal.filters import bandpass, blackman, hamming, hann, highpass, lowpass
from fft_analyzer.signal.generators import chirp, multi_tone, pink_noise, sine_wave, white_noise
from fft_analyzer.visualization.animations import animate_fft_process
from fft_analyzer.visualization.plots import plot_time_and_frequency


def _tone(freq, n=1024, sample_rate=1024.0):
    t = np.arange(n) / sample_rate
    return np.sin(2.0 * np.pi * freq * t)


def _bin_magnitude(signal, sample_rate, freq):
    spec = fft_iterative(signal)
    freqs = fftfreq(len(signal), d=1.0 / sample_rate)
    idx = int(np.argmin(np.abs(freqs - freq)))
    return abs(spec[idx])


def test_windows_match_numpy():
    n = 64
    assert np.allclose(hann(n), np.hanning(n))
    assert np.allclose(hamming(n), np.hamming(n))
    assert np.allclose(blackman(n), np.blackman(n))
    assert hann(1).shape == (1,)


def test_lowpass_highpass_bandpass_remove_the_right_tones():
    sample_rate = 1024.0
    signal = _tone(40) + 0.5 * _tone(300) + 0.25 * _tone(420)
    low = lowpass(signal, sample_rate, 100)
    high = highpass(signal, sample_rate, 350)
    band = bandpass(signal, sample_rate, 250, 350)
    assert _bin_magnitude(low, sample_rate, 40) > 100
    assert _bin_magnitude(low, sample_rate, 300) < 1e-6
    assert _bin_magnitude(high, sample_rate, 420) > 50
    assert _bin_magnitude(high, sample_rate, 40) < 1e-6
    assert _bin_magnitude(band, sample_rate, 300) > 100
    assert _bin_magnitude(band, sample_rate, 40) < 1e-6
    assert _bin_magnitude(band, sample_rate, 420) < 1e-6


def test_sine_and_multitone_peaks():
    signal, _t = sine_wave(50, duration=1.0, sample_rate=1024)
    assert abs(dominant_frequency(signal, sample_rate=1024) - 50.0) <= 1.0
    mixed, _t = multi_tone([40, 120], duration=1.0, sample_rate=1024, amplitudes=[1.0, 0.5])
    freqs, power = periodogram(mixed, sample_rate=1024, window="rect")
    found, _mags = find_peaks(power, freqs=freqs, k=2, min_height=np.max(power) * 0.05)
    assert 40.0 in np.round(found)
    assert 120.0 in np.round(found)


def test_chirp_spectrogram_rises():
    signal, _t = chirp(40, 300, duration=1.0, sample_rate=1024)
    mag, _times, freqs = spectrogram(signal, sample_rate=1024, window_size=128, hop_size=64)
    first = freqs[int(np.argmax(mag[:, 0]))]
    last = freqs[int(np.argmax(mag[:, -1]))]
    assert last > first + 50


def test_noise_shapes():
    white = white_noise(0.25, sample_rate=2048, amplitude=1.5, seed=1)
    assert white.shape == (512,)
    assert abs(np.std(white) - 1.5) < 0.2
    pink = pink_noise(1.0, sample_rate=4096, amplitude=1.0, seed=2)
    freqs, power = periodogram(pink, sample_rate=4096, window="hann")
    low = np.mean(power[(freqs > 20) & (freqs < 80)])
    high = np.mean(power[(freqs > 800) & (freqs < 1600)])
    assert low > high * 2.0


def test_autocorrelation_of_a_sine_and_wav_roundtrip(tmp_path):
    n = 400
    t = np.arange(n)
    signal = np.sin(2.0 * np.pi * t / 20.0)
    ac = autocorrelation(signal)
    direct = np.correlate(signal, signal, mode="full")[n - 1 :]
    assert np.allclose(ac, direct, atol=1e-6)
    path = tmp_path / "tone.wav"
    save_wav(path, 0.5 * signal[:256], 8000)
    loaded, rate = load_wav(path)
    assert rate == 8000
    assert loaded.shape == (256,)
    assert np.allclose(loaded, 0.5 * signal[:256], atol=1e-4)


def test_time_series_period_and_smoothing():
    t = np.arange(240)
    series = np.sin(2.0 * np.pi * t / 12.0) + 0.01 * t
    assert TimeSeriesAnalyzer.dominant_period(series) == 12
    info = TimeSeriesAnalyzer.seasonality(series, sample_rate=1.0)
    assert info["period_samples"] == 12
    freqs, _power = TimeSeriesAnalyzer.periodicities(series, sample_rate=1.0, top_k=1)
    assert abs(freqs[0] - 1.0 / 12.0) < 0.02
    x = np.arange(20, dtype=float)
    smoothed = TimeSeriesAnalyzer.smooth(x, window=5)
    expected = np.convolve(x, np.ones(5) / 5.0, mode="same")
    assert np.allclose(smoothed, expected, atol=1e-8)


def test_plot_and_animation_write_files(tmp_path):
    signal, _t = multi_tone([30, 80], duration=0.5, sample_rate=256)
    plot_path = plot_time_and_frequency(signal, 256, tmp_path / "spec.png", max_freq=120)
    gif_path = animate_fft_process(signal, tmp_path / "fft.gif")
    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert gif_path.exists() and gif_path.stat().st_size > 0
