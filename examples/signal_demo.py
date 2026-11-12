"""Synthetic tones, a chirp spectrogram, and two short GIFs.

Run from the repo root:
    py -3.14 examples/signal_demo.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fft_analyzer.signal.analysis import dominant_frequency, find_peaks, periodogram
from fft_analyzer.signal.generators import chirp, multi_tone, white_noise
from fft_analyzer.visualization.animations import animate_fft_process, animate_filtering
from fft_analyzer.visualization.plots import plot_spectrogram, plot_time_and_frequency


def main():
    out = ROOT / "outputs" / "signal"
    out.mkdir(parents=True, exist_ok=True)
    sample_rate = 1024
    tones, _t = multi_tone(
        [60, 180],
        duration=1.0,
        sample_rate=sample_rate,
        amplitudes=[1.0, 0.4],
    )
    noise = white_noise(1.0, sample_rate=sample_rate, amplitude=0.15, seed=12)
    signal = tones + noise
    plot_time_and_frequency(
        signal,
        sample_rate,
        out / "time_frequency.png",
        title="Two tones plus a little noise",
        max_freq=400,
    )
    freqs, power = periodogram(signal, sample_rate=sample_rate, window="hann")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    mask = freqs <= 400
    ax.plot(freqs[mask], power[mask], color="#1f4e79")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Periodogram of the two-tone signal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "psd.png", dpi=120)
    plt.close(fig)

    sweep, _t = chirp(50, 350, duration=1.0, sample_rate=sample_rate)
    plot_spectrogram(
        sweep,
        sample_rate,
        out / "spectrogram.png",
        window_size=128,
        hop_size=32,
        title="Linear chirp from 50 Hz to 350 Hz",
        max_freq=500,
    )
    animate_fft_process(signal[:256], out / "fft_animation.gif")
    animate_filtering(signal, sample_rate, out / "filter_animation.gif")

    found, mags = find_peaks(power, freqs=freqs, k=2, min_height=np_max_safe(power) * 0.1)
    lines = [
        "signal demo",
        "sample_rate %d" % sample_rate,
        "dominant_hz %.4f" % dominant_frequency(signal, sample_rate=sample_rate),
        "peaks_hz %s" % ", ".join("%.3f" % f for f in found),
        "peak_power %s" % ", ".join("%.6g" % m for m in mags),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, end="")


def np_max_safe(values):
    import numpy as np

    return float(np.max(values))


if __name__ == "__main__":
    main()
