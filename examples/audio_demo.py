"""Build a short synthetic WAV, filter it, and save the spectrogram.

Run from the repo root:
    py -3.14 examples/audio_demo.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fft_analyzer.applications.audio_processor import AudioProcessor
from fft_analyzer.signal.generators import multi_tone, white_noise
from fft_analyzer.visualization.plots import plot_spectrogram, plot_time_and_frequency


def main():
    out = ROOT / "outputs" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    sample_rate = 4096
    duration = 1.0
    chord, _t = multi_tone(
        [220, 277, 330, 1800],
        duration=duration,
        sample_rate=sample_rate,
        amplitudes=[0.5, 0.35, 0.25, 0.3],
    )
    noise = white_noise(duration, sample_rate=sample_rate, amplitude=0.05, seed=7)
    signal = chord + noise
    # 4096 samples is already a power of two, so the filter bins line up.

    proc = AudioProcessor()
    proc.signal = signal
    proc.sample_rate = sample_rate
    proc.save(out / "original.wav")

    low = proc.filter("lowpass", cutoff=600)
    band = proc.filter("bandpass", low=200, high=400)
    proc.save(out / "lowpass.wav", low, sample_rate)
    proc.save(out / "bandpass.wav", band, sample_rate)

    plot_time_and_frequency(signal, sample_rate, out / "original_spectrum.png", title="Synthetic chord", max_freq=2200)
    plot_time_and_frequency(low, sample_rate, out / "lowpass_spectrum.png", title="Low-pass at 600 Hz", max_freq=2200)
    plot_spectrogram(
        signal,
        sample_rate,
        out / "spectrogram.png",
        window_size=512,
        hop_size=128,
        title="Spectrogram of the synthetic chord",
        max_freq=2200,
    )

    ac = proc.autocorrelation(signal)
    lags = np.arange(ac.shape[0])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(lags[:400], ac[:400], color="#1f4e79")
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation of the synthetic chord")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "autocorrelation.png", dpi=120)
    plt.close(fig)

    def energy(x):
        return float(np.dot(x, x))

    lines = [
        "audio demo",
        "sample_rate %d" % sample_rate,
        "samples %d" % signal.shape[0],
        "energy_original %.6f" % energy(signal),
        "energy_lowpass %.6f" % energy(low),
        "energy_bandpass %.6f" % energy(band),
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
