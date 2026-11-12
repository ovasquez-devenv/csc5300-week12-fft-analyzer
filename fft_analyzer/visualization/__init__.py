"""Saved plots and GIF animations. The Agg backend is selected on import."""

import matplotlib

matplotlib.use("Agg")

from .animations import animate_fft_process, animate_filtering
from .plots import (
    plot_condition_numbers,
    plot_fft2_heatmap,
    plot_series,
    plot_spectrogram,
    plot_time_and_frequency,
)

__all__ = [
    "animate_fft_process",
    "animate_filtering",
    "plot_condition_numbers",
    "plot_fft2_heatmap",
    "plot_series",
    "plot_spectrogram",
    "plot_time_and_frequency",
]
