"""Audio, image, and time-series workflows."""

from .audio_processor import AudioProcessor
from .data_analyzer import TimeSeriesAnalyzer
from .image_processor import ImageProcessor

__all__ = ["AudioProcessor", "ImageProcessor", "TimeSeriesAnalyzer"]
