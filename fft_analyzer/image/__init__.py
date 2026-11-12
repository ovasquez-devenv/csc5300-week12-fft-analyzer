"""FFT image filters and JPEG-style DCT compression."""

from .compression import ImageCompressor
from .filters import ImageFilter

__all__ = ["ImageCompressor", "ImageFilter"]
