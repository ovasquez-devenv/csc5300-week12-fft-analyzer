"""FFTAnalyzer: matrix routines, FFTs, and signal and image tools.

The numerical algorithms live in this package. NumPy is the array
container. Transforms are checked against NumPy in the tests.
"""

__version__ = "1.0.0"

from .core.fft import FFT, dft_naive, fft, fft_iterative, fft_recursive, ifft
from .core.matrix import MatrixOps
from .core.numerical import NumericalStability
from .signal.generators import SignalGenerator
from .signal.filters import AudioFilter
from .signal.analysis import SpectralAnalyzer
from .image.filters import ImageFilter
from .image.compression import ImageCompressor
from .applications.audio_processor import AudioProcessor
from .applications.image_processor import ImageProcessor
from .applications.data_analyzer import TimeSeriesAnalyzer

__all__ = [
    "FFT",
    "MatrixOps",
    "NumericalStability",
    "SignalGenerator",
    "AudioFilter",
    "SpectralAnalyzer",
    "ImageFilter",
    "ImageCompressor",
    "AudioProcessor",
    "ImageProcessor",
    "TimeSeriesAnalyzer",
    "dft_naive",
    "fft",
    "fft_iterative",
    "fft_recursive",
    "ifft",
    "__version__",
]
