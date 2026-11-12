# FFTAnalyzer

I'm Omar Vasquez. This is my CSC 5300 Week 12 project, a small toolkit for the matrix routines and the fast transforms in Amakobe, Chapter 11 (Advanced Computational Algorithms, 2nd ed., 2026).

It includes classical and Strassen matrix multiplication, LU with partial pivoting, Householder QR, a power method, condition numbers, a recursive and an iterative Cooley-Tukey FFT, convolution, FFT filters, signal generators, periodograms and spectrograms, FFT image filters, JPEG-style DCT compression, WAV load and save, and a few saved plots and GIFs. The algorithms are in this repo. NumPy is the array container, and the tests compare against NumPy.

The longer write-up is [analysis/chapter11_report.md](analysis/chapter11_report.md). A short pointer also sits at [docs/chapter11_report.md](docs/chapter11_report.md).

## Setup

From the repo root, on Windows:

```
py -3.14 -m pip install -r requirements.txt
```

The same install works with `python -m pip` if `py` is not the launcher you use. Dependencies are numpy, matplotlib, pillow, and pytest. I stayed away from SciPy so the install does not depend on a wheel that might be missing for Python 3.14. Figures use the Agg backend and GIFs are written with Pillow, so ffmpeg is not required.

You can also install the package itself with `py -3.14 -m pip install -e .` The examples add the repo root to the path, so that step is optional.

## Tests

```
py -3.14 -m pytest
```

## Examples

These generate a synthetic tone, a synthetic WAV, and a drawn test image. They do not download anything. Outputs go under `outputs/`.

```
py -3.14 examples/signal_demo.py
py -3.14 examples/audio_demo.py
py -3.14 examples/image_demo.py
```

## Benchmarks

```
py -3.14 benchmarks/run_benchmarks.py
```

That writes `benchmarks/results/benchmarks.csv` plus timing and condition-number plots. The copy in the repo was measured on the build VM (Python 3.12.3, Linux, Intel Xeon, 4 CPUs). I will rerun it on my own PC before I turn the project in. The report quotes those measured numbers.

## Layout

```
fft_analyzer/core          FFT, matrix factorizations, stability helpers
fft_analyzer/signal        windows, filters, generators, spectra
fft_analyzer/image         FFT filters and DCT compression
fft_analyzer/applications  WAV, image, and time-series workflows
fft_analyzer/visualization plots and GIF animations
tests/                     pytest suite
examples/                  headless demos
benchmarks/                timing script and saved results
analysis/                  Chapter 11 report
```
