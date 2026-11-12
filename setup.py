"""Install FFTAnalyzer. The graded algorithms are pure Python plus NumPy."""

from setuptools import find_packages, setup

setup(
    name="fft_analyzer",
    version="1.0.0",
    description="CSC 5300 Week 12 FFT and numerical toolkit",
    packages=find_packages(exclude=("tests", "examples", "benchmarks")),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "matplotlib",
        "pillow",
    ],
)
