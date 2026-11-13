# Chapter 11 report: FFTAnalyzer

Word count: 1446.

## Executive summary

FFTAnalyzer is my Week 12 package for the matrix routines and the fast transforms in Chapter 11. I wrote the algorithms. NumPy stores the arrays, and the tests use np.fft and np.linalg only as a check.

The iterative Cooley-Tukey FFT is the result I want up front. Each butterfly stage is one NumPy operation on the whole block. On my PC it is about 2,490 times faster than the naive DFT at n = 1024, and at n = 2^20, np.fft is about 10 times faster than my iterative FFT. The Hilbert condition number grows from about 19 at n = 2 to about 1.6e13 at n = 10. The CSV and the plots are in benchmarks/results. What follows is what I built, what I measured, and what I think it means.

## Numerical algorithm theory

Amakobe (2026, section 11.5) opens with the triple loop. On a square matrix the work grows with the cube of the side length. Strassen then cuts the matrix into 2 by 2 blocks and replaces eight products with seven, so the recurrence grows a little more slowly than the cube. Extra additions and weaker rounding are why libraries usually skip it. I built it anyway, with the triple loop left as the leaf at 16 by 16. On my PC the classical product wins at n = 16, and from n = 32 up Strassen is faster.

LU with partial pivoting factors a permutation, a lower triangle, and an upper triangle, pivoting on the largest remaining column entry. The determinant is the swap sign times the product of the upper diagonal. A zero pivot returns zero, and the solver raises. Substitution finishes the triangles, and the inverse is one solve per identity column. Householder QR gives an orthogonal factor and an upper triangle. The power method multiplies with the triple loop and stops on the Rayleigh quotient. On the matrix with twos on the diagonal and ones off it, that value is the eigenvalue the matrix is built to have.

Section 11.6 is the floating-point part. I measure absolute and relative error and the one, two, and infinity norms. The matrix two norm is the square root of the dominant eigenvalue of the product of the transpose and the matrix, and the condition number multiplies a norm by the norm of the inverse. I call a matrix ill-conditioned once that number is very large. The Hilbert matrix is the chapter's example, and it crosses that line well before size ten.

## FFT and spectral analysis

From there, section 11.2 writes the DFT as a sum of complex exponentials, one per bin. Done directly, every bin sees every sample, so the work grows with the square of the length. Section 11.3 splits even and odd indexes. The Master Theorem makes that Cooley-Tukey step grow like the length times the log of the length. I coded the recursive form, which follows the split, and the iterative form, which starts in bit-reversed order. The inverse conjugates the spectrum, runs the FFT, conjugates again, and divides by the length.

Circular convolution is a product of FFTs, and both signals have to share a power-of-two length, because padding would change the period. Linear convolution pads to the next power of two and keeps the full ordinary result. Polynomial multiplication is that convolution (section 11.4). Autocorrelation is the inverse FFT of the squared magnitude.

Filters are brick-wall low-pass, high-pass, and band-pass masks, and Hann, Hamming, and Blackman are the cosine windows. The periodogram scales windowed power by the sample rate times the window energy. A short-time FFT makes the spectrogram, so a changing tone becomes a ridge. On synthetic data a low tone plus a higher harmonic peaked on two bins, the lower one taller. The rising chirp is the ridge below, and a sine of a known period came back at that lag.

![Chirp spectrogram](../outputs/signal/spectrogram.png)

The picture code runs that same FFT along each row, then each column.

## Image and audio results

Gaussian blur multiplies by a smooth frequency bump whose width follows the blur. Sobel and Laplacian edges are zero-padded FFT convolutions, and hard denoising keeps a disk of low frequencies. The demo picture is drawn in code: a ramp, a slow sine, a rectangle, a disk, and noise. Sobel magnitude pulls out the outlines.

![Sobel edges](../outputs/image/edges.png)

Compression uses an orthonormal DCT on small blocks and the JPEG luminance table. The table is stretched hard below the middle of the quality scale and eased off above it. Entries never reach zero, so the top quality still rounds. On the noisy demo the compression error drops a lot from quality 10 to quality 90, and quality 10 looks blocky.

![Compression error](../outputs/image/compression_error.png)

Hard low-pass denoising made this sharp-edged test picture worse. The error against the clean drawing rose after the cutoff. The rectangle and the disk are sharp, so the cutoff smears edges more than it removes noise. On a smooth sine in the tests, the same cutoff lowers the error.

Audio is 16-bit mono through the wave module. The clip is a few tones plus a little noise. A low-pass over the lower tones cut the energy, and a band-pass around those tones left it about there. The spectrogram shows the tones as steady lines, which is why I checked the arithmetic next.

## Stability considerations

Adding one to a huge power of ten, then subtracting that power of ten, comes back as zero. One is not a step the format can represent beside a number that large. Adding the huge value, then one, then its negative is also zero, and Kahan misses it too, because the correction falls into the same gap. On a long string of tenths, Kahan lands on the round total and the plain sum misses it.

The schoolbook quadratic formula cancels on a tiny root. I compute the root that stays well scaled, then recover the tiny one from the product of the roots. The error is smaller that way, and the test checks it.

For the Hilbert systems, a tiny nudge in one fixed direction barely moves the solution at n = 2 and moves it much more at n = 10. The path dips at one size in the middle. The condition number keeps climbing through that dip, from about 19 at n = 2 to about 1.6e13 at n = 10. One direction is a single sample, so the change can sit below the condition number times the nudge. Pivoted LU still factors these matrices through n = 10. The identity is perfectly conditioned, and the same nudge barely moves it.

## Benchmarking plots

The times are medians from benchmarks/run_benchmarks.py on my PC (Windows 11, Python 3.14.7, 12 CPUs). The CSV is benchmarks/results/benchmarks.csv. Slow cases are a single run, and the rest use several repeats.

![FFT timing](../benchmarks/results/fft_timing.png)

At n = 1024 the iterative FFT is about 2,490 times faster than the naive DFT, and the recursive form sits between them. At n = 2^20, np.fft is about 10 times faster than my iterative FFT.

![Matrix multiply timing](../benchmarks/results/matmul_timing.png)

Classical multiplication wins at n = 16, where the leaf means the Strassen split never runs. From n = 32 up, Strassen is faster. When the side doubles at the larger sizes, the classical time grows by about the factor a cubic method predicts, and Strassen grows a bit more slowly.

![Hilbert condition numbers and solution change](../benchmarks/results/condition_error.png)

The smooth curve is the condition number, from about 19 at n = 2 to about 1.6e13 at n = 10, and the bumpier curve is the solution change, dip included.

## Final reflections

The FFT is cheap enough to live inside ordinary audio and image code (Amakobe, 2026, sections 11.1 and 11.7). I can walk the recursive form by hand, and the iterative form finishes a long buffer. A frequency product in this package is convolution, polynomial multiplication, and the autocorrelation on the synthetic sine.

The triple loop grew the way a cubic method should when the side length doubled. Strassen was faster from n = 32 up and slower at the n = 16 leaf. I trust the condition number more than a tiny residual, because a Hilbert solve can look fine and still jump when the right-hand side moves a little.

The tests check the FFT against np.fft, including a long round trip, plus convolution, LU and QR reconstruction, the filters, compression error falling as quality rises, and the stability checks. The examples write a synthetic WAV, a drawn image, and the plots under outputs. Nothing is downloaded.

## Reference

Amakobe, M. (2026). *Advanced computational algorithms* (2nd ed.).
