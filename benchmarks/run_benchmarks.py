"""Time the FFTs, the matrix products, and a Hilbert condition check.

Results land in benchmarks/results as a CSV and three PNG plots.
Pass --label to name the machine in machine.txt and on the plot titles.
If you omit it, the label is the hostname, or "my PC" when that name is empty.
"""

import argparse
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fft_analyzer.core.fft import dft_naive, fft_iterative, fft_recursive
from fft_analyzer.core.matrix import matmul_classical, strassen_matmul
from fft_analyzer.core.numerical import condition_number, hilbert, perturbed_solution_change
from fft_analyzer.visualization.plots import plot_condition_numbers, plot_series


def _default_label():
    name = (platform.node() or "").strip()
    return name if name else "my PC"


def _cpu_model():
    model = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return model


def _time_call(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _repeats_for_fft(n, method):
    if method == "naive" and n >= 1024:
        return 1
    if method in ("naive", "recursive"):
        return 3
    if n >= 2**18:
        return 1
    if n >= 2**14:
        return 3
    return 5


def run_fft(rows):
    naive_sizes = [32, 64, 128, 256, 512, 1024]
    recursive_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    iter_sizes = [2**k for k in range(5, 21)]
    rng = np.random.default_rng(0)
    cache = {}

    def signal(n):
        if n not in cache:
            cache[n] = rng.normal(size=n)
        return cache[n]

    plans = (
        ("naive", naive_sizes, dft_naive),
        ("recursive", recursive_sizes, fft_recursive),
        ("iterative", iter_sizes, fft_iterative),
        ("numpy", iter_sizes, np.fft.fft),
    )
    for method, sizes, fn in plans:
        for n in sizes:
            x = signal(n)
            repeats = _repeats_for_fft(n, method)
            # One untimed call so the first measured run is not an import spike.
            if method in ("iterative", "numpy") and n <= 2**16:
                fn(x)
            seconds = _time_call(lambda: fn(x), repeats)
            rows.append(("fft", n, method, seconds, repeats))
            print("fft n=%d method=%s seconds=%.6f" % (n, method, seconds), flush=True)


def run_matmul(rows):
    rng = np.random.default_rng(1)
    sizes = [8, 16, 32, 48, 64, 96, 128]
    for n in sizes:
        A = rng.normal(size=(n, n))
        B = rng.normal(size=(n, n))
        repeats = 5 if n <= 64 else 3
        jobs = (
            ("classical", (lambda A=A, B=B: matmul_classical(A, B))),
            ("strassen", (lambda A=A, B=B: strassen_matmul(A, B, leaf=16))),
        )
        for method, fn in jobs:
            seconds = _time_call(fn, repeats)
            rows.append(("matmul", n, method, seconds, repeats))
            print("matmul n=%d method=%s seconds=%.6f" % (n, method, seconds), flush=True)


def run_condition(rows):
    sizes = []
    conds = []
    errors = []
    for n in range(2, 11):
        matrix = hilbert(n)
        cond = condition_number(matrix, p=2)
        change = perturbed_solution_change(matrix, np.ones(n), noise=1e-8, seed=0)
        rel = change["relative_change"]
        sizes.append(n)
        conds.append(cond)
        errors.append(rel)
        rows.append(("condition", n, "hilbert_cond", cond, 1))
        rows.append(("condition", n, "relative_solution_change", rel, 1))
        print("hilbert n=%d cond=%.6e rel_change=%.6e" % (n, cond, rel), flush=True)
    return sizes, conds, errors


def write_csv(path, rows):
    lines = ["group,size,method,seconds,repeats"]
    for group, size, method, seconds, repeats in rows:
        lines.append("%s,%d,%s,%.10g,%d" % (group, size, method, seconds, repeats))
    path.write_text("\n".join(lines) + "\n")


def _series(rows, group, method, sizes):
    lookup = {(r[1], r[2]): r[3] for r in rows if r[0] == group}
    return [lookup.get((n, method)) for n in sizes]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Time FFTs, matrix products, and a Hilbert check.")
    parser.add_argument(
        "--label",
        default=None,
        help="Name stored in machine.txt. Defaults to the hostname, or 'my PC'.",
    )
    args = parser.parse_args(argv)
    label = args.label.strip() if args.label else _default_label()
    out = ROOT / "benchmarks" / "results"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    run_fft(rows)
    run_matmul(rows)
    sizes, conds, errors = run_condition(rows)
    write_csv(out / "benchmarks.csv", rows)

    fft_sizes = [2**k for k in range(5, 21)]
    plot_series(
        fft_sizes,
        [
            _series(rows, "fft", "naive", fft_sizes),
            _series(rows, "fft", "recursive", fft_sizes),
            _series(rows, "fft", "iterative", fft_sizes),
            _series(rows, "fft", "numpy", fft_sizes),
        ],
        ["naive DFT", "recursive FFT", "iterative FFT", "numpy.fft"],
        out / "fft_timing.png",
        xlabel="length",
        ylabel="seconds (median)",
        title="FFT timing on %s" % label,
        loglog=True,
    )
    mat_sizes = [8, 16, 32, 48, 64, 96, 128]
    plot_series(
        mat_sizes,
        [
            _series(rows, "matmul", "classical", mat_sizes),
            _series(rows, "matmul", "strassen", mat_sizes),
        ],
        ["classical triple loop", "Strassen"],
        out / "matmul_timing.png",
        xlabel="n",
        ylabel="seconds",
        title="Square matrix multiply on %s" % label,
        loglog=True,
    )
    plot_condition_numbers(sizes, conds, errors, out / "condition_error.png")

    note = "\n".join(
        [
            "label: %s" % label,
            "python: %s" % platform.python_version(),
            "system: %s %s" % (platform.system(), platform.release()),
            "machine: %s" % platform.machine(),
            "cpu: %s" % _cpu_model(),
            "cpus: %s" % (os_cpu_count()),
        ]
    )
    (out / "machine.txt").write_text(note + "\n")
    print(note)


def os_cpu_count():
    import os

    return os.cpu_count()


if __name__ == "__main__":
    main()
