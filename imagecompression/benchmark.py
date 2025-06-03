import numpy as np
import time
from .dct import custom_dct2
from scipy.fft import dctn


def benchmark_dct2(sizes, repetitions=10, seed=None):
    """
    Benchmark custom DCT2 vs SciPy's dctn on square matrices of given sizes.

    Args:
        sizes (list[int]): List of matrix sizes N to test (NxN matrices).
        repetitions (int): Number of repetitions per test to average timing.
        seed (int, optional): Seed for random matrix generation (for reproducibility).

    Returns:
        tuple: (custom_times, scipy_times), both lists of averaged execution times.
    """
    if seed is not None:
        np.random.seed(seed)

    custom_times = []
    scipy_times = []

    for N in sizes:
        matrix = np.random.uniform(0, 255, (N, N))

        custom_total = 0.0
        scipy_total = 0.0

        for _ in range(repetitions):
            # Custom DCT2 timing
            start = time.perf_counter()
            custom_dct2(matrix)
            custom_total += time.perf_counter() - start

            # SciPy DCT2 timing
            start = time.perf_counter()
            dctn(matrix, type=2, norm='ortho')
            scipy_total += time.perf_counter() - start

        custom_times.append(custom_total / repetitions)
        scipy_times.append(scipy_total / repetitions)

    return custom_times, scipy_times