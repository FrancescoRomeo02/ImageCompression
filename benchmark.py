import numpy as np
import time
from dct_utilis import custom_dct2
from scipy.fft import dctn


def benchmark_dct2(sizes):
    custom_times = []
    scipy_times = []
    for N in sizes:
        matrix = np.random.rand(N, N)
        start = time.perf_counter()
        custom_dct2(matrix)
        custom_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        dctn(matrix, norm='ortho')
        scipy_times.append(time.perf_counter() - start)
    return custom_times, scipy_times
