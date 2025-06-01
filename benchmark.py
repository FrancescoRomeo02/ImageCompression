import numpy as np
import time
from dct_utilis import custom_dct2
from scipy.fft import dctn


def benchmark_dct2(sizes):
    custom_times = []
    scipy_times = []
    for N in sizes:
        matrix = np.random.uniform(0, 255, (N, N))
        
        # repeat the test 10 times for each size to get an average time
        custom_time = 0
        scipy_time = 0
       
        for _ in range(10):
            start_time = time.time()
            custom_dct2(matrix)
            custom_time += time.time() - start_time
            start_time = time.time()
            dctn(matrix, type=2, norm='ortho')
            scipy_time += time.time() - start_time
       
        custom_times.append(custom_time / 10)
        scipy_times.append(scipy_time / 10)

    return custom_times, scipy_times
