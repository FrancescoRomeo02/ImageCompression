from benchmark import benchmark_dct2
from plot_utils import plot_performance
from log_benchmark_csv import save_benchmark_csv

def main():
    sizes = [2**i for i in range(4, 11)]
    custom_times, scipy_times = benchmark_dct2(sizes)
    plot_performance(sizes, custom_times, scipy_times)
    save_benchmark_csv(custom_times, scipy_times, sizes)

if __name__ == '__main__':
    main()
