from benchmark import benchmark_dct2
from plot_utils import plot_performance


def main():
    sizes = [2**i for i in range(5, 13)]
    custom_times, scipy_times = benchmark_dct2(sizes)
    plot_performance(sizes, custom_times, scipy_times)


if __name__ == '__main__':
    main()
