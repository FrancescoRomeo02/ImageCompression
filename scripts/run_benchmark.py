from imagecompression.benchmark import benchmark_dct2
from imagecompression.plot import plot_performance
from imagecompression.io import save_benchmark_csv
import datetime

def main():
    sizes = [2**i for i in range(3, 13)] 
    seed = 42

    print("Running DCT2 benchmark...")
    custom_times, scipy_times = benchmark_dct2(sizes, repetitions=10, seed=seed)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'results/benchmark_{timestamp}.csv'
    plot_path = f'results/performance_plot_{timestamp}.png'

    save_benchmark_csv(custom_times, scipy_times, sizes, filename=csv_path)
    plot_performance(sizes, custom_times, scipy_times, save_path=plot_path)

    print("Benchmark completed successfully.")

if __name__ == '__main__':
    main()