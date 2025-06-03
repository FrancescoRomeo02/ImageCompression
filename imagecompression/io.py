import csv
import os


def save_benchmark_csv(custom_times, scipy_times, sizes, filename="benchmark_results.csv"):
    """
    Save benchmark results comparing custom DCT vs SciPy DCT to a CSV file in scientific notation.

    Args:
        custom_times (list of float): Execution times for the custom DCT implementation.
        scipy_times (list of float): Execution times for the SciPy DCT implementation.
        sizes (list of int): Sizes of the square matrices tested.
        filename (str): Name of the CSV file (default: 'benchmark_results.csv').

    Returns:
        str: Full path to the saved CSV file.
    """
    # Ensure 'results/' directory exists
    filepath = os.path.join(filename)

    # Write CSV file
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Matrix_Size', 'Custom_Time_s', 'SciPy_Time_s'])

        for i, size in enumerate(sizes):
            custom = custom_times[i]
            scipy_ = scipy_times[i]

            row = [
                size,
                f"{custom:.2e}",
                f"{scipy_:.2e}",
            ]
            writer.writerow(row)

    print(f"✅ Benchmark results saved to: {filepath}")
    return filepath