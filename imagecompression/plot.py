import matplotlib.pyplot as plt
import numpy as np
import os


def plot_performance(sizes, custom_times, scipy_times,
                     save_path='results/performance_plot.png'):
    """
    Generate a semilog performance plot comparing custom DCT2 vs SciPy DCT2,
    with reference curves for O(N^3) and O(N^2 log N) scaled for visual comparison.

    Args:
        sizes (list[int]): Matrix sizes N tested.
        custom_times (list[float]): Execution times for custom DCT2.
        scipy_times (list[float]): Execution times for SciPy DCT2.
        save_path (str): Output path for the PNG plot.

    Returns:
        str: Path to saved plot.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Actual performance data (in shades of blue)
    plt.semilogy(sizes, custom_times, 'o-', color="#1f21b4", label='Custom DCT2')
    plt.semilogy(sizes, scipy_times, 's-', color="#0088cc", label='SciPy DCT2')

    # Reference complexity curves
    n_vals = np.array(sizes)
    n3_scaled = (n_vals**3) / max(n_vals**3) * max(custom_times)
    n2logn_scaled = (n_vals**2 * np.log2(n_vals)) / max(n_vals**2 * np.log2(n_vals)) * max(scipy_times)

    # Plot reference curves (dashed, lighter blues)
    plt.semilogy(n_vals, n3_scaled, '--', color="#1f21b4", label='O(N³) scaled')
    plt.semilogy(n_vals, n2logn_scaled, '--', color="#0088cc", label='O(N² log N) scaled')

    # Plot styling
    plt.title('DCT2 Performance Comparison with Theoretical References', fontsize=14)
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Execution Time (s) [log scale]', fontsize=12)
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Performance plot saved to: {save_path}")
    return save_path