import matplotlib.pyplot as plt
import numpy as np


def plot_performance(sizes, custom_times, scipy_times,
                     save_path='./image.png'):
    """
    Genera un grafico semilogaritmico delle prestazioni della DCT2 personalizzata
    e SciPy, includendo curve teoriche 𝒪(N^3) e 𝒪(N^2 log N) riscalate per confronto visivo.

    Args:
        sizes (list[int]): Dimensioni delle matrici (N) testate.
        custom_times (list[float]): Tempi DCT2 personalizzata.
        scipy_times (list[float]): Tempi DCT2 SciPy.
        save_path (str): Percorso del file PNG del grafico.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    # Dati reali
    plt.semilogy(sizes, custom_times, 'bo-', label='DCT2 Personalizzata')
    plt.semilogy(sizes, scipy_times, 'ro-', label='DCT2 SciPy')

    # Curve teoriche normalizzascipy.fft.tscipy.fft.e
    n_vals = np.array(sizes)
    n3_raw = n_vals**3
    n2logn_raw = n_vals**2 * np.log2(n_vals)

    # Riscalamento per confronto visivo (match massimo reale)
    n3_scaled = n3_raw / max(n3_raw) * max(custom_times)
    n2logn_scaled = n2logn_raw / max(n2logn_raw) * max(scipy_times)

    plt.semilogy(n_vals, n3_scaled, 'b--', alpha=0.5,
                 label='O(N^3) (ref)')
    plt.semilogy(n_vals, n2logn_scaled, 'r--', alpha=0.5,
                 label='O(N^2 log N) (ref)')

    # Asse Y ripulito per semplicità visiva
    plt.tick_params(axis='y', which='minor', left=False)
    plt.tick_params(axis='y', which='major', length=5)

    # Etichette e salvataggio
    plt.title('Confronto Prestazioni DCT2 – Curve Teoriche Riscalate')
    plt.xlabel('Dimensione Matrice (N)')
    plt.ylabel('')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
