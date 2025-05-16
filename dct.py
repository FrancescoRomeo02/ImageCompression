import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fftpack import dct as scipy_dct


def custom_dct2(matrix):
    """
    Implementazione manuale della DCT-2 bidimensionale

    Args:
        matrix (numpy.ndarray): Matrice di input quadrata

    Returns:
        numpy.ndarray: Matrice trasformata DCT-2
    """
    N = matrix.shape[0]
    dct_matrix = np.zeros_like(matrix, dtype=float)

    for u in range(N):
        for v in range(N):
            # Calcolo dei coefficienti C(u) e C(v)
            ck = 1.0 / np.sqrt(N) if u == 0 else np.sqrt(2.0 / N)
            cl = 1.0 / np.sqrt(N) if v == 0 else np.sqrt(2.0 / N)

            # Calcolo della somma dei coefficienti DCT
            sum_value = 0.0
            for x in range(N):
                for y in range(N):
                    cosine_term = np.cos(((2*x + 1) * u * np.pi) / (2*N)) * \
                        np.cos(((2*y + 1) * v * np.pi) / (2*N))
                    sum_value += matrix[x, y] * cosine_term

            dct_matrix[u, v] = ck * cl * sum_value

    return dct_matrix


def benchmark_dct2(sizes):
    """
    Confronta le prestazioni della DCT2 personalizzata e della libreria

    Args:
        sizes (list): Lista di dimensioni delle matrici da testare

    Returns:
        tuple: Tempi per la DCT2 personalizzata e della libreria
    """
    custom_times = []
    scipy_times = []

    for N in sizes:
        # Genera una matrice casuale
        matrix = np.random.rand(N, N)
        # Misura il tempo per la DCT2 personalizzata
        start = time.time()
        custom_dct2(matrix)
        custom_times.append(time.time() - start)
        # Misura il tempo per la DCT2 di SciPy
        start = time.time()
        scipy_dct(scipy_dct(matrix, type=2, norm=None, axis = 0), type=2, norm=None, axis = 1)
        scipy_times.append(time.time() - start)

    return custom_times, scipy_times


def plot_performance():
    """
    Genera un grafico delle prestazioni in scala semilogaritmica
    """
    # Dimensioni delle matrici da testare
    sizes = [8, 16, 32]
    custom_times, scipy_times = benchmark_dct2(sizes)

    plt.figure(figsize=(10, 6))
    plt.semilogy(sizes, custom_times, 'bo-', label='DCT2 Personalizzata')
    plt.semilogy(sizes, scipy_times, 'ro-', label='DCT2 SciPy')
    plt.title('Confronto Prestazioni Implementazioni DCT-2')
    plt.xlabel('Dimensione Matrice (N)')
    plt.ylabel('Tempo di Esecuzione (secondi) - Scala Logaritmica')
    plt.grid(True)
    plt.legend()

    plt.savefig('plot/dct2_performance_comparison.png')
    plt.close()

    print("Dimensioni Matrici:", sizes)
    print("Tempi DCT2 Personalizzata:", custom_times)
    print("Tempi DCT2 SciPy:", scipy_times)


if __name__ == '__main__':
    plot_performance()
