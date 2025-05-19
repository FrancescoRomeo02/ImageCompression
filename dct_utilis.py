import numpy as np
from scipy.fftpack import dct as scipy_dct


def one_dim_dct2(vector):
    """
    Computes the 1D DCT-II of a vector using the direct formula.
    Includes orthogonal normalization.

    Args:
      vector: A 1D NumPy array.

    Returns:
      The 1D DCT-II of the input vector.
    """
    N = len(vector)
    dct_vector = np.zeros(N, dtype=float)

    for k in range(N):
        ck = 1.0 / np.sqrt(N) if k == 0 else np.sqrt(2.0 / N)
        sum_val = 0.0
        for n in range(N):
            sum_val += vector[n] * np.cos(((2 * n + 1) * k * np.pi) / (2 * N))
        dct_vector[k] = ck * sum_val

    return dct_vector


def custom_dct2(matrix):
    """
    Computes the 2D DCT-II of a matrix using the separable property
    and the direct 1D DCT-II implementation.

    Args:
      matrix: A 2D NumPy array (must be square, N x N).

    Returns:
      The 2D DCT-II of the input matrix.
    """
    N = matrix.shape[0]
    if matrix.shape[1] != N:
        raise ValueError("Input matrix must be square.")

    # Step 1: Apply 1D DCT to each row
    intermediate_matrix = np.zeros_like(matrix, dtype=float)
    for i in range(N):
        intermediate_matrix[i, :] = one_dim_dct2(matrix[i, :])

    # Step 2: Apply 1D DCT to each column of the intermediate matrix
    result_matrix = np.zeros_like(matrix, dtype=float)

    for j in range(N):
        result_matrix[:, j] = one_dim_dct2(intermediate_matrix[:, j])

    return result_matrix


def scipy_dct2(matrix):
    """
    Computes 2D DCT of a matrix using the scipy method-

    Args:
      matrix: A 2 Numpy array (must be square, N x N).

    Returns:
      The 2D DCT of the input matrix
    """
    return scipy_dct(scipy_dct(matrix.T, norm='ortho').T, norm='ortho')
