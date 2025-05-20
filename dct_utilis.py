import numpy as np
from scipy.fftpack import dct as scipy_dct
from scipy.fftpack import idct as scipy_idct


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
    vector = np.asarray(vector, dtype=float)
    dct_vector = np.zeros(N, dtype=float)

    # Precompute normalization factors ck
    ck = np.sqrt(2.0 / N) * np.ones(N)
    ck[0] = 1.0 / np.sqrt(N)

    # Precompute cosine terms
    cos_table = np.cos(
        np.pi * np.outer(2 * np.arange(N) + 1, np.arange(N)) / (2 * N))

    # Matrix-style computation for better cache efficiency
    dct_vector = ck * np.dot(cos_table.T, vector)

    return dct_vector


def custom_dct2(matrix):
    """
    Computes the 2D DCT-II of a square matrix using the separable property
    and an optimized 1D DCT-II implementation.
    """
    N, M = matrix.shape
    if N != M:
        raise ValueError("Input matrix must be square.")

    # First DCT along rows
    temp = np.apply_along_axis(one_dim_dct2, axis=1, arr=matrix)

    # Then DCT along columns
    result = np.apply_along_axis(one_dim_dct2, axis=0, arr=temp)

    return result


def scipy_dct2(matrix):
    """
    Computes 2D DCT of a matrix using the scipy method-

    Args:
      matrix: A 2 Numpy array (must be square, N x N).

    Returns:
      The 2D DCT of the input matrix
    """
    return scipy_dct(scipy_dct(matrix.T, norm='ortho').T, norm='ortho')


def idct2(coeffs):
    """
    Computes 2D IDCT of a matrix using the scipy method-

    Args:
      matrix: A 2 Numpy array (must be square, N x N).

    Returns:
      The 2D IDCT of the input matrix
    """

    return scipy_idct(scipy_idct(coeffs.T, norm='ortho').T, norm='ortho')
