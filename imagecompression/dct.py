import numpy as np


def one_dim_dct2(vector):
    """
    Computes the 1D DCT-II of a vector using the direct formula
    with orthogonal normalization (norm='ortho').

    Args:
        vector (np.ndarray): 1D input array.

    Returns:
        np.ndarray: DCT-II transformed vector.
    """
    if vector.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    N = len(vector)
    vector = np.asarray(vector, dtype=float)

    # Normalization factors (orthonormal basis)
    ck = np.sqrt(2.0 / N) * np.ones(N)
    ck[0] = 1.0 / np.sqrt(N)

    # Cosine transform matrix
    cos_table = np.cos(
        np.pi * np.outer(2 * np.arange(N) + 1, np.arange(N)) / (2 * N)
    )

    # Apply transform
    return ck * np.dot(cos_table.T, vector)


def custom_dct2(matrix):
    """
    Computes the 2D DCT-II of a matrix using the separability property:
    apply 1D DCT-II to rows, then to columns.

    Args:
        matrix (np.ndarray): 2D input array (square or rectangular).

    Returns:
        np.ndarray: 2D DCT-II transformed matrix.
    """
    if matrix.ndim != 2:
        raise ValueError("Input must be a 2D array.")

    # Apply DCT-II row-wise, then column-wise
    temp = np.apply_along_axis(one_dim_dct2, axis=1, arr=matrix)
    return np.apply_along_axis(one_dim_dct2, axis=0, arr=temp)