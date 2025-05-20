import numpy as np
from scipy.fftpack import dct as scipy_dct
from scipy.fftpack import idct


def one_dim_dct2(vector):
    N = len(vector)
    vector = np.asarray(vector, dtype=float)
    dct_vector = np.zeros(N, dtype=float)

    alpha = np.sqrt(2.0 / N) * np.ones(N)
    alpha[0] = 1.0 / np.sqrt(N)

    cos_table = np.cos(
        np.pi * np.outer(2 * np.arange(N) + 1, np.arange(N)) / (2 * N))

    dct_vector = alpha * np.dot(cos_table.T, vector)

    return dct_vector

def custom_dct2(matrix):
    N, M = matrix.shape
    if N != M:
        raise ValueError("Input matrix must be square.")
    temp = np.apply_along_axis(one_dim_dct2, axis=1, arr=matrix)
    result = np.apply_along_axis(one_dim_dct2, axis=0, arr=temp)
    return result


def scipy_dct2(matrix):
    return scipy_dct(scipy_dct(matrix.T, norm='ortho').T, norm='ortho')

def idct2(coeffs): 
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')