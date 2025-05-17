
import numpy as np
from scipy.fftpack import dct, idct

A = np.array([
    [231,  32, 233, 161,  24,  71, 140, 245],
    [247,  40, 248, 245, 124, 204,  36, 107],
    [234, 202, 245, 167,   9, 217, 239, 173],
    [193, 190, 100, 167,  43, 180,   8,  70],
    [ 11,  24, 210, 177,  81, 243,   8, 112],
    [ 97, 195, 203,  47, 125, 114, 165, 181],
    [193,  70, 174, 167,  41,  30, 127, 245],
    [ 87, 149,  57, 192,  65, 129, 178, 228]
], dtype=np.float64)

vector = np.array([231, 32, 233, 161, 24, 71, 140, 245], dtype=np.float64)

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

C = dct2(A)
A_rec = idct2(C)
vector = dct(vector, type=2, norm='ortho')

print("APPLICO DCT2\n",C)
print("RICOSTRUISCO \n",A_rec)

print("APPLICO DCT MONO AL VETTORE DI TEST: \n",vector)