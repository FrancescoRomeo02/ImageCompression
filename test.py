
import numpy as np
from dct_utilis import scipy_dct2
input_matrix = np.array([
    [231, 32, 233, 161, 24, 71, 140, 245],
    [247, 40, 248, 245, 124, 204, 36, 107],
    [234, 202, 245, 167, 9, 217, 239, 173],
    [193, 190, 100, 167, 43, 180, 8, 70],
    [11, 24, 210, 177, 81, 243, 8, 112],
    [97, 195, 203, 47, 125, 114, 165, 181],
    [193, 70, 174, 167, 41, 30, 127, 245],
    [87, 149, 57, 192, 65, 129, 178, 228]
], dtype=float)

expected_dct2_matrix = np.array([
    [1.11e+03, 4.40e+01, 7.59e+01, -1.38e+02,
     3.50e+00, 1.22e+02, 1.95e+02, -1.01e+02],
    [7.71e+01, 1.14e+02, -2.18e+01, 4.13e+01,
     8.77e+00, 9.90e+01, 1.38e+02, 1.09e+01],
    [4.48e+01, -6.27e+01, 1.11e+02, -7.63e+01,
     1.24e+02, 9.55e+01, -3.98e+01, 5.85e+01],
    [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01,
     2.66e+01, -3.68e+01, 6.61e+01, 1.25e+02],
    [-1.09e+02, -4.33e+01, -5.55e+01, 8.17e+00,
     3.02e+01, -2.86e+01, 2.44e+00, -9.41e+01],
    [-5.38e+00, 5.66e+01, 1.73e+02, -3.54e+01,
     3.23e+01, 3.34e+01, -5.81e+01, 1.90e+01],
    [7.88e+01, -6.45e+01, 1.18e+02, -1.50e+01,
     -1.37e+02, -3.06e+01, -1.05e+02, 3.98e+01],
    [1.97e+01, -7.81e+01, 9.72e-01, -7.23e+01,
     -2.15e+01, 8.13e+01, 6.37e+01, 5.90e+00]
], dtype=float)

# Prima riga della matrice di input
first_row = input_matrix[0, :]

# Output atteso della DCT-II 1D per la prima riga (normalizzazione ortogonale)
expected_dct1_row1 = np.array([
    4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02, 6.54e+01,
    1.21e+02, 1.16e+02, 2.88e+01
], dtype=float)

print("Eseguendo le trasformazioni con scipy.fftpack.dct e confrontando i risultati...")

# Calcola la DCT-II 1D della prima riga usando scipy, con normalizzazione ortogonale
# type=2 specifica la DCT-II standard
from scipy.fftpack import dct
computed_dct1_row1 = dct(first_row, norm='ortho')

# Calcola la DCT-II 2D dell'intera matrice usando scipy.
# Per la 2D con la funzione 1D dct di fftpack, la si applica prima per le colonne (axis=0)
# e poi sul risultato per le righe (axis=1), mantenendo la normalizzazione.
computed_dct2_matrix = scipy_dct2(input_matrix)
                          

# Confronta i risultati usando np.allclose per gestire le imprecisioni floating point.
rtol = 1e-4
atol = 1e-4
is_dct1_match = np.allclose(
    computed_dct1_row1, expected_dct1_row1, rtol=rtol, atol=atol
)
is_dct2_match = np.allclose(
    computed_dct2_matrix, expected_dct2_matrix, rtol=rtol, atol=atol
)
# --- Output dei risultati della verifica con dettagli in caso di fallimento ---
print(
    f"\n--- Verifica DCT-II 1D (Prima Riga, Tolleranza rtol={rtol}, atol={atol}) ---")
if is_dct1_match:
    print("VERIFICA 1D SUPERATA: Il risultato calcolato corrisponde all'atteso (entro tolleranza).")
else:
    print("VERIFICA 1D FALLITA: Il risultato calcolato NON corrisponde all'atteso.")
    # Stampa le differenze per debugging
    print("\nRisultato Calcolato (1D):\n", computed_dct1_row1)
    print("\nRisultato Atteso (1D):\n", expected_dct1_row1)
    print("\nDifferenza Assoluta Massima (1D):", np.max(
        np.abs(computed_dct1_row1 - expected_dct1_row1)))
print(
    f"\n--- Verifica DCT-II 2D (Matrice Completa, Tolleranza rtol={rtol}, atol={atol}) ---")
if is_dct2_match:
    print("VERIFICA 2D SUPERATA: Il risultato calcolato corrisponde all'atteso (entro tolleranza).")
else:
    print("VERIFICA 2D FALLITA: Il risultato calcolato NON corrisponde all'atteso.")
    # Stampa le differenze per debugging
    print("\nRisultato Calcolato (2D):\n", computed_dct2_matrix)
    print("\nRisultato Atteso (2D):\n", expected_dct2_matrix)
    print("\nDifferenza Assoluta Massima (2D):", np.max(
        np.abs(computed_dct2_matrix - expected_dct2_matrix)))
print("\nVerifica completata.")
