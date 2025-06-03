from scipy.fft import dct, dctn
import numpy as np

# Define 8x8 input matrix (image block)
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

# Expected 2D DCT-II result for the full 8x8 matrix
expected_dct2_matrix = np.array([
    [1111.0000, 44.0000, 75.9000, -138.0000, 3.5000, 122.0000, 195.0000, -101.0000],
    [77.1000, 114.0000, -21.8000, 41.3000, 8.7700, 99.0000, 138.0000, 10.9000],
    [44.8000, -62.7000, 111.0000, -76.3000, 124.0000, 95.5000, -39.8000, 58.5000],
    [-69.9000, -40.2000, -23.4000, -76.7000, 26.6000, -36.8000, 66.1000, 125.0000],
    [-109.0000, -43.3000, -55.5000, 8.1700, 30.2000, -28.6000, 2.4400, -94.1000],
    [-5.3800, 56.6000, 173.0000, -35.4000, 32.3000, 33.4000, -58.1000, 19.0000],
    [78.8000, -64.5000, 118.0000, -15.0000, -137.0000, -30.6000, -105.0000, 39.8000],
    [19.7000, -78.1000, 0.9720, -72.3000, -21.5000, 81.3000, 63.7000, 5.9000]
], dtype=float)

# Expected DCT-II result for the first row only (1D DCT)
expected_dct1_row1 = np.array([
    401.0000, 6.6000, 109.0000, -112.0000, 65.4000, 121.0000, 116.0000, 28.8000
], dtype=float)

# Compute DCT-II 1D on the first row
first_row = input_matrix[0, :]
computed_dct1_row1 = dct(first_row, norm='ortho')

print("Comparing computed DCT-II (1D) on first row against expected values:\n")
for i in range(len(computed_dct1_row1)):
    diff = computed_dct1_row1[i] - expected_dct1_row1[i]
    print(f"Element {i}: Computed = {computed_dct1_row1[i]:.7e}, "
          f"Expected = {expected_dct1_row1[i]:.7e}, "
          f"Difference = {diff:.7e}")

# Compute 2D DCT-II on the full matrix
computed_dct2_matrix = dctn(input_matrix, norm='ortho')

print("\n\nComparing computed DCT-II (2D) top-left row against expected values:\n")
for j in range(8):
    diff = computed_dct2_matrix[0, j] - expected_dct2_matrix[0, j]
    print(f"Element [0, {j}]: Computed = {computed_dct2_matrix[0, j]:.7e}, "
          f"Expected = {expected_dct2_matrix[0, j]:.7e}, "
          f"Difference = {diff:.7e}")

# Single final comparison for bottom-right element
i, j = 7, 7
diff = computed_dct2_matrix[i, j] - expected_dct2_matrix[i, j]
print(f"\nElement [{i}, {j}]: Computed = {computed_dct2_matrix[i, j]:.7e}, "
      f"Expected = {expected_dct2_matrix[i, j]:.7e}, "
      f"Difference = {diff:.7e}")

# Compute error metrics
mse_dct2 = np.mean((computed_dct2_matrix - expected_dct2_matrix) ** 2)
max_rel_err = np.max(np.abs((computed_dct2_matrix - expected_dct2_matrix) / expected_dct2_matrix))

print(f"\nMean Squared Error (MSE) for 2D DCT-II: {mse_dct2:.7e}")
print(f"Max Relative Error for 2D DCT-II: {max_rel_err:.7e}")

# Use np.allclose for overall test assertions
rtol, atol = 1e-2, 1e-2 # Relative and absolute tolerances
is_dct1_match = np.allclose(computed_dct1_row1, expected_dct1_row1, rtol=rtol, atol=atol)
is_dct2_match = np.allclose(computed_dct2_matrix, expected_dct2_matrix, rtol=rtol, atol=atol)

print(f"\n--- DCT-II 1D Verification (First Row) ---")
if is_dct1_match:
    print("✅ PASS: Computed DCT matches expected within tolerance.")
else:
    print("❌ FAIL: Computed DCT does not match expected values.")
    print("Max absolute difference:", np.max(np.abs(computed_dct1_row1 - expected_dct1_row1)))

print(f"\n--- DCT-II 2D Verification (Full Matrix) ---")
if is_dct2_match:
    print("✅ PASS: Computed 2D DCT matches expected within tolerance.")
else:
    print("❌ FAIL: Computed 2D DCT does not match expected values.")
    print("Max absolute difference:", np.max(np.abs(computed_dct2_matrix - expected_dct2_matrix)))

print("\nVerification complete.")