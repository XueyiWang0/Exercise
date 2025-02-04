import matplotlib.pyplot as plt
import numpy as np

# Aufgabe 1.1
x = np.linspace(0,5,200)
theta_vals = np.linspace(0,2,10)

plt.figure(figsize=(10,6))

for theta in theta_vals:
    y = np.cos(np.pi*theta*x) * np.exp(-x)
    plt.plot(x, y, label=f"theta: {theta:.2f}")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()


v = np.array([1, -3, 10])
M = np.array([[1, 3, 7],
              [-5, -10, 2],
              [2, 5, 8]])

# 1.
w = np.cos((np.pi/4.) * v)
print("w:\n", w)

# 2.
z = v + 3 * w
norm_z = np.linalg.norm(z)
print("Norm of z:", norm_z)

# 3.
matrix_product = np.matmul(M,z)
print("Mz=\n", matrix_product)

# 4.
# Transpose
M_transpose = M.T
print("Transpose:\n", M_transpose)

# Determinant
M_determinant = np.linalg.det(M)
print("Determinant:", M_determinant)

# Smallest element
smallest_element = np.min(M)
print("Smallest element:", smallest_element)

# Sum of rows
sum_of_rows = np.sum(M, axis=1)
print("Sum of rows:\n", sum_of_rows)