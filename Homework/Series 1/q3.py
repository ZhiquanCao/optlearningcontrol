import numpy as np

# Define Q
Q = np.array([
    [100, 2, 1],
    [2, 10, 3],
    [1, 3, 1]
])

# Define A and b
A = np.array([[1, 1, 1]])
b = np.array([1])

# Compute Q inverse
Q_inv = np.linalg.inv(Q)

# Compute M = A Q_inv A^T
M = A @ Q_inv @ A.T  # This is a scalar

# Compute lambda
lambda_value = - np.linalg.solve(M, b)  # lambda = - b / M

# Compute x
x = - Q_inv @ A.T @ lambda_value  # x = - Q_inv A^T lambda

# Print results
print("Q inverse:\n", Q_inv)
print("\nM:\n", M)
print("\nLambda:\n", lambda_value)
print("\nx:\n", x.flatten())

# Verify that sum of x equals 1
print("\nSum of x:", x.sum())
