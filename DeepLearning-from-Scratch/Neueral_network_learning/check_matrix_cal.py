import numpy as np

# Define X and W with example values
x1, x2 = 1, 2  # Example values for X
w11, w12, w13 = 0.1, 0.2, 0.3  # Example values for W
w21, w22, w23 = 0.4, 0.5, 0.6

X = np.array([[x1, x2]])  # 1x2 matrix
W = np.array([[w11, w12, w13],
              [w21, w22, w23]])  # 2x3 matrix

# Compute Y
Y = X @ W  # 1x3 matrix
print("Y:", Y)

# Derivative wrt X
dY_dX = W.T  # Transposed W
print("dY/dX:")
print(dY_dX)

# Derivative wrt W
dY_dW = np.zeros_like(W)  # Initialize the derivative matrix
for i in range(W.shape[0]):  # Iterate over rows of W
    for j in range(W.shape[1]):  # Iterate over columns of W
        dY_dW[i, j] = X[0, i]  # Assign the corresponding value of X
print("dY/dW:", dY_dW)
