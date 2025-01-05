# Initial values for Momentum and Nesterov methods
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
epsilon = 0.1  # learning rate
alpha = 0.9  # momentum coefficient
w_momentum = 2.0  # initial w for momentum method
w_nesterov = 2.0  # initial w for Nesterov method
v_momentum = 0.0  # initial velocity for momentum method
v_nesterov = 0.0  # initial velocity for Nesterov method

# To store results for 20 iterations
iterations = 30
momentum_results = []
nesterov_results = []

# Iterative calculations
for i in range(iterations):
    # Momentum Method
    grad_momentum = 2 * w_momentum
    v_momentum = alpha * v_momentum - epsilon * grad_momentum
    w_momentum = w_momentum + v_momentum
    momentum_results.append((i + 1, w_momentum, v_momentum))

    # Nesterov Method
    lookahead = w_nesterov + alpha * v_nesterov
    grad_nesterov = 2 * lookahead
    v_nesterov = alpha * v_nesterov - epsilon * grad_nesterov
    w_nesterov = w_nesterov + v_nesterov
    nesterov_results.append((i + 1, w_nesterov, v_nesterov))

# Create DataFrame for results
df_results = pd.DataFrame({
    "Iteration": [r[0] for r in momentum_results],
    "Momentum_w": [r[1] for r in momentum_results],
    "Momentum_v": [r[2] for r in momentum_results],
    "Nesterov_w": [r[1] for r in nesterov_results],
    "Nesterov_v": [r[2] for r in nesterov_results]
})

print(df_results)
# Plotting Momentum Method
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)  # First subplot
plt.plot(df_results["Iteration"], df_results["Momentum_w"], label="Momentum $w_t$", marker='o')
plt.plot(df_results["Iteration"], df_results["Momentum_v"], label="Momentum $v_t$", marker='x')
plt.title("Momentum Method")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Plotting Nesterov Method
plt.subplot(2, 1, 2)  # Second subplot
plt.plot(df_results["Iteration"], df_results["Nesterov_w"], label="Nesterov $w_t$", marker='o')
plt.plot(df_results["Iteration"], df_results["Nesterov_v"], label="Nesterov $v_t$", marker='x')
plt.title("Nesterov Method")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Show plots
plt.tight_layout()
plt.show()