# Initial values for Momentum and Nesterov methods
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
learing_rate = 0.1  
alpha = 0.9  
theta_momentum = 2.0  
theta_nesterov = 2.0  
theta_new_nesterov = 2.0 

m_t_momentum = 0.0 
m_t_nesterov = 0.0 
m_t_new_nesterov = 0.0 

iterations = 10
momentum_results = []
nesterov_results = []
new_nag_results = []

for i in range(iterations):
    # Momentum Method
    g_t_momentum = 2 * theta_momentum
    m_t_momentum = alpha * m_t_momentum + learing_rate * g_t_momentum
    theta_momentum = theta_momentum - m_t_momentum
    momentum_results.append((i + 1, theta_momentum, m_t_momentum))

    # Nesterov Method
    lookahead = theta_nesterov - alpha * m_t_nesterov
    g_t_nesterov = 2 * lookahead 
    m_t_nesterov = alpha * m_t_nesterov + learing_rate * g_t_nesterov
    theta_nesterov = theta_nesterov - m_t_nesterov
    nesterov_results.append((i + 1, theta_nesterov, m_t_nesterov))

    # New Nesterov Method
    g_t_new_nesterov = 2 * theta_new_nesterov 
    m_t_new_nesterov = alpha * m_t_new_nesterov + learing_rate * g_t_new_nesterov
    theta_new_nesterov = theta_new_nesterov - (alpha *m_t_new_nesterov + learing_rate * g_t_new_nesterov)
    new_nag_results.append((i + 1, theta_new_nesterov, m_t_new_nesterov))

# Create DataFrame for results
df_results = pd.DataFrame({
    "Iteration": [r[0] for r in momentum_results],
    "Momentum_theta": [r[1] for r in momentum_results],
    "Momentum_m_t": [r[2] for r in momentum_results],

    "Nesterov_theta": [r[1] for r in nesterov_results],
    "Nesterov_m_t": [r[2] for r in nesterov_results],

    "new_Nesterov_theta": [r[1] for r in new_nag_results],
    "new_Nesterov_m_t": [r[2] for r in new_nag_results]
})

print(df_results)

# Plotting combined graph for Nesterov and New Nesterov methods
plt.figure(figsize=(10, 6))

# Nesterov Method
plt.plot(df_results["Iteration"], df_results["Nesterov_theta"], label="Nesterov $\\theta$", marker='o', linestyle='-')
plt.plot(df_results["Iteration"], df_results["Nesterov_m_t"], label="Nesterov $m_t$", marker='x', linestyle='-')

# New Nesterov Method
plt.plot(df_results["Iteration"], df_results["new_Nesterov_theta"], label="New Nesterov $\\theta$", marker='o', linestyle='--')
plt.plot(df_results["Iteration"], df_results["new_Nesterov_m_t"], label="New Nesterov $m_t$", marker='x', linestyle='--')

# Graph title and labels
plt.title("Comparison of Nesterov and New Nesterov Methods")
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.grid()

# Show combined plot
plt.tight_layout()
plt.show()