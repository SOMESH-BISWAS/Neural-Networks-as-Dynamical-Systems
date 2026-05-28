import numpy as np
from scipy.integrate import solve_ivp

def f_dynamics(t, h, W, b):
    # The continuous derivative function dh/dt
    return np.tanh(np.dot(W, h) + b)

# Model Parameters and Initial Condition
W = np.array([[-0.5, 0.8], [-1.2, 0.3]])
b = np.array([0.1, -0.2])
h0 = np.array([1.0, 0.5])  

# 1. Discrete ResNet Step (Euler Method, dt=1.0)
h_euler = h0 + 1.0 * f_dynamics(0, h0, W, b)

# 2. Continuous Neural ODE (solve_ivp)
solution = solve_ivp(
    fun=lambda t, h: f_dynamics(t, h, W, b),
    t_span=(0.0, 1.0),
    y0=h0,
    method='RK45'
)
h_continuous = solution.y[:, -1]

print(f"Discrete (ResNet) Output:   {h_euler}")
print(f"Continuous (Neural ODE):    {h_continuous}")
