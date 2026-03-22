import numpy as np
import matplotlib.pyplot as plt

def simulate_hull_white(theta_func, a, sigma, r0, T, dt, K):
    """
    Simulates K realizations of the Hull-White model.
    
    Parameters:
    theta_func : function - Time-dependent drift term theta(t)
    a          : float    - Mean reversion speed
    sigma      : float    - Volatility
    r0         : float    - Initial interest rate
    T          : float    - Total time (years)
    dt         : float    - Time step size
    K          : int      - Number of paths (realizations)
    """
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps)
    rates = np.zeros((num_steps, K))
    rates[0] = r0
    
    for i in range(1, num_steps):
        # Current time for the drift function
        current_t = t[i-1]
        
        # Standard Brownian increment
        dW = np.random.normal(0, np.sqrt(dt), K)
        
        # Euler-Maruyama discretization
        # dr = [theta(t) - a*r]dt + sigma*dW
        dr = (theta_func(current_t) - a * rates[i-1]) * dt + sigma * dW
        rates[i] = rates[i-1] + dr
        
    return t, rates

# --- Example Usage ---

# 1. Define the time-dependent drift function theta(t)
# For example, a sine wave to represent seasonal adjustments
def my_theta(t):
    return  np.abs(0.2*np.sin(t)) + 0.5

# 2. Parameters
a_param = 1.5      # Mean reversion speed
sigma_param = 0.03 # Volatility
r_init = 0.02      # Starting at 2%
total_time = 10    # 10 years
time_step = 0.01   # dt
num_paths = 10     # Number of realizations

# 3. Run Simulation
time_axis, simulated_rates = simulate_hull_white(
    my_theta, a_param, sigma_param, r_init, total_time, time_step, num_paths
)

# 4. Plotting
plt.figure(figsize=(10, 6))
plt.plot(time_axis, simulated_rates, lw=0.8)
plt.title(f"Hull-White Model: {num_paths} Realizations")
plt.xlabel("Time (Years)")
plt.ylabel("Short Rate $r_t$")
plt.grid(True, alpha=0.3)
#plt.ylim([0, 1])
plt.show()
