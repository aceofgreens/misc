import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

def simulate_pid_lag(Kp, Ki, Kd, tau=0.5, dt=0.01, T=20.0, seed=42, smooth_window=101, ax=None, c='m', plot_start=False):
    """
    Simulate 2D trajectory tracking with a PID controller
    driving a first-order lag plant on each axis, and plot only the PID result.
    Uses simple moving-average smoothing for reference trajectory.
    """
    # Time vector
    t = np.arange(0, T, dt)
    
    # Generate wiggly reference trajectory (smoothed noise integration)
    np.random.seed(seed)
    noise_x = np.random.randn(len(t))
    noise_y = np.random.randn(len(t))
    # moving-average smoothing
    window = np.ones(smooth_window) / smooth_window
    smooth_x = np.convolve(noise_x, window, mode='same')
    smooth_y = np.convolve(noise_y, window, mode='same')
    ref_x = np.cumsum(smooth_x) * dt
    ref_y = np.cumsum(smooth_y) * dt
    ref_x = (ref_x - np.mean(ref_x)) * 1.2
    ref_y = (ref_y - np.mean(ref_y)) * 1.2

    # Initialize state and error
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    int_x = int_y = 0.0
    prev_ex = prev_ey = 0.0
    errors = []

    # Simulation loop
    for k in range(len(t) - 1):
        # Compute errors
        ex = ref_x[k] - x[k]
        ey = ref_y[k] - y[k]
        errors.append(np.sqrt(ex**2 + ey**2))
        # Integrate and differentiate
        int_x += ex * dt
        int_y += ey * dt
        der_x = (ex - prev_ex) / dt
        der_y = (ey - prev_ey) / dt
        prev_ex, prev_ey = ex, ey
        # PID control action
        u_x = Kp * ex + Ki * int_x + Kd * der_x
        u_y = Kp * ey + Ki * int_y + Kd * der_y
        # First-order lag dynamics
        vx[k+1] = vx[k] + (u_x - vx[k]) / tau * dt
        vy[k+1] = vy[k] + (u_y - vy[k]) / tau * dt
        # Position update
        x[k+1] = x[k] + vx[k] * dt
        y[k+1] = y[k] + vy[k] * dt

    # Plot only PID-controlled trajectory vs reference
    if plot_start:
        ax.plot(ref_x, ref_y, 'k--', linewidth=2.5, label='Reference')
        ax.scatter(ref_x[0], ref_y[0], c='cyan', edgecolor='k', marker='o', zorder=4, s=100, label='Start (Ref)')
        ax.scatter(x[0], y[0], c='red', marker='^', s=100, edgecolor='k', zorder=4, label='Start (PID)')

    ax.plot(x, y, color=c, linewidth=2.0, label=f'$K_p$ = {Kp}, $K_I$ = {Ki}, $K_d$ = {Kd}, Error = {np.round(np.mean(errors), 3)}')
    plt.xlabel('x position')
    plt.ylabel('y position')

# Example call
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
simulate_pid_lag(Kp=10.0, Ki=0.0, Kd=0.0, ax=axs, c='C1', plot_start=True)
simulate_pid_lag(Kp=10.0, Ki=1., Kd=0.0, ax=axs, c='green')
simulate_pid_lag(Kp=10.0, Ki=1., Kd=2.0, ax=axs, c='C0')
axs.set_ylim([-0.25, 0.25])
plt.legend(loc='lower left')
axs.grid(True)
axs.set_title(f'Different Controllers on a Lagged System')
plt.savefig("pid_control.png", bbox_inches='tight', dpi=300)
plt.show()
