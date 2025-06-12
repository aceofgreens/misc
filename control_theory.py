import numpy as np
import matplotlib.pyplot as plt

# System parameters
K = 1.0     # System gain
tau = 1.0   # Time constant

# Time-domain simulation
dt = 0.001
t = np.arange(0, 30, dt)

# Example input: cosine at low frequency and high frequency
omega_low = 0.75    # low frequency (rad/s)
omega_high = 5.0  # high frequency (rad/s)

A = 3
u_low = A*np.cos(omega_low * t)
u_high = A*np.cos(omega_high * t)

# Impulse response h(t) = (K/tau) * exp(-t/tau) for t >= 0
h = (K / tau) * np.exp(-t / tau)

# Convolve input with impulse response (discrete approximation)
y_low = np.convolve(u_low, h) * dt
y_low = y_low[:len(t)]  # trim to original length

y_high = np.convolve(u_high, h) * dt
y_high = y_high[:len(t)]  # trim to original length

# Plotting time-domain responses
plt.figure(figsize=(12, 8))

# Low-frequency input/output
plt.subplot(2, 2, 1)
plt.plot(t, u_low, label=f'Input: {A}cos({omega_low}t)', linewidth=1.5)
plt.plot(t, y_low, label='Output (low freq)', linestyle='--', linewidth=1.5)
plt.title(f'Low-Frequency Input (ω = {omega_low} rad/s)', fontsize='large')
plt.xlabel('Time [s]', fontsize='large')
plt.ylabel('Amplitude', fontsize='large')
plt.legend()
plt.grid(True)

# High-frequency input/output
plt.subplot(2, 2, 2)
plt.plot(t, u_high, label=f'Input: {A}cos({omega_high}t)', linewidth=1.5)
plt.plot(t, y_high, label='Output (high freq)', linestyle='--', linewidth=1.5)
plt.title(f'High-Frequency Input (ω = {omega_high} rad/s)', fontsize='large')
plt.xlabel('Time [s]', fontsize='large')
plt.ylabel('Amplitude', fontsize='large')
plt.legend()
plt.grid(True)

# plt.tight_layout()
# plt.show()

# Bode plot calculation
frequencies = np.logspace(-2, 2, 500)  # from 0.01 to 100 rad/s
omega = frequencies  # frequencies interpreted as rad/s

H = K / (1 + 1j * omega * tau)

# Magnitude in dB and phase in degrees
magnitude_db = 20 * np.log10(np.abs(H))
phase_deg = np.angle(H, deg=True)

# Plot Bode diagram
# plt.figure(figsize=(10, 8))

# Magnitude plot
plt.subplot(2, 2, 3)
plt.semilogx(frequencies, magnitude_db, linewidth=2.5)
plt.title('Bode Gain Plot for First-Order System', fontsize='large')
plt.ylabel('Magnitude [dB]', fontsize='large')
plt.xlabel('Frequency [rad/s]', fontsize='large')
# plt.vlines([omega_low, omega_high], ymin=-50, ymax=0, linestyles='--', linewidth=1.5, colors='k')
plt.scatter([omega_low, omega_high], y=[-2, -14], s=150, marker='^', edgecolors='k', c=['red', 'cyan'], zorder=4)
plt.text(x=omega_low-0.45, y=-6, s='Low Freq', fontsize='large')
plt.text(x=omega_high-3.7, y=-18, s='High Freq', fontsize='large')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Phase plot
plt.subplot(2, 2, 4)
plt.semilogx(frequencies, phase_deg, linewidth=2.5)
plt.title('Bode Phase Plot for First-Order System', fontsize='large')
plt.xlabel('Frequency [rad/s]', fontsize='large')
plt.ylabel('Phase [degrees]', fontsize='large')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.scatter([omega_low, omega_high], y=[-36, -78], s=150, marker='^', edgecolors='k', c=['red', 'cyan'], zorder=4)
plt.text(x=omega_low-0.55, y=-45, s='Low Freq', fontsize='large')
plt.text(x=omega_high-3.7, y=-86, s='High Freq', fontsize='large')
plt.tight_layout()
plt.savefig("bode_plot.png", bbox_inches='tight', dpi=300)
plt.show()

