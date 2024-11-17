import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.special import sph_harm

# Define the angles for spherical coordinates
phi = np.linspace(0, np.pi, 200)
theta = np.linspace(0, 2 * np.pi, 200)
phi, theta = np.meshgrid(phi, theta)

# Define Cartesian coordinates of the unit sphere
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Calculate the spherical harmonic Y(l,m)
m1, l1 = 1, 3
m2, l2 = 0, 3
m3, l3 = -2, 3

# Calculate the spherical harmonics Y(l,m) for both sets
sh1 = sph_harm(m1, l1, theta, phi).real
sh2 = sph_harm(m2, l2, theta, phi).real
sh3 = sph_harm(m3, l3, theta, phi).real

# Sum the two spherical harmonics

all_sh = np.stack((sh1, sh2, sh3), axis=0)
fmax, fmin = all_sh.max(), all_sh.min()

# Create a figure and an array of 3D subplots
fig, axes = plt.subplots(3, 7, subplot_kw={'projection': '3d'}, figsize=(12, 5))

# Adjust layout for better visualization
fig.tight_layout(pad=2.0)

for i in [1, 2, 3]:
    for ix, j in enumerate(range(-i, i+1)):
        axes[i-1, ix].set_axis_off()

        sh = sph_harm(j, i, theta, phi).real
        axes[i-1, 3 - i + ix].plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.RdBu(1 - (sh - sh.min()) / (sh.max() - sh.min()) ),
                    linewidth=1, antialiased=True)
        axes[i-1, 3 - i + ix].set_title(f"Degree {i}\nOrder {j}", fontsize=10, y=0.85)

for i in range(3):
    for j in range(7):
        axes[i, j].set_box_aspect([1, 1, 1])
        axes[i, j].set_axis_off()

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('sh1.png', dpi=200, bbox_inches='tight')


# HARMONICS WITH RADIALLY-MODULATED PLOT
# Enable interactive plots in Jupyter Notebook using the widget backend
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.special import sph_harm

# Define the angles for spherical coordinates
phi = np.linspace(0, np.pi, 200)
theta = np.linspace(0, 2 * np.pi, 200)
phi, theta = np.meshgrid(phi, theta)

def calc_sh(coeffs, consts, ax):
    sh = 0
    for (m, l), c in zip(coeffs, consts):
        sh += c * sph_harm(m, l, theta, phi).real
    fmax, fmin = sh.max(), sh.min()
    base_radius = 1
    r = base_radius * np.abs(1*sh)  # Adjust 0.5 to control modulation strength
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    fcolors = (sh - fmin) / (fmax - fmin)
    surf = ax.plot_surface(
        x, y, z, rstride=1, cstride=1, facecolors=cm.RdBu(1 - fcolors),
        linewidth=1, antialiased=True)
    ax.set_box_aspect([1, 1, 1])  # Set the aspect ratio of the 3D plot
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(141, projection='3d')
ax2 = fig.add_subplot(142, projection='3d')
ax3 = fig.add_subplot(143, projection='3d')
ax4 = fig.add_subplot(144, projection='3d')
calc_sh([(0, 4), (-1, 2), (3, 4), (-12, 12), (-8, 15)], [1, 2, 3, 0, 1.2], ax)
calc_sh([(1, 2), (1, 8), (13, 14)], [2, 5, 4], ax2)
calc_sh([(-1, 1), (7, 7), (0, 4)], [1, 2.5, 5], ax3)
calc_sh([(2, 2), (0, 3), (-6, 8)], [4, 2, 3], ax4)

# Show the plot
plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig("sh3.png", dpi=250, bbox_inches='tight')
plt.show()