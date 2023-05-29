import matplotlib.pyplot as plt
import numpy as np

# Isometric projection transformation matrix
iso_transform = np.array([[1, -1, 0], [0.5, 0.5, -1], [0.5, 0.5, 1]])

# Define a function to generate the vertices of a cube
def cube_definition(center, size):
    half_size = size / 2.0
    corners = [
        np.array(list(item)) for item in np.array(center) +
        half_size * np.array([(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (1, -1, -1), (1, -1, 1), (1, 1, 1),
                              (1, 1, -1)])
    ]
    return corners

def get_cube_config(levels=5, p=0.99):
    # Create a collection of cubes
    cubes = []
    for k in range(levels):
        for i in range(levels):
            for j in range(levels):
                cube_center = [i, j, k]
                cube_size = 1.0  # no gap between cubes
                cubes.append((sum(cube_center), cube_definition(cube_center, cube_size), (i, j, k)))  # Store depth with each cube

    # Sort for removal
    cubes.sort(key=lambda x: (+x[2][2], +x[0])) # Sort first by height (asc), then by distance (asc)

    removed = np.zeros((levels, levels, levels))
    removed_inds = []

    for ix in range(len(cubes)//1):
        dist, cube, (cx, cy, cz) = cubes[ix]
        # print(cx, cy, cz)

        if cx + cy + cz - 3*levels/2 > 0:
            continue

        if (cx == 0 and cy == 0 and cz == 0): # Remove the first cube
            removed[cx, cy, cz] = 1
            removed_inds.append(ix)
            continue

        if (cx == (levels - 1)) or (cy == (levels - 1) or (cz == (levels - 1))):
            continue
        
        if np.random.rand() < p:

            if cx == 0 and cy == 0:
                if (cz == (levels - 1)) or (cz == 0) \
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)

                elif (removed[cx, cy, cz + 1] == 0) and (removed[cx, cy, cz - 1] == 1) \
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)
            elif cx == 0:    
                if (cz == (levels - 1)) or (cz == 0) \
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0) \
                    and (removed[cx, cy-1, cz] == 1):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)

                elif (removed[cx, cy, cz + 1] == 0) and (removed[cx, cy, cz - 1] == 1) \
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0) \
                    and (removed[cx, cy-1, cz] == 1):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)
            elif cy == 0:    
                if (cz == (levels - 1)) or (cz == 0) \
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0) \
                    and (removed[cx-1, cy, cz] == 1):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)

                elif (removed[cx, cy, cz + 1] == 0) and (removed[cx, cy, cz - 1] == 1) \
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0) \
                    and (removed[cx-1, cy, cz] == 1):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)
            else:
                if (cz == 0) and (removed[cx, cy, cz+1] == 0)\
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0) \
                    and (removed[cx-1, cy, cz] == 1) \
                    and (removed[cx, cy-1, cz] == 1):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)

                elif (cz != (levels - 1)) and (cz != 0) \
                    and (removed[cx, cy, cz + 1] == 0) and (removed[cx, cy, cz - 1] == 1) \
                    and (removed[cx+1, cy, cz] == 0) \
                    and (removed[cx, cy+1, cz] == 0) \
                    and (removed[cx-1, cy, cz] == 1) \
                    and (removed[cx, cy-1, cz] == 1):
                    removed[cx, cy, cz] = 1
                    removed_inds.append(ix)

    for index in sorted(removed_inds, reverse=True):
        del cubes[index]
    return cubes

def plot_cubes(cubes, ax, levels=5, colored=True):
    # # Sort cubes by depth
    cubes.sort(key=lambda x: x[0], reverse=True)

    # Plot each cube with three visible faces
    for ix, (depth, cube, center) in enumerate(cubes):
        cube = np.dot(cube, iso_transform.T)  # Apply isometric projection
        cube = cube[:,:2]  # Ignore z coordinate
        if colored:
            face_color = np.random.rand(3)
        else:
            face_color = np.ones(3)
        ax.add_patch(plt.Polygon(cube[[0, 1, 2, 3]], facecolor=face_color, edgecolor='k', alpha=1.0))
        ax.add_patch(plt.Polygon(cube[[0, 3, 7, 4]], facecolor=face_color*0.7, edgecolor='k', alpha=1.0))
        ax.add_patch(plt.Polygon(cube[[0, 1, 5, 4]], facecolor=face_color*0.9, edgecolor='k', alpha=1.0))

    # Set axes limits to accommodate all cubes and equal aspect
    ax.set_xlim([-levels, levels])
    ax.set_ylim([-levels, levels])
    ax.set_aspect('equal')

N = 3
L = 8
p = 0.99
fig, axs = plt.subplots(N, N, figsize=(12, 12))
cubes = [get_cube_config(levels=L, p=p) for _ in range(N*N)]

for i in range(N):
    for j in range(N):
        plot_cubes(cubes[i*N + j], axs[i, j], levels=L)

        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].set_aspect('equal')
    
plt.subplots_adjust(hspace=0, wspace=0)
plt.show()