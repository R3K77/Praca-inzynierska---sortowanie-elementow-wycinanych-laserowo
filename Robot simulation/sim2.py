from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.animation as animation
import matplotlib.pyplot as plt

# Initial robot coordinates
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 2, 1, 0])
z = np.array([0, 1, 0, 1, 0])

# Create figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 3)
ax.set_zlim(-1, 2)
ax.set_box_aspect([1, 1, 1])

# Initialize robot line
line, = ax.plot([], [], [], 'o-', lw=2)

# Function to initialize animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Function to update animation
def update(frame):
    line.set_data(x[:frame+1], y[:frame+1])
    line.set_3d_properties(z[:frame+1])
    return line,

# Create animation
frames = len(x)
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

# Display animation
plt.show()