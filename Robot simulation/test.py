import roboticstoolbox as rtb
import numpy as np
import math
from spatialmath import *
from spatialmath.base import *
from spatialmath.base.symbolic import *
import time
from roboticstoolbox.tools.trajectory import *
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

sys.path.append('.\\Image preprocessing\\Gcode to image conversion')

from gcode_analize import *
from centroid import *



# Create a robot
robot = rtb.models.DH.Panda()

# Define the joint angles for the robot
q = [0, 0, 0, 0, 0, 0, 0]

# Set up the figure and axes
fig = robot.plot(robot.q, block=True)
ax = fig.ax

# Set the limits of the axes
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])

# Plot the robot in its initial configuration


# Define the lines to be plotted on the XY plane
lines = [[(0, 0), (0.5, 0.5)], [(0, 0), (-0.5, 0.5)], [(0, 0), (0.5, -0.5)], [(0, 0), (-0.5, -0.5)]]

# Plot the lines on the XY plane
for line in lines:
    ax.plot(line, 'r')

# Update the figure
plt.show()
fig.hold()