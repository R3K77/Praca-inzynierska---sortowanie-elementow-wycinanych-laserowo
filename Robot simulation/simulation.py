# ----------------------------------------------------------------
# Autor: Maciej Mróz
# Kod realizuje wizualizację ścieżek cięcia z pliku NC na wykresie wraz z obramowaniem arkusza
# oraz robotem KUKA KR6 R900 używanym do przeniesienia sklasyfikowanych elementów.
# Docelowo kod ma posłużyć do symulacji ruchu robota podczas przeniesienia elementów z arkusza.
# Kod wykorzystuje funkcje z centroid.py i gcode_analize.py. 
# Szczegółowe informacje na temat funkcji znajdują się w plikach centroid.py i gcode_analize.py.
# ----------------------------------------------------------------



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

# pip3 install "scipy<1.12" roboticstoolbox-python sympy imageio qpsolvers[quadprog]

# Inicjalizacja robota KUKA KR6 R900 za pomocą notacji Denavita-Hartenberga
# Wyliczone parametry DH na podstawie dokumentacji technicznej robota:
# https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000205456_en.pdf
# Link do obliczeń: 
# https://drive.google.com/file/d/1YBYsmyM7xhVJO0LZN2hDVbh9otv7J_j0/view?usp=sharing

robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=400, a=25, alpha=-np.pi/2, qlim=[-np.deg2rad(170), np.deg2rad(170)]),
        rtb.RevoluteDH(offset=-np.pi/2, a=455, qlim=[-np.deg2rad(100), np.deg2rad(135)]),
        rtb.RevoluteDH(a=35, alpha=-np.pi/2, qlim=[-np.deg2rad(210), np.deg2rad(66)]),
        rtb.RevoluteDH(d=420, alpha=np.pi/2, qlim=[-np.deg2rad(185), np.deg2rad(185)]),
        rtb.RevoluteDH(alpha=-np.pi/2, qlim=[-np.deg2rad(120), np.deg2rad(120)]),
        rtb.RevoluteDH(d=160, a=0, qlim=[-np.deg2rad(350), np.deg2rad(350)]) # Tool Center Point
    ], name='KUKA KR6 sixx R900'
    )

robot.manufacturer = 'KUKA'
robot.comment = 'Created by Maciej Mróz'

robot.base = SE3(-200, 500, 0) # Ustawienie punktu bazowego robota
robot.default_backend = 'pyplot'
robot.q = [0, 0, 0, 0, 0, 0] # Ustawienie pozycji startowej robota

T = SE3(0, 0, 0) * SE3.OA([0, 1, 0], [0, 0, -1])
solution = robot.ikine_LM(T)
traj = jtraj(robot.q, solution.q, 50)

fig = robot.plot(robot.q)
ax = fig.ax

file_paths = [
    "./Image preprocessing/Gcode to image conversion/NC_files/1.nc"
    ]

cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_paths[0])

offsetX = 0
offsetY = 0

sheetX = x_max
sheetY = y_max

sheet = [[offsetX, offsetX], [offsetY, sheetY + offsetY],
         [offsetX, sheetX + offsetX], [sheetY + offsetY, sheetY + offsetY],
         [offsetX, sheetX + offsetX], [offsetY, offsetY],
         [sheetX + offsetX, sheetX + offsetX], [offsetY, sheetY + offsetY]]


for j in range(len(traj.q)):
    robot.q = traj.q[j]
    fig = robot.plot(robot.q)
    ax = fig.ax

    for i in range(len(cutting_paths)):
        first_element_name = list(cutting_paths.keys())[i]

        if len(first_element_name) == 4:
                continue
        
        first_element_paths = cutting_paths[first_element_name]
        element_paths = first_element_paths[0]

        main_contour, holes = find_main_and_holes(first_element_paths)
        centroid, _ = calculate_centroid(main_contour)
        adjusted_centroid = adjust_centroid_if_outside(centroid, main_contour, 18)
        adjusted_centroid = adjust_centroid_if_in_hole(adjusted_centroid, main_contour, holes, 18)
            
        contours = [main_contour] + holes
        for line in contours:
            for i in range(1, len(line), 1):
                for j in range(len(traj.q)):
                    robot.q = traj.q[j]
                    robot.plot(robot.q)
                    ax.plot([line[i-1][0], line[i][0]], [line[i-1][1], line[i][1]], 'r-')

        ax.plot(*adjusted_centroid, 'b*', label='Adjusted Centroid')


    

for i in range(0, len(sheet), 2):
    ax.plot(sheet[i], sheet[i+1], 'b--')


# for j in range(len(traj.q)):
#     robot.q = traj.q[i]
#     fig(robot.q)

#     time.sleep(0.05)
#     plt.draw()
#     plt.pause(0.001)

ax.set_xlim([-300, 600])
ax.set_ylim([0, 1000])
# ax.set_zlim([0, 1.5])
plt.show()
fig.hold()