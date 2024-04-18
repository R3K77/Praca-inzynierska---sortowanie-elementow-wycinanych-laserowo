import roboticstoolbox as rtb
import numpy as np
import math
from spatialmath import *
from spatialmath.base import *
from spatialmath.base.symbolic import *
import time

# Inicjalizacja robota KUKA KR6 R900 za pomocą notacji Denavita-Hartenberga
# Wyliczone parametry DH na podstawie dokumentacji technicznej robota:
# https://www.kuka.com/-/media/kuka-downloads/imported/8350ff3ca11642998dbdc81dcc2ed44c/0000205456_en.pdf
# Link do obliczeń: 
# https://drive.google.com/file/d/1YBYsmyM7xhVJO0LZN2hDVbh9otv7J_j0/view?usp=sharing

robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(d=0.400, a=0.025, alpha=-np.pi/2, qlim=[-np.deg2rad(170), np.deg2rad(170)]),
        rtb.RevoluteDH(offset=-np.pi/2, a=0.455, qlim=[-np.deg2rad(100), np.deg2rad(135)]),
        rtb.RevoluteDH(a=0.035, alpha=-np.pi/2, qlim=[-np.deg2rad(210), np.deg2rad(66)]),
        rtb.RevoluteDH(d=0.420, alpha=np.pi/2, qlim=[-np.deg2rad(185), np.deg2rad(185)]),
        rtb.RevoluteDH(alpha=-np.pi/2, qlim=[-np.deg2rad(120), np.deg2rad(120)]),
        rtb.RevoluteDH(d=0.160, a=0.000, qlim=[-np.deg2rad(350), np.deg2rad(350)]) # Tool Center Point
    ], name='KUKA KR6 sixx R900'
    )

robot.manufacturer = 'KUKA'
robot.comment = 'Created by Maciej Mróz'

robot.base = SE3(0, 0, 0) # Ustawienie punktu bazowego robota
robot.default_backend = 'pyplot'
robot.q = [0, 0, 0, 0, 0, 0] # Ustawienie pozycji startowej robota




# robot.plot(robot.qd, block = True)
robot.teach(robot.q)

# print(robot.__str__().encode('utf-8', 'replace').decode())

# print(robot.__str__().encode(encoding='utf-8', errors='replace').decode())


# T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
# solution = robot.ikine_LM(T)
# traj = rtb.jtraj(robot.q, solution.q, 100)
# robot.plot(traj.q, backend = 'pyplot', limits=[-0.25, 1.25, -0.5, 0.5, 0, 1], block=True)

