import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon

# Definicja figury z dwoma otworami i średnicą okręgu
outer_shape = Polygon([(0, 0), (40,0), (40,70), (60,70), (60,50), (100, 50), (100, 100), (0, 100)])
hole1 = Polygon([(40, 40), (60, 60), (70, 60), (40, 70)])
# hole2 = Polygon([(20, 80), (30, 80), (30, 90), (20, 90)])
hole2 = Polygon([(8, 62), (25, 62), (25, 90), (8, 90)])
circle_diameter = 10
circle_radius = circle_diameter / 2

# Funkcja sprawdzająca, czy okrąg mieści się w figurze bez nachodzenia na otwory
def is_valid_circle(center, radius, shape, holes):
    circle = Point(center).buffer(radius)
    return shape.contains(circle) and not any(hole.intersects(circle) for hole in holes)

# Tworzenie siatki punktów do sprawdzenia
x_range = np.arange(0, 101, 5)  # Co 5 jednostek na osi X
y_range = np.arange(0, 101, 5)  # Co 5 jednostek na osi Y
holes = [hole2]  # Definicja otworów
# Wyszukiwanie punktów spełniających kryteria
valid_points = []
for x in x_range:
    for y in y_range:
        if is_valid_circle((x, y), circle_radius, outer_shape, holes):
            valid_points.append((x, y))

# Utworzenie figury z otworami jako MultiPolygon
composite_shape = MultiPolygon([outer_shape, holes])

# Obliczanie środka masy całej figury z uwzględnieniem otworów
shape_with_holes = outer_shape  # Początkowa figura bez otworów

for hole in holes:
    # Odejmowanie otworów od figury
    shape_with_holes = shape_with_holes.difference(hole) 
# Obliczenie środka masy za pomocą funkcji centroid
mass_center = shape_with_holes.centroid.coords[0]

# mass_center = composite_shape.centroid.coords[0]
print(f"Center of Mass: {mass_center}")

# Obliczenie odległości od środka masy do każdego z ważnych punktów
distances_to_mass_center = np.linalg.norm(np.array(valid_points) - np.array(mass_center), axis=1)

# Znalezienie indeksu punktu najbliższego środkowi masy
closest_to_mass_center_index = np.argmin(distances_to_mass_center)
closest_to_mass_center = valid_points[closest_to_mass_center_index]

import matplotlib.pyplot as plt

# Rysowanie figury, otworów i możliwych punktów przyłożenia
fig, ax = plt.subplots()
ax.set_facecolor('#F5E6CA')  # Ustawienie tła na #F5E6CA

x_outer, y_outer = outer_shape.exterior.xy
ax.fill(x_outer, y_outer, alpha=1, fc='#343F56', label='Blacha')
for hole in holes:
    x_hole, y_hole = hole.exterior.xy
    ax.fill(x_hole, y_hole, alpha=1, fc='#F5E6CA')
ax.plot(mass_center[0], mass_center[1], 'ro', label='Środek masy')
for point in valid_points:
    circle = plt.Circle(point, circle_radius, color='#019D86', alpha=0.2)
    ax.add_patch(circle)
circle = plt.Circle(closest_to_mass_center, circle_radius, color='yellow', alpha=0.5, label='Najbliższy do środka masy')
ax.add_patch(circle)
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 110)
ax.set_aspect('equal', adjustable='datalim')
ax.set_title('Optymalne umiejscowienie okręgu w pobliżu środka masy')
ax.legend()
# plt.show()


# Zmiana tła poza wykresem
fig.patch.set_facecolor('#F5E6CA')


plt.show()
# fig, ax = plt.subplots()
# x_outer, y_outer = outer_shape.exterior.xy
# ax.fill(x_outer, y_outer, alpha=1, fc='#343F56', label='Blacha')
# for hole in [hole1, hole2]:
#     x_hole, y_hole = hole.exterior.xy
#     ax.fill(x_hole, y_hole, alpha=1, fc='#F5E6CA', label='Otwór')
# ax.plot(mass_center[0], mass_center[1], 'ro', label='Środek masy')
# for point in valid_points:
#     circle = plt.Circle(point, circle_radius, color='#F5E6CA', alpha=0.3)
#     ax.add_patch(circle)
# circle = plt.Circle(closest_to_mass_center, circle_radius, color='yellow', alpha=0.5, label='Najbliższy do środka masy')
# ax.add_patch(circle)
# ax.set_xlim(-10, 110)
# ax.set_ylim(-10, 110)
# ax.set_aspect('equal', adjustable='datalim')
# ax.set_title('Optymalne umiejscowienie okręgu w pobliżu środka masy')
# ax.legend()
# plt.show()
# fig, ax = plt.subplots()
# x_outer, y_outer = outer_shape.exterior.xy
# ax.fill(x_outer, y_outer, alpha=0.5, fc='blue', label='Outer Shape')
# for hole in [hole1, hole2]:
#     x_hole, y_hole = hole.exterior.xy
#     ax.fill(x_hole, y_hole, alpha=1, fc='white', label='Holes')
# ax.plot(mass_center[0], mass_center[1], 'ro', label='Center of Mass')
# for point in valid_points:
#     circle = plt.Circle(point, circle_radius, color='green', alpha=0.1)
#     ax.add_patch(circle)
# circle = plt.Circle(closest_to_mass_center, circle_radius, color='yellow', alpha=0.5, label='Closest to Center of Mass')
# ax.add_patch(circle)
# ax.set_xlim(-10, 110)
# ax.set_ylim(-10, 110)
# ax.set_aspect('equal', adjustable='datalim')
# ax.set_title('Optimal Circle Placement Near Center of Mass with Two Holes')
# ax.legend()
# plt.show()
