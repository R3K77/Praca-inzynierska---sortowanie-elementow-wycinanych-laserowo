import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon as ShapelyPolygon
from matplotlib.patches import Circle, Polygon

# Stałe globalne
SUCTION_DIAMETER = 19                   # Średnica przyssawki
SUCTION_RADIUS = SUCTION_DIAMETER / 2   # Promień przyssawki
MAX_SUCTION_SEARCH_RADIUS = 70          # Maksymalny promień przeszukiwania
MAX_ADJUSTED_DISTANCE = 70              # Maksymalna dopuszczalna odległość od środka ciężkości

# Funkcja sprawdzająca, czy przyssawka w danym punkcie jest w całości na detalu
def is_valid_circle(center_point, radius, main_contour_polygon, holes_polygons):
    circle = Point(center_point).buffer(radius)
    # Sprawdzenie, czy przyssawka nie nachodzi na otwory
    for hole in holes_polygons:
        if circle.intersects(hole):
            return False
    # Sprawdzenie, czy przyssawka jest w całości na detalu
    return circle.within(main_contour_polygon)

# Zmodyfikowana funkcja do wyszukiwania najlepszego punktu przyłożenia przyssawki
def find_best_suction_point(element_name, centroid_point, main_contour_polygon, holes_polygons):
    # Sprawdzenie, czy przyssawka może być umieszczona w środku ciężkości
    if is_valid_circle(centroid_point, SUCTION_RADIUS, main_contour_polygon, holes_polygons):
        return centroid_point
    else:
        # Generowanie punktów w obrębie okręgu przeszukiwania
        angles = np.linspace(0, 2 * np.pi, num=60)
        radii = np.linspace(0, MAX_SUCTION_SEARCH_RADIUS, num=60)

        valid_points = []
        for r in radii:
            for angle in angles:
                x = centroid_point[0] + r * np.cos(angle)
                y = centroid_point[1] + r * np.sin(angle)
                if is_valid_circle((x, y), SUCTION_RADIUS, main_contour_polygon, holes_polygons):
                    valid_points.append((x, y))

        # Jeśli znaleziono jakiekolwiek punkty, wybieramy ten najbliższy do środka ciężkości
        if valid_points:
            distances = [np.linalg.norm(np.array(p) - np.array(centroid_point)) for p in valid_points]
            best_point = valid_points[np.argmin(distances)]
            
            # Sprawdzamy, czy najlepszy punkt jest w akceptowalnej odległości od środka ciężkości
            if np.linalg.norm(np.array(best_point) - np.array(centroid_point)) < MAX_ADJUSTED_DISTANCE:
                return best_point, valid_points
            else:
                print(f"Element name: {element_name}")
                print("\tAdjusted centroid: TOO FAR FROM CENTER")
                return None, valid_points
        else:
            return None, valid_points

# Tworzenie przykładowego elementu
# Definicja głównego obrysu detalu (prostokąt)
main_shape = ShapelyPolygon([
    (0, 0),
    (200, 0),
    (200, 100),
    (0, 100)
])

# Definicja otworów (okręgi)
hole_center_1 = (100, 50)
hole_radius_1 = 20
hole_1 = Point(hole_center_1).buffer(hole_radius_1)

hole_center_2 = (150, 70)
hole_radius_2 = 15
hole_2 = Point(hole_center_2).buffer(hole_radius_2)

holes = [hole_1, hole_2]

# Odejmowanie otworów od głównego obrysu
detail_shape = main_shape.difference(hole_1).difference(hole_2)

# Obliczenie środka ciężkości
centroid_point = detail_shape.centroid.coords[0]

# Wyszukiwanie najlepszego punktu przyłożenia przyssawki
element_name = "test_element"

best_point, valid_points = find_best_suction_point(
    element_name, centroid_point, detail_shape, holes
)

if best_point:
    print(f"Znaleziono najlepszy punkt przyłożenia przyssawki: {best_point}")
else:
    print("Nie znaleziono odpowiedniego punktu przyłożenia przyssawki.")

# Wizualizacja wyników z wszystkimi punktami
fig, ax = plt.subplots(figsize=(10, 6))

# Rysowanie głównego obrysu detalu
x, y = detail_shape.exterior.xy
ax.plot(x, y, color='black', linewidth=2, label='Główny obrys detalu')

# Rysowanie otworów
for hole in holes:
    xh, yh = hole.exterior.xy
    ax.plot(xh, yh, color='black', linewidth=2, linestyle='--', label='Otwór w detalu')

# Rysowanie środka ciężkości
ax.plot(centroid_point[0], centroid_point[1], 'ko', label='Środek ciężkości')


# Rysowanie wszystkich punktów w obszarze poszukiwania
for point in valid_points:
    ax.add_patch(Circle(point, SUCTION_RADIUS, edgecolor='lightgray', fill=False, linestyle='-', linewidth=1))

# Rysowanie najlepszego punktu przyłożenia przyssawki
if best_point:
    ax.plot(best_point[0], best_point[1], 'kx', markersize=12, label='Najlepszy punkt przyssawki')
    # Rysowanie przyssawki - lepsze oznaczenie, ciemniejsza szarość i grubsza linia
    suction_circle = plt.Circle(best_point, SUCTION_RADIUS, edgecolor='black', fill=False, linewidth=2, linestyle='-', label='Przyssawka')
    ax.add_patch(suction_circle)
    # Dodatkowe cieniowanie obszaru przyssawki
    shaded_suction_circle = plt.Circle(best_point, SUCTION_RADIUS, color='lightgray', alpha=0.5)
    ax.add_patch(shaded_suction_circle)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-50, 250)
ax.set_ylim(-50, 150)
ax.legend()
ax.set_title('Wizualizacja wszystkich punktów i najlepszej lokalizacji przyssawki')
plt.show()
