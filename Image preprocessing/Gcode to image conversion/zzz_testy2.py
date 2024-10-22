import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon as ShapelyPolygon
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D

# Stałe globalne
SUCTION_DIAMETER = 19                   # Średnica przyssawki
SUCTION_RADIUS = SUCTION_DIAMETER / 2   # Promień przyssawki
MAX_SUCTION_SEARCH_RADIUS = 40          # Maksymalny promień przeszukiwania
MAX_ADJUSTED_DISTANCE = 40              # Maksymalna dopuszczalna odległość od środka ciężkości

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
def find_best_suction_point(element_name, centroid_point, main_contour_polygon, holes_polygons, num_angles, num_radii):
    # Sprawdzenie, czy przyssawka może być umieszczona w środku ciężkości
    if is_valid_circle(centroid_point, SUCTION_RADIUS, main_contour_polygon, holes_polygons):
        return centroid_point
    else:
        # Generowanie punktów w obrębie okręgu przeszukiwania
        angles = np.linspace(0, 2 * np.pi, num=num_angles)
        radii = np.linspace(0, MAX_SUCTION_SEARCH_RADIUS, num=num_radii)

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
    (200, 50),
    (135, 50),
    (135, 100),
    (200, 100),
    (0, 100)
])

# Definicja otworów (okręgi)
hole_center_1 = (100, 50)
hole_radius_1 = 20
hole_1 = Point(hole_center_1).buffer(hole_radius_1)

holes = [hole_1]

# Odejmowanie otworów od głównego obrysu
detail_shape = main_shape.difference(hole_1)

# Obliczenie środka ciężkości
centroid_point = detail_shape.centroid.coords[0]

# Tworzenie subplots - zmiana układu na 2 wiersze, 1 kolumnę
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))  # figsize został zmieniony dla lepszego wyglądu w pionie

# Parametry dla pierwszego subplotu
num_angles_1 = 15
num_radii_1 = 12
best_point_1, valid_points_1 = find_best_suction_point(
    "test_element_1", centroid_point, detail_shape, holes, num_angles_1, num_radii_1
)

# Wizualizacja wyników dla pierwszego subplotu
ax1.set_title('NUM_SEARCH_ANGLES = 15, NUM_SEARCH_RADII = 12', fontsize=12, fontname='Times New Roman')
# Rysowanie głównego obrysu detalu
if detail_shape.geom_type == 'Polygon':
    x, y = detail_shape.exterior.xy
    ax1.plot(x, y, color='black', linewidth=2, label='Główny obrys detalu')
elif detail_shape.geom_type == 'GeometryCollection':
    for geom in detail_shape.geoms:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax1.plot(x, y, color='black', linewidth=2, label='Główny obrys detalu')
# Rysowanie otworów
for hole in holes:
    xh, yh = hole.exterior.xy
    ax1.plot(xh, yh, color='black', linewidth=2, linestyle='--', label='Otwór w detalu')
# Rysowanie środka ciężkości
ax1.plot(centroid_point[0], centroid_point[1], 'ko', label='Środek ciężkości')
# Rysowanie wszystkich punktów w obszarze poszukiwania
for point in valid_points_1:
    ax1.add_patch(Circle(point, SUCTION_RADIUS, edgecolor='lightgray', fill=False, linestyle='-', linewidth=1))
# Rysowanie najlepszego punktu przyłożenia przyssawki
if best_point_1:
    ax1.plot(best_point_1[0], best_point_1[1], 'kx', markersize=12, label='Najlepszy punkt przyssawki')
    suction_circle = plt.Circle(best_point_1, SUCTION_RADIUS, edgecolor='black', fill=True, facecolor='gray', linewidth=2, linestyle='-')
    ax1.add_patch(suction_circle)

# Parametry dla drugiego subplotu
num_angles_2 = 30
num_radii_2 = 30
best_point_2, valid_points_2 = find_best_suction_point(
    "test_element_2", centroid_point, detail_shape, holes, num_angles_2, num_radii_2
)

# Wizualizacja wyników dla drugiego subplotu
ax2.set_title('NUM_SEARCH_ANGLES = 30, NUM_SEARCH_RADII = 30', fontsize=12, fontname='Times New Roman')
# Rysowanie głównego obrysu detalu
if detail_shape.geom_type == 'Polygon':
    x, y = detail_shape.exterior.xy
    ax2.plot(x, y, color='black', linewidth=2, label='Główny obrys detalu')
elif detail_shape.geom_type == 'GeometryCollection':
    for geom in detail_shape.geoms:
        if geom.geom_type == 'Polygon':
            x, y = geom.exterior.xy
            ax2.plot(x, y, color='black', linewidth=2, label='Główny obrys detalu')
# Rysowanie otworów
for hole in holes:
    xh, yh = hole.exterior.xy
    ax2.plot(xh, yh, color='black', linewidth=2, linestyle='--', label='Otwór w detalu')
# Rysowanie środka ciężkości
ax2.plot(centroid_point[0], centroid_point[1], 'ko', label='Środek ciężkości')
# Rysowanie wszystkich punktów w obszarze poszukiwania
for point in valid_points_2:
    ax2.add_patch(Circle(point, SUCTION_RADIUS, edgecolor='lightgray', fill=False, linestyle='-', linewidth=1))
# Rysowanie najlepszego punktu przyłożenia przyssawki
if best_point_2:
    ax2.plot(best_point_2[0], best_point_2[1], 'kx', markersize=12, label='Najlepszy punkt przyssawki')
    suction_circle = plt.Circle(best_point_2, SUCTION_RADIUS, edgecolor='black', fill=True, facecolor='gray', linewidth=2, linestyle='-')
    ax2.add_patch(suction_circle)

# Dodawanie do legendy obiektów reprezentujących przyssawkę i kandydatów
legend_suction = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=14, label='Obszar przyłożenia przyssawki')
legend_candidate = Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='lightgray', markersize=13, label='Kandydat punktu przyłożenia')

# Dodanie legendy do pierwszego subplotu
handles_1, labels_1 = ax1.get_legend_handles_labels()
handles_1.extend([legend_candidate, legend_suction])
labels_1.extend(["Kandydat punktu przyłożenia", "Obszar przyłożenia przyssawki"])
ax1.legend(handles=handles_1, labels=labels_1, prop={'family': 'Times New Roman'}, loc='upper right')

# Dodanie legendy do drugiego subplotu
handles_2, labels_2 = ax2.get_legend_handles_labels()
handles_2.extend([legend_candidate, legend_suction])
labels_2.extend(["Kandydat punktu przyłożenia", "Obszar przyłożenia przyssawki"])
ax2.legend(handles=handles_2, labels=labels_2, prop={'family': 'Times New Roman'}, loc='upper right')

# Ustawienia dla obu subplotów
for ax in (ax1, ax2):
    ax.set_xlabel('X [mm]', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('Y [mm]', fontsize=12, fontname='Times New Roman')
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-50, 250)
    ax.set_ylim(-25, 125)

# Dodanie tytułu i większego odstępu między subplotami
plt.subplots_adjust(hspace=0.5, left=0.15)  # Dodanie większego odstępu między subplotami (hspace) i marginesu po lewej (left)

plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])

plt.show()
