# --------------------------------------------------------------------------------------------------
# Autor: Bartłomiej Szalwach
# Kod przedstawia przykładowe użycie funkcji do obliczenia centroidu, znalezienia głównego konturu i otworów,
# Jako prosty przykład służy litera "L" z głównym konturem i jednym otworem lub prostokąt z prostokątnym otworem.
# Kod używany do testowania funkcji z centroid.py i gcode_analize.py
# --------------------------------------------------------------------------------------------------


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from centroid import *
from gcode_analize import *

def visualize(contours, centroid, adjusted_centroid):
    fig, ax = plt.subplots()
    main_patch = Polygon(contours[0], closed=True, fill=None, edgecolor='red', linewidth=2)
    ax.add_patch(main_patch)
    for hole in contours[1:]:
        hole_patch = Polygon(hole, closed=True, fill=None, edgecolor='blue', linewidth=2)
        ax.add_patch(hole_patch)
    ax.plot(*centroid, 'go', label='Original Centroid')
    ax.plot(*adjusted_centroid, 'ro', label='Adjusted Centroid + 2 Pixels')
    ax.legend()
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    plt.show()

# Definicja konturów
# contours = [
#     [(10, 10), (10, 50), (50, 50), (50, 10)],  # Główny kontur
#     [(23, 23), (23, 35), (35, 35), (35, 23)],  # Otwór 1
# ]
contours = [
    [(10, 40), (20, 40), (20, 8), (50, 8), (50, 2), (10, 2)],  # Główny kontur litery "L"
    [(13, 25), (18, 25), (18, 10), (13,10)]
]

main_contour, holes = find_main_and_holes(contours)
centroid, _ = calculate_centroid(main_contour)
adjusted_centroid = adjust_centroid_if_outside(centroid, main_contour, 4)
# adjusted_centroid = adjust_centroid_if_in_hole(adjusted_centroid, main_contour, holes, 3)
print(adjusted_centroid)
if point_in_polygon(adjusted_centroid, main_contour):
    print("Point is inside the polygon")
else:
    print("Point is outside the polygon")
    
    
if point_near_edge(adjusted_centroid, main_contour, holes, 3):
    print("Adjusted centroid is inside the polygon and at least 3 pixels from any edge.")
else:
    print("Adjusted centroid does not meet the criteria.")
    
visualize([main_contour] + holes, centroid, adjusted_centroid)








