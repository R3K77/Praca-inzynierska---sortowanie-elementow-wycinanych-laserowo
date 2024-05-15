# ---------------------------------------------------------------------------------------
# Autor: Bartłomiej Szalwach
# Kod realizuje odczytanie ścieżek cięcia z pliku NC i wizualizację ich na wykresie.
# Kod wykorzystuje funkcje z centroid.py i gcode_analize.py do obliczenia centroidu i znalezienia głównego konturu i otworów.
# Szczegółowe informacje na temat funkcji znajdują się w plikach centroid.py i gcode_analize.py.
# ---------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from gcode_analize import *
from centroid import *
import re

if __name__ == "__main__":
    file_paths = [
    "./Image preprocessing/Gcode to image conversion/NC_files/4.nc"
    ]
    
    cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_paths[0])
    
    fig, ax = plt.subplots()
    for i in range(len(cutting_paths)):
        first_element_name = list(cutting_paths.keys())[i]
        if first_element_name == "blacha5_005_001":
            with open("points.txt", "w") as file:
                for point in element_paths:
                    file.write(f"{point[0]}, {point[1]}\n")
        if len(first_element_name) == 4:
            continue
        
        first_element_paths = cutting_paths[first_element_name]
        element_paths = first_element_paths[0]

        main_contour, holes = find_main_and_holes(first_element_paths)
        centroid, _ = calculate_centroid(main_contour)
        adjusted_centroid = adjust_centroid_if_outside(centroid, main_contour, 18)
        adjusted_centroid = adjust_centroid_if_in_hole(adjusted_centroid, main_contour, holes, 18)

        with open("./Robot simulation/centroids.csv", "a") as file:
            file.write(f"{adjusted_centroid[0]},{adjusted_centroid[1]}\n")
        
        contours = [main_contour] + holes
        main_patch = Polygon(contours[0], closed=True, fill=None, edgecolor='red', linewidth=2)
        ax.add_patch(main_patch)
        ax.text(*centroid, first_element_name, fontsize=8, ha='center', va='center')
        for hole in contours[1:]:
            hole_patch = Polygon(hole, closed=True, fill=None, edgecolor='blue', linewidth=2)
            ax.add_patch(hole_patch)

        if point_in_polygon(adjusted_centroid, main_contour):
            for i in range(len(holes)):
                if point_in_polygon(adjusted_centroid, holes[i]):
                    print(f"Element name: {first_element_name} - Centroid: {centroid} - Adjusted centroid: INSIDE HOLE")
                    break
            else:                
                if point_near_edge(adjusted_centroid, main_contour, holes, 17):
                    print(f"Element name: {first_element_name} - Centroid: {centroid} - Adjusted centroid: {adjusted_centroid}")
                    ax.plot(*centroid, 'go', label='Original Centroid')
                    ax.plot(*adjusted_centroid, 'ro', label='Adjusted Centroid')
                else:
                    print(f"Element name: {first_element_name} - Centroid: {centroid} - Adjusted centroid: TOO CLOSE TO EDGE")   
        else:
            print(f"Element name: {first_element_name} - Centroid: {centroid} - Adjusted centroid: OUSIDE")

    # ax.set_xlim(0, 500)
    # ax.set_ylim(0, 1000)
    plt.show()