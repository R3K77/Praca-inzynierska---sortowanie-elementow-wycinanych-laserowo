# ----------------- Plik obliczeniowy do przypisywania elementów do pudełek ----------------- #
# Autor: Bartłomiej Szalwach
# Plik zawiera funkcje do przypisywania elementów do pudełek, obliczania masy elementów oraz
# znajdowania najlepszego punktu przyłożenia przyssawki.
# --------------------------------------------------------------- #

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from matplotlib.patches import Circle, Polygon
from _functions_gcode_analize import visualize_cutting_paths, find_main_and_holes, is_valid_circle, detail_mass
import csv
import matplotlib.lines as mlines  
from matplotlib.patches import Patch
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# ----------------- Stałe globalne ----------------- #
# Stałe używane w programie.
# -------------------------------------------------- #
SUCTION_DIAMETER = 19                   # Średnica przyssawki
SUCTION_RADIUS = SUCTION_DIAMETER / 2   # Promień przyssawki
MAX_SUCTION_SEARCH_RADIUS = 50             # Maksymalna odległość od środka ciężkości
MAX_ADJUSTED_DISTANCE = 60              # Maksymalna odległość od środka ciężkości po korekcie
MAX_DETAIL_MASS = 250                   # Maksymalna masa elementu [g]
MATERIAL_DENSITY = 0.00785              # Gęstość materiału 
MATERIAL_THICKNESS = 1.5                # Grubość materiału [mm]

# ----------------- Funkcja do tworzenia listy pudełek ----------------- #
# Funkcja tworzy listę pudełek (punktów), do których będą przypisane elementy.
# --------------------------------------------------------------- #
def create_boxes():
    box_positions = [
        (421, 50), (421, 100), (421, 150), (421, 200), (421, 250),
        (421, 300), (421, 350), (421, 400), (421, 450), (421, 500),
        (421, 550), (421, 600), (421, 650), (421, 700), (421, 750),
        (421, 800), (421, 850), (421, 900), (421, 950), (421, 1000),
        (421, 1050), (421, 1100), (421, 1150), (421, 1200), (421, 1250)
    ]
    return [Point(x, y) for x, y in box_positions]

# ----------------- Funkcja do przetwarzania pojedynczego elementu ----------------- #
# Funkcja przetwarza pojedynczy element, wyodrębniając jego kształt, otwory, masę
# oraz uwzględniając kryteria dotyczące przyłożenia przyssawki.
# --------------------------------------------------------------- #
def process_element(element_name, element_paths, ax, x_range_center_adj_cache, y_range_center_adj_cache):
    print(f"Processing element: {element_name}")

    if len(element_name) == 4:
        # Ignorowanie obcięcia arkusza
        print("\tIgnored - final trimming of the unused section of sheet metal")
        return None

    # Podział na główny kształt i otwory
    major_shape, holes = find_main_and_holes(element_paths)
    main_contour_polygon = ShapelyPolygon(major_shape)
    holes_polygons = [ShapelyPolygon(hole) for hole in holes]

    # Odejmowanie otworów od głównego kształtu
    for hole in holes:
        main_contour_polygon = main_contour_polygon.difference(ShapelyPolygon(hole))

    # Obliczenie środka ciężkości
    centroid_point = main_contour_polygon.centroid.coords[0]

    # Obliczenie masy elementu
    total_detail_mass = detail_mass(main_contour_polygon, holes_polygons, MATERIAL_DENSITY, MATERIAL_THICKNESS)

    # Ustawienie domyślnych kolorów
    edgecolor = 'gray'
    edgecolor_2 = 'gray'
    circle_color = 'gray'
    text_color = 'gray'

    # Sprawdzenie masy elementu
    if total_detail_mass < MAX_DETAIL_MASS:
        # Znalezienie najlepszego punktu przyłożenia przyssawki
        best_point = find_best_suction_point(
            element_name,
            centroid_point,
            main_contour_polygon,
            holes_polygons,
            x_range_center_adj_cache,
            y_range_center_adj_cache)   
        if best_point:
            # Rysowanie obszaru przyssawki
            #ax.plot(*best_point, 'go')
            ax.add_patch(Circle(best_point, SUCTION_RADIUS, color='green', alpha=1, zorder=3))
            print(f"\tElement name: {element_name}")
            print(f"\tBest suction point found: ({round(best_point[0], 3)}, {round(best_point[1], 3)})")
            # Ustawienie kolorów na wykresie na podstawie stanu
            edgecolor = 'red'
            edgecolor_2 = 'red'
            circle_color = 'orange'
            text_color = 'black'
        else:
            print(f"\tElement name: {element_name}")
            print("\tNo valid point found")
            # Ustawienie kolorów na jasnoszary
            edgecolor = 'lightgray'
            edgecolor_2 = 'lightgray'
            circle_color = 'lightgray'
            text_color = 'lightgray'
            best_point = None  # Brak punktu przyssawki
    else:
        print(f"\tElement name: {element_name}")
        print(f"\tMass is too big ({round(total_detail_mass, 3)} g)")
        best_point = None  # Brak punktu przyssawki z powodu masy

    # Rysowanie elementu
    #   Rysowanie środka ciężkości (jeśli dotyczy)
    if best_point:
        ax.add_patch(Circle(centroid_point, SUCTION_RADIUS, color=circle_color, alpha=0.5, zorder=2))
    #   Rysowanie głównego kształtu
    ax.add_patch(Polygon(major_shape, closed=True, fill=None, edgecolor=edgecolor, linewidth=2))
    #   Wyświetlenie nazwy elementu i masy
    ax.text(*centroid_point, f"{element_name}\n{round(total_detail_mass, 3)} g", fontsize=8, ha='center', va='center', color=text_color)
    #   Rysowanie otworów
    for hole in holes:
        hole_polygon = Polygon(
            np.array(hole),
            closed=True,
            fill=None,
            edgecolor=edgecolor_2,
            linewidth=2,
            linestyle=(0, (1, 0.45))  #  wzór kreski i odstępu
        )
        ax.add_patch(hole_polygon)


    if best_point:
        return (element_name[:-4], best_point)
    else:
        return None

# ----------------- Funkcja do znalezienia najlepszego punktu przyłożenia przyssawki ----------------- #
# Funkcja szuka najlepszego punktu do przyłożenia przyssawki, zaczynając od środka ciężkości.
# Jeśli środek ciężkości nie jest odpowiedni, szuka najbliższego punktu spełniającego kryteria.
# --------------------------------------------------------------- #
def find_best_suction_point(element_name, centroid_point, main_contour_polygon, holes_polygons, x_range_center_adj_cache, y_range_center_adj_cache):
    if is_valid_circle(centroid_point, SUCTION_RADIUS, main_contour_polygon, holes_polygons):
        return centroid_point
    else:
        if element_name not in x_range_center_adj_cache:
            x_range_center_adj_cache[element_name] = np.linspace(
                centroid_point[0] - MAX_SUCTION_SEARCH_RADIUS,
                centroid_point[0] + MAX_SUCTION_SEARCH_RADIUS,
                num=100
            )
        if element_name not in y_range_center_adj_cache:
            y_range_center_adj_cache[element_name] = np.linspace(
                centroid_point[1] - MAX_SUCTION_SEARCH_RADIUS,
                centroid_point[1] + MAX_SUCTION_SEARCH_RADIUS,
                num=100
            )

        x_range_center_adj = x_range_center_adj_cache[element_name]
        y_range_center_adj = y_range_center_adj_cache[element_name]

        valid_points = [
            (x, y) for x in x_range_center_adj for y in y_range_center_adj
            if is_valid_circle((x, y), SUCTION_RADIUS, main_contour_polygon, holes_polygons)
        ]
        if valid_points:
            distances = [np.linalg.norm(np.array(p) - np.array(centroid_point)) for p in valid_points]
            best_point = valid_points[np.argmin(distances)]
            if np.linalg.norm(np.array(best_point) - np.array(centroid_point)) < MAX_ADJUSTED_DISTANCE:
                return best_point
            else:
                print(f"Element name: {element_name}")
                print("\tAdjusted centroid: TOO FAR FROM CENTER")
                return None
        else:
            return None

# ----------------- Funkcja do tworzenia legendy ----------------- #
# Funkcja tworzy własną legendę, wyjaśniając znaczenie kolorów na wykresie.
# --------------------------------------------------------------- #
def add_custom_legend(ax):
    # Tworzenie uchwytów (handles) i etykiet (labels) dla legendy

    legend_elements = [
        Patch(facecolor='none', edgecolor='red', label='Główny obrys detalu'),
        mlines.Line2D([], [], color='red', linestyle='--', label='Otwór w detalu'),
        Patch(facecolor='none', edgecolor='lightgray', label='Nie znaleziono punktu przyssawki'),
        Patch(facecolor='none', edgecolor='gray', label='Element zbyt ciężki'),
        mpatches.Circle((0, 0), radius=5, color='orange', label='Środek ciężkości'),
        mpatches.Circle((0, 0), radius=5, color='green', label='Punkt przyłożenia przyssawki'),
    ]

    ax.legend(handles=legend_elements, loc='upper right')

# ----------------- Główna funkcja programu ----------------- #
# Funkcja przetwarza listę plików NC, wywołując odpowiednie funkcje do przetwarzania i wizualizacji danych.
# ----------------------------------------------------------- #
def main():
    nc_file_paths = [
        "./Image preprocessing/Gcode to image conversion/NC_files/2_FIXME.nc"
    ]

    # Pętla główna przetwarzająca pliki NC
    for nc_file_path in nc_file_paths:
        cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(nc_file_path)
        fig, ax = plt.subplots()
        processed_elements = []
        x_range_center_adj_cache = {}
        y_range_center_adj_cache = {}

        # Przetwarzanie każdego elementu
        for element_name, element_paths in cutting_paths.items():
            element_info = process_element(
                element_name,
                element_paths,
                ax,
                x_range_center_adj_cache,
                y_range_center_adj_cache
            )
            if element_info:
                processed_elements.append(element_info)

        # Przydzielanie elementów do pudełek
        boxes = create_boxes()
        element_details = []
        box_counter = 0

        for element in processed_elements:
            element_name, best_point = element
            if not any(element_name == e[1] for e in element_details):
                box_name = f"box{box_counter + 1}"
                box_point = boxes[box_counter].coords[0]
                box_counter = (box_counter + 1) % len(boxes)
            else:
                for detail in element_details:
                    if detail[1] == element_name:
                        box_name, _, _, box_point = detail
                        break
            element_details.append((box_name, element_name, best_point, box_point))

        # Zapis szczegółów elementów do pliku CSV
        csv_file_path = "./element_details.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Detail - X", "Detail - Y", "Box - X", "Box - Y"])
            for detail in element_details:
                writer.writerow([
                    f"{detail[2][0]:09.4f}",
                    f"{detail[2][1]:09.4f}",
                    f"{detail[3][0]:09.4f}",
                    f"{detail[3][1]:09.4f}"
                ])

        # Rysowanie strzałek od elementów do pudełek
        for detail in element_details:
            best_point = detail[2]
            box_point = detail[3]
            ax.annotate(
                '',
                xy=box_point,
                xytext=best_point,
                arrowprops=dict(facecolor='grey', arrowstyle='->', linewidth=0.2)
            )
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 900)

        # Dodanie własnej legendy do wykresu
        add_custom_legend(ax)

        # Wyświetlenie listy, w którym pudełku znajduje się jaki element
        box_to_elements = {}
        for detail in element_details:
            box_name = detail[0]
            element_name = detail[1]
            if box_name not in box_to_elements:
                box_to_elements[box_name] = []
            box_to_elements[box_name].append(element_name)

        print("\nLista przydziału elementów do pudełek:")
        for box_name, elements in box_to_elements.items():
            print(f"{box_name} zawiera elementy: {', '.join(elements)}")

        plt.show()

if __name__ == "__main__":
    main()
