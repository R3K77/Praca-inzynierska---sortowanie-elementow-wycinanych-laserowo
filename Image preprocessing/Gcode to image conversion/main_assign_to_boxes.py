# ----- Plik wykorzystywany do odczytania elementów z pliku NC, przypisania ich do pudełek i wizualizacji wyników -----------
# Autor: Bartłomiej Szalwach
# -------------------------------------------------------------------------------- #

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from matplotlib.patches import Circle, Polygon
from _functions_computer_vision import visualize_cutting_paths, find_main_and_holes, is_valid_circle, detail_mass
import csv
import matplotlib.lines as mlines  
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from collections import defaultdict

SUCTION_DIAMETER = 40                   # Średnica przyssawki
SUCTION_RADIUS = SUCTION_DIAMETER / 2   # Promień przyssawki
MAX_SUCTION_SEARCH_RADIUS = 80          # Maksymalna odległość od środka ciężkości
MAX_ADJUSTED_DISTANCE = 80              # Maksymalna odległość od środka ciężkości po korekcie
MAX_DETAIL_MASS = 1200                  # Maksymalna masa elementu [g]
MATERIAL_DENSITY = 0.00785              # Gęstość materiału  # ST-3S: 0.00785, AL: 0.0027
MATERIAL_THICKNESS = 3                  # Grubość materiału [mm]
NUM_SEARCH_ANGLES = 80                  # Liczba kątów do przeszukania
NUM_SEARCH_RADII = 80                   # Liczba promieni do przeszukania
Z_INCREMENT = 2                         # Przyrost wysokości w pudełku na każdy element
INITIAL_HEIGHT = 0                      # Początkowa wysokość dla pierwszego elementu
DETAIL_Z = 0                            # Wysokość pobrania detalu
DRAW_SUCTION_CANDIDATES = False          # Czy rysować kandydatów na punkty przyłożenia przyssawki
DRAW_PLACEMENT_ARROWS = True            # Czy rysować strzałki do odłożenia elementów
NC_FILE_PATH = "./Image preprocessing/Gcode to image conversion/NC_files/8.nc"    # Nazwa pliku NC


def create_boxes():
    """
    # Funkcja do tworzenia listy pudełek
    Funkcja tworzy listę pudełek (punktów), do których będą przypisane elementy.
    """
    box_positions = [
        (0, 600), (0, 700), (150, 700), (300, 700), (450, 700),
        (550, 700), (0, 800), (150, 800), (300, 800), (450, 800),
        (550, 600), (421, 600), (421, 650), (421, 700), (421, 750),
        (421, 800), (421, 850), (421, 900), (421, 950), (421, 1000),
        (421, 1050), (421, 1100), (421, 1150), (421, 1200), (421, 1250)
    ]
    return [Point(x, y) for x, y in box_positions]

def process_element(element_name, element_paths, ax, polar_cache):
    """
    # Funkcja do przetwarzania pojedynczego elementu 
    Funkcja przetwarza pojedynczy element, wyodrębniając jego kształt, otwory, masę oraz uwzględniając kryteria dotyczące przyłożenia przyssawki.
    """
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

    # Obliczenie środka ciężkości i masy
    centroid_point = main_contour_polygon.centroid.coords[0]
    total_detail_mass = detail_mass(main_contour_polygon, holes_polygons, MATERIAL_DENSITY, MATERIAL_THICKNESS)

    # Ustawienie domyślnych kolorów
    edgecolor, edgecolor_2, circle_color, text_color  = 'gray', 'gray', 'gray', 'gray'
    best_point = None
    valid_points = []

    if total_detail_mass < MAX_DETAIL_MASS:
        # Znalezienie najlepszego punktu przyłożenia przyssawki
        best_point, valid_points = find_best_suction_point(element_name, centroid_point, main_contour_polygon, holes_polygons, polar_cache)
        if best_point:
            # Rysowanie obszaru przyssawki
            ax.add_patch(Circle(best_point, SUCTION_RADIUS, color='green', alpha=1, zorder=3))
            print(f"\tElement name: {element_name}")
            print(f"\tBest suction point found: ({round(best_point[0], 3)}, {round(best_point[1], 3)})")
            # Ustawienie kolorów na wykresie na podstawie stanu
            edgecolor, edgecolor_2, circle_color, text_color = 'red', 'red', 'orange', 'black'
        else:
            print(f"\tElement name: {element_name}")
            print("\tNo valid point found")
            edgecolor, edgecolor_2, circle_color, text_color = 'lightgray', 'lightgray', 'lightgray', 'lightgray'

    else:
        print(f"\tElement name: {element_name}")
        print(f"\tMass is too big ({round(total_detail_mass, 3)} g)")

    # Rysowanie elementu
    #   Rysowanie środka ciężkości (jeśli dotyczy)
    if best_point:
        ax.add_patch(Circle(centroid_point, SUCTION_RADIUS, color=circle_color, alpha=0.5, zorder=2))
    ax.add_patch(Polygon(major_shape, closed=True, fill=None, edgecolor=edgecolor, linewidth=2))
    ax.text(*centroid_point, f"{element_name}\n{round(total_detail_mass, 3)} g", fontsize=8, ha='center', va='center', color=text_color)
    for hole in holes:
        hole_polygon = Polygon(np.array(hole), closed=True, fill=None, edgecolor=edgecolor_2, linewidth=2, linestyle=(0, (1, 0.45)))
        ax.add_patch(hole_polygon)

    # Rysowanie kandydatów na punkt przyssawki (kontrolowane przez DRAW_SUCTION_CANDIDATES)
    if DRAW_SUCTION_CANDIDATES and valid_points:
        xs, ys = zip(*valid_points)
        ax.scatter(xs, ys, color='blue', s=5, zorder=1)  # Dostosuj rozmiar i kolor według potrzeb

    return (element_name[:-4], best_point, element_name) if best_point else None

def find_best_suction_point(element_name, centroid_point, main_contour_polygon, holes_polygons, polar_cache):
    """
     # Funkcja do znalezienia najlepszego punktu przyłożenia przyssawki 
    Funkcja szuka najlepszego punktu do przyłożenia przyssawki w układzie biegunowym.
    Autor: Bartłomiej Szalwach
    # Wejście:
    - element_name: nazwa elementu - string
    - centroid_point: środek ciężkości elementu - tuple[float, float] 
    - main_contour_polygon: główny kształt elementu - ShapelyPolygon
    - holes_polygons: otwory w elemencie - list(ShapelyPolygon)
    - polar_cache: słownik z zapamiętanymi punktami przeszukiwania
    # Zwraca:
    - best_point: najlepszy punkt przyłożenia przyssawki - tuple[float, float]
    - valid_points: lista punktów przeszukiwania - list[tuple[float, float]]
    """
    # Sprawdzenie, czy przyssawka może być umieszczona w środku ciężkości
    if is_valid_circle(centroid_point, SUCTION_RADIUS, main_contour_polygon, holes_polygons):
        return centroid_point, [centroid_point]

    # Sprawdzenie, czy punkty przeszukiwania są już obliczone
    if element_name not in polar_cache:
        radii = np.linspace(0, MAX_SUCTION_SEARCH_RADIUS, NUM_SEARCH_RADII)
        angles = np.linspace(0, 2 * np.pi, NUM_SEARCH_ANGLES)
        R, Theta = np.meshgrid(radii, angles)
        X = centroid_point[0] + R * np.cos(Theta)
        Y = centroid_point[1] + R * np.sin(Theta)
        candidate_points = np.column_stack((X.ravel(), Y.ravel()))
        polar_cache[element_name] = candidate_points
    else:
        candidate_points = polar_cache[element_name]

    # Filtrowanie punktów przeszukiwania
    valid_points = [
        (x, y) for x, y in candidate_points
        if is_valid_circle((x, y), SUCTION_RADIUS, main_contour_polygon, holes_polygons)
    ]

    # Jeśli znaleziono jakieś punkty, wybierz ten najbliższy środka ciężkości
    if valid_points:
        best_point = min(valid_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(centroid_point)))
        # Sprawdzenie, czy najlepszy punkt jest w dopuszczalnej odległości od środka ciężkości
        if np.linalg.norm(np.array(best_point) - np.array(centroid_point)) < MAX_ADJUSTED_DISTANCE:
            return best_point, valid_points
        print(f"Element name: {element_name}")
        print("\tAdjusted centroid: TOO FAR FROM CENTER")
    return None, valid_points

def add_custom_legend(ax):
    """
    # Funkcja do dodawania legendy na wykresie
    Funkcja dodaje własną legendę, wyjaśniając znaczenie kolorów na wykresie.
    """
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', label='Główny obrys detalu'),
        mlines.Line2D([], [], color='red', linestyle='--', label='Otwór w detalu'),
        Patch(facecolor='none', edgecolor='lightgray', label='Nie znaleziono punktu przyssawki'),
        Patch(facecolor='none', edgecolor='gray', label='Element zbyt ciężki'),
        mpatches.Circle((0, 0), radius=5, color='orange', label='Środek ciężkości'),
        mpatches.Circle((0, 0), radius=5, color='green', label='Punkt przyłożenia przyssawki'),
        mlines.Line2D([], [], color='grey', linestyle='-', linewidth=1, marker='>', markersize=7, label='Strzałka do pudełka')
    ]
    if DRAW_SUCTION_CANDIDATES:
        legend_elements.append(
            mpatches.Circle((0, 0), radius=5, color='blue', label='Kandydaci na punkt przyssawki')
        )
    ax.legend(handles=legend_elements, loc='upper right')

def main():
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    nc_file_paths = [NC_FILE_PATH]

    # Pętla główna przetwarzająca pliki NC
    for nc_file_path in nc_file_paths:
#            return elements, x_min, x_max, y_min, y_max, sheet_size_line, curveCircleData, linearPointsData

        cutting_paths, _, _, _, _ = visualize_cutting_paths(nc_file_path)
        _, ax = plt.subplots()
        processed_elements = []
        polar_cache = {}

        for element_name, element_paths in cutting_paths.items():
            element_info = process_element(element_name, element_paths, ax, polar_cache)
            if element_info:
                processed_elements.append(element_info)
        boxes = create_boxes()
        element_details = []
        box_counter = 0

        # Słownik do śledzenia wysokości w każdym pudełku
        box_heights = {}  # Klucz: nazwa pudełka, wartość: aktualna wysokość

        for element in processed_elements:
            element_name, best_point, full_element_name = element

            # Sprawdź, czy element tego typu został już przypisany do pudełka
            if not any(element_name == e[1] for e in element_details):
                # Przypisz do nowego pudełka
                box_name = f"box{box_counter + 1}"
                box_point = boxes[box_counter].coords[0]
                box_counter = (box_counter + 1) % len(boxes)
                box_heights[box_name] = INITIAL_HEIGHT 
            else:
                # Znajdź istniejące pudełko dla tego typu elementu
                for detail in element_details:
                    if detail[1] == element_name:
                        box_name = detail[0]
                        box_point = boxes[int(box_name.replace("box", "")) - 1].coords[0]
                        break

            # Zaktualizuj wysokość w pudełku
            current_height = box_heights[box_name]
            box_heights[box_name] += Z_INCREMENT
            target_box_point = (box_point[0], box_point[1], current_height)
            element_details.append((box_name, element_name, best_point, target_box_point, full_element_name))

        # Zapis szczegółów elementów do pliku CSV
        csv_file_path = "./element_details.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Nazwa","Detail - X", "Detail - Y", "Detail - Z", "Box - X", "Box - Y", "Box - Z"])
            for detail in element_details:
                name = detail[-1]
                best_point = detail[2]
                target_box_point = detail[3]
                writer.writerow([
                    f"{name}",                           # Nazwa
                    f"{best_point[0]:09.4f}",            # Detail - X
                    f"{best_point[1]:09.4f}",            # Detail - Y
                    f"{DETAIL_Z:09.4f}",                 # Detail - Z (przyjmujemy 0)
                    f"{target_box_point[0]:09.4f}",      # Box - X
                    f"{target_box_point[1]:09.4f}",      # Box - Y
                    f"{target_box_point[2]:09.4f}"       # Box - Z (wysokość)
                ])
                print(f"Zapisano szczegóły elementu: {detail} (Z={target_box_point[2]}) do pliku CSV")

        # Rysowanie strzałek od elementów do pudełek
        if DRAW_PLACEMENT_ARROWS:
            for detail in element_details:
                best_point = detail[2]
                target_box_point = detail[3]
                ax.annotate('', xy=(target_box_point[0], target_box_point[1]), xytext=best_point, arrowprops=dict(facecolor='grey', arrowstyle='->', linewidth=0.2))

        ax.set_xlim(0, 1000), ax.set_ylim(0, 1000)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X [mm]'), ax.set_ylabel('Y [mm]')
        add_custom_legend(ax)

        # Wyświetlenie przydziału elementów do pudełek w konsoli
        box_to_elements = defaultdict(list)
        for detail in element_details:
            box_to_elements[detail[0]].append(detail[1])
        print("\nLista przydziału elementów do pudełek:")
        for box_name, elements in box_to_elements.items():
            print(f"{box_name} zawiera elementy: {', '.join(elements)}")

        plt.show()

if __name__ == "__main__":
    main()
