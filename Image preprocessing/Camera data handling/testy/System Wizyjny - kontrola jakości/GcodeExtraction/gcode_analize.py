import numpy as np
import re
from centroid import calculate_centroid

# ----------------- Funkcja do wizualizacji ścieżek cięcia z pliku NC ----------------- #
# Autor: Bartłomiej Szalwach,
# Funkcja plik .nc z kodem G-kodu i zwraca obraz z wizualizacją ścieżek cięcia.
# Funkcja wykorzystuje bibliotekę matplotlib, numpy i re.
# Funkcja zwraca ścieżki cięcia, minimalne i maksymalne współrzędne X i Y.
# Funkcja przyjmuje ścieżkę do pliku .nc z kodem G-kodu oraz maksymalne współrzędne X i Y.
# Domyślnie maksymalne współrzędne X i Y wynoszą odpowiednio 500 i 1000.
# ------------------------------------------------------------------------------------- #
# Założenia funkcji:
# - Włączenie lasera oznaczone jest jako M10, a wyłączenie jako M11.
# - Ścieżki cięcia są zapisane jako G01X...Y... lub G02X...Y...I...J... lub G03X...Y...I...J...
# - Współrzędne X i Y są zapisane jako liczby zmiennoprzecinkowe.
# - Ruch okrężny w prawo jest zapisany jako G02, a ruch okrężny w lewo jako G03.
# - Współrzędne I i J określają środek okręgu, a współrzędne X i Y określają punkt końcowy.
# - Współrzędne I i J są zapisane jako liczby zmiennoprzecinkowe.
# - Wszystkie współrzędne są mniejsze niż podane maksymalne współrzędne X i Y.
# ------------------------------------------------------------------------------------- #
# Przykład użycia:
# cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths("./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-6001.nc")
# ------------------------------------------------------------------------------------- #
# Modyfikacje: Rafał Szygenda
# - liczba punktów łuków jest argumentem wejściowym funkcji, większa rozdzielczość
# - Zwracana jest linijka z rozmiarem blachy z gcode

def visualize_cutting_paths(file_path, x_max=500, y_max=1000, arc_pts_len = 200):
    sheet_size_line = None
    with open(file_path, 'r') as file:
        file_content = file.read().splitlines()

    pattern_cnc_commands_extended = re.compile(
        r'(M10|M11|G01X([0-9.]+)Y([0-9.]+)|G0[23]X([0-9.]+)Y([0-9.]+)I([0-9.-]+)J([0-9.-]+))')

    pattern_element_name = re.compile(r';@@\[DetailName\((.*?)\)\]')
    laser_on = False
    elements = {}
    current_element_name = 'Unnamed'
    element_index = {}  # Słownik do przechowywania indeksów dla każdej nazwy elementu
    current_path = []
    current_position = (0, 0)
    curveCircleData = {}
    linearPointsData = {}

    for line in file_content:
        element_match = pattern_element_name.search(line)
        if "*SHEET" in line:
            sheet_size_line = line
        if element_match:
            name = element_match.group(1)
            if name not in element_index:
                element_index[name] = 1  # Rozpocznij liczenie od 1
            else:
                element_index[name] += 1

            # Formatowanie nazwy z zerami wiodącymi
            current_element_name = f"{name}_{element_index[name]:03d}"  # Dodaje zera wiodące do indeksu
            if current_path:
                if current_element_name not in elements:
                    elements[current_element_name] = []

                elements[current_element_name].append(current_path)
                current_path = []

        else:
            matches_cnc = pattern_cnc_commands_extended.findall(line)
            for match in matches_cnc:
                command = match[0]
                if command == 'M10':  # Laser ON
                    laser_on = True
                elif command == 'M11':  # Laser OFF
                    if laser_on and current_path:  # Zapis ścieżki do bieżącego elementu
                        if current_element_name not in elements:
                            elements[current_element_name] = []

                        elements[current_element_name].append(current_path)
                        current_path = []
                    laser_on = False
                elif laser_on:  # Dodaj punkty do ścieżki, jeśli laser jest włączony
                    # Obsługa instrukcji cięcia...
                    if command.startswith('G01'):  # Linia prosta
                        x, y = float(match[1]), float(match[2])
                        current_path.append((x, y))
                        current_position = (x, y)

                        try:
                            linearPointsData[current_element_name].append(current_position)
                        except Exception as e:
                            linearPointsData[current_element_name] = [current_position]

                    elif command.startswith('G02') or command.startswith('G03'):  # Ruch okrężny
                        x, y, i, j = (float(match[3]), float(match[4]),
                                      float(match[5]), float(match[6]))
                        center_x = current_position[0] + i  # Środek łuku na osi X
                        center_y = current_position[1] + j  # Środek łuku na osi Y
                        radius = np.sqrt(i ** 2 + j ** 2)  # Promień łuku
                        start_angle = np.arctan2(current_position[1] - center_y,
                                                 current_position[0] - center_x)  # Kąt początkowy łuku (w radianach)
                        end_angle = np.arctan2(y - center_y, x - center_x)  # Kąt końcowy łuku (w radianach)

                        if command.startswith('G02'):  # Zgodnie z ruchem wskazówek zegara
                            if end_angle > start_angle:
                                end_angle -= 2 * np.pi
                        else:  # Przeciwnie do ruchu wskazówek zegara
                            if end_angle < start_angle:
                                end_angle += 2 * np.pi

                        angles = np.linspace(start_angle, end_angle, num=arc_pts_len)  # Generowanie punktów łuku (50 punktów)
                        arc_points = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in
                                      angles]  # Obliczenie punktów łuku
                        try:
                            curveCircleData[current_element_name].append((center_x,center_y,radius))
                        except:
                            curveCircleData[current_element_name] = [(center_x,center_y,radius)]

                        current_path.extend(arc_points)  # Dodanie punktów łuku do ścieżki
                        current_position = (x, y)  # Aktualizacja pozycji

    # Jeśli laser został wyłączony po ostatnim cięciu, dodajemy ścieżkę do listy
    if current_path:
        if current_element_name not in elements:
            elements[current_element_name] = []
        elements[current_element_name].append(current_path)

    # Rozmiar arkusza
    x_min, y_min = 0, 0

    return elements, x_min, x_max, y_min, y_max, sheet_size_line, curveCircleData, linearPointsData


# ----------------- Funkcja do znalezienia głównego konturu i otworów ----------------- #
# Autor: Bartłomiej Szalwach
# Funkcja przyjmuje listę konturów i zwraca główny kontur i otwory.
# ------------------------------------------------------------------------------------- #
# Założenia funkcji:
# - Główny kontur jest konturem z największym polem powierzchni.
# - Otwory są konturami z mniejszym polem powierzchni.
# ------------------------------------------------------------------------------------- #
# Przykład użycia:
# main_contour, holes = find_main_and_holes(contours)
# ------------------------------------------------------------------------------------- #

def find_main_and_holes(contours):
    areas = [(calculate_centroid(contour)[1], contour) for contour in contours]
    areas.sort(reverse=True, key=lambda x: x[0])
    main_contour = areas[0][1]
    holes = [area[1] for area in areas[1:]]
    return main_contour, holes

# ------------- Funkcja do sprawdzenia, czy punkt znajduje się w wielokącie ----------- #
# Funkcja przyjmuje punkt i wielokąt i zwraca True, jeśli punkt znajduje się wewnątrz wielokąta.
# Źródło: https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
# ------------------------------------------------------------------------------------- #
# Przykład użycia:
# point_in_polygon(adjusted_centroid, main_contour)
# ------------------------------------------------------------------------------------- #

def point_in_polygon(point, polygon):
    num_vertices = len(polygon)
    x, y = point[0], point[1]
    inside = False
    p1 = polygon[0]  # Pierwszy punkt wielokąta

    # Iteruj przez wszystkie wierzchołki wielokąta
    for i in range(1, num_vertices + 1):
        p2 = polygon[i % num_vertices]  # Drugi punkt wielokąta

        # Sprawdź, czy punkt jest poniżej maksymalnej współrzędnej y krawędzi
        if y > min(p1[1], p2[1]):
            # Sprawdź, czy punkt jest poniżej maksymalnej współrzędnej y krawędzi
            if y <= max(p1[1], p2[1]):
                # Sprawdź, czy punkt jest na lewo od maksymalnej współrzędnej x krawędzi
                if x <= max(p1[0], p2[0]):
                    # Sprawdź, czy krawędź przecina linię poziomą przechodzącą przez punkt
                    x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    # Sprawdź, czy przecięcie jest na lewo od punktu
                    if p1[0] == p2[0] or x <= x_intersection:
                        # Zmień flagę wewnętrzną
                        inside = not inside
        # Przesuń punkt drugi do punktu pierwszego
        p1 = p2

    # Zwróć True, jeśli punkt znajduje się wewnątrz wielokąta, w przeciwnym razie False
    return inside

