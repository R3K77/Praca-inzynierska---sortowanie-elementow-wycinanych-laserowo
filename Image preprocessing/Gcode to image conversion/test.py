import matplotlib.pyplot as plt
import numpy as np
import re # Obsługa wyrażeń regularnych
import cv2 
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# ----------------- Funkcja do wizualizacji ścieżek cięcia z pliku NC ----------------- #
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

def visualize_cutting_paths(file_path, x_max=500, y_max=1000):
    pattern_cnc_commands_extended = re.compile(r'(M10|M11|G01X([0-9.]+)Y([0-9.]+)|G0[23]X([0-9.]+)Y([0-9.]+)I([0-9.-]+)J([0-9.-]+))')
    pattern_detail_name = re.compile(r';@@\[DetailName\((.*?)\)\]')

    elements = []  # Lista do przechowywania elementów i ich ścieżek
    current_path = []
    laser_on = False
    current_position = (0, 0)
    current_element_name = ""  # Zmienna do przechowywania nazwy bieżącego elementu

    with open(file_path, 'r') as file:
        for line in file:
            detail_match = pattern_detail_name.search(line)
            if detail_match:
                # Jeśli jest to nowy element, dodaj poprzedni do listy (jeśli istnieje)
                if current_path:
                    elements.append({"name": current_element_name, "path": current_path})
                    current_path = []
                current_element_name = detail_match.group(1)
                laser_on = False
                continue

            match = pattern_cnc_commands_extended.search(line)
            if match:
                command = match.group(1)
                if command == 'M10':  # Laser ON
                    laser_on = True
                elif command == 'M11':  # Laser OFF
                    laser_on = False
                if laser_on:  # Dodaj punkty tylko jeśli laser jest włączony
                    if command.startswith('G01'):  # Linia prosta
                        x, y = float(match.group(2)), float(match.group(3))
                        current_path.append((x, y))
                        current_position = (x, y)
                    elif command.startswith('G02') or command.startswith('G03'):  # Ruch okrężny
                        # Logika dla ruchu okrężnego
                        x, y, i, j = float(match.group(4)), float(match.group(5)), float(match.group(6)), float(match.group(7))
                        center_x = current_position[0] + i
                        center_y = current_position[1] + j
                        radius = np.sqrt(i**2 + j**2)
                        
                        start_angle = np.arctan2(current_position[1] - center_y, current_position[0] - center_x)
                        end_angle = np.arctan2(y - center_y, x - center_x)
                        
                        if command.startswith('G02'):  # Zgodnie z ruchem wskazówek zegara
                            if end_angle > start_angle:
                                end_angle -= 2 * np.pi
                        else:  # Przeciwnie do ruchu wskazówek zegara
                            if end_angle < start_angle:
                                end_angle += 2 * np.pi
                        
                        angles = np.linspace(start_angle, end_angle, num=50)
                        arc_points = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles]
                        current_path.extend(arc_points)
                        current_position = (x, y)
                        
    # Dodanie ostatniego elementu do listy, jeśli ścieżka nie jest pusta
    if current_path:
        elements.append({"name": current_element_name, "path": current_path})

    x_min, y_min = 0, 0
    
    return elements, x_min, x_max, y_min, y_max
    
    
def generate_unique_colors(n):
    # Generowanie n unikalnych kolorów
    colors = plt.cm.get_cmap('hsv', n)
    return [colors(i) for i in range(n)]
    

if __name__ == "__main__":
    file_paths = ["./Image preprocessing/Gcode to image conversion/NC_files/Arkusz-1001.nc", 
                  "./Image preprocessing/Gcode to image conversion/NC_files/arkusz-2001.nc", 
                  "./Image preprocessing/Gcode to image conversion/NC_files/arkusz-3001.nc", 
                  "./Image preprocessing/Gcode to image conversion/NC_files/Arkusz-4001.nc", 
                  "./Image preprocessing/Gcode to image conversion/NC_files/Arkusz-5001.nc", 
                  "./Image preprocessing/Gcode to image conversion/NC_files/Arkusz-6001.nc", 
                  "./Image preprocessing/Gcode to image conversion/NC_files/Arkusz-7001.nc"]
    
    for file_path in file_paths:
        elements, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_path)
        
        # Utworzenie mapowania nazw elementów do kolorów
        unique_names = list({element['name'] for element in elements})
        colors = generate_unique_colors(len(unique_names))
        name_to_color = dict(zip(unique_names, colors))
        
        # Wizualizacja ścieżek cięcia z kolorem bazującym na nazwie elementu
        fig, ax = plt.subplots()
        for element in elements:
            name = element['name']
            path = element['path']
            color = name_to_color[name]  # Wybór koloru na podstawie nazwy
            polygon = Polygon(path, closed=True, edgecolor=color, facecolor=color, alpha=0.5)
            ax.add_patch(polygon)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')

        # Zapis do pliku
        plt.savefig(f"./Image preprocessing/Gcode to image conversion/visualisation/{file_path.split('/')[-1].split('.')[0]}.png", bbox_inches='tight', pad_inches=0, facecolor='black', dpi=300)
        plt.close(fig)

    # Opcjonalnie: wyświetlenie zapisanych obrazów za pomocą OpenCV
    for file_path in file_paths:
        img_path = f"./Image preprocessing/Gcode to image conversion/visualisation/{file_path.split('/')[-1].split('.')[0]}.png"
        img = cv2.imread(img_path)
        cv2.imshow('Visualized Cutting Paths', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    