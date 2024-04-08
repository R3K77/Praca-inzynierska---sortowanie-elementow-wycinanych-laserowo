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
    # Odczyt pliku
    with open(file_path, 'r') as file:
        file_content = file.read().splitlines()

    # Wyszukanie wszystkich instrukcji CNC (M10, M11, G01, G02, G03) w pliku przy użyciu wyrażenia regularnego
    pattern_cnc_commands_extended = re.compile(r'(M10|M11|G01X([0-9.]+)Y([0-9.]+)|G0[23]X([0-9.]+)Y([0-9.]+)I([0-9.-]+)J([0-9.-]+))')
    pattern_element_name = re.compile(r';@@\[DetailName\((.*?)\)\]')

    laser_on = False
    elements = {}
    current_element_name = 'Unnamed'
    current_path = []
    current_position = (0, 0)

    for line in file_content:
        element_match = pattern_element_name.search(line)
        if element_match:  # Nowy element
            # Zapisz bieżącą ścieżkę, jeśli istnieje
            if current_path:
                if current_element_name not in elements:
                    elements[current_element_name] = []
                elements[current_element_name].append(current_path)
                current_path = []
            # Ustaw nazwę nowego elementu
            current_element_name = element_match.group(1)
        else:
            matches_cnc = pattern_cnc_commands_extended.findall(line)
            for match in matches_cnc:
                command = match[0]
                if command == 'M10':  # Laser ON
                    laser_on = True 
                elif command == 'M11':  # Laser OFF
                    if laser_on and current_path: # Dodajemy ścieżkę do bieżącego elementu
                        if current_element_name not in elements:
                            elements[current_element_name] = []
                        elements[current_element_name].append(current_path)
                        current_path = []
                    laser_on = False
                elif laser_on: # Dodaj punkty do ścieżki, jeśli laser jest włączony
                    # Obsługa instrukcji cięcia...
                    if command.startswith('G01'):  # Linia prosta
                        x, y = float(match[1]), float(match[2])
                        current_path.append((x, y))
                        current_position = (x, y)
                    elif command.startswith('G02') or command.startswith('G03'):  # Ruch okrężny
                        x, y, i, j = float(match[3]), float(match[4]), float(match[5]), float(match[6])  # Ruch okrężny (G02 - zgodnie z ruchem wskazówek zegara, G03 - przeciwnie do ruchu wskazówek zegara)
                        x, y, i, j = float(match[3]), float(match[4]), float(match[5]), float(match[6])
                        center_x = current_position[0] + i  # Środek łuku na osi X
                        center_y = current_position[1] + j  # Środek łuku na osi Y
                        radius = np.sqrt(i**2 + j**2)  # Obliczenie promienia łuku
                        
                        start_angle = np.arctan2(current_position[1] - center_y, current_position[0] - center_x) # Kąt początkowy łuku (w radianach)
                        end_angle = np.arctan2(y - center_y, x - center_x) # Kąt końcowy łuku (w radianach) 
                        
                        if command.startswith('G02'):  # Zgodnie z ruchem wskazówek zegara
                            if end_angle > start_angle:
                                end_angle -= 2 * np.pi
                        else:  # Przeciwnie do ruchu wskazówek zegara
                            if end_angle < start_angle:
                                end_angle += 2 * np.pi
                        
                        angles = np.linspace(start_angle, end_angle, num=50)  # Generowanie punktów łuku (50 punktów) 
                        arc_points = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles] # Obliczenie punktów łuku
                        current_path.extend(arc_points)  # Dodanie punktów łuku do ścieżki
                        current_position = (x, y) # Aktualizacja pozycji

    # Jeśli laser został wyłączony po ostatnim cięciu, dodajemy ścieżkę do listy
    if current_path:
        if current_element_name not in elements:
            elements[current_element_name] = []
        elements[current_element_name].append(current_path)

    # Rozmiar arkusza
    x_min, y_min = 0, 0
    
    return elements, x_min, x_max, y_min, y_max



if __name__ == "__main__":
    
    file_paths = ["./Image preprocessing/Gcode to image conversion/NC_files/arkusz-2001.nc"]
    
    
    
    for file_path in file_paths:
        cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_path)
        fig, ax = plt.subplots()
        patches = []
        for element_name, paths in cutting_paths.items():
            for path in paths:
                polygon = Polygon(path, closed=True)
                patches.append(polygon)
                
        

        
        # Define a list of colors

        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'navy', 'maroon', 'olive', 'silver', 'aqua', 'indigo', 'gold']

        for i, (element_name, paths) in enumerate(cutting_paths.items()):
            color = colors[i % len(colors)]  # Get a color from the list based on the index
            for path in paths:
                polygon = Polygon(path, closed=True, facecolor=color, edgecolor="black", alpha=0.6)  # Set the edgecolor to the selected color
                patches.append(polygon)
                
        # Create a PatchCollection with all the polygons and add it to the plot
        collection = PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

        # Set the plot limits based on the minimum and maximum coordinates
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
       
        
        
        # Show the plot
        plt.show()
        # Remove blue fill color from shapes
        for patch in patches:
            patch.set_facecolor('none')
            
                # # Add black dot at the center of gravity for each shape
        # for element_name, paths in cutting_paths.items():
        #     for path in paths:
        #         x_coords = [point[0] for point in path]
        #         y_coords = [point[1] for point in path]
        #         center_x = sum(x_coords) / len(x_coords)
        #         center_y = sum(y_coords) / len(y_coords)
        #         ax.plot(center_x, center_y, 'ko')


    