import matplotlib.pyplot as plt
import numpy as np
import re # Obsługa wyrażeń regularnych
import cv2
import os
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
        file_content = file.read()

    # Wyszukanie wszystkich instrukcji CNC (M10, M11, G01, G02, G03) w pliku przy użyciu wyrażenia regularnego
    pattern_cnc_commands_extended = re.compile(r'(M10|M11|G01X([0-9.]+)Y([0-9.]+)|G0[23]X([0-9.]+)Y([0-9.]+)I([0-9.-]+)J([0-9.-]+))')
    matches_cnc = pattern_cnc_commands_extended.findall(file_content)

    # Inicjalizacja zmiennych
    laser_on = False
    cutting_paths = []
    current_path = []
    current_position = (0, 0)

    # Szukanie ścieżek cięcia
    for match in matches_cnc: # Iteracja po wszystkich instrukcjach CNC znalezionych za pomocą wyrażenia regularnego
        command = match[0]
        if command == 'M10':  # Laser ON
            laser_on = True 
        elif command == 'M11':  # Laser OFF
            if laser_on and current_path: # Jeśli laser był włączony i ścieżka nie jest pusta, dodajemy ścieżkę do listy
                cutting_paths.append(current_path)
                current_path = []
            laser_on = False
        elif laser_on: # Jeśli laser jest włączony
            if command.startswith('G01'):  # Linia prosta
                x, y = float(match[1]), float(match[2])
                current_path.append((x, y))
                current_position = (x, y)
            elif command.startswith('G02') or command.startswith('G03'):  # Ruch okrężny (G02 - zgodnie z ruchem wskazówek zegara, G03 - przeciwnie do ruchu wskazówek zegara)
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
        cutting_paths.append(current_path)

    # Rozmiar arkusza
    x_min, y_min = 0, 0
    
    return cutting_paths, x_min, x_max, y_min, y_max 



if __name__ == "__main__":
    print(os.getcwd())
    file_paths = ["NC_files/Arkusz-1001.nc"]
    
    
    for file_path in file_paths:
        
        # Przykładowe wywołanie funkcji
        cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_path)

        # Wizualizacja i wypełnianie ścieżek
        fig, ax = plt.subplots(figsize=(x_max / 100, y_max / 100))
        patches = []
        for path in cutting_paths:
            polygon = Polygon(path, closed=True, color='green')  # Poprawka tutaj
            patches.append(polygon)

        p = PatchCollection(patches, alpha=0.4)
        p.set_color('green')
        ax.add_collection(p)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.axis('off')

        # Zapisanie do pliku, z uwzględnieniem wymagań dotyczących braku marginesu i koloru tła oraz nazwy pliku źródłowego
        plt.savefig(f"visualisation/{file_path.split('/')[-1].split('.')[0]}.png", pad_inches=0, facecolor='black')
    
    # Wyświetlenie wszystkich obrazów (przełączanie za pomocą slidera)
    for file_path in file_paths:
        cv2.imshow('image', cv2.imread(f"visualisation/{file_path.split('/')[-1].split('.')[0]}.png"))
    cv2.createTrackbar('Trackbar', 'image', 0, 6, lambda x: cv2.imshow('image', cv2.imread(f"visualisation/{file_paths[x].split('/')[-1].split('.')[0]}.png")))
    cv2.waitKey(0)
    


    