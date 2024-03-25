import matplotlib.pyplot as plt
import numpy as np
import re
import cv2

# ----------------- Funkcja do wizualizacji ścieżek cięcia z pliku NC ----------------- #
# Funkcja plik .nc z kodem G-kodu i zwraca obraz z wizualizacją ścieżek cięcia.
# Funkcja wykorzystuje bibliotekę matplotlib, numpy i re. 
# Funkcja zwraca ścieżki cięcia, minimalne i maksymalne współrzędne X i Y.
# Funkcja przyjmuje ścieżkę do pliku .nc z kodem G-kodu oraz maksymalne współrzędne X i Y.
# Przykład użycia:
# cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths("./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-6001.nc")
# ----------------------------------------------------------------------------------- #

def visualize_cutting_paths(file_path, x_max=500, y_max=1000):
    # Odczyt pliku
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Wyszukanie wszystkich instrukcji CNC (M10, M11, G01) z współrzędnymi X i Y za pomocą Regex
    pattern_cnc_commands_extended = re.compile(r'(M10|M11|G01X([0-9.]+)Y([0-9.]+)|G0[23]X([0-9.]+)Y([0-9.]+)I([0-9.-]+)J([0-9.-]+))')
    matches_cnc = pattern_cnc_commands_extended.findall(file_content)

    # Inicjalizacja zmiennych
    laser_on = False
    cutting_paths = []
    current_path = []
    current_position = (0, 0)

    # Szukanie ścieżek cięcia
    for match in matches_cnc:
        command = match[0]
        if command == 'M10':  # Laser ON
            laser_on = True
        elif command == 'M11':  # Laser OFF
            if laser_on and current_path:
                cutting_paths.append(current_path)
                current_path = []
            laser_on = False
        elif laser_on:
            if command.startswith('G01'):  # Linia prostą
                x, y = float(match[1]), float(match[2])
                current_path.append((x, y))
                current_position = (x, y)
            elif command.startswith('G02') or command.startswith('G03'):  # Ruch okrężny
                x, y, i, j = float(match[3]), float(match[4]), float(match[5]), float(match[6])
                center_x = current_position[0] + i  # Środek łuku na osi X
                center_y = current_position[1] + j  # Środek łuku na osi Y
                radius = np.sqrt(i**2 + j**2)  # Obliczamy promień łuku
                
                start_angle = np.arctan2(current_position[1] - center_y, current_position[0] - center_x)
                end_angle = np.arctan2(y - center_y, x - center_x)
                
                if command.startswith('G02'):  # Zgodnie z ruchem wskazówek zegara
                    if end_angle > start_angle:
                        end_angle -= 2 * np.pi
                else:  # Przeciwnie do ruchu wskazówek zegara
                    if end_angle < start_angle:
                        end_angle += 2 * np.pi
                
                angles = np.linspace(start_angle, end_angle, num=50)  # Generujemy kąty dla punktów łuku
                arc_points = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in angles]
                current_path.extend(arc_points)
                current_position = (x, y)

    # Jeśli laser został wyłączony po ostatnim cięciu, dodajemy ścieżkę do listy
    if current_path:
        cutting_paths.append(current_path)

    # Rozmiar arkusza
    x_min, y_min = 0, 0
    
    return cutting_paths, x_min, x_max, y_min, y_max



if __name__ == "__main__":
    
    file_paths = ["./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-1001.nc", 
                  "./Przygotowanie obrazu/gcode2image/NC_files/arkusz-2001.nc", 
                  "./Przygotowanie obrazu/gcode2image/NC_files/arkusz-3001.nc", 
                  "./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-4001.nc", 
                  "./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-5001.nc", 
                  "./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-6001.nc", 
                  "./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-7001.nc"]
    
    
    for file_path in file_paths:
        # x_min, x_max, y_min, y_max = 0, 500, 0, 1000
        cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_path)

        plt.figure(figsize=(x_max/70, y_max/70), facecolor='black')

        # Rysowanie ścieżek cięcia
        for path in cutting_paths:
            plt.plot([x for x, y in path], [y for x, y in path], linestyle='-', marker=None, color='white')

        plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], linestyle='-', marker=None, color='white')
        # Wyłączenie tytułów i osi
        plt.axis('off')
        ylim = plt.ylim([y_min, y_max])
        xlim = plt.xlim([x_min, x_max]) 
        
        # Zapisanie do pliku, z uwzględnieniem wymagań dotyczących braku marginesu i koloru tła oraz nazwy pliku źródłowego
        plt.savefig(f"./Przygotowanie obrazu/gcode2image/visualisation/{file_path.split('/')[-1].split('.')[0]}.png", pad_inches=0, facecolor='black')
    
    # Wyświetlenie wszystkich obrazów (przełączanie za pomocą slidera)
    for file_path in file_paths:
        cv2.imshow('image', cv2.imread(f"./Przygotowanie obrazu/gcode2image/visualisation/{file_path.split('/')[-1].split('.')[0]}.png"))
    cv2.createTrackbar('Trackbar', 'image', 0, 6, lambda x: cv2.imshow('image', cv2.imread(f"./Przygotowanie obrazu/gcode2image/visualisation/{file_paths[x].split('/')[-1].split('.')[0]}.png")))
    cv2.waitKey(0)
    


    