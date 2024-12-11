import numpy as np
from stl import mesh
import sys
import csv

sys.path.append('.\\Image preprocessing\\Gcode to image conversion')

from _functions_gcode_analize_do_usuniecia import *
import matplotlib.pyplot as plt

def extrude_contours(contours, thickness):
    # Tworzenie listy punktów dla konturów wraz z dodatkowymi punktami dla grubości
    vertices = []
    for contour in contours:
        if len(contour) < 2:  # Sprawdź, czy kontur ma co najmniej 2 punkty
            continue
        for i in range(len(contour)):
            vertices.append((contour[i][0], contour[i][1], 0.0))
            vertices.append((contour[(i + 1) % len(contour)][0], contour[(i + 1) % len(contour)][1], 0.0))

    # Tworzenie trójkątów na podstawie punktów
    faces = []
    num_vertices = len(vertices)
    for i in range(0, num_vertices, 4):  # Zmiana zakresu na co czwarty punkt
        if i + 3 >= num_vertices:  # Pomiń iterację, jeśli indeks jest poza zakresem
            continue
        v0 = vertices[i]
        v1 = vertices[i + 1]
        v2 = vertices[i + 2]
        v3 = vertices[i + 3]
        faces.append([i, i + 1, i + 3])
        faces.append([i, i + 3, i + 2])

    # Tworzenie obiektu mesh
    mesh_data = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[f[j]]

    return mesh_data

# Funkcja do wczytania danych z pliku CSV
def load_csv_data(csv_file):
    csv_data = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Pomijamy nagłówek
        for row in reader:
            csv_data.append(row)
    return csv_data

# Wczytanie danych z pliku CSV
csv_file = 'element_details.csv'  # Podaj ścieżkę do pliku CSV
csv_data = load_csv_data(csv_file)

file_paths = [
    "./Image preprocessing/Gcode to image conversion/NC_files/4.nc"
]

cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_paths[0])

elements = []

for i in range(len(cutting_paths)):
    first_element_name = list(cutting_paths.keys())[i]
    if len(first_element_name) == 4:
        continue

    first_element_paths = cutting_paths[first_element_name]
    element_paths = first_element_paths[0]

    main_contour, holes = find_main_and_holes(first_element_paths)

    # Konwersja współrzędnych na wartości zmiennoprzecinkowe
    main_contour = [(float(main_contour[i][0]) / 1000, float(main_contour[i][1]) / 1000) for i in range(len(main_contour))]
    holes = [[(float(hole[i][0]) / 1000, float(hole[i][1]) / 1000) for i in range(len(hole))] for hole in holes]

    # Tworzenie konturów
    contours = [main_contour] #+ holes

    # Wyznaczanie czterech punktów reprezentujących rogi prostokąta
    x_min = min(main_contour, key=lambda p: p[0])[0]
    x_max = max(main_contour, key=lambda p: p[0])[0]
    y_min = min(main_contour, key=lambda p: p[1])[1]
    y_max = max(main_contour, key=lambda p: p[1])[1]

    rectangle_corners = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
    elements.append(rectangle_corners)

# Funkcja do obliczania centroidu prostokąta
def calculate_centroid(rectangle):
    x_coords = [point[0] for point in rectangle]
    y_coords = [point[1] for point in rectangle]
    centroid_x = sum(x_coords) / len(rectangle)
    centroid_y = sum(y_coords) / len(rectangle)
    return centroid_x, centroid_y

# Sortowanie elementów według współrzędnej X centroidu
# elements.sort(key=lambda rect: calculate_centroid(rect)[0])

# Pętla generująca pliki STL
for i in range(len(elements)):
    # Grubość konturów
    thickness = 2.0

    # Sprawdzenie, czy jest wystarczająca ilość danych w CSV
    if i >= len(csv_data):
        print(f'Brak wystarczającej ilości danych w pliku CSV dla elementu {i}')
        break

    # Tworzenie obiektu mesh
    stl_mesh = extrude_contours([elements[i]], thickness)

    # Zapisywanie do pliku STL z nazwą z CSV
    stl_name = csv_data[i][0]  # Nazwa elementu z pierwszej kolumny CSV
    stl_mesh.save(f'.\\Robot simulation\\meshes\\{stl_name}.stl')

print('Pliki STL zostały wygenerowane pomyślnie')