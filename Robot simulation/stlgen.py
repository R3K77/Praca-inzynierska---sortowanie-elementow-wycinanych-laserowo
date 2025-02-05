import numpy as np
import sys, os
from stl import mesh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

path_to_modules = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Image preprocessing', 'Gcode to image conversion'))
if path_to_modules not in sys.path:
    sys.path.append(path_to_modules)

from main_assign_to_boxes import find_main_and_holes, ShapelyPolygon
from _functions_computer_vision import visualize_cutting_paths

NC_FILE_PATH = './Image preprocessing/Gcode to image conversion/NC_files/8.nc'
HEIGHT = 2 # grubość arkusza [mm]

data_points = []

def process_element(element_name, element_paths):
    # print(f"Processing element: {element_name}")
    if len(element_name) == 4:
        # Ignorowanie obcięcia arkusza
        # print("\tIgnored - final trimming of the unused section of sheet metal")
        return None

    # Podział na główny kształt i otwory
    major_shape, holes = find_main_and_holes(element_paths)
    main_contour_polygon = ShapelyPolygon(major_shape)
    # holes_polygons = [ShapelyPolygon(hole) for hole in holes

    # Odejmowanie otworów od głównego kształtu
    for hole in holes:
        main_contour_polygon = main_contour_polygon.difference(ShapelyPolygon(hole))
        
    return element_name, main_contour_polygon


def generate_stl(data_points, height, output_file):
    """
    Tworzy plik STL z danych 2D jako wyciągnięcie wzdłuż osi Z, 
    gdzie górna i dolna podstawa są ograniczone przez ściany boczne.
    Środek bryły będzie w punkcie (0, 0).

    :param data_points: Lista współrzędnych 2D [(x1, y1), (x2, y2), ...]
    :param height: Wysokość bryły
    :param output_file: Nazwa wyjściowego pliku STL
    """
    # Konwertuj dane na macierz NumPy
    points = np.array(data_points)

    # Oblicz przesunięcie środka
    # centroid = np.mean(points, axis=0)
    # print(f"Centroid: {centroid}")
    # points -= centroid

    # Dodaj trzeci wymiar (Z=0 i Z=height)
    bottom = np.column_stack((points, np.zeros(len(points))))
    top = np.column_stack((points, np.full(len(points), height)))

    # Stwórz listę trójkątów
    vertices = []
    center_bottom = [0, 0, 0]
    center_top = [0, 0, height]

    for i in range(len(points)):
        # Indeksy kolejnych punktów
        next_i = (i + 1) % len(points)

        # Trójkąty dolnej podstawy
        vertices.append([bottom[i], bottom[next_i], center_bottom])

        # Trójkąty górnej podstawy
        vertices.append([top[i], center_top, top[next_i]])

        # Ściany boczne
        vertices.append([bottom[i], top[i], top[next_i]])
        vertices.append([bottom[i], top[next_i], bottom[next_i]])

    # Zamień listę na NumPy array i utwórz mesh
    vertices = np.array(vertices)
    surface = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(vertices):
        for j in range(3):
            surface.vectors[i][j] = f[j]

    # Set color to light gray
    light_gray = np.array([211, 211, 211], dtype=np.uint8)
    surface.colors = np.tile(light_gray, (len(surface.vectors), 1))

    # Zapisz mesh do pliku STL
    surface.save(f'./Robot simulation/meshes/{output_file}.stl')
    print(f"Plik STL zapisano jako: {output_file} w folderze /Robot simulation/meshes")

def calculate_centroid_area(polygon):
    """
    Oblicza środek masy (centroid obszaru) wielokąta w oparciu o jego współrzędne.
    
    :param polygon: Obiekt Shapely Polygon
    :return: Krotka (centroid_x, centroid_y) reprezentująca środek masy
    """
    x_coords, y_coords = polygon.exterior.coords.xy
    n = len(x_coords)
    area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    
    for i in range(n - 1):
        common_term = x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
        area += common_term
        centroid_x += (x_coords[i] + x_coords[i + 1]) * common_term
        centroid_y += (y_coords[i] + y_coords[i + 1]) * common_term

    area /= 2.0
    centroid_x /= (6.0 * area)
    centroid_y /= (6.0 * area)

    return centroid_x, centroid_y

def main():
    nc_file_paths = [NC_FILE_PATH]
    # Pętla główna przetwarzająca pliki NC
    for nc_file_path in nc_file_paths:
        cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(nc_file_path)
        processed_elements = []

        for element_name, element_paths in cutting_paths.items():
            processed_element = process_element(element_name, element_paths)
            if processed_element is not None:
                processed_elements.append(processed_element)

    for element in processed_elements:
        x_coords, y_coords = element[1].exterior.coords.xy

        # print(np.mean(x_coords), np.mean(y_coords))

        element_name = element[0]
        
        # Tworzenie wielokąta Shapely
        polygon = Polygon(zip(x_coords, y_coords))
        
        # Obliczanie środka masy
        centroid_x, centroid_y = calculate_centroid_area(polygon)

        # print(f"Centroid for {element_name}: ({centroid_x}, {centroid_y})")

        # Przesunięcie współrzędnych względem środka masy
        x_coords = [(x - centroid_x) / 1000 for x in x_coords]
        y_coords = [(y - centroid_y) / 1000 for y in y_coords]

        # Tworzenie listy punktów z przesuniętymi współrzędnymi
        data_points = list(zip(x_coords, y_coords))

        plt.plot(x_coords, y_coords)
        plt.show()

        # Generowanie pliku STL
        generate_stl(data_points, HEIGHT / 1000, element_name)

if __name__ == "__main__":
    main()