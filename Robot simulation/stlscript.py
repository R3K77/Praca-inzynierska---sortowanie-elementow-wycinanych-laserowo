import numpy as np
from stl import mesh
import sys

sys.path.append('.\\Image preprocessing\\Gcode to image conversion')

from gcode_analize import *
from centroid import *
import matplotlib.pyplot as plt

def extrude_contours(contours, thickness):
    # Tworzenie listy punktów dla konturów wraz z dodatkowymi punktami dla grubości
    vertices = []
    for contour in contours:
        if len(contour) < 2:  # check if contour has at least 2 points
            continue
        for i in range(len(contour)):
            vertices.append((contour[i][0], contour[i][1], 0.0))
            vertices.append((contour[(i + 1) % len(contour)][0], contour[(i + 1) % len(contour)][1], 0.0))

    # Tworzenie trójkątów na podstawie punktów
    faces = []
    num_vertices = len(vertices)
    for i in range(0, num_vertices, 4):  # zmiana zakresu na co czwarty punkt
        if i + 3 >= num_vertices:  # Skip iteration if index is out of range
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

file_paths = [
    "./Image preprocessing/Gcode to image conversion/NC_files/4.nc"
    ]

cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_paths[0])

for i in range(len(cutting_paths)):
    first_element_name = list(cutting_paths.keys())[i]
    # if first_element_name == "blacha5_005_001":
    #     with open("points.txt", "w") as file:
    #         for point in element_paths:
    #             file.write(f"{point[0]}, {point[1]}\n")
    if len(first_element_name) == 4:
        continue
        
    first_element_paths = cutting_paths[first_element_name]
    element_paths = first_element_paths[0]

    main_contour, holes = find_main_and_holes(first_element_paths)

    # Convert coordinates to floats
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

    
    print(rectangle_corners)

    

    # Plotting rectangle corners
    x_coords = [corner[0] for corner in rectangle_corners]
    y_coords = [corner[1] for corner in rectangle_corners]

    plt.plot(x_coords, y_coords, 'ro')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rectangle Corners')
    plt.show()

    
    # Grubość konturów
    thickness = 2.0

    # Tworzenie obiektu mesh
    stl_mesh = extrude_contours([rectangle_corners], thickness)

    # Zapisywanie do pliku STL
    stl_mesh.save(f'.\\Robot simulation\\meshes\\output_{i}.stl')
        



# # Tworzenie konturów
# contours = [
#     [(10 / 1000, 10 / 1000), (10 / 1000, 50 / 1000), (50 / 1000, 50 / 1000), (50 / 1000, 10 / 1000)],  # Główny kontur
#     [(23 / 1000, 23 / 1000), (23 / 1000, 35 / 1000), (35 / 1000, 35 / 1000), (35 / 1000, 23 / 1000)],  # Otwór 1
# ]

# # Grubość konturów
# thickness = 3.0

# # Tworzenie obiektu mesh
# stl_mesh = extrude_contours(contours, thickness)

# # Zapisywanie do pliku STL
# stl_mesh.save('.\\Robot simulation\\meshes\\output.stl')
