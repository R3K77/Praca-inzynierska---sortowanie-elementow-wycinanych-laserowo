import numpy as np
from stl import mesh

def extrude_contours(contours, thickness):
    # Tworzenie listy punktów dla konturów wraz z dodatkowymi punktami dla grubości
    vertices = []
    for contour in contours:
        for i in range(len(contour)):
            vertices.append((contour[i][0], contour[i][1], 0.0))
            vertices.append((contour[(i + 1) % len(contour)][0], contour[(i + 1) % len(contour)][1], 0.0))

    # Tworzenie trójkątów na podstawie punktów
    faces = []
    num_vertices = len(vertices)
    for i in range(0, num_vertices, 4):  # zmiana zakresu na co czwarty punkt
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

# Tworzenie konturów
contours = [
    [(10, 10), (10, 50), (50, 50), (50, 10)],  # Główny kontur
    [(23, 23), (23, 35), (35, 35), (35, 23)],  # Otwór 1
]

# Grubość konturów
thickness = 3.0

# Tworzenie obiektu mesh
stl_mesh = extrude_contours(contours, thickness)

# Zapisywanie do pliku STL
stl_mesh.save('output.stl')
