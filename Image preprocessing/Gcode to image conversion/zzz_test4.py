import numpy as np
import matplotlib.pyplot as plt

# Parametry wejściowe prostokąta
top_left = np.array([1, 1])  # Lewy górny róg prostokąta
width = 5  # Szerokość prostokąta
height = 3  # Wysokość prostokąta

# Definiowanie wierzchołków prostokąta
bottom_left = top_left + np.array([0, -height])
bottom_right = top_left + np.array([width, -height])
top_right = top_left + np.array([width, 0])

# Listy współrzędnych dla rysowania prostokąta
rectangle_x = [top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]]
rectangle_y = [top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]]

# Wykres
plt.figure(figsize=(8, 8))
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(rectangle_x, rectangle_y, color='gray', linestyle='-', label='Ścieżka cięcia detalu')
plt.plot(top_left[0], top_left[1], marker='o', color='black', markersize=8, linestyle='None', label='Lewy górny róg')
plt.plot(bottom_left[0], bottom_left[1], marker='s', color='black', markersize=8, linestyle='None', label='Lewy dolny róg')
plt.plot(bottom_right[0], bottom_right[1], marker='^', color='black', markersize=8, linestyle='None', label='Prawy dolny róg')
plt.plot(top_right[0], top_right[1], marker='x', color='black', markersize=8, linestyle='None', label='Prawy górny róg')
plt.axis('equal')
plt.legend()
#plt.title('Wizualizacja prostokątnego detalu na podstawie G-code')
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.show()