import numpy as np
import matplotlib.pyplot as plt

# Definiowanie wierzchołków kwadratu
top_left = np.array([0, 100])
bottom_left = np.array([0, 0])
bottom_right = np.array([100, 0])
top_right = np.array([100, 100])

# Listy współrzędnych dla rysowania kwadratu
rectangle_x = [top_left[0], top_right[0], bottom_right[0], bottom_left[0], top_left[0]]
rectangle_y = [top_left[1], top_right[1], bottom_right[1], bottom_left[1], top_left[1]]

# Wykres
plt.figure(figsize=(8, 8))
plt.rcParams['font.family'] = 'Times New Roman'

# Rysowanie kwadratu
plt.plot(rectangle_x, rectangle_y, color='gray', linestyle='-', label='Ścieżka cięcia detalu')

# Rysowanie punktów jako kółka
plt.plot(top_left[0], top_left[1], marker='o', color='black', markersize=8, linestyle='None', label='Punkt odczytany z G-code')
plt.plot(bottom_left[0], bottom_left[1], marker='o', color='black', markersize=8, linestyle='None')
plt.plot(bottom_right[0], bottom_right[1], marker='o', color='black', markersize=8, linestyle='None')
plt.plot(top_right[0], top_right[1], marker='o', color='black', markersize=8, linestyle='None')

plt.axis('equal')
plt.legend()
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.show()
