import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

# Zdefiniowanie punktów na podstawie podanej listy
points = np.array([
    [0, 0],
    [0, 10],
    [3, 10],
    [3, 3],
    [10, 3],
    [10, 0],
    [0, 0]
])

# Usunięcie duplikatu punktu, który zamyka pętlę
points = points[:-1]

# Użycie triangulacji Delaunay'a
tri = Delaunay(points)



tri.simplices, tri.points  # Wyświetlenie indeksów i punktów triangulacji

def calculate_geometric_center(points):
    """Funkcja do obliczania środka geometrycznego zestawu punktów."""
    return np.mean(points, axis=0)

def calculate_centroid(points):
    """Funkcja do obliczania środka masy trójkąta."""
    return np.mean(points, axis=0)

centroids = np.array([calculate_centroid(points[tri.simplices[i]]) for i in range(len(tri.simplices))])

# Obliczenie środka geometrycznego dla wszystkich punktów figury
geometric_center = calculate_geometric_center(points)

# Obliczenie odległości od każdego centroidu do środka geometrycznego
distances = np.linalg.norm(centroids - geometric_center, axis=1)

# Znalezienie indeksu centroidu najbliższego środkowi geometrycznemu
closest_centroid_index = np.argmin(distances)
closest_centroid = centroids[closest_centroid_index]

# Rysowanie wykresu z zaznaczonym środkiem geometrycznym i najbliższym centroidem
plt.figure()
plt.triplot(points[:, 0], points[:, 1], tri.simplices)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.plot(centroids[:, 0], centroids[:, 1], '^', color='red')
plt.plot(geometric_center[0], geometric_center[1], 'go', label='Geometric Center')
plt.plot(closest_centroid[0], closest_centroid[1], 'mo', label='Closest Centroid')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Closest Centroid to Geometric Center')
plt.legend()
plt.show()

closest_centroid, distances[closest_centroid_index]  # Wyswietlenie współrzędnych najbliższego centroidu i jego odległości do środka geometrycznego

