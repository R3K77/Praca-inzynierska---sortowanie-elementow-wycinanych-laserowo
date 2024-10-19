# ----------- Plik zawierający funkcje do obliczania środka ciężkości wielokąta ------------ #
# Autor: Bartłomiej Szalwach
# ------------------------------------------------------------------------------------------ #

import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString



# --------- Funkcja do dostosowania środka ciężkości, jeśli znajduje się w otworze --------- #
# Funkcja przyjmuje centroid, główny kontur, otwory i dystans dodatkowego przesunięcia.
# Jeśli centroid znajduje się wewnątrz otworu, funkcja zwraca nowy centroid przesunięty o określony dystans.
# Nowy centroid jest obliczany na podstawie najbliższego punktu na krawędzi otworu.
# ------------------------------------------------------------------------------------------ #
# Założenia funkcji:
# - Funkcja sprawdza, czy centroid znajduje się wewnątrz otworu.
# - Jeśli centroid znajduje się wewnątrz otworu, obliczany jest dystans do najbliższego punktu na krawędzi otworu.
# - Nowy centroid obliczany jest na podstawie przesunięcia centroidu w kierunku najbliższego punktu + określony dystans.
# - Funkcja zwraca nowy centroid jako zestaw punktów (centroid_x, centroid_y).
# ------------------------------------------------------------------------------------------ #
# Przykład użycia funkcji:
# new_centroid = adjust_centroid_if_in_hole(centroid, main_contour, holes, 15)
# ------------------------------------------------------------------------------------------ #

def adjust_centroid_if_in_hole(centroid, main_poly, holes, offset_distance=2):
    if len(main_poly) < 4:
        return centroid
    point = Point(centroid)
    main_polygon = ShapelyPolygon(main_poly)
    closest_point = None
    min_distance = float('inf')
    for hole in holes:
        if len(hole) < 4:
            continue
        hole_polygon = ShapelyPolygon(hole)
        if point.within(hole_polygon):
            boundary = LineString(hole_polygon.boundary)
            projected_point = boundary.interpolate(boundary.project(point))
            distance = point.distance(projected_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = projected_point
    if closest_point:
        dir_vector = np.array(closest_point.coords[0]) - np.array(centroid)
        norm_vector = dir_vector / np.linalg.norm(dir_vector)
        new_centroid = np.array(centroid) + norm_vector * (min_distance + offset_distance)
        return tuple(new_centroid)
    return centroid





# -------- Funkcja do dostosowania środka ciężkości, jeśli znajduje się na zewnątrz -------- #
# Funkcja przyjmuje centroid, główny kontur i dystans dodatkowego przesunięcia.
# Jeśli centroid znajduje się na zewnątrz głównego konturu, funkcja zwraca nowy przesunięty centroid.
# Nowy centroid jest obliczany na podstawie najbliższego punktu na krawędzi głównego konturu.
# ------------------------------------------------------------------------------------------ #
# Założenia funkcji:
# - Funkcja sprawdza, czy centroid znajduje się na zewnątrz głównego konturu.
# - Jeśli centroid znajduje się na zewnątrz głównego konturu, obliczany jest dystans do najbliższego punktu na krawędzi.
# - Nowy centroid obliczany jest na podstawie przesunięcia centroidu w kierunku najbliższego punktu + określony dystans.
# - Funkcja zwraca nowy centroid jako zestaw punktów (centroid_x, centroid_y).
# ------------------------------------------------------------------------------------------ #
# Przykład użycia funkcji:
# new_centroid = adjust_centroid_if_outside(centroid, main_contour, 15)
# ------------------------------------------------------------------------------------------ #

def adjust_centroid_if_outside(centroid, main_poly, offset_distance=2):
    point = Point(centroid)
    main_polygon = ShapelyPolygon(main_poly)
    closest_point = None
    min_distance = float('inf')
    if not point.within(main_polygon):
        boundary = LineString(main_polygon.boundary)
        projected_point = boundary.interpolate(boundary.project(point))
        distance = point.distance(projected_point)
        if distance < min_distance:
            min_distance = distance
            closest_point = projected_point
    if closest_point:
        dir_vector = np.array(closest_point.coords[0]) - np.array(centroid)
        norm_vector = dir_vector / np.linalg.norm(dir_vector)
        new_centroid = np.array(centroid) + norm_vector * (min_distance + offset_distance)
        return tuple(new_centroid)
    return centroid





# ------------ Funkcja do sprawdzenia, czy punkt znajduje się wewnątrz wielokąta ------------ #
# Funkcja przyjmuje punkt i wielokąt i zwraca True, jeśli punkt znajduje się wewnątrz wielokąta.
# ------------------------------------------------------------------------------------------ #
# Założenia funkcji:
# - Funkcja wykorzystuje bibliotekę Shapely do sprawdzenia, czy punkt znajduje się wewnątrz wielokąta.
# - Funkcja zwraca wartość logiczną True, jeśli punkt znajduje się wewnątrz wielokąta.
# ------------------------------------------------------------------------------------------ #
# Przykład użycia funkcji:
# point_in_polygon(adjusted_centroid, main_contour)
# ------------------------------------------------------------------------------------------ #

def point_near_edge(point, main_polygon, holes, min_distance=3):
    shapely_point = Point(point)
    shapely_main_polygon = ShapelyPolygon(main_polygon)

    # Sprawdzenie dystansu do głównego kształtu
    distance_to_main_boundary = shapely_main_polygon.boundary.distance(shapely_point)
    if distance_to_main_boundary < min_distance:
        return False  # Za blisko krawędzi

    # Sprawdzenie dystansu od otworów
    for hole in holes:
        shapely_hole_polygon = ShapelyPolygon(hole)
        distance_to_hole_boundary = shapely_hole_polygon.boundary.distance(shapely_point)
        if distance_to_hole_boundary < min_distance:
            return False  # Za blisko krawędzi
    return True