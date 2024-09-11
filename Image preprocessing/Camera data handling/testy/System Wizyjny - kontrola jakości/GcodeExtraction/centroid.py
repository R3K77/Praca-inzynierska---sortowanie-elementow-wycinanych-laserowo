import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString


# ------------- Funkcja do obliczania środka ciężkości i powierzchni wielokąta ------------- #
# Autor: Bartłomiej Szalwach
# Funkcja przyjmuje listę punktów definiujących wierzchołki wielokąta (jako zestawy punktów (x, y))
# i zwraca centroid oraz powierzchnię tego wielokąta.
# Do obliczeń wykorzystywana jest biblioteka numpy, która umożliwia operacje na tablicach.
# Centroid zwracany jest jako zestaw punktów (centroid_x, centroid_y), a powierzchnia jako pojedyncza wartość.
# ------------------------------------------------------------------------------------------ #
# Założenia funkcji:
# - Powierzchnia wielokąta obliczana jest przy użyciu wzoru polegającego na wykorzystaniu iloczynu skalarnego
#   oraz funkcji przesunięcia indeksu elementów tablicy (np.roll).
# - Centroid obliczany jest jako średnia ważona współrzędnych punktów, z wagą proporcjonalną do struktury wielokąta.
# - Wartości centroidu są zwracane jako wartości bezwzględne, co jest specyficznym zachowaniem tej funkcji.
# - Powierzchnia zawsze jest zwracana jako wartość dodatnia.
# ------------------------------------------------------------------------------------------ #
# Przykład użycia funkcji:
# centroid, area = calculate_centroid(main_contour)
# ------------------------------------------------------------------------------------------ #

def calculate_centroid(poly):
    if len(poly) < 3:
        return (None, None), 0
    x, y = zip(*poly)
    x = np.array(x)
    y = np.array(y)

    # Obliczanie powierzchni wielokąta (A) przy użyciu formuły Shoelace:
    # A = 0.5 * abs(sum(x_i * y_(i+1) - y_i * x_(i+1)))
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    if area == 0:
        return (None, None), 0

    # Obliczanie współrzędnej x centroidu (C_x) wielokąta:
    # C_x = (1 / (6 * A)) * sum((x_i + x_(i+1)) * (x_i * y_(i+1) - x_(i+1) * y_i))
    centroid_x = (np.sum((x + np.roll(x, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y)) / (6.0 * area))

    # Obliczanie współrzędnej y centroidu (C_y) wielokąta:
    # C_y = (1 / (6 * A)) * sum((y_i + y_(i+1)) * (x_i * y_(i+1) - x_(i+1) * y_i))
    centroid_y = (np.sum((y + np.roll(y, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y)) / (6.0 * area))
    return (abs(centroid_x), abs(centroid_y)), area


# --------- Funkcja do dostosowania środka ciężkości, jeśli znajduje się w otworze --------- #
# Autor: Bartłomiej Szalwach
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
# Autor: Bartłomiej Szalwach
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
# Autor: Bartłomiej Szalwach
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