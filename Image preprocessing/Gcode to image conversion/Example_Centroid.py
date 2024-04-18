import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString

def calculate_centroid(poly):
    x, y = zip(*poly)
    x = np.array(x)
    y = np.array(y)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    centroid_x = (np.sum((x + np.roll(x, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y)) / (6.0 * area))
    centroid_y = (np.sum((y + np.roll(y, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y)) / (6.0 * area))
    return (centroid_x, centroid_y), area

def adjust_centroid_if_in_hole(centroid, main_poly, holes, offset_distance=2):
    point = Point(centroid)
    main_polygon = ShapelyPolygon(main_poly)
    closest_point = None
    min_distance = float('inf')
    for hole in holes:
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

def find_main_and_holes(contours):
    areas = [(calculate_centroid(contour)[1], contour) for contour in contours]
    areas.sort(reverse=True, key=lambda x: x[0])
    main_contour = areas[0][1]
    holes = [area[1] for area in areas[1:]]
    return main_contour, holes

def point_near_edge(point, main_polygon, holes, min_distance=3):
    shapely_point = Point(point)
    shapely_main_polygon = ShapelyPolygon(main_polygon)

    # Check distance to the main polygon boundary
    distance_to_main_boundary = shapely_main_polygon.boundary.distance(shapely_point)
    if distance_to_main_boundary < min_distance:
        return False  # Too close to the main boundary
    
    # Now check each hole to ensure the point is not inside and is sufficiently distant
    for hole in holes:
        shapely_hole_polygon = ShapelyPolygon(hole)
        distance_to_hole_boundary = shapely_hole_polygon.boundary.distance(shapely_point)
        if distance_to_hole_boundary < min_distance:
            return False  # Too close to a hole boundary

    return True  # Passed all checks


def visualize(contours, centroid, adjusted_centroid):
    fig, ax = plt.subplots()
    main_patch = Polygon(contours[0], closed=True, fill=None, edgecolor='red', linewidth=2)
    ax.add_patch(main_patch)
    for hole in contours[1:]:
        hole_patch = Polygon(hole, closed=True, fill=None, edgecolor='blue', linewidth=2)
        ax.add_patch(hole_patch)
    ax.plot(*centroid, 'go', label='Original Centroid')
    ax.plot(*adjusted_centroid, 'ro', label='Adjusted Centroid + 2 Pixels')
    ax.legend()
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    plt.show()

def point_in_polygon(point, polygon, distance=3):
    num_vertices = len(polygon)
    x, y = point[0], point[1]
    inside = False

    # Przechowaj pierwszy punkt jako poprzedni punkt
    p1 = polygon[0]
 
    # Iteruj przez wszystkie wierzchołki wielokąta
    for i in range(1, num_vertices + 1):
        # Przechowaj drugi punkt jako obecny punkt
        p2 = polygon[i % num_vertices]
 
        # Sprawdź, czy punkt jest poniżej maksymalnej współrzędnej y krawędzi
        if y > min(p1[1], p2[1]):
            # Sprawdź, czy punkt jest poniżej maksymalnej współrzędnej y krawędzi
            if y <= max(p1[1], p2[1]):
                # Sprawdź, czy punkt jest na lewo od maksymalnej współrzędnej x krawędzi
                if x <= max(p1[0], p2[0]):
                    # Sprawdź, czy krawędź przecina linię poziomą przechodzącą przez punkt
                    x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
 
                    # Sprawdź, czy przecięcie jest na lewo od punktu
                    if p1[0] == p2[0] or x <= x_intersection:
                        # Sprawdź, czy punkt jest co najmniej 3 piksele od krawędzi
                        if min(abs(x - p1[0]), abs(x - p2[0]), abs(y - p1[1]), abs(y - p2[1])) >= distance:
                            # Zmień flagę wewnętrzną
                            inside = not inside
 
        # Przesuń punkt drugi do punktu pierwszego
        p1 = p2
 
    # Zwróć True, jeśli punkt znajduje się wewnątrz wielokąta, w przeciwnym razie False
    return inside


# Definicja konturów
# contours = [
#     [(10, 10), (10, 50), (50, 50), (50, 10)],  # Główny kontur
#     [(23, 23), (23, 35), (35, 35), (35, 23)],  # Otwór 1
# ]

contours = [
    [(10, 40), (20, 40), (20, 8), (50, 8), (50, 2), (10, 2)],  # Główny kontur litery "L"
    [(13, 25), (18, 25), (18, 10), (13,10)]
]



main_contour, holes = find_main_and_holes(contours)
centroid, _ = calculate_centroid(main_contour)
adjusted_centroid = adjust_centroid_if_outside(centroid, main_contour, 4)
# adjusted_centroid = adjust_centroid_if_in_hole(adjusted_centroid, main_contour, holes, 3)
print(adjusted_centroid)
if point_in_polygon(adjusted_centroid, main_contour):
    print("Point is inside the polygon")
else:
    print("Point is outside the polygon")
    
    
if point_near_edge(adjusted_centroid, main_contour, holes, 3):
    print("Adjusted centroid is inside the polygon and at least 3 pixels from any edge.")
else:
    print("Adjusted centroid does not meet the criteria.")
    
    
visualize([main_contour] + holes, centroid, adjusted_centroid)








