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

def find_main_and_holes(contours):
    areas = [(calculate_centroid(contour)[1], contour) for contour in contours]
    areas.sort(reverse=True, key=lambda x: x[0])
    main_contour = areas[0][1]
    holes = [area[1] for area in areas[1:]]
    return main_contour, holes

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

# Define contours
contours = [
    [(10, 10), (10, 50), (50, 50), (50, 10)],  # Possible main contour
    [(23, 23), (23, 35), (35, 35), (35, 23)],  # Possible hole
]

main_contour, holes = find_main_and_holes(contours)
centroid, _ = calculate_centroid(main_contour)
adjusted_centroid = adjust_centroid_if_in_hole(centroid, main_contour, holes)
visualize([main_contour] + holes, centroid, adjusted_centroid)
