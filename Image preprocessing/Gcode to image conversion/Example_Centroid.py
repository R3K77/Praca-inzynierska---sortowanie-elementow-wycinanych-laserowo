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
            # Calculate nearest point on the hole boundary
            boundary = LineString(hole_polygon.boundary)
            projected_point = boundary.interpolate(boundary.project(point))
            distance = point.distance(projected_point)
            if distance < min_distance:
                min_distance = distance
                closest_point = projected_point
    
    if closest_point:
        # Calculate direction vector from centroid to closest point and normalize it
        dir_vector = np.array(closest_point.coords[0]) - np.array(centroid)
        norm_vector = dir_vector / np.linalg.norm(dir_vector)
        # Move centroid away by the min_distance + additional offset
        new_centroid = np.array(centroid) + norm_vector * (min_distance + offset_distance)
        return tuple(new_centroid)

    return centroid

# Define contours including main contour and holes
contours = [
    [(10, 10), (10, 50), (50, 50), (50, 10)],  # Main contour
    [(23, 23), (23, 35), (35, 35), (35, 23)],  # Hole
]

main_contour = contours[0]
holes = contours[1:]

centroid, _ = calculate_centroid(main_contour)
adjusted_centroid = adjust_centroid_if_in_hole(centroid, main_contour, holes)

# Visualization
fig, ax = plt.subplots()
main_patch = Polygon(main_contour, closed=True, fill=None, edgecolor='red', linewidth=2)
ax.add_patch(main_patch)
for hole in holes:
    hole_patch = Polygon(hole, closed=True, fill=None, edgecolor='blue', linewidth=2)
    ax.add_patch(hole_patch)

# Plot centroids
ax.plot(*centroid, 'go', label='Original Centroid')
ax.plot(*adjusted_centroid, 'ro', label='Adjusted Centroid + 2 Pixels')
ax.legend()

ax.set_xlim(0, 60)
ax.set_ylim(0, 60)
plt.show()
