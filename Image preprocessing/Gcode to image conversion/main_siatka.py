import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from matplotlib.patches import Circle, Polygon
from gcode_analize import *
from centroid import *
import re
import csv  # Import the csv module

def is_valid_circle(center, radius, shape, holes):
    """Sprawdza, czy okrąg mieści się w figury bez nakładania na otwory."""
    circle = Point(center).buffer(radius)
    return shape.contains(circle) and all(not hole.intersects(circle) for hole in holes)


element_list = []

file_paths = [
    "./Image preprocessing/Gcode to image conversion/NC_files/8.nc",
    # "./Image preprocessing/Gcode to image conversion/NC_files/7.nc",
    # "./Image preprocessing/Gcode to image conversion/NC_files/6.nc",
    # "./Image preprocessing/Gcode to image conversion/NC_files/5.nc",
    # "./Image preprocessing/Gcode to image conversion/NC_files/4.nc",
    # "./Image preprocessing/Gcode to image conversion/NC_files/3.nc",
    "./Image preprocessing/Gcode to image conversion/NC_files/2.nc",
    "./Image preprocessing/Gcode to image conversion/NC_files/1.nc"
    ]

for file_path in file_paths:
    cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths(file_path)
    
    fig, ax = plt.subplots()
    circle_diameter = 19  # ŚREDNICA PRZYSSAWKI
    circle_radius = circle_diameter / 2

    for i in range(len(cutting_paths)):
        first_element_name = list(cutting_paths.keys())[i]
        if len(first_element_name) == 4:
            continue
        first_element_paths = cutting_paths[first_element_name]
        element_paths = first_element_paths[0]

        main_contour, holes = find_main_and_holes(first_element_paths)
        
        shapely_main_contour = ShapelyPolygon(main_contour)
        shapely_holes = [ShapelyPolygon(hole) for hole in holes]
        shapely_polygon = ShapelyPolygon(main_contour)
        for hole in holes:
            shapely_polygon = shapely_polygon.difference(ShapelyPolygon(hole))
        centroid = shapely_polygon.centroid.coords[0]
        
        valid_points = []
        x_range = np.linspace(min(x for x, _ in main_contour), max(x for x, _ in main_contour), num=50)
        y_range = np.linspace(min(y for _, y in main_contour), max(y for _, y in main_contour), num=50)
        for x in x_range:
            for y in y_range:
                if is_valid_circle((x, y), circle_radius, shapely_main_contour, shapely_holes):
                    valid_points.append((x, y))
        if valid_points:
            distances = [np.linalg.norm(np.array(p) - np.array(centroid)) for p in valid_points]
            best_point = valid_points[np.argmin(distances)]
            if np.linalg.norm(np.array(best_point) - np.array(centroid)) < 300:
                ax.plot(*best_point, 'go', label='Valid Circle Center')
                circle = Circle(best_point, circle_radius, color='green', alpha=1, label='Suction Cup Area')
                ax.add_patch(circle)
                print(f"Element name: {first_element_name} - Centroid: {centroid} - Adjusted centroid: {best_point}")
                element_info = (first_element_name[:-4], best_point)
                element_list.append(element_info)
            else:
                print(f"Element name: {first_element_name} - Centroid: {centroid} - Adjusted centroid: TOO FAR FROM CENTER")
        else:
            print(f"Element name: {first_element_name} - Centroid: {centroid} - No valid point found")

        main_patch = Polygon(main_contour, closed=True, fill=None, edgecolor='red', linewidth=2)
        ax.add_patch(main_patch)
        ax.text(*centroid, first_element_name, fontsize=8, ha='center', va='center')
        for hole in shapely_holes:
            hole_patch = Polygon(np.array(hole.exterior.coords), closed=True, fill=None, edgecolor='blue', linewidth=2)
            ax.add_patch(hole_patch)
            

    # Pojemniki na te same kształty
    box1 = Point(-100, 0)
    box2 = Point(-100, 100)
    box3 = Point(-100, 200)
    box4 = Point(-100, 300)
    box5 = Point(-100, 400)
    box6 = Point(-200, 0)
    box7 = Point(-200, 100)
    box8 = Point(-200, 200)
    box9 = Point(-200, 300)
    box10 = Point(-200, 400)
    box11 = Point(-300, 0)
    box12 = Point(-300, 100)
    box13 = Point(-300, 200)
    box14 = Point(-300, 300)
    box15 = Point(-300, 400)
    box16 = Point(-400, 0)
    box17 = Point(-400, 100)
    box18 = Point(-400, 200)
    box19 = Point(-400, 300)
    box20 = Point(-400, 400)
    box21 = Point(-500, 0) 
    box22 = Point(-500, 100)
    box23 = Point(-500, 200)
    box24 = Point(-500, 300)
    box25 = Point(-500, 400)
    box26 = Point(-600, 0)
    box27 = Point(-600, 100)
    box28 = Point(-600, 200)
    box29 = Point(-600, 300)
    box30 = Point(-600, 400)

    boxes = [box1, box2, box3, box4, box5, box6, box7, box8, box9, box10,
            box11, box12, box13, box14, box15, box16, box17, box18, box19, box20,
                box21, box22, box23, box24, box25, box26, box27, box28, box29, box30]

    element_details = []
    box_counter = 0

    for element in element_list:
        element_name, best_point = element
        if not any(element_name == e[1] for e in element_details):
            box_name = f"box{box_counter + 1}" 
            box_point = boxes[box_counter].coords[0] 
            box_counter = (box_counter + 1) % len(boxes)  
        else:
            for detail in element_details:
                if detail[1] == element_name:
                    box_name, _, _, box_point = detail
                    break
        element_details.append((box_name, element_name, best_point, box_point))

    for detail in element_details:
        print(f"Box Name: {detail[0]}, Element Name: {detail[1]}, Best Point: {detail[2]}, Box Point: {detail[3]}")
        
 # Saving detail[2] to first and second columns and detail[3] to third and fourth columns in CSV
    csv_file_path = "./element_details.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Detail - X", "Detail - Y", "Box - X", "Box - Y"])  # Header row
        for detail in element_details:
            writer.writerow([f"{detail[2][0]:09.4f}", f"{detail[2][1]:09.4f}", f"{detail[3][0]:09.4f}", f"{detail[3][1]:09.4f}"])
        
    for detail in element_details:
        best_point = detail[2]
        box_point = detail[3]
        arrow = plt.arrow(best_point[0], best_point[1], box_point[0] - best_point[0], box_point[1] - best_point[1],
                            length_includes_head=True, head_width=10, head_length=10, fc='black', ec='black')
        ax.add_patch(arrow)
    plt.show()
    
    # Czyszczenie listy elementów
    element_list = []
    element_details = []
    box_counter = 0
    
    #Czyszczenie arrow
    arrow.remove()