import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from matplotlib.patches import Circle, Polygon
from gcode_analize import visualize_cutting_paths, find_main_and_holes
import csv  


def is_valid_circle(center, radius, shape, holes):
    """Sprawdza, czy okrąg mieści się w figury bez nakładania na otwory."""
    circle = Point(center).buffer(radius)
    return shape.contains(circle) and all(not hole.intersects(circle) for hole in holes)

# Deklaracja listy elementów
processedElements = []

# Pliki NC do analizy
ncFilePaths = [
    #"./Image preprocessing/Gcode to image conversion/NC_files/8_fixed.nc",
    # "./Image preprocessing/Gcode to image conversion/NC_files/7.nc", # nieużywany
    # "./Image preprocessing/Gcode to image conversion/NC_files/6.nc", # nieużywany
    # "./Image preprocessing/Gcode to image conversion/NC_files/5.nc", # nieużywany
    # "./Image preprocessing/Gcode to image conversion/NC_files/4.nc", # nieużywany
    # "./Image preprocessing/Gcode to image conversion/NC_files/3_fixed.nc", # nieużywany
    #"./Image preprocessing/Gcode to image conversion/NC_files/2.nc",
     "./Image preprocessing/Gcode to image conversion/NC_files/1.nc" # nieużywany
    ]

# Deklaracja rozmiaru przyssawki
suctionDiameter = 19                        # ŚREDNICA PRZYSSAWKI
suctionRadius = suctionDiameter / 2         # PROMIEŃ PR
xRangeCenterAdjCache = {}
yRangeCenterAdjCache = {}

maxDistanceToCenter = 50

# Główna pętla
for ncFilePath in ncFilePaths:
    cuttingPaths, x_min, x_max, y_min, y_max = visualize_cutting_paths(ncFilePath)
    fig, ax = plt.subplots()

    for currentElementName, currentElementPaths in cuttingPaths.items():
        print(f"Processing element: {currentElementName}")
        if len(currentElementName) == 4:
            # Ignorowanie obcięcia arkusza. Końcowe obcięcie nie ma nazwy, więc zostaje w nim tylko nazwa o 4 znakach
            print(f"Element name: {currentElementName} - Ignored - final trimming of the unused section of sheet metal")
            continue
        
        # Podział na główny kształt i otwory
        majorShape, holes = find_main_and_holes(currentElementPaths)
        mainContourPolygon = ShapelyPolygon(majorShape)
        holesPolygons = [ShapelyPolygon(hole) for hole in holes]

        # Obliczenie środka ciężkości
        for hole in holes:
            mainContourPolygon = mainContourPolygon.difference(ShapelyPolygon(hole))
        centroidPoint = mainContourPolygon.centroid.coords[0]
        circle2 = Circle(centroidPoint, suctionRadius, color='blue', alpha=0.5, label='Center of Mass')
        ax.add_patch(circle2)
        
        if is_valid_circle((centroidPoint[0], centroidPoint[1]), suctionRadius, mainContourPolygon, holesPolygons):
            # Środek ciężkości jest wewnątrz kształtu i nie nakłada się na żaden otwór
            ax.plot(*centroidPoint, 'ro', label='Valid Circle Center')
            circle = Circle(centroidPoint, suctionRadius, color='red', alpha=1, label='Suction Cup Area')
            ax.add_patch(circle)
            print(f"Element name: {currentElementName} - Centroid: {centroidPoint} (valid on center of mass)")
            element_info = (currentElementName[:-4], centroidPoint)
            processedElements.append(element_info)
        else:
            # Środek ciężkości jest poza kształtem lub nakłada się na otwór
            # Wybór najlepszego punktu wewnątrz kształtu do przyłożenia przyssawki
            if currentElementName not in xRangeCenterAdjCache:
                xRangeCenterAdjCache[currentElementName] = np.linspace(centroidPoint[0] - maxDistanceToCenter, centroidPoint[0] + maxDistanceToCenter, num=100)
            if currentElementName not in yRangeCenterAdjCache:
                yRangeCenterAdjCache[currentElementName] = np.linspace(centroidPoint[1] - maxDistanceToCenter, centroidPoint[1] + maxDistanceToCenter, num=100)

            xRangeCenterAdj = xRangeCenterAdjCache[currentElementName]
            yRangeCenterAdj = yRangeCenterAdjCache[currentElementName]
            
            # Znalezienie punktów wewnątrz kształtu, które nie nachodzą na otwory i znajdują się wewnątrz kształtu
            validCoordinates = [(x, y) for x in xRangeCenterAdj for y in yRangeCenterAdj if is_valid_circle((x, y), suctionRadius, mainContourPolygon, holesPolygons)]
            if validCoordinates:
                distances = [np.linalg.norm(np.array(p) - np.array(centroidPoint)) for p in validCoordinates]
                best_point = validCoordinates[np.argmin(distances)]
                if np.linalg.norm(np.array(best_point) - np.array(centroidPoint)) < 60:
                    ax.plot(*best_point, 'go', label='Valid Circle Center')
                    circle = Circle(best_point, suctionRadius, color='green', alpha=1, label='Suction Cup Area')
                    ax.add_patch(circle)
                    print(f"Element name: {currentElementName} - Centroid: {centroidPoint} - Adjusted centroid: {best_point}")
                    element_info = (currentElementName[:-4], best_point)
                    processedElements.append(element_info)
                else:
                    print(f"Element name: {currentElementName} - Centroid: {centroidPoint} - Adjusted centroid: TOO FAR FROM CENTER")
            else:
                print(f"Element name: {currentElementName} - Centroid: {centroidPoint} - No valid point found")

        main_patch = Polygon(majorShape, closed=True, fill=None, edgecolor='red', linewidth=2)
        ax.add_patch(main_patch)
        ax.text(*centroidPoint, currentElementName, fontsize=8, ha='center', va='center')
        for hole in holesPolygons:
            hole_patch = Polygon(np.array(hole.exterior.coords), closed=True, fill=None, edgecolor='blue', linewidth=2)
            ax.add_patch(hole_patch)

            
    # Pojemniki na te same kształty
    box1 = Point(338, 300)
    box2 = Point(338, 450)
    box3 = Point(338, 600)
    box4 = Point(338, 750)
    box5 = Point(338, 500)
    box6 = Point(338, 500)
    box7 = Point(338, 500)
    box8 = Point(338, 500)
    box9 = Point(338, 500)
    box10 = Point(338, 500)
    box11 = Point(338, 500)
    box12 = Point(338, 500)
    box13 = Point(338, 500)
    box14 = Point(338, 500)
    box15 = Point(338, 500)
    box16 = Point(338, 500)
    box17 = Point(338, 500)
    box18 = Point(338, 500)
    box19 = Point(338, 500)
    box20 = Point(338, 500)
    box21 = Point(338, 500)
    box22 = Point(338, 500)
    box23 = Point(338, 500)
    box24 = Point(338, 500)
    box25 = Point(338, 500)
    box26 = Point(338, 500)
    box27 = Point(338, 500)
    box28 = Point(338, 500)
    box29 = Point(338, 500)
    box30 = Point(338, 500)
    

    boxes = [box1, box2, box3, box4, box5, box6, box7, box8, box9, box10,
            box11, box12, box13, box14, box15, box16, box17, box18, box19, box20,
            box21, box22, box23, box24, box25, box26, box27, box28, box29, box30]

    element_details = []
    box_counter = 0

    for element in processedElements:
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
    processedElements = []
    element_details = []
    box_counter = 0
    
    #Czyszczenie arrow
    arrow.remove()