import numpy as np
from gcode_analize import visualize_cutting_paths, find_main_and_holes
import cv2
import random
import scipy.interpolate as sc
from scipy.optimize import least_squares
from scipy.optimize import minimize

# Funkcja generuje obraz cv2 wyciętych elementów:
# in - sheet_path scieżka do .nc; output_res_xy - rozdzielczość wyjściowa obrazu; arc_pts_len - liczba punktów z której generowane są łuki/okręgi.
# out - images_dict elementy wycięte cv2; pts_dict punkty konturu; sheet_size gcode'owy wymiar blachy.


# TODO przebudowa funkcji do dynamicznej rozdzielczości pod kamere
def singleGcodeElementsCV2(sheet_path, scale = 5, arc_pts_len = 300):
    """
    Creates cv2 image of sheet element from gcode.
    Output image size is the same as bounding box of element times scale

    Args:
        sheet_path (string): absolute path to gcode .nc file
        scale (int): output images resize scale
        arc_pts_len (int): amount of points generated for each circular move from gcode

    Returns:
        images_dict (dictionary): object of "blacha_xx_xx" keys with cv2 generated image from element contours

        pts_dict (dictionary): =||= with element contour points

        pts_hole_dict (dictionary): =||= with element holes contour points

        adjustedCircleLineData (): =||= with circular moves made for each element

        adjustedLinearData (): =||= with linear moves made for each element

        sheet_size (int tuple): tuple of x,y sheet size

    """
    cutting_paths, x_min, x_max, y_min, y_max, sheet_size_line, circleLineData, linearPointsData = visualize_cutting_paths(sheet_path, arc_pts_len= arc_pts_len)
    if sheet_size_line is not None:
        #Rozkodowanie linii na wymiary
        split = sheet_size_line.split()
        sheet_size = (split[1],split[2])
        images_dict = {}
        pts_dict = {}
        pts_hole_dict = {}
        adjustedCircleLineData = {}
        adjustedLinearData = {}
        for key, value in cutting_paths.items():
            pts_hole_dict[f'{key}'] = [] # do zachowania punktów konturu z gcode
            adjustedCircleLineData[f'{key}'] = []
            adjustedLinearData[f'{key}'] = []
            main_contour, holes, = find_main_and_holes(value)
            #min maxy do przeskalowania obrazu
            max_x = max(main_contour, key=lambda item: item[0])[0]
            max_y = max(main_contour, key=lambda item: item[1])[1]
            min_x = min(main_contour, key=lambda item: item[0])[0]
            min_y = min(main_contour, key=lambda item: item[1])[1]

            # Przeskalowanie punktów konturu do zfitowania obrazu
            dx = scale
            dy = scale
            output_res_x = int((max_x-min_x+1)*scale)
            output_res_y = int((max_y-min_y+1)*scale)

            img = np.zeros((output_res_y, output_res_x, 3), dtype=np.uint8)
            adjusted_main = [(int((x-min_x)*dx), int((y-min_y)*dy)) for x, y in main_contour]
            pts = np.array(adjusted_main, np.int32)
            cv2.fillPoly(img, [pts], color=(255,255,255))

            # do narysowania dziur
            adjusted_holes = [[(int((x - min_x)*dx), int((y - min_y)*dy)) for x, y in hole] for hole in holes]
            for hole in adjusted_holes:
                pts2 = np.array(hole, np.int32)
                pts2_resh = pts2.reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts2_resh], color=(0,0,0))
                pts_hole_dict[f'{key}'].append(pts2)
            # adjust circle line'ow
            # try bo może byc kontur bez kół
            try:
                for c in circleLineData[f'{key}']:
                    adjustedCircleLineData[f'{key}'].append(((c[0] - min_x)*dx,(c[1] - min_y)*dy,c[2]*np.sqrt(dy*dx))) #a,b,r
            except:
                pass
            # adjust linear punktów
            try:
                for l in linearPointsData[f'{key}']:
                    adjustedLinearData[f'{key}'].append(((l[0] - min_x)*dx,(l[1] - min_y)*dy))
            except:
                pass
            #binaryzacja
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 30, 255, 0)
            images_dict[f"{key}"] = thresh
            pts_dict[f"{key}"] = pts
            # cv2.imshow('winname',thresh)
            # cv2.waitKey(0)
        return images_dict, pts_dict, sheet_size, pts_hole_dict, adjustedCircleLineData, adjustedLinearData
    else:
        return None, None, None, None,None

def imageBInfoExtraction(imageB):
    """
    Extracts data for quality control from camera image
    Args:
        imageB (nd.array): gray and threshholded camera image
    Returns:
        corners (np.array): array of tuples containing corner info (x,y,corner_val).

        corner_val (int) is evaluation of how "sure of a pick" the corner is (range 0-255)

        contours (nd.array): array of contour points

        contourImage (nd.array): cv2 image with drawn detected contour points

        polyImage (nd.array): cv2 image with drawn detected corner points TO BE UPDATED

        hullImage (nd.array): cv2 image with drawn corners from cv2.convexHull()

    """
    imageBFloat = np.float32(imageB)
    contours, _ = cv2.findContours(imageB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourImage = cv2.cvtColor(imageB, cv2.COLOR_GRAY2BGR)
    polyImage = contourImage.copy()
    hullImage = contourImage.copy()
    for contour in contours:
        # Przybliżenie
        arc = 0.015 * cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, arc, -1, True)
        #oryginalny kontur
        cv2.drawContours(contourImage, [contour], -1, (255, 100, 0), 2)
        for point in contour:
            cv2.circle(contourImage, (point[0][0], point[0][1]), 3, (0, 0, 255), 2)
        #kontur przybliżony (punkty kluczowe)
        cv2.drawContours(polyImage, [poly], -1, (155, 155, 0), 2)
        for point in poly:
            cv2.circle(polyImage, (point[0][0],point[0][1]), 3, (0, 0, 255), 2)

        #Hull punkty bogate
        cv2.drawContours(hullImage, cv2.convexHull(contour),-1, (0,0, 255), 10)
    #Harris Corner detection
    dst = cv2.cornerHarris(imageBFloat, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    rows = dst.shape[0]
    columns = dst.shape[1]
    corners_f32 = np.array([])
    corners_index = np.array([])
    threshhold = 0.01*dst.max()
    for i in range(rows):
        for j in range(columns):
            point = dst[i][j]
            if point>threshhold:
                corners_f32 = np.append(corners_f32,dst[i][j])
                corners_index = np.array([np.append(corners_index,np.array([i,j]))])
            else:
                continue
    corners_index = corners_index.reshape(-1, 2)
    corners_uint8 = corners_f32.astype(np.uint8)

    corners = np.array([])
    for i in range(corners_uint8.shape[0]):
        buf = np.append(corners_index[i],corners_uint8[i])
        corners = np.append(corners,buf)
    corners = corners.reshape(-1,3) # elementy to [[x,y,corner_val],...]
    return corners, contours, contourImage, polyImage, hullImage

def lineFromPoints(x1, y1, x2, y2):
    """
    calculates linear function parameters for 2 points
    Args:
        x1 (int): point 1 x val

        y1 (int): point 1 y val

        x2 (int): point 2 x val

        y2 (int): point 2 y val

    Returns:
        A,B,C (float): linear function parameters denoted as Ax+By+C = 0
    """
    if x1 == x2:  # prosta pionowa
        A = 1
        B = 0
        C = -x1
    elif y1 == y2:  # prosta pozioma
        A = 0
        B = 1
        C = -y1
    else:
        m = (y2 - y1) / (x2 - x1)  
        b = y1 - m * x1 
        # Równanie prostej w postaci ogólnej: Ax + By + C = 0
        A = m
        B = -1
        C = b
    return A, B, C

def linesContourCompare(imageB,gcode_data):
    """
        Compares image from camera to gcode image.
        Prints RMSE error
    Args:
        imageB (nd.array): gray and threshholded camera image

        gcode_data (dictionary): Object with linear and circular movements from gcode element
    """
    #imageB - przetworzony obraz, po thresholdzie, biała figura czarne tło
    cornersB, contoursB, contourBImage, _, _ = imageBInfoExtraction(imageB)
    gcodeLines = {
        "circle": gcode_data['circleData'],
        "linear": [],
    }
    lean = gcode_data['linearData']
    imgCopy = imageB.copy()
    imgCopy = cv2.cvtColor(imgCopy,cv2.COLOR_GRAY2BGR)
    #tworzenie lini z "punktow liniowych"
    #FIXME litera A ma brakującą linie? -- gcode bledy juz oznaczony
    for i in range(len(gcode_data['linearData'])): # tuple i lista tupli
        if i == 1:
            continue
        x1,y1 = gcode_data['linearData'][i]
        x2,y2 = gcode_data['linearData'][i-1]
        cv2.line(imgCopy,(int(x1),int(y1)),(int(x2),int(y2)),random_color(),3)
        A,B,C = lineFromPoints(x1,y1,x2,y2)
        gcodeLines['linear'].append((A,B,C))

    cntrErrors = []

    for i in range(len(contoursB)):
        for j in range(len(contoursB[i])):
            xCntr,yCntr = contoursB[i][j][0] #we love numpy with this one
            #porównanie do lini prostej
            d_minimal = 1000
            for l in range(len(gcodeLines["linear"])):
                A,B,C = gcodeLines["linear"][l] # (a,b) y=ax+b
                d = np.abs(A*xCntr + B*yCntr + C)/(np.sqrt(A**2 + B**2))
                if d < d_minimal:
                    d_minimal = d

            #porównanie do kół
            for k in range(len(gcodeLines["circle"])):
                a, b, r = gcodeLines["circle"][k]
                cv2.circle(imgCopy, (int(a),int(b)),int(np.abs(r)),(0,0,255), 3)
                d_circ = np.abs(np.sqrt((xCntr - a)**2 + (yCntr - b)**2) - r)
                if d_circ < d_minimal:
                    d_minimal = d_circ

            cntrErrors.append(d_minimal)
    cntrErrorMAX = max(cntrErrors)
    RMSE = sum(np.sqrt(e*e) for e in cntrErrors)/len(cntrErrors)
    imageBGR = cv2.cvtColor(imageB,cv2.COLOR_GRAY2BGR)
    bigPhoto = cv2.hconcat([imageBGR,imgCopy])
    bigPhoto2 = cv2.hconcat([contourBImage,imgCopy])
    cv2.imshow("CircleNation",bigPhoto2)
    print("----------\n")
    contourSum = 0
    for contour in contoursB:
        contourSum += len(contour)
    print("ilosc punktow: ",contourSum)
    print("RMSE:",RMSE)

def random_color():
    """
        Generates a random color in BGR format.
    Returns:
        tuple: A tuple representing the color (B, G, R).
    """
    b = random.randint(0, 255)  # Random blue value
    g = random.randint(0, 255)  # Random green value
    r = random.randint(0, 255)  # Random red value
    return (b, g, r)


if __name__ == "__main__":
    # Test czy spakowana funkcja działa
    images, pts, sheet_size, pts_hole, circleLineData, linearData = singleGcodeElementsCV2(
        sheet_path='../../../../Gcode to image conversion/NC_files/2_FIXME.NC',
        arc_pts_len=300)
    for key, value in images.items():
        # Porównanie contours i approx poly w znajdowaniu punktów.
        corners, contours, contourImage, polyImage, hullImage = imageBInfoExtraction(value)
        # wyciąganie ze słownika
        points = pts[f'{key}']
        points_hole = pts_hole[f'{key}']

        try:
            linData = linearData[f'{key}']
        except:
            linData = []

        try:
            circData = circleLineData[f'{key}']
        except:
            circData = []

        gcode_data_packed = {
            "linearData": linData,
            "circleData": circData,
        }
        linesContourCompare(value, gcode_data_packed)

        buf = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        #========== COLOR SCHEME PUNKTOW =============
        #czerwony - punkty proste głównego obrysu
        #żółty - punkty koliste głównego obrysu
        #różowy - punkty proste otworu
        #zielony - punkty koliste otworu

        # Wizualizacja punktów głównego konturu + punktow kolistych
        for i in range(len(points)):
            cv2.circle(buf, (points[i][0],points[i][1]),3 ,(0,0,255),3)

        # # Wizualizacja punktów wycięć + punktow kolistych
        for j in range(len(points_hole)):
            for i in range(len(points_hole[j])):
                cv2.circle(buf, (points_hole[j][i][0],points_hole[j][i][1]),2,(0, 255, 0), 2)


        # cv2.polylines(buf,[points],True,(0,0,255),thickness = 3)
        # cv2.imshow("imageB Contour", contourImage) # niebieski - obrys, czerwony - punkty konturu
        # cv2.imshow("imageB Poly", polyImage) # morski - linie konturu, czerwony - uproszczone punkty konturu
        # cv2.imshow("imageB Hull", hullImage)
        # gigazdjecie
        # imgTop = cv2.hconcat([buf, contourImage])
        # imgBottom = cv2.hconcat([polyImage, hullImage])
        # imgMerge = cv2.vconcat([imgTop, imgBottom])
        # cv2.imshow("BIG MERGE", imgMerge)

        cv2.imshow("Gcode image + punkty Gcode", buf)
        cv2.waitKey(0)
        cv2.destroyAllWindows()






