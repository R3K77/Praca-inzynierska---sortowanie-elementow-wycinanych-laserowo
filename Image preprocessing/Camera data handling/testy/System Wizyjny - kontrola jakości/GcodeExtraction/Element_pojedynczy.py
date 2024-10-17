from collections import defaultdict
import numpy as np
from gcode_analize import visualize_cutting_paths_extended, find_main_and_holes
import cv2
import random

# TODO Inzynierka:
# - quality control shape comparison DONE
# - porównywanie linii obrazu prostego FIXED poprzez korzystanie tylko z tego wyżej XD
# - perspektive transform dla prostego kształtu, badanie bledu w zaleznosci od obrotu elementu względem płaszczyzny obrazu
# - zrobić zly obraz do porównania żeby było zle rmse FIXED po czesci, zle gcode obrazy maja błąd (ale moge zrobic jeszcze)
# - SIFT, rotacja elementow przy odkładaniu DONE
# - Rotacja translacja blachy TODO
# - mocowanie do stolu TODO

# TODO przebudowa funkcji do dynamicznej rozdzielczości pod kamere
def allGcodeElementsCV2(sheet_path, scale = 3, arc_pts_len = 300):
    """
    Creates cv2 images of sheet elements from gcode.
    Output images size is the same as bounding box of element times scale

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
    cutting_paths, x_min, x_max, y_min, y_max, sheet_size_line, circleLineData, linearPointsData = visualize_cutting_paths_extended(sheet_path, arc_pts_len= arc_pts_len)
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

def singleGcodeElementCV2(cutting_path,circle_line_data,linear_points_data):
    new_image = "placeholder"
    adjusted_circle_data = "placeholder2"
    adjusted_linear_data = "placeholder3"
    return new_image,adjusted_circle_data,adjusted_linear_data

def cameraImage():
    image = "placeholder"
    return image

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
    contours, _ = cv2.findContours(imageB, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
    try:
        img = gcode_data['image']
        contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cornersB, contoursB, contourBImage, _, _ = imageBInfoExtraction(imageB)
        ret = cv2.matchShapes(contours[0],contoursB[0],1,0.0)
        for contour in contours:
            for contourB in contoursB:
                ret = cv2.matchShapes(contour, contourB, 1, 0.0)
                if ret > 0.02:
                    print(f'Odkształcenie, ret: {ret} \n')
                    return False
        gcodeLines = {
            "circle": gcode_data['circleData'],
            "linear": [],
        }
        imgCopy = imageB.copy()
        imgCopy = cv2.cvtColor(imgCopy,cv2.COLOR_GRAY2BGR)
        for i in range(len(gcode_data['linearData'])+1): # tuple i lista tupli
            if i == 0:
                continue
            if i == len(gcode_data['linearData']): # obsluga ostatni + pierwszy punkty linia łącząca
                x1, y1 = gcode_data['linearData'][i-1]
                x2, y2 = gcode_data['linearData'][0]
                cv2.line(imgCopy, (int(x1), int(y1)), (int(x2), int(y2)), randomColor(), 3)
                A, B, C = lineFromPoints(x1, y1, x2, y2)
                gcodeLines['linear'].append((A, B, C))
                break

            x1,y1 = gcode_data['linearData'][i]
            x2,y2 = gcode_data['linearData'][i-1]
            cv2.line(imgCopy,(int(x1),int(y1)),(int(x2),int(y2)),randomColor(),3)
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
        RMSE = np.sqrt(sum(e*e for e in cntrErrors)/len(cntrErrors))
        print("----------")
        contourSum = 0
        for contour in contoursB:
            contourSum += len(contour)
        print("ilosc punktow: ",contourSum)
        print("RMSE:",RMSE)
        if RMSE > 1.1:
            print("Detal posiada błąd wycięcia")
            print(f'ret: {ret} \n')
            return False
        else:
            print("Detal poprawny")
            print(f'ret: {ret} \n')
            return True
    except Exception as e:
        print("Podczas przetwarzania obrazu wystąpił błąd")
        print(e)
        return False

def randomColor():
    """
        Generates a random color in BGR format.
    Returns:
        tuple: A tuple representing the color (B, G, R).
    """
    b = random.randint(0, 255)  # Random blue value
    g = random.randint(0, 255)  # Random green value
    r = random.randint(0, 255)  # Random red value
    return (b, g, r)

def elementStackingRotation(images):
    """
        Calculates rotation between objects of the same class.
    Args:
        images: Dict of generated cv2 images from gcode
    Returns:
        output_rotation: Dict of rotations when putting the element away
    """
    hash = defaultdict(list)
    output_rotation = {}
    for key, value in images.items():
        cut_key = key[0:-4]
        if cut_key not in hash:
            hash[cut_key].append(value)
            output_rotation[key] = 0
        else:
            sift = cv2.SIFT_create()
            kp_first,des_first = sift.detectAndCompute(hash[cut_key][0], None)
            kp_other,des_other = sift.detectAndCompute(value, None)
            if len(kp_first) == 0 or len(kp_other) == 0: # brak matchy sift, głównie złe klasy
                output_rotation[key] = 0
                continue
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True)
            matches = bf.match(des_first,des_other)
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([kp_first[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_other[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            if len(src_pts) < 4 or len(dst_pts) < 4: # brak matchy sift
                #powody braku matcha
                # złe klasowanie
                # błąd metody?
                output_rotation[key] = 0
                continue
            M,mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
            if M is not None:
                angle = np.degrees(np.arctan2(M[1,0],M[0,0]))
                print(f"Kąt obrotu między obrazami: {angle} stopni")
                hash[cut_key].append(value)
                output_rotation[key] = (angle)
            else:
                print("Homografia nie została znaleziona.")
                output_rotation[key] = 0
    return output_rotation

def testAlgorithmFunction(key, value, pts, pts_hole, linearData, circleLineData):
    """
    Funkcja testowa do sprawdzania działania funkcji obróbki obrazu

    Args:
        key: Nazwa klucza obrazu.
        value: Obraz do przetwarzania.
        pts: Słownik punktów konturu.
        pts_hole: Słownik punktów otworu.
        linearData: Dane dotyczące punktów prostych.
        circleLineData: Dane dotyczące punktów kolistych.
    """
    # Porównanie contours i approx poly w znajdowaniu punktów.
    corners, contours, contourImage, polyImage, hullImage = imageBInfoExtraction(value)
    # wyciąganie ze słownika
    points = pts[f'{key}']
    points_hole = pts_hole[f'{key}']
    try:
        linData = linearData[f'{key}']
    except KeyError:
        linData = []

    try:
        circData = circleLineData[f'{key}']
    except KeyError:
        circData = []

    gcode_data_packed = {
        "linearData": linData,
        "circleData": circData,
        "image": value,
    }
    linesContourCompare(value, gcode_data_packed)

    buf = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)

    # ========== COLOR SCHEME PUNKTOW =============
    # czerwony - punkty proste głównego obrysu
    # żółty - punkty koliste głównego obrysu
    # różowy - punkty proste otworu
    # zielony - punkty koliste otworu

    # Wizualizacja punktów głównego konturu + punktow kolistych
    for i in range(len(points)):
        cv2.circle(buf, (points[i][0], points[i][1]), 3, (0, 0, 255), 3)

    # # Wizualizacja punktów wycięć + punktow kolistych
    for j in range(len(points_hole)):
        for i in range(len(points_hole[j])):
            cv2.circle(buf, (points_hole[j][i][0], points_hole[j][i][1]), 2, (0, 255, 0), 2)

    # Wyświetlanie obrazów, zakomentowane linie można odkomentować w razie potrzeby
    # cv2.polylines(buf, [points], True, (0, 0, 255), thickness=3)
    # cv2.imshow("imageB Contour", contourImage)
    # cv2.imshow("imageB Poly", polyImage)
    # cv2.imshow("imageB Hull", hullImage)
    # # Gigazdjęcie
    # imgTop = cv2.hconcat([buf, contourImage])
    # imgBottom = cv2.hconcat([polyImage, hullImage])
    # imgMerge = cv2.vconcat([imgTop, imgBottom])
    # cv2.imshow("BIG MERGE", imgMerge)
    cv2.imshow("Gcode image + punkty Gcode", buf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testTilt(imageB,gcode_data):
    """
    Quality control vision system test, simulates tilted element
    Args:
        imageB:
        gcode_data:
    Returns:

    """
    height_new, width_new = imageB.shape[:2]
    src_points_new = np.float32([[0, 0], [width_new, 0], [width_new, height_new], [0, height_new]])
    dst_points_new = np.float32([[50, 0], [width_new - 50, 0], [width_new, height_new], [0, height_new]])
    matrix_new = cv2.getPerspectiveTransform(src_points_new, dst_points_new)
    transformed_image_new = cv2.warpPerspective(imageB, matrix_new, (width_new, height_new))
    return transformed_image_new, linesContourCompare(transformed_image_new,gcode_data)

def testAdditionalHole(imageB,gcode_data):
    """
    Draws a random black shape (circle or rectangle) on the given image.

    Args:
        imageB: Input image on which the shape will be drawn.

    Returns:
        Image with a random black shape drawn on it.
    """
    # Kopia obrazu, aby nie modyfikować oryginału
    image_copy = imageB.copy()

    # Wybór losowego kształtu: 0 dla koła, 1 dla prostokąta
    shape_type = random.choice([0, 1])

    # Wymiary obrazu
    height, width = image_copy.shape[:2]

    # Losowanie współrzędnych dla kształtu
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)

    # Losowanie rozmiaru kształtu
    size = random.randint(10, 50)  # Losowy rozmiar kształtu

    if shape_type == 0:  # Okrąg
        cv2.circle(image_copy, (x, y), size, (0, 0, 0), -1)  # Czarny kolor
    else:  # Prostokąt
        top_left = (x, y)
        bottom_right = (min(x + size, width - 1), min(y + size, height - 1))
        cv2.rectangle(image_copy, top_left, bottom_right, (0, 0, 0), -1)  # Czarny kolor

    return image_copy, linesContourCompare(image_copy,gcode_data)

if __name__ == "__main__":
    # Test czy spakowana funkcja działa
    images, pts, sheet_size, pts_hole, circleLineData, linearData = allGcodeElementsCV2(
        sheet_path='../../../../Gcode to image conversion/NC_files/8.NC',
        arc_pts_len=300)
    rotations = elementStackingRotation(images)
    for key, value in images.items():
        try:
            linData = linearData[f'{key}']
        except KeyError:
            linData = []
        try:
            circData = circleLineData[f'{key}']
        except KeyError:
            circData = []
        gcode_data_packed = {
            "linearData": linData,
            "circleData": circData,
            "image": value,
        }
        # img_transformed,is_image_good = testAdditionalHole(value,gcode_data_packed) # podmienic funkcje w zaleznosci od testu
        # img = cv2.hconcat([value,img_transformed])
        # cv2.imshow("obraz",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        testAlgorithmFunction(key,value,pts,pts_hole,linearData,circleLineData)














