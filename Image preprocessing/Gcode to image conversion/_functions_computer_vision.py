import math
from collections import defaultdict
from _functions_gcode_analize import find_main_and_holes
import numpy as np
import cv2
import random
import re
import json

def visualize_cutting_paths_extended(file_path, x_max=500, y_max=1000, arc_pts_len = 200):
    """
    # ----------------- Funkcja do wizualizacji ścieżek cięcia z pliku NC ----------------- #
    # Autor: Bartłomiej Szalwach,
    # Funkcja plik .nc z kodem G-kodu i zwraca obraz z wizualizacją ścieżek cięcia.
    # Funkcja wykorzystuje bibliotekę matplotlib, numpy i re.
    # Funkcja zwraca ścieżki cięcia, minimalne i maksymalne współrzędne X i Y.
    # Funkcja przyjmuje ścieżkę do pliku .nc z kodem G-kodu oraz maksymalne współrzędne X i Y.
    # Domyślnie maksymalne współrzędne X i Y wynoszą odpowiednio 500 i 1000.
    # ------------------------------------------------------------------------------------- #
    # Założenia funkcji:
    # - Włączenie lasera oznaczone jest jako M10, a wyłączenie jako M11.
    # - Ścieżki cięcia są zapisane jako G01X...Y... lub G02X...Y...I...J... lub G03X...Y...I...J...
    # - Współrzędne X i Y są zapisane jako liczby zmiennoprzecinkowe.
    # - Ruch okrężny w prawo jest zapisany jako G02, a ruch okrężny w lewo jako G03.
    # - Współrzędne I i J określają środek okręgu, a współrzędne X i Y określają punkt końcowy.
    # - Współrzędne I i J są zapisane jako liczby zmiennoprzecinkowe.
    # - Wszystkie współrzędne są mniejsze niż podane maksymalne współrzędne X i Y.
    # ------------------------------------------------------------------------------------- #
    # Przykład użycia:
    # cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths_extended("./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-6001.nc")
    # ------------------------------------------------------------------------------------- #
    # Modyfikacje: Rafał Szygenda
    # - liczba punktów łuków jest argumentem wejściowym funkcji, większa rozdzielczość
    # - zwrotka rozmiaru blachy, dane koła i punktow liniowych (do systemu wizyjnego)
    """
    sheet_size_line = None
    with open(file_path, 'r') as file:
        file_content = file.read().splitlines()

    pattern_cnc_commands_extended = re.compile(
        r'(M10|M11|G01X([0-9.]+)Y([0-9.]+)|G0[23]X([0-9.]+)Y([0-9.]+)I([0-9.-]+)J([0-9.-]+))')

    pattern_element_name = re.compile(r';@@\[DetailName\((.*?)\)\]')
    laser_on = False
    elements = {}
    current_element_name = 'Unnamed'
    element_index = {}  # Słownik do przechowywania indeksów dla każdej nazwy elementu
    current_path = []
    current_position = (0, 0)
    curveCircleData = defaultdict(list)
    linearPointsData = defaultdict(list)

    for line in file_content:
        element_match = pattern_element_name.search(line)
        if "*SHEET" in line:
            sheet_size_line = line
        if element_match:
            name = element_match.group(1)
            if name not in element_index:
                element_index[name] = 1  # Rozpocznij liczenie od 1
            else:
                element_index[name] += 1

            # Formatowanie nazwy z zerami wiodącymi
            current_element_name = f"{name}_{element_index[name]:03d}"  # Dodaje zera wiodące do indeksu
            if current_element_name == "_001":
                current_element_name = prev_element_name
                continue
            prev_element_name = current_element_name
            if current_path:
                if current_element_name not in elements:
                    elements[current_element_name] = []

                elements[current_element_name].append(current_path)
                current_path = []

        else:
            matches_cnc = pattern_cnc_commands_extended.findall(line)
            for match in matches_cnc:
                command = match[0]
                if command == 'M10':  # Laser ON
                    laser_on = True
                elif command == 'M11':  # Laser OFF
                    if laser_on and current_path:  # Zapis ścieżki do bieżącego elementu
                        if current_element_name not in elements:
                            elements[current_element_name] = []

                        elements[current_element_name].append(current_path)
                        current_path = []
                    laser_on = False
                elif laser_on:  # Dodaj punkty do ścieżki, jeśli laser jest włączony
                    # Obsługa instrukcji cięcia...
                    if command.startswith('G01'):  # Linia prosta
                        x, y = float(match[1]), float(match[2])
                        current_path.append((x, y))
                        current_position = (x, y)

                        linearPointsData[current_element_name].append(current_position)

                    elif command.startswith('G02') or command.startswith('G03'):  # Ruch okrężny
                        x, y, i, j = (float(match[3]), float(match[4]),
                                      float(match[5]), float(match[6]))
                        center_x = current_position[0] + i  # Środek łuku na osi X
                        center_y = current_position[1] + j  # Środek łuku na osi Y
                        radius = np.sqrt(i ** 2 + j ** 2)  # Promień łuku
                        start_angle = np.arctan2(current_position[1] - center_y,
                                                 current_position[0] - center_x)  # Kąt początkowy łuku (w radianach)
                        end_angle = np.arctan2(y - center_y, x - center_x)  # Kąt końcowy łuku (w radianach)

                        if command.startswith('G02'):  # Zgodnie z ruchem wskazówek zegara
                            if end_angle > start_angle:
                                end_angle -= 2 * np.pi
                        else:  # Przeciwnie do ruchu wskazówek zegara
                            if end_angle < start_angle:
                                end_angle += 2 * np.pi

                        angles = np.linspace(start_angle, end_angle, num=arc_pts_len)  # Generowanie punktów łuku (50 punktów)
                        arc_points = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in
                                      angles]  # Obliczenie punktów łuku
                        curveCircleData[current_element_name].append((center_x,center_y,radius))
                        linearPointsData[current_element_name].append(arc_points[-1])
                        current_path.extend(arc_points)  # Dodanie punktów łuku do ścieżki
                        current_position = (x, y)  # Aktualizacja pozycji

    # Jeśli laser został wyłączony po ostatnim cięciu, dodajemy ścieżkę do listy
    if current_path:
        if current_element_name not in elements:
            elements[current_element_name] = []
        elements[current_element_name].append(current_path)

    # Rozmiar arkusza
    x_min, y_min = 0, 0
    split = sheet_size_line.split()
    sheet_size = (split[1], split[2])
    json_object = {
        "elements" : elements,
        "sheet_size" : sheet_size,
        "curveCircleData" : curveCircleData,
        "linearPointsData" : linearPointsData,
    }
    with open(f"elements_data_json/{current_element_name[:7]}.json","w") as f:
        json.dump(json_object,f)

    return elements, x_min, x_max, y_min, y_max, sheet_size_line, curveCircleData, linearPointsData

def allGcodeElementsCV2(sheet_path, scale = 5, arc_pts_len = 300):
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

def singleGcodeElementCV2(cutting_path,circle_line_data,linear_points_data,bounding_box_size):
    #przeskalowanie
    main_contour, holes, = find_main_and_holes(cutting_path)
    max_x = max(main_contour, key=lambda item: item[0])[0]
    max_y = max(main_contour, key=lambda item: item[1])[1]
    min_x = min(main_contour, key=lambda item: item[0])[0]
    min_y = min(main_contour, key=lambda item: item[1])[1]
    width,height = bounding_box_size

    if max_x == min_x and max_y == min_y:
        raise ValueError("Błąd GCODE obrazu")
    if max_x == min_x:
        scale_x = None
    else:
        scale_x = width / (max_x - min_x)

    if max_y == min_y:
        scale_y = None
    else:
        scale_y = width / (max_y - min_y)

    if scale_x is not None and scale_y is not None:
        scale = max(scale_x, scale_y)
    elif scale_x is None:
        scale = scale_y
    else:
        scale = scale_x

    dx = scale
    dy = scale
    output_res_x = width
    output_res_y = height
    img = np.zeros((output_res_y, output_res_x, 3), dtype=np.uint8)

    #przeskalowanie danych do ucietego obrazu
    adjusted_main = [(int((x - min_x) * dx), int((y - min_y) * dy)) for x, y in main_contour]
    pts = np.array(adjusted_main, np.int32)
    cv2.fillPoly(img, [pts], color=(255, 255, 255))

    adjusted_holes = [[(int((x - min_x) * dx), int((y - min_y) * dy)) for x, y in hole] for hole in holes]
    for hole in adjusted_holes:
        pts2 = np.array(hole, np.int32)
        pts2_resh = pts2.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts2_resh], color=(0, 0, 0))

    adjusted_circle_data = []
    for c in circle_line_data:
        adjusted_circle_data.append(((c[0] - min_x) * dx, (c[1] - min_y) * dy, c[2] * np.sqrt(dx * dy)))  # a, b, r

    adjusted_linear_data = []
    for l in linear_points_data:
        adjusted_linear_data.append(((l[0] - min_x) * dx, (l[1] - min_y) * dy))

    # Binary thresholding to convert the image to black and white
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, 0)
    # Pack the results
    gcode_data_packed = {
        "image": thresh,
        "linearData": adjusted_linear_data,
        "circleData": adjusted_circle_data,
    }
    return gcode_data_packed

def capture_median_frame(crop_values):
    frames = 100
    BgrSubtractor = cv2.createBackgroundSubtractorMOG2(history = frames, varThreshold=50,detectShadows=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while frames > 0:
        ret,frame = cap.read()
        frame = camera_calibration(frame)
        h,w,_ = frame.shape
        top_px = crop_values["top"]
        bottom_px = crop_values["bottom"]
        left_px = crop_values["left"]
        right_px = crop_values["right"]
        sliced_frame = frame[top_px:h - bottom_px, left_px:w - right_px]
        if not ret:
            break
        BgrSubtractor.apply(sliced_frame,learningRate = 0.1)
        frames -=1
    cap.release()
    return BgrSubtractor

def cameraImage(BgrSubtractor,crop_values):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret,frame = cap.read()
    while not ret:
        ret,frame = cap.read()

    frame = camera_calibration(frame)
    h, w, _ = frame.shape
    top_px = crop_values["top"]
    bottom_px = crop_values["bottom"]
    left_px = crop_values["left"]
    right_px = crop_values["right"]
    sliced_frame = frame[top_px:h - bottom_px, left_px:w - right_px]

    fg_mask = BgrSubtractor.apply(sliced_frame,learningRate = 0)
    _,fg_mask = cv2.threshold(fg_mask, 130, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    cleaned_thresholded = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours)
    crop = cleaned_thresholded[y:y + h, x:x + w].copy()
    img_pack = [crop,cleaned_thresholded,frame]
    return crop, (w,h), img_pack

def camera_calibration(frame):
    # Wczytanie parametrów kamery z pliku
    loaded_mtx = np.loadtxt('settings/mtx_matrix.txt', delimiter=',')
    loaded_dist = np.loadtxt('settings/distortion_matrix.txt', delimiter=',')
    loaded_newcameramtx = np.loadtxt('settings/new_camera_matrix.txt', delimiter=',')
    loaded_roi = np.loadtxt('settings/roi_matrix.txt', delimiter=',')

    # Kalibracja kamery
    frame = cv2.undistort(frame, loaded_mtx, loaded_dist, None, loaded_newcameramtx)
    x, y, w, h = map(int, loaded_roi)
    frame = frame[y:y + h, x:x + w]
    return frame

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
        contoursB,_ = cv2.findContours(imageB,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        ret = cv2.matchShapes(contours[0],contoursB[0],1,0.0)
        if ret > 0.05:
            print(f'Odkształcenie, ret: {ret} \n')
            return False,0, img,img
        gcodeLines = {
            "circle": gcode_data['circleData'],
            "linear": [],
        }
        imgCopy = imageB.copy()
        imgCopy_lines = cv2.cvtColor(imgCopy,cv2.COLOR_GRAY2BGR)
        imgCopy_circ = imgCopy_lines.copy()
        for i in range(len(gcode_data['linearData'])+1): # tuple i lista tupli
            if i == 0:
                continue
            if i == len(gcode_data['linearData']): # obsluga ostatni + pierwszy punkty linia łącząca
                x1, y1 = gcode_data['linearData'][i-1]
                x2, y2 = gcode_data['linearData'][0]
                A, B, C = lineFromPoints(x1, y1, x2, y2)
                gcodeLines['linear'].append((A, B, C))
                break

            x1,y1 = gcode_data['linearData'][i]
            x2,y2 = gcode_data['linearData'][i-1]
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
                    cv2.circle(imgCopy_circ, (int(a),int(b)),int(np.abs(r)),(0,0,255), 3)
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
            return False,RMSE, imgCopy_lines,imgCopy_circ
        else:
            print("Detal poprawny")
            print(f'ret: {ret} \n')
            return True,RMSE, imgCopy_lines,imgCopy_circ
    except Exception as e:
        print("Podczas przetwarzania obrazu wystąpił błąd")
        print(e)
        return False,0, imageB, imageB

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

def sheetRotationTranslation(bgr_subtractor):
    REFPOINT = (1444, 517) # robocie refpoint to (0,0)
    _,_,img_pack = cameraImage(bgr_subtractor)
    thresh = img_pack[1]
    org_img = img_pack[2]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("thresh",img_pack[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours = max(contours, key = cv2.contourArea)
    final_contours = contours
    if len(final_contours)>4:
        eps = 0.05 * cv2.arcLength(final_contours,True)
        approx = cv2.approxPolyDP(final_contours,eps,True)
        final_contours = approx
    final_contours = final_contours.reshape(-1,2)
    sorted_x = sorted(final_contours, key= lambda point: point[0], reverse=True)
    right_most = sorted_x[:2]
    right_most = sorted(right_most, key=lambda point: point[1])
    xt,yt  = right_most[0]
    xb,yb = right_most[1]
    A,B,_ = lineFromPoints(xt,yt,xb,yb)
    a = -A/B
    alpha = np.arctan(a)
    alpha = np.rad2deg(alpha)
    if alpha > 0:
        alpha = 90 - alpha
    else:
        alpha = -90-alpha
    diff_x = abs(REFPOINT[0] - xb)
    diff_y = abs(REFPOINT[1] - yb)

    return alpha,(diff_x,diff_y)

def nothing(x):
    pass

def get_crop_values():
    # Otwórz dostęp do kamery
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Nie można otworzyć kamery")
        return None

    # Tworzenie okna
    cv2.namedWindow('Podgląd z kamery')

    # Dodanie suwaków
    cv2.createTrackbar('Góra', 'Podgląd z kamery', 0, 100, nothing)
    cv2.createTrackbar('Dół', 'Podgląd z kamery', 0, 100, nothing)
    cv2.createTrackbar('Lewa', 'Podgląd z kamery', 0, 100, nothing)
    cv2.createTrackbar('Prawa', 'Podgląd z kamery', 0, 100, nothing)

    # Zmienna przechowująca końcowe wartości
    crop_values = {"top": 0, "bottom": 0, "left": 0, "right": 0}

    while True:
        # Przechwyć klatkę z kamery
        ret, frame = cap.read()
        frame = camera_calibration(frame)
        # Jeśli nie udało się pobrać klatki, zakończ
        if not ret:
            print("Nie można pobrać klatki")
            break

        # Pobierz wartości z suwaków
        top = cv2.getTrackbarPos('Góra', 'Podgląd z kamery')
        bottom = cv2.getTrackbarPos('Dół', 'Podgląd z kamery')
        left = cv2.getTrackbarPos('Lewa', 'Podgląd z kamery')
        right = cv2.getTrackbarPos('Prawa', 'Podgląd z kamery')

        # Oblicz nowe wymiary obrazu
        h, w, _ = frame.shape
        top_px = int(h * (top / 100))
        bottom_px = int(h * (bottom / 100))
        left_px = int(w * (left / 100))
        right_px = int(w * (right / 100))

        # Wycinanie obrazu (slicing)
        sliced_frame = frame[top_px:h-bottom_px, left_px:w-right_px]

        # Wyświetl obraz po przycięciu
        cv2.imshow('Podgląd z kamery', sliced_frame)

        # Wyjście po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Zapisz ostateczne wartości
            crop_values["top"] = top_px
            crop_values["bottom"] = bottom_px
            crop_values["left"] = left_px
            crop_values["right"] = right_px
            break

    # Zakończ nagrywanie i zamknij okna
    cap.release()
    cv2.destroyAllWindows()

    return crop_values

if __name__ == "__main__":
    paths = [
        "NC_files/1.NC",
        "NC_files/2_FIXME.NC",
        "NC_files/3.NC",
        "NC_files/4.NC",
        "NC_files/5.NC",
        "NC_files/6.NC",
        "NC_files/7.NC",
        "NC_files/8.NC",
    ]
    for path in paths:
        visualize_cutting_paths_extended(path)
