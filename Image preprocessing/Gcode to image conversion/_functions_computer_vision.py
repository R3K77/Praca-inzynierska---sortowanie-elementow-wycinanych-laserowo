from collections import defaultdict

from shapely.lib import reverse

from _functions_gcode_analize import find_main_and_holes
import numpy as np
import cv2
import re
import json
import base64

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
    # - zapis danych o elementach do json
    # - zdjecia elementów np.array (zgodne z cv2)
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
    cv2_images = gcodeToImageCV2(elements)
    angles = elementRotationByTemplateMatching(cv2_images)
    json_object = {
        "elements" : elements,
        "sheet_size" : sheet_size,
        "curveCircleData" : curveCircleData,
        "linearPointsData" : linearPointsData,
        "rotation" : angles
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

def gcodeToImageCV2(cutting_paths,scale=3):
    images_dict = {}
    for key, value in cutting_paths.items():
        main_contour, holes, = find_main_and_holes(value)
        # min maxy do przeskalowania obrazu
        max_x = max(main_contour, key=lambda item: item[0])[0]
        max_y = max(main_contour, key=lambda item: item[1])[1]
        min_x = min(main_contour, key=lambda item: item[0])[0]
        min_y = min(main_contour, key=lambda item: item[1])[1]

        # Przeskalowanie punktów konturu do zfitowania obrazu
        dx = scale
        dy = scale
        output_res_x = int((max_x - min_x + 1) * scale)
        output_res_y = int((max_y - min_y + 1) * scale)

        img = np.zeros((output_res_y, output_res_x, 3), dtype=np.uint8)
        adjusted_main = [(int((x - min_x) * dx), int((y - min_y) * dy)) for x, y in main_contour]
        pts = np.array(adjusted_main, np.int32)
        cv2.fillPoly(img, [pts], color=(255, 255, 255))

        # do narysowania dziur
        adjusted_holes = [[(int((x - min_x) * dx), int((y - min_y) * dy)) for x, y in hole] for hole in holes]
        for hole in adjusted_holes:
            pts2 = np.array(hole, np.int32)
            pts2_resh = pts2.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts2_resh], color=(0, 0, 0))

        # binaryzacja
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 30, 255, 0)
        # gcode_image_base64 = base64.b64encode(thresh).decode('utf-8')
        images_dict[f"{key}"] = thresh
    return images_dict

def capture_median_frame(crop_values,camera_id):
    frames = 100
    BgrSubtractor = cv2.createBackgroundSubtractorMOG2(history = frames, varThreshold=50,detectShadows=True)
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
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

def cameraImage(BgrSubtractor,crop_values,camera_id):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
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
    img_pack = [crop,cleaned_thresholded,sliced_frame,fg_mask]
    cap.release()
    # cv2.imshow("fg_mask",fg_mask)
    # cv2.imshow("cleaned_thresholded",cleaned_thresholded)
    # cv2.imshow("crop",crop)
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
        if ret > 0.5:
            print(f'Odkształcenie, ret: {ret} \n')
            return False,0,ret
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

        certainErrors = [e for e in cntrErrors if e > 1]
        accuracy = 100 * (len(certainErrors)/len(cntrErrors))
        RMSE = np.sqrt(sum(e*e for e in cntrErrors)/len(cntrErrors))
        print("----------")
        contourSum = 0
        for contour in contoursB:
            contourSum += len(contour)
        print("ilosc punktow: ",contourSum)
        print("RMSE:",RMSE)
        if accuracy < 99.5:
            print("Detal posiada błąd wycięcia")
            print(f'ret: {ret} \n')
            return False,RMSE, ret

        print("Detal poprawny")
        print(f'ret: {ret} \n')
        return True,RMSE, ret

    except Exception as e:
        print("Podczas przetwarzania obrazu wystąpił błąd")
        print(e)
        return False,0, ret

def elementRotationSIFT(images):
    """
    Calculates rotation between objects of the same class using SIFT.

    Args:
        images: Dict of generated cv2 images from gcode
    Returns:
        output_rotation: Dict of rotations when putting the element away
    """
    hash = defaultdict(list)
    output_rotation = {}

    for key, value in images.items():
        cut_key = key[:-4]

        if cut_key not in hash:
            hash[cut_key].append(value)
            output_rotation[key] = 0
        else:
            # Initialize SIFT
            sift = cv2.SIFT_create()
            kp_first, des_first = sift.detectAndCompute(hash[cut_key][0], None)
            kp_other, des_other = sift.detectAndCompute(value, None)

            if len(kp_first) == 0 or len(kp_other) == 0:
                output_rotation[key] = 0
                continue

            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = bf.match(des_first, des_other)
            matches = sorted(matches, key=lambda x: x.distance)
            src_pts = np.float32([kp_first[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_other[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            if len(src_pts) < 4 or len(dst_pts) < 4:
                output_rotation[key] = 0
                continue

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
                print(f"Rotation angle between images: {angle} degrees")
                hash[cut_key].append(value)
                output_rotation[key] = angle
            else:
                print("Homography could not be found.")
                output_rotation[key] = 0

    return output_rotation

def elementRotationPCU(images):
    """
    Calculates rotation between binary images of simple shapes using PCA on contours.

    Args:
        images: Dict of binary cv2 images from gcode.
    Returns:
        output_rotation: Dict of rotations between pairs of images.
    """
    hash = defaultdict(list)
    output_rotation = {}

    for key, value in images.items():
        cut_key = key[:-4]  # Exclude file extension

        # Check if the base image is stored in hash
        if cut_key not in hash:
            hash[cut_key].append(value)
            output_rotation[key] = 0
        else:
            # Get the contours of both images
            contours1, _ = cv2.findContours(hash[cut_key][0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if contours were found in both images
            if len(contours1) == 0 or len(contours2) == 0:
                output_rotation[key] = 0
                continue

            # Perform PCA on the contour points to get the orientation angle
            def get_orientation_angle(contour):
                contour_points = np.squeeze(contour)
                if len(contour_points.shape) < 2:  # Check if there are enough points
                    return None
                mean, eigenvectors = cv2.PCACompute(contour_points.astype(np.float32), mean=None)
                # Calculate the angle of the primary eigenvector
                angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])  # radians
                return np.degrees(angle)  # convert to degrees

            # Get angles for each contour
            angle1 = get_orientation_angle(contours1[0])
            angle2 = get_orientation_angle(contours2[0])

            # Ensure both angles are valid
            if angle1 is None or angle2 is None:
                output_rotation[key] = 0
                continue

            # Calculate relative rotation
            relative_rotation = angle2 - angle1
            # relative_rotation = (relative_rotation + 360) % 360  # Normalize to [0, 360)

            # Handle cases close to 180-degree difference (for flipped shapes)
            if abs(relative_rotation - 180) < 5:  # Tolerance of 5 degrees for 180-degree rotations
                relative_rotation = 180

            output_rotation[key] = relative_rotation
            hash[cut_key].append(value)

    return output_rotation

def elementRotationMoments(images):
    """
    Calculates rotation between binary images of simple shapes using contours.

    Args:
        images: Dict of binary cv2 images from gcode.
    Returns:
        output_rotation: Dict of rotations between pairs of images.
    """
    hash = defaultdict(list)
    output_rotation = {}

    for key, value in images.items():
        cut_key = key[:-4]  # Exclude file extension

        # Check if the base image is stored in hash
        if cut_key not in hash:
            hash[cut_key].append(value)
            output_rotation[key] = 0
        else:
            # Get the contours of both images
            contours1, _ = cv2.findContours(hash[cut_key][0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if contours were found in both images
            if len(contours1) == 0 or len(contours2) == 0:
                output_rotation[key] = 0
                continue

            # Compute image moments to find the center of each shape
            M1 = cv2.moments(contours1[0])
            M2 = cv2.moments(contours2[0])

            if M1["m00"] == 0 or M2["m00"] == 0:
                output_rotation[key] = 0
                continue

            # Calculate the centroid of each contour
            cX1, cY1 = int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"])
            cX2, cY2 = int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"])

            # Shift contours so centroids overlap
            shifted_contour1 = contours1[0] - [cX1, cY1]
            shifted_contour2 = contours2[0] - [cX2, cY2]

            # Calculate the rotation angle using principal component analysis (PCA)
            _, _, angle1 = cv2.fitEllipse(shifted_contour1)
            _, _, angle2 = cv2.fitEllipse(shifted_contour2)

            # Calculate relative rotation
            relative_rotation = angle2 - angle1
            relative_rotation = (relative_rotation + 360) % 360  # Normalize to [0, 360)

            # Handle cases close to 180-degree difference (for flipped shapes)
            if abs(relative_rotation - 180) < 5:  # Tolerance of 5 degrees for 180-degree rotations
                relative_rotation = 180

            output_rotation[key] = relative_rotation
            hash[cut_key].append(value)

    return output_rotation

def elementRotationByTemplateMatching(images):
    """
    Calculates rotation between binary images of simple shapes by using template matching with rotated templates.

    Args:
        images: Dict of binary cv2 images.
    Returns:
        output_rotation: Dict of rotations between pairs of images.
    """
    hash = defaultdict(list)
    output_rotation = {}

    for key, value in images.items():
        cut_key = key[:-4]  # Exclude file extension

        if cut_key not in hash:
            hash[cut_key].append(value)
            output_rotation[key] = 0
        else:
            # Get the template image (first image in the hash)
            template = hash[cut_key][0]
            # Prepare the current image (second image in the comparison)
            target = value

            # Initialize variables for maximum match
            max_match = -1  # To keep track of the best match value
            best_angle = 0  # Angle that gives the best match

            # Loop over a range of angles (e.g., 0 to 360 degrees)
            for angle in range(0, 360, 5):  # Test every 5 degrees (adjust as needed)
                # Rotate the template by the current angle
                rotated_template = rotate_image(template, angle)

                # Check if the rotated template is larger than the target image
                if rotated_template.shape[0] > target.shape[0] or rotated_template.shape[1] > target.shape[1]:
                    # Resize the rotated template to fit within the target image size
                    scale_factor = min(target.shape[0] / rotated_template.shape[0], target.shape[1] / rotated_template.shape[1])
                    rotated_template = cv2.resize(rotated_template, (0, 0), fx=scale_factor, fy=scale_factor)

                # Perform template matching
                result = cv2.matchTemplate(target, rotated_template, cv2.TM_CCOEFF_NORMED)

                # Handle edge case where the result is empty or doesn't match
                if result is None or result.size == 0:
                    continue

                # Get the maximum match value and location
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                # Check if the current match is better than the previous ones
                if max_val > max_match:
                    max_match = max_val
                    best_angle = angle

            # Store the best angle found
            output_rotation[key] = best_angle
            print(f"Rotation angle for {key}: {best_angle} degrees")

            # Add the current image to the hash for future comparisons
            hash[cut_key].append(value)

    return output_rotation

def rotate_image(image, angle):
    """
    Rotates an image by a given angle.

    Args:
        image: The input image to rotate.
        angle: The angle by which to rotate the image.

    Returns:
        The rotated image.
    """
    # Get the image center, which will be the center of rotation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Create the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def sheetRotationTranslation(bgr_subtractor,camera_id,crop_values,sheet_length_mm):
    REFPOINT = (1429, 523) # punkt (0,0,z) bazy robota
    _,_,img_pack = cameraImage(bgr_subtractor,crop_values,camera_id)
    thresh = img_pack[1]
    org_img = img_pack[2]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("thresh",img_pack[1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    contours = max(contours, key = cv2.contourArea)
    final_contours = contours
    if len(final_contours)>4:
        eps = 0.05 * cv2.arcLength(final_contours,True)
        approx = cv2.approxPolyDP(final_contours,eps,True)
        final_contours = approx
    final_contours = final_contours.reshape(-1,2)
    sorted_x = sorted(final_contours, key= lambda point: point[0], reverse=True)
    sorted_y = sorted(final_contours, key=lambda point: point[1],reverse=True)

    bottom_points = sorted_y[:2]
    bottom_points = sorted(bottom_points, key=lambda point: point[0])
    xl,yl = bottom_points[0]
    right_most = sorted_x[:2]
    right_most = sorted(right_most, key=lambda point: point[1])
    xt,yt  = right_most[0]
    xb,yb = right_most[1]
    A,B,C = lineFromPoints(xt,yt,xb,yb)
    try:
        a = -A/B
        alpha = np.arctan(a)
        alpha = np.rad2deg(alpha)
    except:
        alpha = 0

    if alpha > 0:
        alpha = 90 - alpha
    else:
        alpha = -90-alpha
    diff_x_px = REFPOINT[0] - xb
    diff_y_px = REFPOINT[1] - yb

    scalePxMm = sheet_length_mm / np.sqrt((xl - xb)**2 + (yl - yb)**2)
    diff_x = diff_x_px * scalePxMm
    diff_y = diff_y_px * scalePxMm
    data_out = [(xl,yl),(xt,yt),(xb,yb),(A,B,C),img_pack,final_contours]
    return -alpha,(diff_x,diff_y),data_out

def recalibratePoint(point,angle,translation):
    angle_rad = np.radians(angle)
    SE2_rotation = np.array([
        [np.cos(angle_rad),-np.sin(angle_rad)],
        [np.sin(angle_rad),np.cos(angle_rad)]
    ])
    point_np = np.array([point[0],point[1]])
    transformed_point = np.dot(SE2_rotation,point_np) + np.array(translation)
    return tuple(transformed_point)

def nothing(x):
    pass

def get_crop_values(camera_id):
    # Otwórz dostęp do kamery
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
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

    return crop_values, sliced_frame

def draw_circle_on_click(image):
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Rysowanie kółka o promieniu 1 piksel w miejscu kliknięcia
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            print(f"Kliknięto punkt: ({x}, {y})")
            cv2.imshow("Obraz", image)

    # Wyświetlenie obrazu i ustawienie funkcji obsługi zdarzenia kliknięcia
    cv2.imshow("Obraz", image)
    cv2.setMouseCallback("Obraz", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_sheet_json():
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

if __name__ == "__main__":
    generate_sheet_json()
