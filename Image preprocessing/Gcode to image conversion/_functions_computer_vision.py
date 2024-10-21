import math
from collections import defaultdict
import numpy as np
import cv2
import random
import re

def calculate_centroid(poly):
    """
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
    """
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
            if current_path:
                if current_element_name not in elements:
                    elements[current_element_name] = []

                elements[current_element_name].append(current_path)
                current_path = []
                linearPointsData[current_element_name].append(current_position)

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

                        linearPointsData[current_element_name].append(current_position)

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

    return elements, x_min, x_max, y_min, y_max, sheet_size_line, curveCircleData, linearPointsData

def find_main_and_holes(contours):
    """
    # ----------------- Funkcja do znalezienia głównego konturu i otworów ----------------- #
    # Autor: Bartłomiej Szalwach
    # Funkcja przyjmuje listę konturów i zwraca główny kontur i otwory.
    # ------------------------------------------------------------------------------------- #
    # Założenia funkcji:
    # - Główny kontur jest konturem z największym polem powierzchni.
    # - Otwory są konturami z mniejszym polem powierzchni.
    # ------------------------------------------------------------------------------------- #
    # Przykład użycia:
    # main_contour, holes = find_main_and_holes(contours)
    # ------------------------------------------------------------------------------------- #
    """
    areas = [(calculate_centroid(contour)[1], contour) for contour in contours]
    areas.sort(reverse=True, key=lambda x: x[0])
    main_contour = areas[0][1]
    holes = [area[1] for area in areas[1:]]
    return main_contour, holes

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

def capture_median_frame():
    while True:
        cap = cv2.VideoCapture(1)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not cap.isOpened():
            print("Nie udało się otworzyć kamerki")
            continue
        else:
            break

    frames = []
    while len(frames) < 10:
        ret, frame = cap.read()
        frame = camera_calibration(frame)
        if not ret:
            continue
        frames.append(frame)

    cap.release()
    median_frame = np.median(np.array(frames), axis=0).astype(np.uint8)

    return median_frame

def cameraImage(median_background_frame):
    median_frame = capture_median_frame()
    gray_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    gray_background  = cv2.cvtColor(median_background_frame,cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_background,gray_frame)
    _, thresholded = cv2.threshold(diff, 70, 150, cv2.THRESH_BINARY)
    #odszumienie
    kernel = np.ones((5, 5), np.uint8)
    cleaned_thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(cleaned_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    crop = cleaned_thresholded[y:y + h, x:x + w]
    binarised = cv2.threshold(crop,50,255,cv2.THRESH_BINARY)[1]
    cv2.imshow("gray_image",gray_frame)
    cv2.imshow("diff",diff)
    cv2.imshow('thr',cleaned_thresholded)
    cv2.imshow("binarised",binarised)
    cv2.imshow("przetworzony", crop)
    img_pack = [median_frame,gray_frame,diff,cleaned_thresholded,crop]
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return binarised , (w,h), img_pack

def camera_calibration(frame):
    # Wczytanie parametrów kamery z pliku
    loaded_mtx = np.loadtxt('../../../settings/mtx_matrix.txt', delimiter=',')
    loaded_dist = np.loadtxt('../../../settings/distortion_matrix.txt', delimiter=',')
    loaded_newcameramtx = np.loadtxt('../../../settings/new_camera_matrix.txt', delimiter=',')
    loaded_roi = np.loadtxt('../../../settings/roi_matrix.txt', delimiter=',')

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
        cornersB, contoursB, contourBImage, _, _ = imageBInfoExtraction(imageB)
        ret = cv2.matchShapes(contours[0],contoursB[0],1,0.0)
        if ret > 0.05:
            print(f'Odkształcenie, ret: {ret} \n')
            return False, img,img
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
                cv2.line(imgCopy_lines, (int(x1), int(y1)), (int(x2), int(y2)), randomColor(), 3)
                A, B, C = lineFromPoints(x1, y1, x2, y2)
                gcodeLines['linear'].append((A, B, C))
                break

            x1,y1 = gcode_data['linearData'][i]
            x2,y2 = gcode_data['linearData'][i-1]
            cv2.line(imgCopy_lines,(int(x1),int(y1)),(int(x2),int(y2)),randomColor(),3)
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
            return False, imgCopy_lines,imgCopy_circ
        else:
            print("Detal poprawny")
            print(f'ret: {ret} \n')
            return True, imgCopy_lines,imgCopy_circ
    except Exception as e:
        print("Podczas przetwarzania obrazu wystąpił błąd")
        print(e)
        return False, imageB, imageB

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

def sheetRotationTranslation(background_frame):
    REFPOINT = (100,100) # robocie refpoint to (0,0)
    gray_background =  cv2.cvtColor(background_frame,cv2.COLOR_BGR2GRAY)
    _,_,img_pack = cameraImage(background_frame)
    thresh = img_pack[4]
    org_img = img_pack[0]
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = contours
    if len(contours)>4:
        eps = 0.05 * cv2.arcLength(contours,True)
        approx = cv2.approxPolyDP(contours,eps,True)
        final_contours = approx
    final_contours = final_contours.reshape(-1,2)
    sorted_x = sorted(final_contours, key= lambda point: point[0], reverse=True)
    right_most = sorted_x[:2]
    right_most = sorted(right_most, key=lambda point: point[1])
    xt,yt  = right_most[0]
    xb,yb = right_most[1]
    A,B,C = lineFromPoints(xt,yt,xb,yb)
    a = -A/B
    alpha = math(a)
    diff_x = abs(REFPOINT[0] - xb)
    diff_y = abs(REFPOINT[1] - yb)
    cv2.circle(org_img, REFPOINT,2,(0,0,255), 3)
    cv2.imshow("camera_img",org_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return alpha,(diff_x,diff_y)

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
    _,img_lines,img_circles = linesContourCompare(value, gcode_data_packed)
    buf = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)

    # ========== COLOR SCHEME PUNKTOW =============
    # czerwony - punkty proste głównego obrysu
    # żółty - punkty koliste głównego obrysu
    # różowy - punkty proste otworu
    # zielony - punkty koliste otworu

    # Wizualizacja punktów głównego konturu + punktow kolistych
    # for i in range(len(points)):
    #     cv2.circle(buf, (points[i][0], points[i][1]), 3, (0, 0, 255), 3)
    #
    # # # Wizualizacja punktów wycięć + punktow kolistych
    # for j in range(len(points_hole)):
    #     for i in range(len(points_hole[j])):
    #         cv2.circle(buf, (points_hole[j][i][0], points_hole[j][i][1]), 2, (0, 255, 0), 2)

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
    value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    value_with_border = cv2.copyMakeBorder(value, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_lines_with_border = cv2.copyMakeBorder(img_lines, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_circles_with_border = cv2.copyMakeBorder(img_circles, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.hconcat([value_with_border, img_lines_with_border])
    img2 = cv2.hconcat([img, img_circles_with_border])
    cv2.imwrite(f"{key}.jpg", value_with_border)
    cv2.imwrite(f"{key}_lines.jpg",img_lines_with_border)
    cv2.imwrite(f"{key}_circles.jpg",img_circles_with_border)
    # cv2.imshow("Gcode image + punkty Gcode", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
