from collections import defaultdict
from shapely.lib import reverse
import numpy as np
import cv2
import json
import random
import os
import csv
import numpy as np
import re
import base64
from shapely.geometry import Point

REFPOINT = (1423, 549)


def visualize_cutting_paths_extended(file_path, x_max=500, y_max=1000, arc_pts_len=200):
  """
    # Funkcja do wizualizacji ścieżek cięcia z pliku NC
    Autor: Bartłomiej Szalwach,
    Funkcja plik .nc z kodem G-kodu i zwraca obraz z wizualizacją ścieżek cięcia.
    Funkcja wykorzystuje bibliotekę matplotlib, numpy i re.
    Funkcja zwraca ścieżki cięcia, minimalne i maksymalne współrzędne X i Y.
    Funkcja przyjmuje ścieżkę do pliku .nc z kodem G-kodu oraz maksymalne współrzędne X i Y.
    Domyślnie maksymalne współrzędne X i Y wynoszą odpowiednio 500 i 1000.
    Założenia funkcji:
    - Włączenie lasera oznaczone jest jako M10, a wyłączenie jako M11.
    - Ścieżki cięcia są zapisane jako G01X...Y... lub G02X...Y...I...J... lub G03X...Y...I...J...
    - Współrzędne X i Y są zapisane jako liczby zmiennoprzecinkowe.
    - Ruch okrężny w prawo jest zapisany jako G02, a ruch okrężny w lewo jako G03.
    - Współrzędne I i J określają środek okręgu, a współrzędne X i Y określają punkt końcowy.
    - Współrzędne I i J są zapisane jako liczby zmiennoprzecinkowe.
    - Wszystkie współrzędne są mniejsze niż podane maksymalne współrzędne X i Y.
    # Przykład użycia:
    cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths_extended("./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-6001.nc")
    # Modyfikacje
    Rafał Szygenda
    - liczba punktów łuków jest argumentem wejściowym funkcji, większa rozdzielczość
    - zwrotka rozmiaru blachy, dane koła i punktow liniowych (do systemu wizyjnego)
    - Kąt obrotu końcowy
    - zapis danych o elementach do json
    - zdjecia elementów np.array (zgodne z cv2)
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
  element_index = {}
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
        element_index[name] = 1
      else:
        element_index[name] += 1
      current_element_name = f"{name}_{element_index[name]:03d}"
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
            curveCircleData[current_element_name].append((center_x, center_y, radius))
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
    "elements": elements,
    "sheet_size": sheet_size,
    "curveCircleData": curveCircleData,
    "linearPointsData": linearPointsData,
    "rotation": angles
  }
  with open(f"Image preprocessing/Gcode to image conversion/elements_data_json/{current_element_name[:7]}.json",
            "w") as f:
    json.dump(json_object, f)
  return elements, x_min, x_max, y_min, y_max, sheet_size_line, curveCircleData, linearPointsData

def allGcodeElementsCV2(sheet_path, scale=5, arc_pts_len=300):
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
  cutting_paths, x_min, x_max, y_min, y_max, sheet_size_line, circleLineData, linearPointsData = visualize_cutting_paths_extended(
    sheet_path, arc_pts_len=arc_pts_len)
  if sheet_size_line is not None:
    # Rozkodowanie linii na wymiary
    split = sheet_size_line.split()
    sheet_size = (split[1], split[2])
    images_dict = {}
    pts_dict = {}
    pts_hole_dict = {}
    adjustedCircleLineData = {}
    adjustedLinearData = {}
    for key, value in cutting_paths.items():
      pts_hole_dict[f'{key}'] = []  # do zachowania punktów konturu z gcode
      adjustedCircleLineData[f'{key}'] = []
      adjustedLinearData[f'{key}'] = []
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
        pts_hole_dict[f'{key}'].append(pts2)
      # adjust circle line'ow
      # try bo może byc kontur bez kół
      try:
        for c in circleLineData[f'{key}']:
          adjustedCircleLineData[f'{key}'].append(
            ((c[0] - min_x) * dx, (c[1] - min_y) * dy, c[2] * np.sqrt(dy * dx)))  # a,b,r
      except:
        pass
      # adjust linear punktów
      try:
        for l in linearPointsData[f'{key}']:
          adjustedLinearData[f'{key}'].append(((l[0] - min_x) * dx, (l[1] - min_y) * dy))
      except:
        pass
      # binaryzacja
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(gray, 30, 255, 0)
      images_dict[f"{key}"] = thresh
      pts_dict[f"{key}"] = pts
      # cv2.imshow('winname',thresh)
      # cv2.waitKey(0)
    return images_dict, pts_dict, sheet_size, pts_hole_dict, adjustedCircleLineData, adjustedLinearData
  else:
    return None, None, None, None, None

def singleGcodeElementCV2(cutting_path, circle_line_data, linear_points_data, bounding_box_size):
  """
        Funkcja tworząca przeskalowane dane o elemencie wzorcowym
    Args:
        cutting_path: dane z gcode jednego elementu o obrysie elementu
        circle_line_data: dane z gcode o wycinkach nieliniowych
        linear_points_data: dane z gcode o wycinkach liniowych
        bounding_box_size: rozmiar wykrytego elementu jako (width x height)
    Returns:
        gcode_data_packed: słownik danych elementu wzorcowego (do użycia z funkcją linesContourCompare)
    """
  # przeskalowanie
  main_contour, holes, = find_main_and_holes(cutting_path)
  max_x = max(main_contour, key=lambda item: item[0])[0]
  max_y = max(main_contour, key=lambda item: item[1])[1]
  min_x = min(main_contour, key=lambda item: item[0])[0]
  min_y = min(main_contour, key=lambda item: item[1])[1]
  width, height = bounding_box_size

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

  # przeskalowanie danych do ucietego obrazu
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

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, thresh = cv2.threshold(gray, 30, 255, 0)
  gcode_data_packed = {
    "image": thresh,
    "linearData": adjusted_linear_data,
    "circleData": adjusted_circle_data,
  }
  return gcode_data_packed

def gcodeToImageCV2(cutting_paths, scale=3):
  """
        Funkcja generująca obrazy elementów wzorcowych zgodne z formatem opencv
    Args:
        cutting_paths: słownik elementów wzorcowych
        scale: przeskalowanie obrazu względem oryginału
    Returns:
    """
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

def capture_median_frame(crop_values, camera_id):
  """
    Captures and processes frames from a camera to compute the median background subtractor.

    Args:
        crop_values (dict): Dictionary containing cropping values with keys 'top', 'bottom', 'left', 'right'.
        camera_id (int): The ID of the camera to capture frames from.

    Returns:
        cv2.BackgroundSubtractor: A background subtractor trained with the captured frames.
    """
  frames = 100
  BgrSubtractor = cv2.createBackgroundSubtractorMOG2(history=frames, varThreshold=50, detectShadows=True)
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
  cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
  cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
  cap.set(cv2.CAP_PROP_EXPOSURE, -4)

  fgmask_vec = []
  frames_vec = []
  while frames > 0:
    ret, frame = cap.read()
    frame = camera_calibration(frame)
    h, w, _ = frame.shape
    top_px = crop_values["top"]
    bottom_px = crop_values["bottom"]
    left_px = crop_values["left"]
    right_px = crop_values["right"]
    sliced_frame = frame[top_px:h - bottom_px, left_px:w - right_px]
    if not ret:
      break
    fg_mask = BgrSubtractor.apply(sliced_frame, learningRate=0.1)
    fgmask_vec.append(fg_mask)
    frames_vec.append(sliced_frame)
    frames -= 1
  cap.release()
  for i in range(len(fgmask_vec)):
    cv2.imwrite(f"Image preprocessing/Gcode to image conversion/camera_images_debug/fgmask_{i}.png", fgmask_vec[i])
    cv2.imwrite(f"Image preprocessing/Gcode to image conversion/camera_images_debug/frame_{i}.png", frames_vec[i])
  return BgrSubtractor

def cameraImage(BgrSubtractor, crop_values, camera_id):
  """
    Captures a single frame from the camera, processes it, and applies background subtraction.

    Args:
        BgrSubtractor (cv2.BackgroundSubtractor): Pre-trained background subtractor.
        crop_values (dict): Dictionary containing cropping values with keys 'top', 'bottom', 'left', 'right'.
        camera_id (int): The ID of the camera to capture frames from.

    Returns:
        tuple: Contains the cropped frame with foreground mask, its dimensions (width, height),
               and a list of intermediate processing steps.
    """
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
  cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
  cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
  cap.set(cv2.CAP_PROP_EXPOSURE, -4)
  ret, frame = cap.read()
  while not ret:
    ret, frame = cap.read()

  frame = camera_calibration(frame)
  h, w, _ = frame.shape
  top_px = crop_values["top"]
  bottom_px = crop_values["bottom"]
  left_px = crop_values["left"]
  right_px = crop_values["right"]
  sliced_frame = frame[top_px:h - bottom_px, left_px:w - right_px]

  fg_mask = BgrSubtractor.apply(sliced_frame, learningRate=0)
  _, fg_mask = cv2.threshold(fg_mask, 130, 255, cv2.THRESH_BINARY)
  kernel = np.ones((5, 5), np.uint8)
  cleaned_thresholded = fg_mask  # cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
  contours, _ = cv2.findContours(cleaned_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = max(contours, key=cv2.contourArea)
  x, y, w, h = cv2.boundingRect(contours)
  crop = cleaned_thresholded[y:y + h, x:x + w].copy()
  img_pack = [crop, cleaned_thresholded, sliced_frame, fg_mask]
  cap.release()
  # cv2.imshow("fg_mask",fg_mask)
  # cv2.imshow("cleaned_thresholded",cleaned_thresholded)
  # cv2.imshow("crop",crop)
  return crop, (w, h), img_pack

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

def camera_calibration(frame):
  """
    Calibrates the input frame using pre-saved camera parameters.

    Args:
        frame (numpy.ndarray): The input image/frame to be calibrated.

    Returns:
        numpy.ndarray: The calibrated image/frame.
    """
  # Wczytanie parametrów kamery z pliku
  loaded_mtx = np.loadtxt('settings/mtx_matrix.txt', delimiter=',')
  loaded_dist = np.loadtxt('settings/distortion_matrix.txt',
                           delimiter=',')
  loaded_newcameramtx = np.loadtxt('settings/new_camera_matrix.txt',
                                   delimiter=',')
  loaded_roi = np.loadtxt('settings/roi_matrix.txt', delimiter=',')

  # Kalibracja kamery
  frame = cv2.undistort(frame, loaded_mtx, loaded_dist, None, loaded_newcameramtx)
  x, y, w, h = map(int, loaded_roi)
  frame = frame[y:y + h, x:x + w]
  return frame

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

def linesContourCompare(imageB, gcode_data):
  """
        Compares image from camera to gcode image.
        Prints RMSE error
    Args:
        imageB (nd.array): gray and threshholded camera image

        gcode_data (dictionary): Object with linear and circular movements from gcode element
    """
  # try:
  img = gcode_data['image']
  contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contoursB, _ = cv2.findContours(imageB, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  ret = cv2.matchShapes(contours[0], contoursB[0], 1, 0.0)
  # if ret > 0.06:
  #     print(f'Odkształcenie, ret: {ret} \n')
  #     return False,0,ret,None,None
  gcodeLines = {
    "circle": gcode_data['circleData'],
    "linear": [],
  }
  new_imageB = cv2.flip(imageB, 1)
  # new_imageB = findRotation(img, imageB)
  contoursB, _ = cv2.findContours(new_imageB, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  imgCopy = img.copy()
  imgCopy = cv2.cvtColor(imgCopy,cv2.COLOR_GRAY2BGR)
  new_imageB = cv2.cvtColor(new_imageB,cv2.COLOR_GRAY2BGR)
  imgBCopy = imageB.copy()
  cv2.drawContours(new_imageB, contoursB, -1, (0, 0, 255), 1)
  # imgCopy_lines = cv2.cvtColor(imgCopy,cv2.COLOR_GRAY2BGR)
  # imgCopy_circ = imgCopy_lines.copy()
  for i in range(len(gcode_data['linearData']) + 1):  # tuple i lista tupli
    if i == 0:
      continue
    if i == len(gcode_data['linearData']):  # obsluga ostatni + pierwszy punkty linia łącząca
      x1, y1 = gcode_data['linearData'][i - 1]
      x2, y2 = gcode_data['linearData'][0]
      A, B, C = lineFromPoints(x1, y1, x2, y2)
      gcodeLines['linear'].append((A, B, C))
      cv2.line(imgCopy, (int(x1), int(y1)), (int(x2), int(y2)), randomColor(), 2)
      break

    x1, y1 = gcode_data['linearData'][i]
    x2, y2 = gcode_data['linearData'][i - 1]
    A, B, C = lineFromPoints(x1, y1, x2, y2)
    gcodeLines['linear'].append((A, B, C))
    cv2.line(imgCopy, (int(x1), int(y1)), (int(x2), int(y2)), randomColor(), 2)
  cntrErrors = []
  for i in range(len(contoursB)):
    for j in range(len(contoursB[i])):
      xCntr, yCntr = contoursB[i][j][0]  # we love numpy with this one
      # porównanie do lini prostej
      d_minimal = 1000
      for l in range(len(gcodeLines["linear"])):
        A, B, C = gcodeLines["linear"][l]  # (a,b) y=ax+b
        d = np.abs(A * xCntr + B * yCntr + C) / (np.sqrt(A ** 2 + B ** 2))
        if d < d_minimal:
          d_minimal = d

      # porównanie do kół
      for k in range(len(gcodeLines["circle"])):
        a, b, r = gcodeLines["circle"][k]
        cv2.circle(imgCopy, (int(a), int(b)), int(np.abs(r)), (255, 0, 255), 2)
        d_circ = np.abs(np.sqrt((xCntr - a) ** 2 + (yCntr - b) ** 2) - r)
        if d_circ < d_minimal:
          d_minimal = d_circ

      scale_px_to_mm = 0.6075949367088608  # stala wartosc obliczona recznie lol
      d_minimal = d_minimal * scale_px_to_mm
      cntrErrors.append(d_minimal)

  # Punkty, które nie przekraczają błędu 2:
  goodPoints = [e for e in cntrErrors if e <= 2]
  accuracy = 100 * (len(goodPoints) / len(cntrErrors))

  RMSE = np.sqrt(sum(e * e for e in cntrErrors) / len(cntrErrors))
  print("----------")

  # contourSum = 0
  # for contour in contoursB:
  #     contourSum += len(contour)
  # print("ilosc punktow: ",contourSum)
  print("RMSE:", RMSE)
  if accuracy < 99:
    print("Detal posiada błąd wycięcia")
    print(f'ret: {ret} \n')
    print('accuracy: ', accuracy)
    return False, RMSE, ret, imgCopy, new_imageB

  print("Detal poprawny")
  print(f'ret: {ret} \n')
  print('accuracy: ', accuracy)
  return True, RMSE, ret, imgCopy, new_imageB

  # except Exception as e:
  #     print("Podczas przetwarzania obrazu wystąpił błąd")
  #     print(e)
  #     return False,0, ret

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

def findRotation(imageA, imageB):
  """
    Finds the best rotation of imageB (0 to 360 degrees in steps of 0.5) that best matches imageA using template matching.

    Args:
        imageA: Target image (numpy array, single channel).
        imageB: Template image to be rotated (numpy array, single channel).

    Returns:
        best_rotated_imageB: imageB rotated to the angle that gave the best match.
    """
  max_match = -1
  best_angle = 0
  best_rotated_image = None

  for angle in np.arange(0, 360, 0.5):
    # Rotate imageB by the current angle
    rotated_template = rotate_image(imageB, angle)

    # If rotated_template is larger than imageA, resize it
    if (rotated_template.shape[0] > imageA.shape[0]) or (rotated_template.shape[1] > imageA.shape[1]):
      scale_factor = min(imageA.shape[0] / rotated_template.shape[0],
                         imageA.shape[1] / rotated_template.shape[1])
      rotated_template = cv2.resize(rotated_template, (0, 0), fx=scale_factor, fy=scale_factor)

    # Perform template matching
    result = cv2.matchTemplate(imageA, rotated_template, cv2.TM_CCOEFF_NORMED)

    # Check if we got a valid result
    if result is None or result.size == 0:
      continue

    # Get the maximum match value from the result
    _, max_val, _, _ = cv2.minMaxLoc(result)

    # Update best angle and max match if current is better
    if max_val > max_match:
      max_match = max_val
      best_angle = angle
      best_rotated_image = rotated_template.copy()

  print(f"Best rotation angle found: {best_angle} degrees with match: {max_match}")
  return best_rotated_image

def sheetRotationTranslation(bgr_subtractor, camera_id, crop_values, sheet_length_mm):
  """
    Determines the rotation and translation of a sheet based on its contours in the frame.

    Args:
        bgr_subtractor (cv2.BackgroundSubtractor): Pre-trained background subtractor.
        camera_id (int): The ID of the camera to capture frames from.
        crop_values (dict): Dictionary containing cropping values with keys 'top', 'bottom', 'left', 'right'.
        sheet_length_mm (float): The length of the sheet in millimeters for scaling.

    Returns:
        tuple: Contains the rotation angle in degrees, translation in millimeters (x, y),
               and additional processed data including key points and contours.
    """
  _, _, img_pack = cameraImage(bgr_subtractor, crop_values, camera_id)
  thresh = img_pack[1]
  org_img = img_pack[2]
  contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.imshow("thresh",img_pack[1])
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  contours = max(contours, key=cv2.contourArea)
  final_contours = contours
  if len(final_contours) > 4:
    eps = 0.05 * cv2.arcLength(final_contours, True)
    approx = cv2.approxPolyDP(final_contours, eps, True)
    final_contours = approx
  final_contours = final_contours.reshape(-1, 2)
  sorted_x = sorted(final_contours, key=lambda point: point[0], reverse=True)
  sorted_y = sorted(final_contours, key=lambda point: point[1], reverse=True)

  bottom_points = sorted_y[:2]
  bottom_points = sorted(bottom_points, key=lambda point: point[0])
  xl, yl = bottom_points[0]
  right_most = sorted_x[:2]
  right_most = sorted(right_most, key=lambda point: point[1])
  xt, yt = right_most[0]
  xb, yb = right_most[1]
  A, B, C = lineFromPoints(xt, yt, xb, yb)
  try:
    a = -A / B
    alpha = np.arctan(a)
    alpha = np.rad2deg(alpha)
  except:
    alpha = 0

  if alpha > 0:
    alpha = 90 - alpha
  else:
    alpha = -90 - alpha
  diff_x_px = REFPOINT[0] - xb
  diff_y_px = REFPOINT[1] - yb

  scalePxMm = sheet_length_mm / np.sqrt((xl - xb) ** 2 + (yl - yb) ** 2)  # 0.720607
  diff_x = diff_x_px * scalePxMm
  diff_y = diff_y_px * scalePxMm
  data_out = [(xl, yl), (xt, yt), (xb, yb), (A, B, C), img_pack, final_contours]
  return alpha, (diff_x, diff_y), data_out

def recalibratePoint(point, angle, translation):
  """
    Recalibrates a point based on a given rotation angle and translation vector.

    Args:
        point (tuple): The original point as (x, y).
        angle (float): Rotation angle in degrees.
        translation (tuple): Translation vector as (x, y).

    Returns:
        tuple: The recalibrated point as (x', y').
    """
  angle_rad = np.radians(angle)
  SE2_rotation = np.array([
    [np.cos(angle_rad), -np.sin(angle_rad)],
    [np.sin(angle_rad), np.cos(angle_rad)]
  ])
  point_np = np.array([point[0], point[1]])
  transformed_point = np.dot(SE2_rotation, point_np) + np.array(translation)
  return tuple(transformed_point)

def nothing(x):
  pass

def get_crop_values(camera_id):
  """
    Interactive function to determine cropping values for a camera feed.

    Args:
        camera_id (int): The ID of the camera to capture frames from.

    Returns:
        tuple: Contains the cropping values as a dictionary and the sliced frame.
    """
  # Otwórz dostęp do kamery
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
  cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
  cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
  # cap.set(cv2.CAP_PROP_EXPOSURE, -1)
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
    sliced_frame = frame[top_px:h - bottom_px, left_px:w - right_px]

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

def testAdditionalHole(imageB, gcode_data):
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

  return image_copy, linesContourCompare(image_copy, gcode_data)

def AutoAdditionalHoleTest():
  # Path to the NC file
  path = "NC_files/8.NC"

  # Directory to save images
  save_dir = "CV_program_photos/Additional_hole_test"
  os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

  # Load all G-code elements
  images, pts, sheet_size, pts_hole, circleLineData, linearData = allGcodeElementsCV2(
    sheet_path=path,
    arc_pts_len=300
  )

  # Process each image
  for key, value in images.items():
    gcode_data = {
      "image": value,
      "circleData": circleLineData[key],
      "linearData": linearData[key],
    }

    # Perform additional hole testing
    image_dziura, quality_values = testAdditionalHole(value, gcode_data)

    # Display images
    cv2.imshow("original", value)
    cv2.imshow("zmodyfikowane zdjecie", image_dziura)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save images
    original_image_path = os.path.join(save_dir, f"{key}_original.jpg")
    modified_image_path = os.path.join(save_dir, f"{key}_modified.jpg")

    cv2.imwrite(original_image_path, value)
    cv2.imwrite(modified_image_path, image_dziura)
    print("siema")

def readRobotCVJsonData(json_name):
  """
      Reads JSON file, shows element images, saves them, and generates a CSV file with specific data.
  Args:
      json_name: json file name path

  Returns:

  """
  dir = "H:"
  csv_file_path = f"{dir}/CV_program_photos/data_summary.csv"

  # Open CSV file for writing
  with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header row
    csv_writer.writerow(["Element", "RMSE", "Deformation", "Palletizing Angle"])

    with open(f'{json_name}', 'r') as f:
      data = json.load(f)

    for key, value in data.items():
      if key == "sheet":
        print(f'obrót: {value["rotation"]}')
        print(f'translacja: {value["translation"]}')
        continue

      buf_vec = {}
      for key2, value2 in value['bonusImages'].items():
        img_bytes = base64.b64decode(value2)
        img_real = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        buf_vec[key2] = img_real

      image_bytes2 = base64.b64decode(value['gcode_data']['image'])
      image_gcode = cv2.imdecode(np.frombuffer(image_bytes2, np.uint8), cv2.IMREAD_COLOR)
      image_bytes3 = base64.b64decode(value['object_image'])
      image_item = cv2.imdecode(np.frombuffer(image_bytes3, np.uint8), cv2.IMREAD_COLOR)

      # Extract and print element data
      print(f'Element: {key}')
      print(f'rmse : {value["RMSE"]}')
      print(f"deformation : {value['deformation']}")
      print(f'palletizing_angle : {value["palletizing_angle"]}')
      print("\n \n")

      # Write data to CSV
      csv_writer.writerow([key, value["RMSE"], value["deformation"], value["palletizing_angle"]])

      # Save images
      cv2.imwrite(
        f"{dir}/CV_program_photos/zdjecia_przebieg/{key}_object1.png",
        image_item)
      cv2.imwrite(
        f"{dir}/CV_program_photos/zdjecia_przebieg/{key}_gcode1.png",
        image_gcode)

      for key2, value2 in buf_vec.items():
        cv2.imwrite(
          f"{dir}/CV_program_photos/zdjecia_przebieg/{key}_{key2}.png",
          value2)

      # Process contours
      gcode_data = value['gcode_data']
      image_gcode = cv2.cvtColor(image_gcode, cv2.COLOR_BGR2GRAY)
      image_item = cv2.cvtColor(image_item, cv2.COLOR_BGR2GRAY)
      gcode_data['image'] = image_gcode
      correct, RMSE, RET, wzor, cam = linesContourCompare(image_item, gcode_data)
      cv2.imwrite(
        f"{dir}/CV_program_photos/zdjecia_przebieg/{key}_wzor.png",
        wzor)
      saved = cv2.imwrite(
        f"{dir}/CV_program_photos/zdjecia_przebieg/{key}_contours.png",
        cam)
      print(saved)

def readRobotSheetCVJSONData(json_file):
  """
        Reads JSON file, shows sheet images and saves them
    Args:
        json_name: json file name path

    Returns:

    """
  # Wczytaj dane z pliku JSON
  with open(json_file, 'r', encoding='utf8') as f:
    cv_data = json.load(f)

  # Wyodrębnij dane z JSONa
  sheet_data = cv_data['sheet']
  right_down_point = tuple(sheet_data['right_down_point'])
  right_up_point = tuple(sheet_data['right_up_point'])
  left_down_point = tuple(sheet_data['left_down_point'])
  bonus_images = sheet_data['bonusImages']

  # Punkt referencyjny

  # Odtwórz obrazy z Base64
  camera_image = cv2.imdecode(np.frombuffer(base64.b64decode(bonus_images['camera_image']), np.uint8),
                              cv2.IMREAD_COLOR)
  mog2_image = cv2.imdecode(np.frombuffer(base64.b64decode(bonus_images['MOG2_image']), np.uint8), cv2.IMREAD_COLOR)
  object_full_image = cv2.imdecode(np.frombuffer(base64.b64decode(bonus_images['object_full_image']), np.uint8),
                                   cv2.IMREAD_COLOR)

  # Zaznacz punkty na obrazie MOG2
  marked_image = mog2_image.copy()
  # Zaznaczenie punktów
  cv2.circle(marked_image, right_down_point, 5, (0, 0, 255), -1)  # Czerwony
  cv2.putText(marked_image, 'Right Down', (right_down_point[0] + 10, right_down_point[1]), cv2.FONT_HERSHEY_SIMPLEX,
              0.5, (0, 0, 255), 1)

  cv2.circle(marked_image, right_up_point, 5, (0, 255, 0), -1)  # Zielony
  cv2.putText(marked_image, 'Right Up', (right_up_point[0] + 10, right_up_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
              (0, 255, 0), 1)

  cv2.circle(marked_image, left_down_point, 5, (255, 0, 0), -1)  # Niebieski
  cv2.putText(marked_image, 'Left Down', (left_down_point[0] + 10, left_down_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
              (255, 0, 0), 1)

  # Zaznaczenie punktu referencyjnego
  cv2.circle(marked_image, REFPOINT, 5, (0, 255, 255), -1)  # Żółty
  cv2.putText(marked_image, 'REFPOINT', (REFPOINT[0] + 10, REFPOINT[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),
              1)

  # Rysowanie linii od REFPOINT do Right Down
  cv2.line(marked_image, REFPOINT, right_down_point, (255, 255, 0), 2)  # Jasnoniebieska linia

  # Wyświetl obrazy
  cv2.imshow("Camera Image", camera_image)
  cv2.imshow("MOG2 Image with Points", marked_image)
  cv2.imshow("Object Full Image", object_full_image)

  # Poczekaj na klawisz ESC, aby zamknąć
  print("Naciśnij ESC, aby zamknąć...")
  while True:
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC
      cv2.imwrite("CV_program_photos/zdjecia_przebieg/sheet_camera.png", camera_image)
      cv2.imwrite("CV_program_photos/zdjecia_przebieg/sheet_points.png", marked_image)
      cv2.imwrite("CV_program_photos/zdjecia_przebieg/sheet_full_image.png", object_full_image)
      break

  cv2.destroyAllWindows()


def visualize_cutting_paths(file_path, x_max=500, y_max=1000):
  """
    # Funkcja do wizualizacji ścieżek cięcia z pliku NC
    Funkcja plik .nc z kodem G-kodu i zwraca obraz z wizualizacją ścieżek cięcia.
    Funkcja wykorzystuje bibliotekę matplotlib, numpy i re.
    Funkcja zwraca ścieżki cięcia, minimalne i maksymalne współrzędne X i Y.
    Funkcja przyjmuje ścieżkę do pliku .nc z kodem G-kodu oraz maksymalne współrzędne X i Y.
    Domyślnie maksymalne współrzędne X i Y wynoszą odpowiednio 500 i 1000.
    # Założenia funkcji:
    - Włączenie lasera oznaczone jest jako M10, a wyłączenie jako M11.
    - Ścieżki cięcia są zapisane jako G01X...Y... lub G02X...Y...I...J... lub G03X...Y...I...J...
    - Współrzędne X i Y są zapisane jako liczby zmiennoprzecinkowe.
    - Ruch okrężny w prawo jest zapisany jako G02, a ruch okrężny w lewo jako G03.
    - Współrzędne I i J określają środek okręgu, a współrzędne X i Y określają punkt końcowy.
    - Współrzędne I i J są zapisane jako liczby zmiennoprzecinkowe.
    - Wszystkie współrzędne są mniejsze niż podane maksymalne współrzędne X i Y.
    # Przykład użycia:
    cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths("./Przygotowanie obrazu/gcode2image/NC_files/Arkusz-6001.nc")
    """
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
  for line in file_content:
    element_match = pattern_element_name.search(line)
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
            angles = np.linspace(start_angle, end_angle, num=50)  # Generowanie punktów łuku (50 punktów)
            arc_points = [(center_x + radius * np.cos(a), center_y + radius * np.sin(a)) for a in
                          angles]  # Obliczenie punktów łuku
            current_path.extend(arc_points)  # Dodanie punktów łuku do ścieżki
            current_position = (x, y)  # Aktualizacja pozycji
  # Jeśli laser został wyłączony po ostatnim cięciu, dodajemy ścieżkę do listy
  if current_path:
    if current_element_name not in elements:
      elements[current_element_name] = []
    elements[current_element_name].append(current_path)
  # Rozmiar arkusza
  x_min, y_min = 0, 0
  return elements, x_min, x_max, y_min, y_max


def calculate_centroid(poly):
  """
    #  Funkcja do obliczania środka ciężkości i powierzchni wielokąta
    Funkcja przyjmuje listę punktów definiujących wierzchołki wielokąta (jako zestawy punktów (x, y)) i zwraca centroid oraz powierzchnię tego wielokąta. Do obliczeń wykorzystywana jest biblioteka numpy, która umożliwia operacje na tablicach. Centroid zwracany jest jako zestaw punktów (centroid_x, centroid_y), a powierzchnia jako pojedyncza wartość.
    # Założenia funkcji:
    - Powierzchnia wielokąta obliczana jest przy użyciu wzoru polegającego na wykorzystaniu iloczynu skalarnego
      oraz funkcji przesunięcia indeksu elementów tablicy (np.roll).
    - Centroid obliczany jest jako średnia ważona współrzędnych punktów, z wagą proporcjonalną do struktury wielokąta.
    - Wartości centroidu są zwracane jako wartości bezwzględne, co jest specyficznym zachowaniem tej funkcji.
    - Powierzchnia zawsze jest zwracana jako wartość dodatnia.
    # Przykład użycia funkcji:
    centroid, area = calculate_centroid(main_contour)
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


def find_main_and_holes(contours):
  """
    #  Funkcja do znalezienia głównego konturu i otworów
    Funkcja przyjmuje listę konturów i zwraca główny kontur i otwory.
    # Założenia funkcji:
    - Główny kontur jest konturem z największym polem powierzchni.
    - Otwory są konturami z mniejszym polem powierzchni.
    # Przykład użycia:
    main_contour, holes = find_main_and_holes(contours)
    """
  areas = [(calculate_centroid(contour)[1], contour) for contour in contours]
  areas.sort(reverse=True, key=lambda x: x[0])
  main_contour = areas[0][1]
  holes = [area[1] for area in areas[1:]]
  return main_contour, holes


def point_in_polygon(point, polygon):
  """
    #  Funkcja do sprawdzenia, czy punkt znajduje się w wielokącie
    Funkcja przyjmuje punkt i wielokąt i zwraca True, jeśli punkt znajduje się wewnątrz wielokąta.
    Źródło: https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
    # Przykład użycia:
    point_in_polygon(adjusted_centroid, main_contour)
    """
  num_vertices = len(polygon)
  x, y = point[0], point[1]
  inside = False
  p1 = polygon[0]
  for i in range(1, num_vertices + 1):
    p2 = polygon[i % num_vertices]
    if y > min(p1[1], p2[1]):
      if y <= max(p1[1], p2[1]):
        if x <= max(p1[0], p2[0]):
          x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
          if p1[0] == p2[0] or x <= x_intersection:
            inside = not inside
    p1 = p2
  return inside


def is_valid_circle(center, radius, shape, holes):
  """
    # Funkcja do sprawdzenia, czy okrąg mieści się w figurze
    Funkcja przyjmuje środek okręgu, promień, kształt i otwory.
    Funkcja zwraca True, jeśli okrąg mieści się w figurze bez nakładania na otwory.
    # Przykład użycia:
    is_valid_circle(center, radius, shape, holes)
    """
  circle = Point(center).buffer(radius)
  return shape.contains(circle) and all(not hole.intersects(circle) for hole in holes)


def detail_mass(shape, holes, material_density=0.0027, material_thickness=1.5):
  """
    #  Funkcja do sprawdzenia jaką mase ma detal
    Funkcja przyjmuje kształt, otwory, gęstość materiału i grubość materiału.
    Funkcja zwraca masę detalu na podstawie kształtu i otworów.
    # Przykład użycia:
    detail_mass(shape, holes, material_density=0.0027, material_thickness=1.5)
    """
  total_area = shape.area - sum(hole.area for hole in holes)
  total_volume = total_area * material_thickness
  total_mass = total_volume * material_density
  return total_mass


if __name__ == "__main__":
  readRobotCVJsonData('../../cv_data_blacha8_sheetTest_18-12_USZKODZENIA.json')
  # AutoAdditionalHoleTest()
  # generate_sheet_json()



