import math
import os
from collections import defaultdict
import numpy as np
import cv2
import random
from gcode_analize import visualize_cutting_paths_extended, find_main_and_holes


# TODO Inzynierka:
# - quality control shape comparison DONE
# - porównywanie linii obrazu prostego FIXED poprzez korzystanie tylko z tego wyżej XD
# - perspektive transform dla prostego kształtu, badanie bledu w zaleznosci od obrotu elementu względem płaszczyzny obrazu DONE
# - zrobić zly obraz do porównania żeby było zle rmse DONE
# - SIFT, rotacja elementow przy odkładaniu DONE
# - Rotacja translacja blachy DONE


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


# przebudowa funkcji do dynamicznej rozdzielczości pod kamere DONE
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
                    adjustedLinearData[f'{key}'].append(((l[0] - min_x)*dx,(l[1] - min_y)*dy,l[2]))
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
                x1, y1,_ = gcode_data['linearData'][i-1]
                x2, y2,_ = gcode_data['linearData'][0]
                cv2.line(imgCopy_lines, (int(x1), int(y1)), (int(x2), int(y2)), randomColor(), 3)
                A, B, C = lineFromPoints(x1, y1, x2, y2)
                gcodeLines['linear'].append((A, B, C))
                break

            x1,y1,_ = gcode_data['linearData'][i]
            x2,y2,_ = gcode_data['linearData'][i-1]
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

        certainErrors = [e for e in cntrErrors if e > 1]
        accuracy = 100 * (len(certainErrors)/len(cntrErrors))
        RMSE = np.sqrt(sum(e*e for e in cntrErrors)/len(cntrErrors))


        print("----------")
        contourSum = 0
        for contour in contoursB:
            contourSum += len(contour)
        print("ilosc punktow: ",contourSum)
        print("Accuracy: ", accuracy)
        print("RMSE: ",RMSE)
        if accuracy < 99.5:
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
    A,B,C = lineFromPoints(xt,yt,xb,yb)
    a = -A/B
    alpha = np.arctan(a)
    alpha = np.rad2deg(alpha)
    if alpha > 0:
        alpha = 90 - alpha
    else:
        alpha = -90-alpha
    diff_x = abs(REFPOINT[0] - xb)
    diff_y = abs(REFPOINT[1] - yb)
    print(f'rotacja: {alpha}, translation: {diff_x,diff_y}')

    cv2.circle(org_img, REFPOINT,12,(0,0,255), -1)
    cv2.line(org_img, (xt, yt), (xb, yb), (0, 0, 255), 6)
    cv2.circle(org_img, (xt,yt), 10,(0,255,255), -1)
    cv2.circle(org_img,(xb,yb), 10,(255,255,0),-1)
    cv2.imwrite('resized.jpg',org_img)
    cv2.imshow('resized',org_img)
    cv2.imwrite('resized_thresh.jpg',thresh)
    cv2.imshow('resized_thresh',thresh)
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
    contours,_ = cv2.findContours(value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    value = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    cntrs_img = value.copy()
    cv2.drawContours(cntrs_img,contours,-1,(0,0,255), 3)
    value_with_border = cv2.copyMakeBorder(value, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_lines_with_border = cv2.copyMakeBorder(img_lines, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_circles_with_border = cv2.copyMakeBorder(img_circles, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cntrs_img_with_border = cv2.copyMakeBorder(cntrs_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # img = cv2.hconcat([value_with_border, img_lines_with_border])
    # img2 = cv2.hconcat([img, img_circles_with_border])
    cv2.imwrite(f"{key}.jpg", value_with_border)
    cv2.imwrite(f"{key}_lines.jpg",img_lines_with_border)
    cv2.imwrite(f"{key}_circles.jpg",img_circles_with_border)
    cv2.imwrite(f"{key}_contours.jpg",cntrs_img_with_border)
    # cv2.imshow("Gcode image + punkty Gcode", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def testTilt(imageB,gcode_data,key):
    """
    Quality control vision system test, simulates tilted element
    Args:
        imageB:
        gcode_data:
    Returns:
    """
    imageCopy = imageB.copy()
    height_new, width_new = imageB.shape[:2]
    tranform_value = int(width_new*0.08)
    src_points_new = np.float32([[0, 0], [width_new, 0], [width_new, height_new], [0, height_new]])
    dst_points_new = np.float32([[tranform_value, 0], [width_new - tranform_value, 0], [width_new, height_new], [0, height_new]])
    matrix_new = cv2.getPerspectiveTransform(src_points_new, dst_points_new)
    transformed_image_new = cv2.warpPerspective(imageB, matrix_new, (width_new, height_new))

    copy_borders = cv2.copyMakeBorder(imageCopy, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    transformed_borders = cv2.copyMakeBorder(transformed_image_new, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # img = cv2.hconcat([copy_borders,transformed_borders])
    cv2.imwrite(f"{key}.jpg",copy_borders)
    cv2.imwrite(f"{key}_tilt.jpg",transformed_borders)
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

def singleGcodeTest():
    (cutting_paths, _, _, _, _, sheet_size_line, circleLineData,
     linearPointsData) = visualize_cutting_paths_extended('../../../Gcode to image conversion/NC_files/8.NC')
    median_background_frame = capture_median_frame()
    cv2.imshow("bgr",median_background_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for key, value in cutting_paths.items():
        camera_image, bounding_box, img_pack = cameraImage(median_background_frame)
        gcode_data = singleGcodeElementCV2(cutting_paths[key],circleLineData[key],linearPointsData[key],bounding_box)
        is_ok,_ = linesContourCompare(camera_image,gcode_data)
        cv2.imshow(key + "camera", camera_image)
        cv2.imshow(key, gcode_data['image'])
        for i in range(len(img_pack)):
            cv2.imshow(f'{i}',img_pack[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def SheetInfoTest():
    substractor = capture_median_frame()
    print('background image zrobiony')
    alpha, translation = sheetRotationTranslation(substractor)
    print(f'rotacja: {alpha}, translation: {translation}')

def draw_circle_on_click():
    # Zmienna do przechowywania współrzędnych kliknięcia
    click_coordinates = []

    # Funkcja obsługująca kliknięcie myszą
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Gdy klikniemy lewym przyciskiem myszy, dodaj współrzędne
            click_coordinates.append((x, y))
            print(f"Kliknięto na koordynatach: ({x}, {y})")

    # Uruchomienie kamery
    cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # Ustawienie funkcji callback dla okna z obrazem
    cv2.namedWindow('Kamera')
    cv2.setMouseCallback('Kamera', mouse_callback)

    while True:
        # Odczyt obrazu z kamery
        ret, frame = cap.read()
        frame = camera_calibration(frame)
        # Rysowanie kółek w miejscach kliknięcia
        for coord in click_coordinates:
            cv2.circle(frame, coord, 1, (0, 0, 255), 2)  # Zielone kółko z obwódką

        # Wyświetlanie obrazu z kamery
        cv2.imshow('Kamera', frame)

        # Warunek wyjścia (klawisz 'q' zamyka aplikację)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zwolnienie zasobów
    cap.release()
    cv2.destroyAllWindows()

def AutoAdditionalHoleTest():
    path = "../../../Gcode to image conversion/NC_files/8.NC"
    images, pts, sheet_size, pts_hole, circleLineData, linearData = allGcodeElementsCV2(
        sheet_path=path,
        arc_pts_len=300)
    for key,value in images.items():
        gcode_data = {
            "image": images[key],
            "circleData": circleLineData[key],
            "linearData": linearData[key],
        }
        image_dziura, quality_values =  testAdditionalHole(value,gcode_data)
        print("siema")

if __name__ == "__main__":
    # draw_circle_on_click()
    # SheetInfoTest()
    # # singleGcodeTest()
    # Test czy spakowana funkcja działa
    AutoAdditionalHoleTest()
    # paths = [
    #     "../../../Gcode to image conversion/NC_files/1.NC",
    #     "../../../Gcode to image conversion/NC_files/2.NC",
    #     "../../../Gcode to image conversion/NC_files/3_fixed.NC",
    #     "../../../Gcode to image conversion/NC_files/4.NC",
    #     "../../../Gcode to image conversion/NC_files/5.NC",
    #     "../../../Gcode to image conversion/NC_files/6.NC",
    #     "../../../Gcode to image conversion/NC_files/7.NC",
    #     "../../../Gcode to image conversion/NC_files/8.NC",
    # ]
    # for path in paths:
    #     # os.chdir(r'C:\Users\Rafał\Documents\GitHub\Projekt-Przejsciowy---sortowanie-elementow-wycinanych-laserowo\Image preprocessing\Camera data handling\testy\System Wizyjny - kontrola jakości\GcodeExtraction')
    #     images, pts, sheet_size, pts_hole, circleLineData, linearData = allGcodeElementsCV2(
    #         sheet_path=path,
    #         arc_pts_len=300)
    #     # os.chdir(
    #     #     r'C:\Users\Rafał\Documents\GitHub\Projekt-Przejsciowy---sortowanie-elementow-wycinanych-laserowo\Image preprocessing\Camera data handling\testy\System Wizyjny - kontrola jakości\GcodeExtraction\ZdjeciaElementy')
    #     rotations = elementRotationByTemplateMatching(images)
    #     print(rotations)
    #     if path == "../../../Gcode to image conversion/NC_files/8.NC":
    #         for key,value in images.items():
    #             cv2.imshow(key,value)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows
        # for key, value in images.items():
        #     try:
        #         linData = linearData[f'{key}']
        #     except KeyError:
        #         linData = []
        #     try:
        #         circData = circleLineData[f'{key}']
        #     except KeyError:
        #         circData = []
        #     gcode_data_packed = {
        #         "linearData": linData,
        #         "circleData": circData,
        #         "image": value,
        #     }
        #     testAlgorithmFunction(key,value,pts,pts_hole,linearData,circleLineData)
        #     testTilt(value,gcode_data_packed,key) # podmienic funkcje w zaleznosci od testu
        #     img = cv2.hconcat([value,img_transformed])
        #     cv2.imshow("obraz",img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()















