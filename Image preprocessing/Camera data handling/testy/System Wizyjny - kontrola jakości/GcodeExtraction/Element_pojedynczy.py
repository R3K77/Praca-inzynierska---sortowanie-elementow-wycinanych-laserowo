import numpy as np
from gcode_analize import visualize_cutting_paths, find_main_and_holes
import cv2


# Funkcja generuje obraz cv2 wyciętych elementów:
# in - sheet_path scieżka do .nc; output_res_xy - rozdzielczość wyjściowa obrazu; arc_pts_len - liczba punktów z której generowane są łuki/okręgi.
# out - images_dict elementy wycięte cv2; pts_dict punkty konturu; sheet_size gcode'owy wymiar blachy.


def single_gcode_elements_cv2(sheet_path, output_res_x = 500, output_res_y = 500, arc_pts_len = 300):
    cutting_paths, x_min, x_max, y_min, y_max, sheet_size_line, isPathCircle = visualize_cutting_paths(sheet_path, arc_pts_len= arc_pts_len)
    if sheet_size_line is not None:
        #Rozkodowanie linii na wymiary
        split = sheet_size_line.split()
        sheet_size = (split[1],split[2])
        images_dict = {}
        pts_dict = {}
        pts_isCircle_dict = {}
        pts_hole_dict = {}
        pts_holeIsCircle_dict = {}
        for key, value in cutting_paths.items():
            pts_hole_dict[f'{key}'] = [] # do zachowania punktów konturu z gcode
            main_contour,main_bool, holes, holes_bool = find_main_and_holes(value, isPathCircle[f'{key}'])

            # print('\n ------------------------------------------------------------------')
            # print(f'\n{key} : {value}')
            #min maxy do przeskalowania obrazu
            max_x = max(main_contour, key=lambda item: item[0])[0]
            max_y = max(main_contour, key=lambda item: item[1])[1]
            min_x = min(main_contour, key=lambda item: item[0])[0]
            min_y = min(main_contour, key=lambda item: item[1])[1]

            # Przeskalowanie punktów konturu do zfitowania obrazu 500x500
            dx = output_res_x/(max_x-min_x)
            dy = output_res_y/(max_y-min_y)
            img = np.zeros((output_res_x, output_res_y, 3), dtype=np.uint8)
            adjusted_main = [(int((x-min_x)*dx), int((y-min_y)*dy)) for x, y in main_contour]
            pts = np.array(adjusted_main, np.int32)
            pts_reshape = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=(255,255,255))
            pts_isCircle_dict[f'{key}'] = []
            pts_isCircle_dict[f'{key}'].append(main_bool)

            # do narysowania dziur
            adjusted_holes = [[(int((x - min_x)*dx), int((y - min_y)*dy)) for x, y in hole] for hole in holes]
            for hole in adjusted_holes:
                pts2 = np.array(hole, np.int32)
                pts2_resh = pts2.reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts2_resh], color=(0,0,0))
                pts_hole_dict[f'{key}'].append(pts2)
            pts_holeIsCircle_dict[f'{key}'] = []
            pts_holeIsCircle_dict[f'{key}'].append(holes_bool)


            #binaryzacja
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 30, 255, 0)
            images_dict[f"{key}"] = thresh
            pts_dict[f"{key}"] = pts
            # cv2.imshow('winname',thresh)
            # cv2.waitKey(0)
        return images_dict, pts_dict, sheet_size, pts_hole_dict, pts_isCircle_dict, pts_holeIsCircle_dict
    else:
        return None, None, None

def imageBInfoExtraction(imageB, pts):

    contours, _ = cv2.findContours(imageB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourImage = cv2.cvtColor(imageB, cv2.COLOR_GRAY2BGR)
    polyImage = contourImage.copy()
    hullImage = contourImage.copy()
    print(f'pts shape[0]: {pts.shape[0]}')
    sciany = pts.shape[0]

    #todo dodać punkty harrisa cv2 do rogów
    for contour in contours:
        arc = 0.015 * cv2.arcLength(contour, True)
        poly = cv2.approxPolyDP(contour, arc, -1, True)


        cv2.drawContours(contourImage, [contour], -1, (255, 100, 0), 2)
        for point in contour:
            cv2.circle(contourImage, (point[0][0], point[0][1]), 3, (0, 0, 255), 2)

        cv2.drawContours(polyImage, [poly], -1, (155, 155, 0), 2)
        for point in poly:
            cv2.circle(polyImage, (point[0][0],point[0][1]), 3, (0, 0, 255), 2)

        cv2.drawContours(hullImage, cv2.convexHull(contour),-1, (0,0, 255), 10)

        # punkty haris
        # ImageB = buf.copy()
        # f32ImageB = np.float32(value)
        # dst = cv2.cornerHarris(f32ImageB, 2, 3, 0.04)
        # dst = cv2.dilate(dst, None)
        # buf[dst > 0.01 * dst.max()] = [0, 0, 255]
        # #
        # cv2.imshow('dst', buf)

    return contourImage, polyImage, hullImage

#TODO funkcje tworzące proste oraz funkcje nieliniowe wraz z porównaniem.
#Jest Dobrze

#TODO po funkcjach wyzej, zrobić testy porównania :/
#Ogólnie bardzo możliwe, że będe musiał przenieść coś z parametrów na zmienną zewnetrzną,
#W pętli rzeczywistej, będę musiał dynamicznie dobierać rozdzielczość obrazu, więc przydałoby się to cacko zmieniać jakoś.


if __name__ == "__main__":
    # Test czy spakowana funkcja działa
    images, pts, sheet_size, pts_hole, pts_isCircle_dict, pts_holeIsCircle_dict = single_gcode_elements_cv2(
        '../../../../Gcode to image conversion/NC_files/arkusz-2001.nc',
        600, 600,
        200)
    for key, value in images.items():
        # Porównanie contours i approx poly w znajdowaniu punktów.
        contourImage, polyImage, hullImage = imageBInfoExtraction(value, pts[key])
        # wyciąganie ze słownika
        pts_isCircle = pts_isCircle_dict[f'{key}']
        pts_holeIsCircle = pts_holeIsCircle_dict[f'{key}']
        points = pts[f'{key}']
        points_hole = pts_hole[f'{key}']

        buf = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)


        #========== COLOR SCHEME PUNKTOW =============
        #czerwony - punkty proste głównego obrysu
        #żółty - punkty koliste głównego obrysu
        #różowy - punkty proste otworu
        #zielony - punkty koliste otworu

        # Wizualizacja punktów głównego konturu + punktow kolistych
        for i in range(len(points)):
            if pts_isCircle[0][i]:
                cv2.circle(buf, (points[i][0],points[i][1]),3 ,(0,255,255),3)
            else:
                cv2.circle(buf, (points[i][0],points[i][1]),3 ,(0,0,255),3)

        # # Wizualizacja punktów wycięć + punktow kolistych
        for j in range(len(points_hole)):
            for i in range(len(points_hole[j])):
                if pts_holeIsCircle[0][j][i]:
                    cv2.circle(buf, (points_hole[j][i][0],points_hole[j][i][1]),2,(0, 255, 0), 2)
                else:
                    cv2.circle(buf, (points_hole[j][i][0], points_hole[j][i][1]), 2, (255, 0, 255), 2)

        # cv2.polylines(buf,[points],True,(0,0,255),thickness = 3)

        cv2.imshow("Gcode image + punkty Gcode", buf)
        # cv2.imshow("imageB Contour", contourImage) # niebieski - obrys, czerwony - punkty konturu
        # cv2.imshow("imageB Poly", polyImage) # morski - linie konturu, czerwony - uproszczone punkty konturu
        # cv2.imshow("imageB Hull", hullImage)
        # gigazdjecie
        # imgTop = cv2.hconcat([buf, contourImage])
        # imgBottom = cv2.hconcat([polyImage, hullImage])
        # imgMerge = cv2.vconcat([imgTop, imgBottom])
        # cv2.imshow("BIG MERGE", imgMerge)

        cv2.waitKey(0)
        cv2.destroyAllWindows()



