import numpy as np
from gcode_analize import visualize_cutting_paths, find_main_and_holes
import cv2


# Funkcja generuje obraz cv2 wyciętych elementów:
# in - sheet_path scieżka do .nc; output_res_xy - rozdzielczość wyjściowa obrazu; arc_pts_len - liczba punktów z której generowane są łuki/okręgi.
# out - images_dict elementy wycięte cv2; pts_dict punkty konturu; sheet_size gcode'owy wymiar blachy.


def single_gcode_elements_cv2(sheet_path, output_res_x = 500, output_res_y = 500, arc_pts_len = 300):
    cutting_paths, x_min, x_max, y_min, y_max, sheet_size_line = visualize_cutting_paths(sheet_path, arc_pts_len= arc_pts_len)
    if sheet_size_line is not None:
        #Rozkodowanie linii na wymiary
        split = sheet_size_line.split()
        sheet_size = (split[1],split[2])
        images_dict = {}
        pts_dict = {}
        pts_hole_dict = {}
        for key, value in cutting_paths.items():
            pts_hole_dict[f'{key}'] = [] # do zachowania punktów konturu z gcode

            main_contour, holes = find_main_and_holes(value)
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

            # do narysowania dziur
            adjusted_holes = [[(int((x - min_x)*dx), int((y - min_y)*dy)) for x, y in hole] for hole in holes]
            for hole in adjusted_holes:
                pts2 = np.array(hole, np.int32)
                pts2_resh = pts2.reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts2_resh], color=(0,0,0))
                pts_hole_dict[f'{key}'].append(pts2)



            #binaryzacja
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 30, 255, 0)
            images_dict[f"{key}"] = thresh
            pts_dict[f"{key}"] = pts
            # cv2.imshow('winname',thresh)
            # cv2.waitKey(0)
        return images_dict, pts_dict, sheet_size, pts_hole_dict
    else:
        return None, None, None

def imageBInfoExtraction(imageB, pts):

    contours, _ = cv2.findContours(imageB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourImage = cv2.cvtColor(imageB, cv2.COLOR_GRAY2BGR)
    polyImage = contourImage.copy()
    hullImage = contourImage.copy()
    print(f'pts shape[0]: {pts.shape[0]}')
    sciany = pts.shape[0]


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


    return contourImage, polyImage, hullImage




if __name__ == "__main__":
    #Test czy spakowana funkcja działa
    images,pts,sheet_size, pts_hole = single_gcode_elements_cv2(
    '../../../../Gcode to image conversion/NC_files/arkusz-2001.nc',
    500,500,
    100)
    for key, value in images.items():
        #Porównanie contours i approx poly w znajdowaniu punktów.
        contourImage, polyImage, hullImage = imageBInfoExtraction(value, pts[key])
        buf = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        points = pts[f'{key}'] #.reshape((-1,1,2)) <- dodać jak używane jest polylines
        points_hole = pts_hole[f'{key}']

        # Wizualizacja punktów głównego konturu
        for point in points:
            cv2.circle(buf, (point[0],point[1]),3 ,(0,0,255),3)
        # Wizualizacja punktów wycięć
        for arr in points_hole:
            for point in arr:
                cv2.circle(buf, (point[0],point[1]),2,(255, 0, 255), 2)

        # cv2.polylines(buf,[points],True,(0,0,255),thickness = 3)


        cv2.imshow("Gcode image + punkty Gcode", buf)
        cv2.imshow("imageB Contour", contourImage) # niebieski - obrys, czerwony - punkty konturu
        cv2.imshow("imageB Poly", polyImage) # morski - linie konturu, czerwony - uproszczone punkty konturu
        cv2.imshow("imageB Hull", hullImage)
        # gigazdjecie
        imgTop = cv2.hconcat([buf, contourImage])
        imgBottom = cv2.hconcat([polyImage, hullImage])
        imgMerge = cv2.vconcat([imgTop, imgBottom])
        cv2.imshow("BIG MERGE", imgMerge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



