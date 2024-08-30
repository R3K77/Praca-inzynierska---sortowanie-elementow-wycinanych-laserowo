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
        for key, value in cutting_paths.items():
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
                pts2 = pts2.reshape((-1, 1, 2))
                cv2.fillPoly(img, [pts2], color=(0,0,0))

            #binaryzacja
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 30, 255, 0)
            images_dict[f"{key}"] = thresh
            pts_dict[f"{key}"] = pts
            # cv2.imshow('winname',thresh)
            # cv2.waitKey(0)
        return images_dict, pts_dict, sheet_size
    else:
        return None, None, None


if __name__ == "__main__":
    #Test czy spakowana funkcja działa
    images,pts,sheet_size = single_gcode_elements_cv2('../../../../Gcode to image conversion/NC_files/arkusz-2001.nc',500,500)
    print(f'sheet size: {sheet_size}')
    for key, value in images.items():
        # key - nazwa ze slownika, taka sama jak blacha + threshold
        # value - dane obrazu
        buf = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        points = pts[f'{key}'].reshape((-1,1,2))
        # print(f'points: {points} \n')
        # for point in points:
        #     # print(f'single point: {point}')
        #     cv2.circle(buf, (point[0],point[1]),10 ,(0,0,255),1)
        #
        cv2.polylines(buf,[points],True,(0,0,255),thickness = 3)
        img = cv2.resize(buf, (value.shape[0],value.shape[1]))
        cv2.imshow(key, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



