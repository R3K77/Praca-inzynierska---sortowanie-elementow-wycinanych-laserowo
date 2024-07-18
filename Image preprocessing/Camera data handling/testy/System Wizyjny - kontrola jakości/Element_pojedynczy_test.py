import numpy as np
from gcode_analize import visualize_cutting_paths, find_main_and_holes
import cv2


cutting_paths, x_min, x_max, y_min, y_max = visualize_cutting_paths('../../../Gcode to image conversion/NC_files/arkusz-1001.nc')


for key, value in cutting_paths.items():
    # Tutaj będą rysowane kontury elementu

    main_contour, holes = find_main_and_holes(value)
    max_x = max(main_contour, key=lambda item: item[0])[0]
    max_y = max(main_contour, key=lambda item: item[1])[1]
    min_x = min(main_contour, key=lambda item: item[0])[0]
    min_y = min(main_contour, key=lambda item: item[1])[1]
    print(f'Main contour: {main_contour}')
    print(f'Max x: {max_x} Max y: {max_y} Min x: {min_x} Min y: {min_y}')
    img = np.zeros((int(max_y - min_y), int(max_x - min_x), 3), dtype=np.uint8)
    adjusted_main = [(int(x-min_x), int(y-min_y)) for x, y in main_contour]
    pts = np.array(adjusted_main, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color=(255,255,255))
    # Rysowanie otworów elementu
    adjusted_holes = [[(int(x - min_x), int(y - min_y)) for x, y in hole] for hole in holes]
    for hole in adjusted_holes:
        pts = np.array(hole, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=(0,0,0))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 30, 255, 0)
    thresh = cv2.resize(thresh, (thresh.shape[1] * 4, thresh.shape[0] * 4))
    cv2.imshow('image', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

