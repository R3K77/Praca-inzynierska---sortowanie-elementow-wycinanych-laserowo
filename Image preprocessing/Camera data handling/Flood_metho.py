import cv2
import numpy as np


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

# Na podstawie problemu z odbijaniem światła,
# przedstawionego na: https://stackoverflow.com/questions/67099645/contour-detection-reduce-glare-on-image-opencv-python
# Funkcja służy do wykrycia największego elementu na obrazie.
# Warunek konieczny działania tej funkcji to jednolite tło, którego kolor nie jest podobny do koloru obiektu.
# (np. blacha na jasnoszarym stole). Z racji na statyczne położenie kamery, można dodać

# TODO - ponownie przetestować po update
# W przypadku kiedy nie działa najlepiej, dodać WarpPerspective, przed flood aby zmniejszyć obszar roboczy systemu wizyjnego.
def Workspace_detection(frame):
    lowerDiff = [2, 2, 2, 5]
    upperDiff = [2, 2, 2, 5]
    birdEye = np.maximum(frame, 10)
    foreground = birdEye.copy()
    seed = (10, 10)
    cv2.floodFill(foreground, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=lowerDiff, upDiff=upperDiff)

    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cntrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Tu można zamienić na simple, jeżeli ma być wykrywany prostokątny element

    max_area = 0
    max_area_index = 0
    for i, cntr in enumerate(cntrs):
        area = cv2.contourArea(cntr)
        if area > max_area:
            max_area = area
            max_area_index = i


    # kąt obrotu
    x, y, w, h = cv2.boundingRect(cntrs[max_area_index])
    rect = cv2.minAreaRect(cntrs[max_area_index])
    (width_box, height_box), angle = rect[1], rect[2]

    if width_box > height_box:
        angle += 90

    if cntrs[max_area_index][0][0][0] < cntrs[max_area_index][1][0][0]:
        angle += 180

    cv2.drawContours(birdEye, cntrs, max_area_index, (0,255,0), 2)

    # Rogi prostokąta
    left_upper_corner = (x, y)
    left_bottom_corner = (x, y+h)
    right_upper_corner = (x+w, y)
    right_bottom_corner = (x+w, y+h)
    cv2.circle(birdEye, left_upper_corner, 5, (0, 0, 255), -1)
    cv2.circle(birdEye, left_bottom_corner, 5, (0, 0, 255), -1)
    cv2.circle(birdEye, right_upper_corner, 5, (0, 0, 255), -1)
    cv2.circle(birdEye, right_bottom_corner, 5, (0, 0, 255), -1)




    pts_org = np.array([left_upper_corner, right_upper_corner, right_bottom_corner, left_bottom_corner], np.float32)
    pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    matrix = cv2.getPerspectiveTransform(pts_org, pts_dst)
    workplace = cv2.warpPerspective(birdEye, matrix, (w, h))

    return birdEye, workplace, thresh, blur, gray, right_bottom_corner


cap = cv2.VideoCapture(0)

while True:
    rd, frame = cap.read()
    frame = camera_calibration(frame)
    birdEye, workplace, thresh, blur, gray, _ = Workspace_detection(frame)
    cv2.imshow('thresh', thresh)
    cv2.imshow('blur', blur)
    cv2.imshow('gray', gray)
    if workplace is not None:
        cv2.imshow('workplace', workplace)

    cv2.imshow('birdEye', birdEye)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()