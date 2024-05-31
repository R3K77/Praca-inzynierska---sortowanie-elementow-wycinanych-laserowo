import cv2
import numpy as np

# Na podstawie problemu z odbijaniem światłą przedstawionego
# na: https://stackoverflow.com/questions/67099645/contour-detection-reduce-glare-on-image-opencv-python

cap = cv2.VideoCapture(1)
frame_pre = cap.read()
height, width = frame_pre[1].shape[:2]
lowerDiff = [2,2,2,2]
upperDiff = [2,2,2,2]
lowerDiff2 = [2,2,2,2]
upperDiff2 = [2,2,2,2]
lowerDiff3 = [3,3,3,3]
upperDiff3 = [3,3,3,3]

while True:
    frame = cap.read()
    birdEye = np.maximum(frame[1], 10)
    foreground = birdEye.copy()
    seed = (10, 10)
    cv2.floodFill(foreground, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=lowerDiff, upDiff=upperDiff)

    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    cntrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Tu można zamienić na simple, jeżeli ma być wykrywany prostokątny element

    max_area = 0
    max_area_index = 0
    for i , cntr in enumerate(cntrs):
        area = cv2.contourArea(cntr)
        if area > max_area:
            max_area = area
            max_area_index = i

    cv2.drawContours(birdEye, cntrs, -1, (255, 0, 255), 3)

    cv2.imshow('foreground', foreground)
    cv2.imshow('thresh', thresh)

    cv2.imshow('birdEye', birdEye)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Show images for testing


cap.release()
cv2.destroyAllWindows()