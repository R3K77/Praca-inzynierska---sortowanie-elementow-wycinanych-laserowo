import cv2
import numpy as np

# Na podstawie problemu z odbijaniem światłą przedstawionego
# na: https://stackoverflow.com/questions/67099645/contour-detection-reduce-glare-on-image-opencv-python

birdEye = cv2.VideoCapture(0)

while True:
    birdEye = np.maximum(birdEye, 10)
    foreground = birdEye.copy()
    seed = (10, 10)  # tUse he top left corner as a "background" seed color (assume pixel [10,10] is not in an object).
    cv2.floodFill(foreground, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=(5, 5, 5, 5), upDiff=(5, 5, 5, 5))
    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    # Apply threshold
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    # Use opening for removing small outliers
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # Find contours
    cntrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # Tu można zamienić na simple, jeżeli ma być wykrywany prostokątny element
    # Draw contours
    cv2.drawContours(birdEye, cntrs, -1, (255, 0, 255), 3)
    cv2.imshow('foreground', foreground)
    cv2.imshow('gray', gray)
    cv2.imshow('thresh', thresh)
    cv2.imshow('birdEye', birdEye)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Show images for testing


birdEye.release()
cv2.destroyAllWindows()