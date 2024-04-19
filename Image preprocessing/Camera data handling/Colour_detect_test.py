import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
lower_red = np.array([129, 125, 187])
upper_red = np.array([179, 255, 255])

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over all detected contours
    for contour in contours:
        # Get convex hull of the contour
        hull = cv2.convexHull(contour)

        # Draw the contour and its hull
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

        # srednia wykryta (srodek czerwonego punktu)
        sum_x = sum_y = 0
        for point in hull:
            sum_x += point[0][0]
            sum_y += point[0][1]
        avg_x = sum_x / len(hull)
        avg_y = sum_y / len(hull)

        print(f'Average point: ({avg_x}, {avg_y})')
        print("---------------------")

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()