import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow('HSV Adjust')

# Create trackbars for color change
cv2.createTrackbar('Lower H','HSV Adjust', 129, 179, nothing)
cv2.createTrackbar('Lower S','HSV Adjust',125,255,nothing)
cv2.createTrackbar('Lower V','HSV Adjust',187,255,nothing)
cv2.createTrackbar('Upper H','HSV Adjust',179,179,nothing)
cv2.createTrackbar('Upper S','HSV Adjust',255,255,nothing)
cv2.createTrackbar('Upper V','HSV Adjust',255,255,nothing)
#Czerwone
lower_red = np.array([114, 130, 95])
upper_red = np.array([179, 255, 255])
#Niebieskie
lower_blue = np.array([51, 128, 172])
upper_blue = np.array([141, 255, 255])

#Zielone
lower_green = np.array([35, 49, 154])
upper_green = np.array([71, 255, 255])

# Maska do otwarcia obrazu
kernel = np.ones((3,3),np.uint8)
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get current positions of six trackbars
    l_h = cv2.getTrackbarPos('Lower H', 'HSV Adjust')
    l_s = cv2.getTrackbarPos('Lower S', 'HSV Adjust')
    l_v = cv2.getTrackbarPos('Lower V', 'HSV Adjust')
    u_h = cv2.getTrackbarPos('Upper H', 'HSV Adjust')
    u_s = cv2.getTrackbarPos('Upper S', 'HSV Adjust')
    u_v = cv2.getTrackbarPos('Upper V', 'HSV Adjust')

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
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
    cv2.imshow('mask2', mask2)
    cv2.imshow('res', res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()