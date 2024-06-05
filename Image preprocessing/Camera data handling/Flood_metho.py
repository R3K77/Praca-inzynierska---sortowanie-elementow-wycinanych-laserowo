import cv2
import numpy as np
import time # Tylko do printowania co jakiś czas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def RefPoint_Detection(img, lower, upper):
    if img.size == 0:
        return 0, 0

    kernel = np.ones((3, 3), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_area_index = -1
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_area_index = i

    if max_area_index == -1:
        return 0, 0, mask

    # Obliczanie punktu środkowego dla największego konturu
    M = cv2.moments(contours[max_area_index])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY, mask


def Calculate_angle(base_x, base_y, xAxis_x, xAxis_y, rec_edge_points):
    if rec_edge_points is None:
        return 0, 0, 0

    base = np.array([base_x, base_y])
    xAxis = np.array([xAxis_x, xAxis_y])
    # Wektor osi X
    vec_x = xAxis - base
    vec_rec = rec_edge_points[3] - rec_edge_points[2]
    # Obliczanie kąta między wektorami
    angle_z = np.arctan2(np.linalg.det([vec_x, vec_rec]), np.dot(vec_x, vec_rec))
    angle_z = np.degrees(angle_z)
    # translacja
    TR_x = rec_edge_points[2][0] - base_x
    TR_y = rec_edge_points[2][1] - base_y
    return TR_x, TR_y, angle_z


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
def Workspace_detection(frame):
    lowerDiff = [3, 3, 3, 5]
    upperDiff = [3, 3, 3, 5]
    birdEye = np.maximum(frame, 10)
    foreground = birdEye.copy()

    height, width = frame.shape[:2]
    seeds = [(10, 10), (10, height - 10), (width - 10, 10), (width - 10, height - 10)]
    for seed in seeds:
        cv2.floodFill(foreground, None, seedPoint=seed, newVal=(0, 0, 0), loDiff=lowerDiff, upDiff=upperDiff)

    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)[1]
    thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    thresh3 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    cntrs, _ = cv2.findContours(thresh3, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # Tu można zamienić na simple, jeżeli ma być wykrywany prostokątny element

    max_area = 0
    max_area_index = -1
    for i, cntr in enumerate(cntrs):
        area = cv2.contourArea(cntr)
        if area > max_area:
            max_area = area
            max_area_index = i

    # wyznaczenie rogów prostokąta
    # https://stackoverflow.com/questions/7263621/how-to-find-corners-on-a-image-using-opencv
    arc = 0.04 * cv2.arcLength(cntrs[max_area_index], True)
    poly = cv2.approxPolyDP(cntrs[max_area_index], arc, True)

    if len(poly) != 4:
        return birdEye, None, thresh, blur, gray, None, None, None
    if len(poly) == 4:
        cv2.drawContours(birdEye, [poly], 0, (0), 5)
        # cv2.drawContours(birdEye, cntrs, max_area_index, (0, 255, 0), 2)
        poly = poly[np.argsort(poly[:, 0][:, 1])]
        # Rogi prostokąta
        left_upper_corner = poly[0][0]
        left_bottom_corner = poly[1][0]
        right_upper_corner = poly[2][0]
        right_bottom_corner = poly[3][0]
        w = int(np.sqrt(
            (right_upper_corner[0] - left_upper_corner[0]) ** 2 + (right_upper_corner[1] - left_upper_corner[1]) ** 2))
        h = int(np.sqrt((right_upper_corner[0] - right_bottom_corner[0]) ** 2 + (
                    right_upper_corner[1] - right_bottom_corner[1]) ** 2))
        cv2.circle(birdEye, left_bottom_corner, 5, (255, 0, 255), -1) # różowy
        cv2.circle(birdEye, right_upper_corner, 5, (0, 0, 255), -1) # czerwony
        cv2.circle(birdEye, right_bottom_corner, 5, (255, 255, 255), -1) # biały
        cv2.circle(birdEye, left_upper_corner, 5, (0, 255, 255), -1) # żółty
        pts_org = np.array([left_upper_corner, right_upper_corner, right_bottom_corner, left_bottom_corner], np.float32)
        pts_dst = np.array([[0, 0], [w, 0],[w, h], [0, h]], np.float32)
        matrix = cv2.getPerspectiveTransform(pts_org, pts_dst)
        workplace = cv2.warpPerspective(birdEye, matrix, (w, h))
        rect = cv2.minAreaRect(pts_org)
        angle = rect[2]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = -(90 - angle)
        return birdEye, workplace, thresh3, blur, gray, right_bottom_corner, angle, pts_org

def draw_line_through_points(image, point1, point2):
    # Współrzędne dwóch punktów

    # Obliczanie współczynnika kierunkowego m
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])

    # Obliczanie współczynnika b
    b = point1[1] - m * point1[0]

    # Obliczanie punktów na krańcach ekranu, które leżą na linii
    y_at_x_min = int(m * 0 + b)
    y_at_x_max = int(m * image.shape[1] + b)
    cv2.line(image, (0, y_at_x_min), (image.shape[1], y_at_x_max), (0, 255, 0), 2)

## Dolna i górna granica kolorów
# Czerwone
lower_red = np.array([167, 100, 97])
upper_red = np.array([179, 255, 255])
# Niebieskie
lower_blue = np.array([77, 127, 102])
upper_blue = np.array([111, 255, 255])
# Zielone
lower_green = np.array([34, 49, 154])
upper_green = np.array([71, 255, 255])

cap = cv2.VideoCapture(0)
last_print_time = time.time()
while True:
    rd, frame = cap.read()
    frame = camera_calibration(frame)
    # Detekcja Workspace
    birdEye, workplace, thresh, blur, gray, _, angle, pts_org = Workspace_detection(frame)
    # punkty układu bazowego + rysowanie
    base_x, base_y, mask_red = RefPoint_Detection(frame, lower_red, upper_red)
    xAxis_x, xAxis_y, mask_blue = RefPoint_Detection(frame, lower_blue, upper_blue)
    yAxis_x, yAxis_y, mask_green = RefPoint_Detection(frame, lower_green, upper_green)
    cv2.circle(birdEye, (int(base_x), int(base_y)), 5, (255, 0, 0), -1)  # Base point in red
    cv2.circle(birdEye, (int(xAxis_x), int(xAxis_y)), 5, (0, 255, 0), -1)  # X-axis point in blue
    cv2.circle(birdEye, (int(yAxis_x), int(yAxis_y)), 5, (0, 0, 255), -1)  # Y-axis point in green
    TR_x, TR_y, angle_2 = Calculate_angle(base_x, base_y, xAxis_x, xAxis_y, pts_org)

    # Linia do translacji
    # start_point = (int(base_x), int(base_y))
    # end_point = (int(base_x + TR_x), int(base_y + TR_y))
    # cv2.line(birdEye, start_point, end_point, (0, 255, 0), 2)
    # Linie osi układu bazowego
    if pts_org is not None:
        draw_line_through_points(birdEye, (base_x, base_y), (xAxis_x, xAxis_y))
        draw_line_through_points(birdEye, tuple(pts_org[2]), tuple(pts_org[1]))

    cv2.imshow('thresh', thresh)
    cv2.imshow('blur', blur)
    # cv2.imshow('gray', gray)
    # if workplace is not None:
    #     cv2.imshow('workplace', workplace)
    cv2.imshow('mask_red', mask_red)
    cv2.imshow('mask_blue', mask_blue)
    cv2.imshow('mask_green', mask_green)
    cv2.imshow('birdEye', birdEye)
    current_time = time.time()
    if current_time - last_print_time >= 0.5:
        print("---------------------------")
        print(f'Angle: {angle}')
        print(f'Angle2 : {angle_2}')
        print(f"TR_x: {TR_x}")
        print(f"TR_y: {TR_y}")
        last_print_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Create a PdfPages object
pdf = PdfPages('output.pdf')

# List of images to save
birdEye = cv2.cvtColor(birdEye, cv2.COLOR_BGR2RGB)
images = [thresh, blur, mask_red, mask_blue, mask_green, birdEye]
image_names = ['thresh', 'blur', 'mask_red', 'mask_blue', 'Wykrycie zielonego', 'Kąt']

# For each image
for img, name in zip(images, image_names):
    # Create a new figure
    fig = plt.figure()

    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title(name)

    # Save the figure to the SVG file with no extra white space
    plt.savefig(f'Images/{name}.svg', format='svg', bbox_inches='tight')

    # Close the figure
    plt.close(fig)

for img, name in zip(images, image_names):
    # Create a PdfPages object with the image name
    pdf = PdfPages(f'Images/{name}.pdf')

    # Create a new figure
    fig = plt.figure()

    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title(name)

    # Save the figure to the pdf file with no extra white space
    pdf.savefig(fig, bbox_inches='tight')

    # Close the pdf file
    pdf.close()
# TODO - translacja powinna być podana w mm, więc trzeba przeskalować na podstawie obsługiwanej blachy, może info z CSV?






