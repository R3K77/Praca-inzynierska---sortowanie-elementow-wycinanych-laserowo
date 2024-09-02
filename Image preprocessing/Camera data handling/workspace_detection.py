import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
# ----------------- Funkcja do detekcji krawędzi i transformacji perspektywicznej ----------------- #
# Funkcja przyjmuje obraz, wartości progów detekcji krawędzi oraz parametry filtracji obrazu
# Funkcja zwraca obraz wynikowy, obraz z detekcją krawędzi oraz obraz po filtracji
# W przypadku braku znalezienia konturów zwracany jest pusty obraz
# Zwracane obrazy:
# img_output, edged, gray = workspace_detection(img, Canny_threshold1, Canny_threshold2, bilateralFilter_d, bilateralFilter_SigmaColor, bilateralFilter_SigmaSpace, own_size=False, own_size_width=0, own_size_height=0)
#  - img_output - obraz wynikowy
#  - edged - obraz z detekcją krawędzi
#  - gray - obraz po filtracji w skali szarości
# Wejściowe parametry:
#  - img - obraz wejściowy
#  - Canny_threshold1 - próg detekcji krawędzi
#  - Canny_threshold2 - próg detekcji krawędzi
#  - bilateralFilter_d - parametr filtracji bilateralnej 
#  - bilateralFilter_SigmaColor - parametr filtracji bilateralnej 
#  - bilateralFilter_SigmaSpace - parametr filtracji bilateralnej 
#  - own_size - własne wymiary obrazu wynikowego (True/False), domyślnie False
#  - own_size_width - szerokość obrazu wynikowego, domyślnie 0
#  - own_size_height - wysokość obrazu wynikowego, domyślnie 0
# ------------------------------------------------------------------------------------------------- #

#TODO przebudować funkcję do wykrywania blachy na background substraction
def workspace_detection(img, Canny_threshold1, Canny_threshold2, bilateralFilter_d, bilateralFilter_SigmaColor, bilateralFilter_SigmaSpace, own_size=False, own_size_width=0, own_size_height=0):
    # Filtracja obrazu w skali szarości 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, bilateralFilter_d, bilateralFilter_SigmaColor, bilateralFilter_SigmaSpace)
    edged = cv2.Canny(gray, Canny_threshold1, Canny_threshold2) 
    
    # Detekcja konturów
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    
    # Znalezienie największego konturu
    biggest = np.array([])
    if len(contours) != 0:
        max_area = 0
        for i in contours:                                          # Przechodzenie przez wszystkie kontury
            area = cv2.contourArea(i)                               # Obliczenie pola powierzchni
            if area > 500:                                          # Odfiltrowanie małych konturów
                peri = cv2.arcLength(i, True)                       # Obliczenie obwodu
                approx = cv2.approxPolyDP(i, 0.015 * peri, True)    # Znalezienie wierzchołków
                if area > max_area and len(approx) == 4:            # Sprawdzenie czy kontur jest czworokątem
                    biggest = approx                                
                    max_area = area

    # Oznaczenie największego konturu na obrazie
    cv2.drawContours(img, biggest, -1, (0, 255, 0), 5)
    
    # W przypadku braku znalezienia konturów zwracany jest pusty obraz
    if biggest.size == 0:
        # cv2.imshow('Detekcja krawędzi', edged)
        # cv2.imshow('Detekcja krawędziw', gray)

        return np.zeros((1, 1, 3), dtype="uint8"), edged, gray, None
    # W przeciwnym wypadku wyznaczane są współrzędne narożników obrazu wynikowego
    else:
        points = biggest.reshape((4, 2))
        input_pts = np.zeros((4, 2), dtype="float32")

        # Wyznaczenie koordynatów narożników 
        points_sum = points.sum(axis=1)
        input_pts[0] = points[np.argmin(points_sum)]        # top-left 
        input_pts[3] = points[np.argmax(points_sum)]        # bottom-right
        points_diff = np.diff(points, axis=1) 
        input_pts[1] = points[np.argmin(points_diff)]       # top-right
        input_pts[2] = points[np.argmax(points_diff)]       # bottom-left
        
        # Wyznaczenie wymiarów obrazu
        (top_left, top_right, bottom_left, bottom_right) = input_pts # Rozpakowanie punktów
        bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))   # Obliczenie dolnej szerokości
        top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))                  # Obliczenie górnej szerokości
        right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))       # Obliczenie prawej wysokości
        left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))            # Obliczenie lewej wysokości

        max_width = 0
        max_height = 0        
        # Rozmiar obrazu wynikowego
        if own_size == True:
            max_width = own_size_width
            max_height = own_size_height
        else:
            max_width = max(int(bottom_width), int(top_width))          # Maksymalna szerokość
            # max_height = int(max_width * 1.414)                       # Dla kartki A4 stosunek szerokości do wysokości wynosi 1.414
            max_height = max(int(right_height), int(left_height))       # Maksymalna wysokość

        # Współrzędne narożników obrazu wynikowego
        converted_points = np.array([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]], dtype="float32")

        # Transformacja perspektywiczna
        matrix = cv2.getPerspectiveTransform(input_pts, converted_points)
        img_output = cv2.warpPerspective(img, matrix, (max_width, max_height))

        # Dopasowanie wymiarów obrazów w celu użycia funkcji hstack
        gray = np.stack((gray,)*3, axis=-1)
        edged = np.stack((edged,)*3, axis=-1)

        return img_output, edged, gray, input_pts
    

# ----------------- Funkcja do kalibracji kamery ----------------- #
# Funkcja przyjmuje obraz z kamery
# Funkcja zwraca obraz po kalibracji kamery
# Funkcja wczytuje z plików macierze kalibracji kamery
# ---------------------------------------------------------------- #
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

# ----------------- Funkcja do detekcji punktu referencyjnego ----------------- #
# Funkcja przyjmuje obraz, dolną i górną granicę koloru HSV
# Funkcja zwraca współrzędne punktu referencyjnego o danym kolorze
# --------------------------------------------------------------------------- #
def RefPoint_Detection(img, lower, upper):
    if img.size == 0:
        return 0, 0

    kernel = np.ones((3,3),np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    avg_x = avg_y = 0
    for contour in contours:
        hull = cv2.convexHull(contour)
        sum_x = sum_y = 0
        for point in hull:
            sum_x += point[0][0]
            sum_y += point[0][1]
        avg_x = sum_x / len(hull)
        avg_y = sum_y / len(hull)
    return avg_x, avg_y, mask

def Calculate_angle(base_x, base_y, xAxis_x, xAxis_y, yAxis_x, yAxis_y, rec_edge_points):
    if rec_edge_points is None:
        return 0, 0, 0

    base = np.array([base_x, base_y])
    xAxis = np.array([xAxis_x, xAxis_y])
    # Wektor osi X
    vec_x = xAxis - base
    vec_rec = rec_edge_points[1] - rec_edge_points[0]
    # Obliczanie kąta między wektorami
    angle_z = np.arctan2(np.linalg.det([vec_x, vec_rec]), np.dot(vec_x, vec_rec))
    angle_z = np.degrees(angle_z)
    #translacja
    TR_x = rec_edge_points[1][0] - base_x
    TR_y = rec_edge_points[1][1] - base_y
    angle_z = 180 + angle_z
    return TR_x, TR_y, angle_z

# Utworzenie obiektu kamery
cap = cv2.VideoCapture(0)

## Dolna i górna granica kolorów
#Czerwone
lower_red = np.array([132, 94, 127])
upper_red = np.array([179, 255, 255])
#Niebieskie
lower_blue = np.array([77, 127, 102])
upper_blue = np.array([111, 255, 255])
#Zielone
lower_green = np.array([34, 49, 154])
upper_green = np.array([71, 255, 255])

# Zapisz aktualny czas
last_print_time = time.time()

# Główna pętla programu
while True:
    ret, frame = cap.read()
    frame = camera_calibration(frame)
    if not ret:
        print('Nie można wczytać obrazu z kamery')
        break
    img = frame.copy()
    img_original = img.copy()

    # Pobranie wartości progów detekcji krawędzi i filtracji
    # Punkt referencyjny początku układu współrzędnych koncówki oraz punkty osi X i Y
    base_x, base_y, mask_red = RefPoint_Detection(img, lower_red, upper_red)
    xAxis_x, xAxis_y, mask_blue = RefPoint_Detection(img, lower_blue, upper_blue)
    yAxis_x, yAxis_y, mask_green = RefPoint_Detection(img, lower_green, upper_green)
    # Draw circles at the points of interest
    cv2.circle(img, (int(base_x), int(base_y)), 5, (255, 0, 0), -1)  # Base point in red
    cv2.circle(img, (int(xAxis_x), int(xAxis_y)), 5, (0, 255, 0), -1)  # X-axis point in blue
    cv2.circle(img, (int(yAxis_x), int(yAxis_y)), 5, (0, 0, 255), -1)  # Y-axis point in green

    # Wywołanie funkcji do detekcji krawędzi i transformacji perspektywicznej
    img_output, edged, gray, rec_edge_points = workspace_detection(img, 18, 53, 5, 10, 10)
    # Translacja oraz kąt obrotu
    TR_x, TR_y, angle = Calculate_angle(base_x, base_y, xAxis_x, xAxis_y, yAxis_x, yAxis_y, rec_edge_points)
    # Wyświetlenie obrazów
    # cv2.imshow('edged', edged)
    cv2.imshow('img', img)
    cv2.imshow('img_output', img_output)

    current_time = time.time()
    if current_time - last_print_time >= 0.5:
        print(f"\rTR_x: {TR_x} TR_y: {TR_y} angle: {angle}", end=' '*10)
        last_print_time = current_time

    # Oczekiwanie na klawisz 'q' do zakończenia pętli
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # if img_output.size > 100:
    #     break

# Zwolnienie kamery i zamknięcie wszystkich okien
cap.release()
cv2.destroyAllWindows()
print("------------------------------------------------")
print("wyjscie z petli")

# Translacja i obrot w osi z.
