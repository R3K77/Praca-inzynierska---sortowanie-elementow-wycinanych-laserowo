import cv2
import numpy as np
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
        cv2.imshow('Detekcja krawędzi', edged)
        cv2.imshow('Detekcja krawędziw', gray)

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
        # Obliczenie kąta obrotu figury
        angle = np.arctan2(input_pts[1][1] - input_pts[0][1], input_pts[1][0] - input_pts[0][0]) * 180 / np.pi
        return img_output, edged, gray, angle
    

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

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
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
    return avg_x, avg_y

# Utworzenie obiektu kamery
cap = cv2.VideoCapture(0)
lower_red = np.array([129, 125, 187])
upper_red = np.array([179, 255, 255])

# Utworzenie okien do wyświetlania obrazów i suwaków do zmiany parametrów detekcji krawędzi i filtracji 
cv2.namedWindow('Detekcja krawędzi')
cv2.createTrackbar('Threshold 1', 'Detekcja krawędzi', 18, 100, lambda x: None)
cv2.createTrackbar('Threshold 2', 'Detekcja krawędzi', 53, 200, lambda x: None)
cv2.namedWindow('Detekcja krawędziw')
cv2.createTrackbar('Threshold 3', 'Detekcja krawędziw', 5, 100, lambda x: None)
cv2.createTrackbar('Threshold 4', 'Detekcja krawędziw', 10, 100, lambda x: None)
cv2.createTrackbar('Threshold 5', 'Detekcja krawędziw', 10, 200, lambda x: None)

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
    threshold1 = cv2.getTrackbarPos('Threshold 1', 'Detekcja krawędzi')
    threshold2 = cv2.getTrackbarPos('Threshold 2', 'Detekcja krawędzi')
    threshold3 = cv2.getTrackbarPos('Threshold 3', 'Detekcja krawędziw')
    threshold4 = cv2.getTrackbarPos('Threshold 4', 'Detekcja krawędziw')
    threshold5 = cv2.getTrackbarPos('Threshold 5', 'Detekcja krawędziw')
    # Punkt referencyjny początku układu współrzędnych robota

    avg_x, avg_y = RefPoint_Detection(img, lower_red, upper_red)
    # Wywołanie funkcji do detekcji krawędzi i transformacji perspektywicznej
    img_output, edged, gray, angle = workspace_detection(img, threshold1, threshold2, threshold3, threshold4, threshold5)
    # Wyświetlenie obrazów
    cv2.imshow('edged', edged)
    cv2.imshow('img', img)

    print(f'Obrot = {angle}')
    print(f'Punkt referencyjny: ({avg_x}, {avg_y})')
    print("------------------------------")
    cv2.imshow('img_output', img_output)
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
