import cv2
import numpy as np

# Funkcja zwracająca największy kontur
def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:                                          # Przechodzenie przez wszystkie kontury
        area = cv2.contourArea(i)                               # Obliczenie pola powierzchni
        if area > 500:                                          # Odfiltrowanie małych konturów
            peri = cv2.arcLength(i, True)                       # Obliczenie obwodu
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)    # Znalezienie wierzchołków
            if area > max_area and len(approx) == 4:            # Sprawdzenie czy kontur jest czworokątem
                biggest = approx                                
                max_area = area
    return biggest

cap = cv2.VideoCapture(1)
cv2.namedWindow('Detekcja krawędzi')
cv2.createTrackbar('Threshold 1', 'Detekcja krawędzi', 10, 100, lambda x: None)
cv2.createTrackbar('Threshold 2', 'Detekcja krawędzi', 70, 200, lambda x: None)
cv2.namedWindow('Detekcja krawędziw')
cv2.createTrackbar('Threshold 3', 'Detekcja krawędziw', 5, 100, lambda x: None)
cv2.createTrackbar('Threshold 4', 'Detekcja krawędziw', 10, 100, lambda x: None)
cv2.createTrackbar('Threshold 5', 'Detekcja krawędziw', 10, 200, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Nie można wczytać obrazu z kamery')
        break
    img = frame.copy()
    img_original = img.copy()

    threshold1 = cv2.getTrackbarPos('Threshold 1', 'Detekcja krawędzi')
    threshold2 = cv2.getTrackbarPos('Threshold 2', 'Detekcja krawędzi')
    threshold3 = cv2.getTrackbarPos('Threshold 3', 'Detekcja krawędziw')
    threshold4 = cv2.getTrackbarPos('Threshold 4', 'Detekcja krawędziw')
    threshold5 = cv2.getTrackbarPos('Threshold 5', 'Detekcja krawędziw')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, threshold3, threshold4, threshold5)
    edged = cv2.Canny(gray, threshold1, threshold2)

    # Detekcja konturów
    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]

    # Znalezienie największego konturu
    if len(contours) == 0:
        print("Nie znaleziono konturów")
        biggest = np.array([])
    else:
        biggest = biggest_contour(contours)

    # Rysowanie największego konturu
    cv2.drawContours(img, biggest, -1, (0, 255, 0), 5)

    if biggest.size == 0:
        print("Nie znaleziono czworokątnego konturu")
        cv2.imshow('Detekcja krawędzi', edged)
        cv2.imshow('Detekcja krawędziw', gray)
    else:
        points = biggest.reshape((4, 2))
        input_pts = np.zeros((4, 2), dtype="float32")

        # Wyznaczenie koordynatów narożników
        points_sum = points.sum(axis=1)
        input_pts[0] = points[np.argmin(points_sum)] # top-left 
        input_pts[3] = points[np.argmax(points_sum)] # bottom-right
        points_diff = np.diff(points, axis=1)
        input_pts[1] = points[np.argmin(points_diff)] # top-right
        input_pts[2] = points[np.argmax(points_diff)] # bottom-left

        # Wyznaczenie wymiarów obrazu
        (top_left, top_right, bottom_left, bottom_right) = input_pts # Rozpakowanie punktów
        bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))   # Obliczenie dolnej szerokości
        top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))                  # Obliczenie górnej szerokości
        right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))       # Obliczenie prawej wysokości
        left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))            # Obliczenie lewej wysokości

        # Rozmiar obrazu wynikowego
        max_width = max(int(bottom_width), int(top_width)) # Maksymalna szerokość
        # max_height = int(max_width * 1.414)                # Dla kartki A4 stosunek szerokości do wysokości wynosi 1.414
        max_height = max(int(right_height), int(left_height)) # Maksymalna wysokość

        # Współrzędne narożników obrazu wynikowego
        converted_points = np.array([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]], dtype="float32")

        # Transformacja perspektywiczna
        matrix = cv2.getPerspectiveTransform(input_pts, converted_points)
        img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))

        # Dopasowanie wymiarów obrazów w celu użycia funkcji hstack
        gray = np.stack((gray,)*3, axis=-1)
        edged = np.stack((edged,)*3, axis=-1)

        # Wyświetlenie obrazów
        cv2.imshow('Detekcja krawędzi', edged)
        cv2.imshow('Detekcja krawędziw', img)
        
        # img_hor = np.hstack((img_original, gray, edged, img))
        # cv2.imshow('Detekcja krawędzi', img_hor)
        cv2.imshow('Obraz wynikowy', img_output)

    # Oczekiwanie na klawisz 'q' do zakończenia pętli
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnienie kamery i zamknięcie wszystkich okien
cap.release()
cv2.destroyAllWindows()