import cv2
import numpy as np
import sys
from workspace_detection import workspace_detection
from workspace_detection import camera_calibration
# Utworzenie obiektu kamery

cap = cv2.VideoCapture(0)

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
    print("wszedlem do petli")
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

    # Wywołanie funkcji do detekcji krawędzi i transformacji perspektywicznej
    img_output, edged, _ = workspace_detection(img, threshold1, threshold2, threshold3, threshold4, threshold5)
    # Wyświetlenie obrazów
    cv2.imshow('Obraz z kamery', img)
    cv2.imshow('Obraz wynikowy', img_output)

    # Oczekiwanie na klawisz 'q' do zakończenia pętli
    if cv2.waitKey(1) & 0xFF == ord('q') or img_output.size > 100:
        break


cap.release()
cv2.destroyAllWindows()

# Zwolnienie kamery i zamknięcie wszystkich okien
