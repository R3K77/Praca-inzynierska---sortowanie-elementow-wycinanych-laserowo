import cv2

def nothing(x):
    pass

# Otwórz dostęp do kamery
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Nie można otworzyć kamery")
    exit()

# Tworzenie okna
cv2.namedWindow('Podgląd z kamery')

# Dodanie suwaków
cv2.createTrackbar('Góra', 'Podgląd z kamery', 0, 100, nothing)
cv2.createTrackbar('Dół', 'Podgląd z kamery', 0, 100, nothing)
cv2.createTrackbar('Lewa', 'Podgląd z kamery', 0, 100, nothing)
cv2.createTrackbar('Prawa', 'Podgląd z kamery', 0, 100, nothing)

while True:
    # Przechwyć klatkę z kamery
    ret, frame = cap.read()

    # Jeśli nie udało się pobrać klatki, zakończ
    if not ret:
        print("Nie można pobrać klatki")
        break

    # Pobierz wartości z suwaków
    top = cv2.getTrackbarPos('Góra', 'Podgląd z kamery')
    bottom = cv2.getTrackbarPos('Dół', 'Podgląd z kamery')
    left = cv2.getTrackbarPos('Lewa', 'Podgląd z kamery')
    right = cv2.getTrackbarPos('Prawa', 'Podgląd z kamery')

    # Oblicz nowe wymiary obrazu
    h, w, _ = frame.shape
    top = int(h * (top / 100))
    bottom = int(h * (bottom / 100))
    left = int(w * (left / 100))
    right = int(w * (right / 100))

    # Wycinanie obrazu (slicing)
    sliced_frame = frame[top:h-bottom, left:w-right]

    # Wyświetl obraz po przycięciu
    cv2.imshow('Podgląd z kamery', sliced_frame)

    # Wyjście po naciśnięciu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zakończ nagrywanie i zamknij okna
cap.release()
cv2.destroyAllWindows()
print(f'Top: {top}, Bottom: {bottom}, Left: {left}, Right: {right}')
