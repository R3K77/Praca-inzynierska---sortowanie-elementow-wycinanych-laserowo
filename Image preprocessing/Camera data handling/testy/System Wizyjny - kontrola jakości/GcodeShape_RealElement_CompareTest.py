from Element_pojedynczy_test import single_gcode_elements_cv2
import cv2
import numpy as np

### background jako pojedyncze zdjęcie
# background = cv2.imread('./ZdjeciaElementy/poprawne/background.jpg')
# grayMedianFrame = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
### background jako suma z wideo

cap = cv2.VideoCapture('./ZdjeciaElementy/bgr_video.mp4')
# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=80)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)


# Lista poprawnych i niepoprawnych numerów
niepoprawne = [6,7,10,11,14,19,20]
poprawne = [1,2,3,4,5,8,9,12,13,15,16,17,18,21,22]

threshold_value = 60

# Trackbar callback function
def empty_callback(value):
    pass

# Iteracja przez numery poprawnych
for numer in poprawne:
    # Jeżeli numer jest na liście niepoprawnych, pomiń iterację
    if numer in niepoprawne:
        continue
    # Create a window
    cv2.namedWindow('Combined Image')
    # Create a trackbar
    cv2.createTrackbar('Threshold', 'Combined Image', threshold_value, 255, empty_callback)
    # Tworzenie nazwy pliku
    while True:
        nazwa_pliku = f'.\ZdjeciaElementy\poprawne\IMG{numer}.jpg'
        threshold_value = cv2.getTrackbarPos('Threshold', 'Combined Image')


        # Wczytanie i wyświetlenie obrazu
        frame = cv2.imread(nazwa_pliku)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (grayMedianFrame.shape[1], grayMedianFrame.shape[0]))
        dframe = cv2.absdiff(frame, grayMedianFrame)
        th, dframe = cv2.threshold(dframe, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_eroded = cv2.morphologyEx(dframe, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_contour_area = 200
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        frame_out = frame.copy()
        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 200), 3)

        if dframe.shape[0] != frame.shape[0]:
            height = min(dframe.shape[0], frame.shape[0])
            dframe = cv2.resize(dframe, (dframe.shape[1], height))
            frame = cv2.resize(frame, (frame.shape[1], height))

        combined_image = cv2.vconcat([mask_eroded, frame_out])
        combined_image = cv2.resize(combined_image, (1500,900))

        cv2.imshow('Combined Image', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

