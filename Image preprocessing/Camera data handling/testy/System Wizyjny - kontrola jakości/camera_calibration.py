import numpy as np
import cv2
import glob
from pathlib import Path

# ----------------------------- KALIBRACJA KAMERY -----------------------------
# W celu kalibracji kamery należy wykonać zdjęcia szachownicy z różnych pozycji i z różnych kątów.
# Aby zapisać zdjęcia do kalibracji, należy uruchomić program i nacisnąć spację, aż do zrobienia odpowiedniej ilości zdjęć.
# Zalecana ilość zdjęć to około 40.
# Po zrobieniu zdjęć, należy wyłączyć kamerę naciskając klawisz 'q'.
# Następnie program wykona kalibrację kamery i zapisze parametry kamery do plików.
# Parametry kamery zostaną zapisane w folderze "settings" w plikach:
# - mtx_matrix.txt
# - distortion_matrix.txt
# - new_camera_matrix.txt
# - roi_matrix.txt
# Zapisane parametry kamery należy wczytać do właściwego programu, aby uzyskać poprawny obraz.
# Przykład wczytania parametrów kamery znajduje się w funkcji load_camera_calibration().
#
# Przed rozpoczęciem kalibracji należy usunąć pliki z folderu "frames"!
# ----------------------------------------------------------------------------

# Wczytanie parametrów kamery z pliku
def load_camera_calibration():
    loaded_mtx = np.loadtxt('../../settings/mtx_matrix.txt', delimiter=',')
    loaded_dist = np.loadtxt('../../settings/distortion_matrix.txt', delimiter=',')
    loaded_newcameramtx = np.loadtxt('../../settings/new_camera_matrix.txt', delimiter=',')
    loaded_roi = np.loadtxt('../../settings/roi_matrix.txt', delimiter=',')

    return loaded_mtx, loaded_dist, loaded_newcameramtx, loaded_roi

def main():
    CALIBRATION_DIR = "./frames/"

    # Inicjalizacja kamery
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    dir = Path(CALIBRATION_DIR)
    dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    
    # Wykonanie zdjęć szachownicy w celu kalibracji kamery
    while (True):
        if not cap.grab():
            break

        _, frame = cap.retrieve()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        
        # Wyjście z programu po naciśnięciu klawisza 'q'
        if key & 0xFF == ord('q'):
            break
        # Zapis zdjęcia po naciśnięciu spacji
        elif key & 0xFF == 32:
            cv2.imwrite(f"{str(dir)}/{counter:06d}.png", frame)
            counter += 1
            print(counter)
            
    # Zakończenie pracy kamery
    cap.release()

    # Kryteria zakończenia optymalizacji
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Przygotowanie "punktów" w przestrzeni 3D
    objp = np.zeros((13 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:13, 0:9].T.reshape(-1, 2)

    # Macierze do przechowywania punktów obiektów i punktów obrazu ze wszystkich obrazów
    objpoints = []
    imgpoints = []

    images = glob.glob(f"{str(dir)}/*.png") 

    # Przeszukanie wszystkich obrazów w celu znalezienia narożników szachownicy
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Znalazienie narożników szachownicy
        ret, corners = cv2.findChessboardCorners(gray, (13, 9), None)

        # Jeśli znaleziono narożniki, dodaj punkty do tablic
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            # Zaznacz narożniki na obrazie
            cv2.drawChessboardCorners(img, (13, 9 ), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(200)
            
    # Zakończenie zaznaczania narożników
    cv2.destroyAllWindows()

    print('Kalibrowanie w trakcie...')
    
    # Kalibracja kamery na podstawie znalezionych punktów obiektów i punktów obrazu 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # Zapis parametrów kamery do pliku
    np.savetxt('../../settings/mtx_matrix.txt', mtx, delimiter=',')
    np.savetxt('../../settings/distortion_matrix.txt', dist, delimiter=',')
    np.savetxt('../../settings/new_camera_matrix.txt', newcameramtx, delimiter=',')
    np.savetxt('../../settings/roi_matrix.txt', roi, delimiter=',')

    # Przykład wczytania parametrów kamery z pliku 
    load_mtx, load_dist, load_newcameramtx, load_roi = load_camera_calibration()
    
def calibrate_camera():
    # Inicjalizacja kamery
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Przygotowanie "punktów" w przestrzeni 3D
    objp = np.zeros((13 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:13, 0:9].T.reshape(-1, 2)

    # Tablice do przechowywania punktów obiektów i obrazu
    objpoints = []
    imgpoints = []

    # Kryteria zakończenia optymalizacji
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("Naciśnij spację, aby zapisać klatkę, lub 'q' aby zakończyć.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Nie udało się odczytać klatki z kamery.")
            break

        cv2.imshow('Kamera', frame)
        key = cv2.waitKey(1)

        # Wyjście z programu
        if key & 0xFF == ord('q'):
            break

        # Przechwycenie klatki po naciśnięciu spacji
        elif key & 0xFF == 32:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (13, 9), None)

            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # Zaznacz narożniki
                cv2.drawChessboardCorners(frame, (13, 9), corners2, ret)
                cv2.imshow('Kamera', frame)
                print("Znaleziono narożniki, dodano klatkę do kalibracji.")

    # Zakończenie pracy kamery
    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("Nie zebrano wystarczającej liczby danych do kalibracji.")
        return

    print('Kalibrowanie kamery...')
    # Kalibracja kamery
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    return mtx, dist, newcameramtx, roi

if __name__ == "__main__":
    main()
    