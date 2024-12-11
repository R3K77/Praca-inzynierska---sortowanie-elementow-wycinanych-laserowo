# ----------- Plik wykorzystywany do komunikacji z robotem KUKA ------------ #
# Autorzy: Bartłomiej Szalwach, Maciej Mróz, Rafał Szygenda
# -------------------------------------------------------------------------- #
import base64
import socket
import csv
import json

import cv2
# import orjson

import keyboard
from collections import defaultdict
from _functions_computer_vision import *
REFPOINT = (1423, 549)
# Konfiguracja serwera
HOST = '0.0.0.0'  # Nasłuchiwanie na wszystkich interfejsach sieciowych
PORT = 59152  # Port zgodny z konfiguracją w robocie KUKA


def main(json_name):
    # Bufor pod system wizyjny
    cv_data = {}
    crop_values_sheet = {'bottom': 499, 'left': 0, 'right': 380, 'top': 0} #right 23 down 47
    crop_values = {'top': 127, 'bottom': 361, 'left': 285, 'right': 817} #gora 12 dol 34 lewa 15 prawa 43
    print("zbieram informacje o tle dla quality control \n")
    BgrSubstractor_Quality = capture_median_frame(crop_values,2)
    print("otwieram dane o elementach z jsona \n")
    with open(f'Image preprocessing/Gcode to image conversion/elements_data_json/{json_name}.json','r') as f:
        data = json.load(f)
    elements = data['elements']
    SHEET_SIZE = 570
    curveData = data['curveCircleData']
    linearData = data['linearPointsData']
    angles_elements = data['rotation'] 
    # # Tworzenie gniazda serwera
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Serwer nasłuchuje na {HOST}:{PORT}")
    try:
        client_socket, client_address = server_socket.accept()
        print(f"Połączono z {client_address}")
    except Exception as e:
        print(f"Błąd: \n {e}")
        return
    print("zbieranie informacji o stanowisku roboczym")
    BgrSubstractor_Sheet = capture_median_frame(crop_values_sheet, 1)
    print("Umieść blachę w stanowisku roboczym (spacja)")
    keyboard.wait('space')
    print("Zbieranie informacji o położeniu blachy")
    angle_sheet,translation_mm,sheetData = sheetRotationTranslation(BgrSubstractor_Sheet,1,crop_values_sheet,SHEET_SIZE)
    for i in range(len(sheetData[4])):
        cv2.imwrite(f"Image preprocessing/Gcode to image conversion/camera_images_debug/AAAAsheet{i}.png",sheetData[4][i])
    while translation_mm[0] < 0 or translation_mm[1] < 0: 
        print("popraw położenie blachy, tak aby translacja byla ujemna")
        print(f"angle:{angle_sheet} ")
        print(f"translation: {translation_mm}")
        keyboard.wait('space')
        print("Zbieranie informacji o położeniu blachy")
        angle_sheet,translation_mm,sheetData = sheetRotationTranslation(BgrSubstractor_Sheet,1,crop_values_sheet,SHEET_SIZE)
    print(f"angle:{angle_sheet} ")
    print(f"translation: {translation_mm}")
    photos1 = []
    for img in sheetData[4]:
        _, buf = cv2.imencode('.jpg', img)
        imgbase64 = base64.b64encode(buf).decode('utf-8')
        photos1.append(imgbase64)
    cv_data['sheet'] = {
        "right_down_point": [int(sheetData[2][0]),int(sheetData[2][1])],
        "right_up_point": [int(sheetData[1][0]),int(sheetData[1][1])],
        "left_down_point": [int(sheetData[0][0]),int(sheetData[0][1])],
        "right_side_linear_fcn": [int(sheetData[3][0]),int(sheetData[3][1]),int(sheetData[3][2])],
        "rotation": angle_sheet,
        "translation": translation_mm,
        "bonusImages": {
            "camera_image": photos1[2],
            "MOG2_image": photos1[3],
            "object_full_image": photos1[1],
        }
    }
    with open(f'cv_data_{json_name}_sheetTest.json','w',encoding ='utf8') as f:
        json.dump(cv_data,f,ensure_ascii=False)


    with open('element_details.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pominięcie nagłówka
        for row in reader:
            if not row:
                break
            # ------------- POBRANIE DETALU -------------

            name = row[0]
            detail_x = float(row[1])
            detail_y = float(row[2])
            detail_z = float(row[3])
            print(f"Odczytano dane z csv: {detail_x}, {detail_y}, {detail_z}")
            print(f"Nazwa elementu: {name}")
            box_x = float(row[4])
            box_y = float(row[5])
            box_z = float(row[6])
            calibrated_point = recalibratePoint((detail_y,detail_x),angle_sheet,translation_mm)
            print(f"skalibrowany punkty: Y: {calibrated_point[1]},X: {calibrated_point[0]}")
            # Wartości do wysłania
            send_valueX = calibrated_point[0]
            send_valueY = calibrated_point[1]
            send_valueZ = detail_z
            send_valueAngle = angles_elements[name]

            # Formatowanie danych do wysłania
            response = f"{send_valueX:09.4f}{send_valueY:09.4f}{send_valueZ:09.4f}{send_valueAngle:09.4f}a"
            client_socket.send(response.encode('ascii'))
            print(f"Wysłano dane do ruchu A \n {response }")
            # Oczekiwanie na informację zwrotną od robota
            data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            print(f"Robot Dane: {data}")

            # System wizyjny
            # print("odpalam system wizyjny")

            crop, bounding_box,img_pack = cameraImage(BgrSubstractor_Quality,crop_values,2)
            
            try:
                curves = curveData[name]
            except KeyError:
                curves = []
            try:
                linear = linearData[name]
            except KeyError:
                linear = []
            
            gcode_data = singleGcodeElementCV2(elements[name],curves,linear,bounding_box)
            correct,RMSE,ret = linesContourCompare(crop,gcode_data)
            
            # fix do enkodowania obrazow do jsona
            _,buffer = cv2.imencode('.jpg', gcode_data['image'])
            _,buffer2 = cv2.imencode('.jpg', crop)
            photos = []
            for img in img_pack:
                _,buf = cv2.imencode('.jpg',img)
                imgbase64 = base64.b64encode(buf).decode('utf-8')
                photos.append(imgbase64)
            
            gcode_image_base64 = base64.b64encode(buffer).decode('utf-8')
            camera_image_base64 = base64.b64encode(buffer2).decode('utf-8')
            
            cv_data[name] = {
                "gcode_data": {
                    'image': gcode_image_base64,
                    "linearData": gcode_data['linearData'],
                    "circleData": gcode_data['circleData'],
                },
                "correct": correct,
                "RMSE": RMSE,
                "deformation": ret,
                "object_image": camera_image_base64,
                "palletizing_angle": send_valueAngle,
                "bonusImages":{
                    "camera_image": photos[2],
                    "MOG2_image": photos[3],
                    "object_full_image": photos[1],
                }
            }

            # data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            # print(f"Robot dane: {data}")
            # ------------- ODŁOŻENIE DETALU -------------
            # Wartości do wysłania
            # if correct:
            send_valueY = box_x
            send_valueX = box_y
            send_valueZ = box_z
            # else:
            #     #TODO DODAĆ BOXA Z BŁĘDNYMI ELEMENTAMI
            #     send_valueY = box_x
            #     send_valueX = box_y
            #     send_valueZ = box_z


            # Formatowanie danych do wysłania
            response = f"{send_valueX:09.4f}{send_valueY:09.4f}{send_valueZ:09.4f}{send_valueAngle:09.4f}b"
            client_socket.send(response.encode('ascii'))
            print(f"Wysłano dane do ruchu B: {response}")

            # Oczekiwanie na informację zwrotną od robota
            data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            print(f"Otrzymane dane: {data}")

    print("Koniec pliku CSV")
    send_valueX = 2000.0
    send_valueY = 2000.0
    send_valueZ = 2000.0
    # Formatowanie danych do wysłania
    response = f"{send_valueX:09.4f}{send_valueY:09.4f}{send_valueZ:09.4f}{send_valueAngle:09.4f}c"
    print(f"Przygotowano dane: {response}")
    client_socket.send(response.encode('ascii'))
    print(f"Wysłano dane: {response}")

    # Oczekiwanie    na informację zwrotną od robota
    data = client_socket.recv(1024).decode('utf-8', errors='ignore')
    client_socket.close()
    print("Połączenie zamknięte")
    with open(f'cv_data_{json_name}_sheetTest.json','w',encoding ='utf8') as f:
        json.dump(cv_data,f,ensure_ascii=False)

def readRobotCVJsonData(json_name):
    with open(f'{json_name}.json','r') as f:
        data = json.load(f)

    for key,value in data.items():
        if key == "sheet":
            continue

        buf_vec = {}
        for key2,value2 in value['bonusImages'].items():
            img_bytes = base64.b64decode(value2)
            img_real = cv2.imdecode(np.frombuffer(img_bytes,np.uint8),cv2.IMREAD_COLOR)
            buf_vec[key2] = img_real

        image_bytes2 = base64.b64decode(value['gcode_data']['image'])
        image_gcode = cv2.imdecode(np.frombuffer(image_bytes2, np.uint8), cv2.IMREAD_COLOR)
        print(f'Element: {key}')
        print(f'rmse : {value["RMSE"]}')
        print(f"deformation : {value['deformation']}")
        print(f'palletizing_angle : {value["palletizing_angle"]}')
        print("\n \n")
        # image_gcode = cv2.resize(image_gcode)
        cv2.imshow("gcode", image_gcode)
        for key2,value2 in buf_vec.items():
            cv2.imshow(key2,value2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def readRobotSheetCVJSONData(json_file):
    # Wczytaj dane z pliku JSON
    with open(json_file, 'r', encoding='utf8') as f:
        cv_data = json.load(f)

    # Wyodrębnij dane z JSONa
    sheet_data = cv_data['sheet']
    right_down_point = tuple(sheet_data['right_down_point'])
    right_up_point = tuple(sheet_data['right_up_point'])
    left_down_point = tuple(sheet_data['left_down_point'])
    bonus_images = sheet_data['bonusImages']

    # Punkt referencyjny
    

    # Odtwórz obrazy z Base64
    camera_image = cv2.imdecode(np.frombuffer(base64.b64decode(bonus_images['camera_image']), np.uint8), cv2.IMREAD_COLOR)
    mog2_image = cv2.imdecode(np.frombuffer(base64.b64decode(bonus_images['MOG2_image']), np.uint8), cv2.IMREAD_COLOR)
    object_full_image = cv2.imdecode(np.frombuffer(base64.b64decode(bonus_images['object_full_image']), np.uint8), cv2.IMREAD_COLOR)

    # Zaznacz punkty na obrazie MOG2
    marked_image = mog2_image.copy()
    # Zaznaczenie punktów
    cv2.circle(marked_image, right_down_point, 5, (0, 0, 255), -1)  # Czerwony
    cv2.putText(marked_image, 'Right Down', (right_down_point[0] + 10, right_down_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.circle(marked_image, right_up_point, 5, (0, 255, 0), -1)  # Zielony
    cv2.putText(marked_image, 'Right Up', (right_up_point[0] + 10, right_up_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.circle(marked_image, left_down_point, 5, (255, 0, 0), -1)  # Niebieski
    cv2.putText(marked_image, 'Left Down', (left_down_point[0] + 10, left_down_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Zaznaczenie punktu referencyjnego
    cv2.circle(marked_image, REFPOINT, 5, (0, 255, 255), -1)  # Żółty
    cv2.putText(marked_image, 'REFPOINT', (REFPOINT[0] + 10, REFPOINT[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Rysowanie linii od REFPOINT do Right Down
    cv2.line(marked_image, REFPOINT, right_down_point, (255, 255, 0), 2)  # Jasnoniebieska linia

    # Wyświetl obrazy
    cv2.imshow("Camera Image", camera_image)
    cv2.imshow("MOG2 Image with Points", marked_image)
    cv2.imshow("Object Full Image", object_full_image)

    # Poczekaj na klawisz ESC, aby zamknąć
    print("Naciśnij ESC, aby zamknąć...")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # crop,sliced_frame = get_crop_values(2) 
    # print(crop)
    # draw_circle_on_click(sliced_frame)
    main('blacha8') 
    readRobotSheetCVJSONData("cv_data_blacha8_sheetTest.json")
    readRobotCVJsonData('cv_data_blacha8_sheetTest')