# ----------- Plik wykorzystywany do komunikacji z robotem KUKA ------------ #
# Autorzy: Bartłomiej Szalwach, Maciej Mróz, Rafał Szygenda
# -------------------------------------------------------------------------- #
import base64
import socket
import csv
import json

import cv2
import orjson

import keyboard
from collections import defaultdict
from _functions_computer_vision import *

# Konfiguracja serwera
HOST = '0.0.0.0'  # Nasłuchiwanie na wszystkich interfejsach sieciowych
PORT = 59152  # Port zgodny z konfiguracją w robocie KUKA


def main(json_name):
    # Bufor pod system wizyjny
    cv_data = {}
    # crp = get_crop_values()
    crop_values = {'bottom': 38, 'left': 127, 'right': 120, 'top': 156}
    crop_values_sheet = {}
    BgrSubstractor_Quality = capture_median_frame(crop_values)
    BgrSubstractor_Sheet = capture_median_frame()
    with open(f'elements_data_json/{json_name}.json','r') as f:
        data = json.load(f)
    elements = data['elements']
    sheet_size = data['sheet_size']
    curveData = data['curveCircleData']
    linearData = data['linearPointsData']
    print("Umieść blachę w stanowisku roboczym ...")
    keyboard.wait('space')
    print("Zbieranie informacji o położeniu blachy")
    angle,translation_mm = sheetRotationTranslation(BgrSubstractor_Quality)
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
    with open('element_details.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Pominięcie nagłówka
        for row in reader:
            if not row:
                break
            # ------------- POBRANIE DETALU -------------
            detail_x = float(row[1])
            detail_y = float(row[2])
            detail_z = float(row[3])
            print(f"Odczytano dane z csv: {detail_x}, {detail_y}, {detail_z}")
            box_x = float(row[4])
            box_y = float(row[5])
            box_z = float(row[6])

            # Wartości do wysłania
            send_valueY = detail_x
            send_valueX = detail_y
            send_valueZ = detail_z

            # Formatowanie danych do wysłania
            response = f"{send_valueX:09.4f}{send_valueY:09.4f}{send_valueZ:09.4f}a"
            client_socket.send(response.encode('ascii'))
            print(f"Wysłano dane do ruchu A")
            # Oczekiwanie na informację zwrotną od robota
            data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            print(f"Robot Dane: {data}")

            # System wizyjny
            # print("odpalam system wizyjny")
            # name = row[0]
            # crop, bounding_box,_ = cameraImage(BgrSubstractor_Quality,crop_values)
            #
            # try:
            #     curves = curveData[name]
            # except KeyError:
            #     curves = []
            # try:
            #     linear = linearData[name]
            # except KeyError:
            #     linear = []
            #
            # gcode_data = singleGcodeElementCV2(elements[name],curves,linear,bounding_box)
            # correct,RMSE,ret = linesContourCompare(crop,gcode_data)
            # palletizing_angle = elementStackingRotation(cv_data,name,gcode_data['image'])
            # # temporary fix do enkodowania obrazow do jsona
            # _,buffer = cv2.imencode('.jpg', gcode_data['image'])
            # _,buffer2 = cv2.imencode('.jpg', crop)
            # gcode_image_base64 = base64.b64encode(buffer).decode('utf-8')
            # camera_image_base64 = base64.b64encode(buffer2).decode('utf-8')
            #
            # cv_data[name] = {
            #     "gcode_data": {
            #         'image': gcode_image_base64,
            #         "linearData": gcode_data['linearData'],
            #         "circleData": gcode_data['circleData'],
            #     },
            #     "correct": correct,
            #     "RMSE": RMSE,
            #     "deformation": ret,
            #     "camera_image": camera_image_base64,
            #     "palletizing_angle": palletizing_angle
            # }

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

            #TODO Dodać bazę do robota na inny stół do paletyzacji :)

            # Formatowanie danych do wysłania
            response = f"{send_valueX:09.4f}{send_valueY:09.4f}{send_valueZ:09.4f}b"
            print(f"Przygotowano dane: {response}")
            client_socket.send(response.encode('ascii'))
            print(f"Wysłano dane: {response}")

            # Oczekiwanie na informację zwrotną od robota
            data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            print(f"Otrzymane dane: {data}")

    print("Koniec pliku CSV")
    send_valueX = 2000.0
    send_valueY = 2000.0
    send_valueZ = 2000.0
    # Formatowanie danych do wysłania
    response = f"{send_valueX:09.4f}{send_valueY:09.4f}{send_valueZ:09.4f}c"
    print(f"Przygotowano dane: {response}")
    client_socket.send(response.encode('ascii'))
    print(f"Wysłano dane: {response}")

    # Oczekiwanie    na informację zwrotną od robota
    data = client_socket.recv(1024).decode('utf-8', errors='ignore')
    client_socket.close()
    print("Połączenie zamknięte")
    with open(f'cv_data_{json_name}.json','w',encoding ='utf8') as f:
        json.dump(cv_data,f,ensure_ascii=False)

def readRobotCVJsonData(json_name):
    with open(f'cv_data_{json_name}.json','r') as f:
        data = json.load(f)

    for key,value in data.items():
        image_bytes1 = base64.b64decode(value['camera_image'])
        image_bytes2 = base64.b64decode(value['gcode_data']['image'])
        image_gcode = cv2.imdecode(np.frombuffer(image_bytes2, np.uint8), cv2.IMREAD_COLOR)
        image_camera = cv2.imdecode(np.frombuffer(image_bytes1, np.uint8), cv2.IMREAD_COLOR)
        print(f'Element: {key}')
        print(f'rmse : {value["RMSE"]}')
        print(f"deformation : {value['deformation']}")
        print(f'palletizing_angle : {value["palletizing_angle"]}')
        print("\n \n")
        # image_gcode = cv2.resize(image_gcode)
        cv2.imshow("gcode", image_gcode)
        cv2.imshow("camera", image_camera)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    crop,sliced_frame = get_crop_values(1)
    draw_circle_on_click(sliced_frame)

    # main('blacha8')
    # readRobotCVJsonData('blacha8')
    #FIX
    # Domyślnie w gcode elementy maja swoj "obrot", aby uniknac trduniejszego,
    # dodać do kamery obrót obrazu o 90/180 stopni aby wyrownac obroty miedzy gcode-real image
    # do quality control włączyć światło