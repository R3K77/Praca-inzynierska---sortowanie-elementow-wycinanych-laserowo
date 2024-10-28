# ----------- Plik wykorzystywany do komunikacji z robotem KUKA ------------ #
# Autorzy: Bartłomiej Szalwach, Maciej Mróz, Rafał Szygenda
# -------------------------------------------------------------------------- #

import socket
import csv
import json
import keyboard
from collections import defaultdict
from _functions_computer_vision import *

# Konfiguracja serwera
HOST = '0.0.0.0'  # Nasłuchiwanie na wszystkich interfejsach sieciowych
PORT = 59152  # Port zgodny z konfiguracją w robocie KUKA


def main(json_name):
    # Bufor pod system wizyjny
    cv_data = {}
    crop_values = {'bottom': 0, 'left': 127, 'right': 76, 'top': 152}
    BgrSubstractor = capture_median_frame(crop_values)
    with open(f'elements_data_json/{json_name}.json','r') as f:
        data = json.load(f)
    elements = data['elements']
    sheet_size = data['sheet_size']
    curveData = data['curveCircleData']
    linearData = data['linearPointsData']
    print("Umieść blachę w stanowisku roboczym ...")
    keyboard.wait('space')
    print("Zbieranie informacji o położeniu blachy")
    angle,translation_mm = sheetRotationTranslation(BgrSubstractor)
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
            print(f"Przygotowano dane: {response}")
            client_socket.send(response.encode('ascii'))
            print(f"Wysłano dane: {response}")

            # Oczekiwanie na informację zwrotną od robota
            data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            print(f"Otrzymane dane: {data}")

            # System wizyjny
            name = row[0]
            crop, bounding_box,_ = cameraImage(BgrSubstractor,crop_values)
            gcode_data = singleGcodeElementCV2(elements[name],curveData[name],linearData[name],bounding_box)
            correct,RMSE,ret = linesContourCompare(crop,gcode_data)
            palletizing_angle = elementStackingRotation(cv_data,name,gcode_data['image'])
            cv_data[name] = {
                "gcode_data": gcode_data,
                "correct": correct,
                "RMSE": RMSE,
                "deformation": ret,
                "camera_image": crop,
                "palletizing_angle": palletizing_angle
            }
            #TODO W ROBOCIE DODAĆ WAITFOR DO ODEBRANIA KOLEJNEJ RAMKI!!!!!!!!!!!!

            # ------------- ODŁOŻENIE DETALU -------------
            print(f"Odczytano dane z csv: {box_x}, {box_y}, {box_z}")
            # Wartości do wysłania
            if correct:
                send_valueY = box_x
                send_valueX = box_y
                send_valueZ = box_z
            else:
                #TODO DODAĆ BOXA Z BŁĘDNYMI ELEMENTAMI
                send_valueY = box_x
                send_valueX = box_y
                send_valueZ = box_z

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

    # Oczekiwanie na informację zwrotną od robota
    data = client_socket.recv(1024).decode('utf-8', errors='ignore')
    client_socket.close()
    print("Połączenie zamknięte")
    with open(f'cv_data_{json_name}.json','w',encoding ='utf8') as f:
        json.dump(cv_data,f,ensure_ascii=False)


if __name__ == "__main__":
    main()
