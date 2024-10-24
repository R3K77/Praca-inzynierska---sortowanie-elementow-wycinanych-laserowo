# ----------- Plik wykorzystywany do komunikacji z robotem KUKA ------------ #
# Autorzy: Bartłomiej Szalwach, Maciej Mróz, Rafał Szygenda
# -------------------------------------------------------------------------- #

import socket
import csv
import json
from collections import defaultdict
from _functions_computer_vision import *
# Konfiguracja serwera
HOST = '0.0.0.0'  # Nasłuchiwanie na wszystkich interfejsach sieciowych
PORT = 59152      # Port zgodny z konfiguracją w robocie KUKA

def main():
    # Bufor pod system wizyjny
    crop_values = {'bottom': 0, 'left': 127, 'right': 76, 'top': 152}
    # crop_values = get_crop_values()
    computer_vision_data = []
    images = defaultdict(list)
    print("Przygotowanie systemu wizyjnego")
    cutting_paths, _, _, _, _, sheet_size_line, circleLineData, linearPointsData = visualize_cutting_paths_extended(
        "NC_files/8.nc")
    median_background_frame = capture_median_frame(crop_values)
    print("Przygotowanie gotowe")
    # # Tworzenie gniazda serwera
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Serwer nasłuchuje na {HOST}:{PORT}")
    # if data:
    #     background_image = capture_median_frame()
    #     alpha, translation = sheetRotationTranslation(background_image)
    while True:
        # Akceptowanie połączenia od klienta (robota KUKA)
        client_socket, client_address = server_socket.accept()
        print(f"Połączono z {client_address}")
        try:
            # Odczyt danych z pliku CSV
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
                    element_name = row[0]
                    camera_image,bound_box_size = cameraImage(median_background_frame,crop_values)
                    gcode_data = singleGcodeElementCV2(cutting_paths[element_name],circleLineData[element_name],linearPointsData[element_name],bound_box_size)
                    is_element_correct,RMSE,_,_ = linesContourCompare(camera_image,gcode_data)
                    computer_vision_data.append({
                        "element_name": element_name,
                        "camera_image": camera_image,
                        "RMSE": RMSE,
                        "is_correct": is_element_correct,
                    })
                    images[element_name].append(camera_image)
                    images[element_name].append(gcode_data['image'])
                    # ------------- ODŁOŻENIE DETALU -------------
                    # if is_element_correct:
                    print(f"Odczytano dane z csv: {box_x}, {box_y}, {box_z}")
                    # Wartości do wysłania
                    send_valueY = box_x
                    send_valueX = box_y
                    send_valueZ = box_z
                    # else:
                    #
                    #     #TODO zastąpić boxem dla niepoprawnych elementów
                    #     send_valueY = box_x
                    #     send_valueX = box_y
                    #     send_valueZ = box_z


                    # Formatowanie danych do wysłania
                    response = f"{send_valueX:09.4f}{send_valueY:09.4f}{send_valueZ:09.4f}b"
                    print(f"Przygotowano dane: {response}")
                    client_socket.send(response.encode('ascii'))
                    print(f"Wysłano dane: {response}")

                    # Oczekiwanie na informację zwrotną od robota
                    data = client_socket.recv(1024).decode('utf-8', errors='ignore')
                    print(f"Otrzymane dane: {data}")
                
            # Sprawdzenie warunku zakończenia połączenia
            if not row:
                
                break

            # Kontynuuj dalszą część pętli

        except Exception as e:
            print(f"Wystąpił błąd: {e}")

        finally:
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
            output_file = "computer_vision_data.json"
            # Zapisywanie danych do pliku JSON
            with open(output_file, 'w') as f:
                json.dump(computer_vision_data, f, indent=4)
            for key,lst in images.items():
                for i in range(len(lst)):
                    cv2.imwrite(f"saved_images/{key}_{i}.jpg", lst[i])


if __name__ == "__main__":
    main()
    
    # 2  - otwarcie
    # 5  - zamkniecie
    
    # Dwupołożeniowy bistabilny
    
    # Włączenie ciśnienia:
    # 2 - ON
    # 5 - OFF
    
    # Wyłączenie ciśnienia:
    # 2 - OFF
    # 5 - ON
    
    
    
    #  # Wartości do wysłania
    #         send_valueX = 101.0
    #         send_valueY = 187.005

    #         # Formatowanie danych do wysłania
    #         response = f"{send_valueX:09.4f}{send_valueY:09.4f}"
    #         print(f"Przygotowano dane: {response}")
    #         client_socket.send(response.encode('ascii'))
    #         # print(f"Wysłano dane: {response}")

    #         # Odbieranie danych od robota
    #         data = client_socket.recv(1024).decode('utf-8', errors='ignore')
    #         if not data:
    #             break