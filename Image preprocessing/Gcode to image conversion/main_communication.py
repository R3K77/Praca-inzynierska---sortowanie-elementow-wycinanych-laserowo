# ----------- Plik wykorzystywany do komunikacji z robotem KUKA ------------ #
# Autorzy: Bartłomiej Szalwach, Maciej Mróz, Rafał Szygenda
# -------------------------------------------------------------------------- #

import socket
import time
import csv
import os
import sys
# do importu funkcji
sys.path.append(os.path.join(os.path.dirname(__file__), '..','Camera data handling', 'testy', 'System Wizyjny - kontrola jakości', 'GcodeExtraction'))
from Element_pojedynczy import *
from gcode_analize import visualize_cutting_paths_extended
# Konfiguracja serwera
HOST = '0.0.0.0'  # Nasłuchiwanie na wszystkich interfejsach sieciowych
PORT = 59152      # Port zgodny z konfiguracją w robocie KUKA




def main():
    # Bufor pod system wizyjny
    print("Przygotowanie systemu wizyjnego")
    cutting_paths, _, _, _, _, sheet_size_line, circleLineData, linearPointsData = visualize_cutting_paths_extended(
        "NC_files/8.nc")
    print("Przygotowanie gotowe")
    # Tworzenie gniazda serwera
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Serwer nasłuchuje na {HOST}:{PORT}")

    #TODO przed startem robota powinien być ruch do docelowego punktu
    # w ktorym dzieje sie quality control zeby zebrac median_frame
    #snippet:
    client_socket, client_address = server_socket.accept()
    data = client_socket.recv(1024).decode('utf-8',errors='ignore')
    if data:
        median_background_frame = capture_median_frame()
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
                    camera_image,bound_box_size = cameraImage(median_background_frame)
                    gcode_data = singleGcodeElementCV2(cutting_paths[element_name],circleLineData[element_name],linearPointsData[element_name],bound_box_size)
                    is_element_correct = linesContourCompare(camera_image,gcode_data)

                    # ------------- ODŁOŻENIE DETALU -------------
                    if is_element_correct:
                        print(f"Odczytano dane z csv: {box_x}, {box_y}, {box_z}")
                        # Wartości do wysłania
                        send_valueY = box_x
                        send_valueX = box_y
                        send_valueZ = box_z
                    else:

                        #TODO zastąpić boxem dla niepoprawnych elementów
                        send_valueY = box_x
                        send_valueX = box_y
                        send_valueZ = box_z


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