import socket
import time
import csv

# Konfiguracja serwera
HOST = '0.0.0.0'  # Nasłuchiwanie na wszystkich interfejsach sieciowych
PORT = 59152      # Port zgodny z konfiguracją w robocie KUKA

def main():
    # Tworzenie gniazda serwera
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print(f"Serwer nasłuchuje na {HOST}:{PORT}")

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
                    detail_x = float(row[0])
                    detail_y = float(row[1])
                    print(f"Odczytano dane z csv: {detail_x}, {detail_y}")
                    box_x = float(row[2])
                    box_y = float(row[3])

                    # Wartości do wysłania
                    send_valueY = detail_x
                    send_valueX = detail_y

                    # Formatowanie danych do wysłania
                    response = f"{send_valueX:09.4f}{send_valueY:09.4f}"
                    print(f"Przygotowano dane: {response}")
                    client_socket.send(response.encode('ascii'))
                    print(f"Wysłano dane: {response}")

                    # Oczekiwanie na informację zwrotną od robota
                    data = client_socket.recv(1024).decode('utf-8', errors='ignore')
                    print(f"Otrzymane dane: {data}")
                    # while data.strip() != "koniec":
                    #     data = client_socket.recv(1024).decode('utf-8', errors='ignore')
                    
                    # Sprawdzenie warunku zakończenia połączenia
                    if not row:
                        print("Koniec pliku CSV")
                        send_valueX = 2000.0
                        send_valueY = 2000.0

                        # Formatowanie danych do wysłania
                        response = f"{send_valueX:09.4f}{send_valueY:09.4f}"
                        print(f"Przygotowano dane: {response}")
                        client_socket.send(response.encode('ascii'))
                        # print(f"Wysłano dane: {response}")

                        # Oczekiwanie na informację zwrotną od robota
                        data = client_socket.recv(1024).decode('utf-8', errors='ignore')

            # Sprawdzenie warunku zakończenia połączenia
            if not row:
                break

            # Kontynuuj dalszą część pętli

        except Exception as e:
            print(f"Wystąpił błąd: {e}")

        finally:
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