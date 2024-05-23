import socket
import time

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
            # Wartości do wysłania
            send_valueX = 1234.5778
            send_valueY = 4245.4777

            # Formatowanie danych do wysłania
            response = f"{send_valueX},{send_valueY}\n"
            client_socket.send(response.encode('utf-8'))
            print(f"Wysłano dane: {response}")

            # Odbieranie danych od robota
            data = client_socket.recv(1024).decode('utf-8', errors='ignore')
            if not data:
                break
            # Przetwarzanie odebranych danych (zakładamy, że są oddzielone przecinkami)
            values = data.split(',')
            if len(values) != 2:
                print(f"Nieoczekiwany format danych: {data}")
                continue

            valueX = float(values[0])
            valueY = float(values[1])

            print(f"Odebrano dane: {valueX}, {valueY}")

        except Exception as e:
            print(f"Wystąpił błąd: {e}")

        finally:
            client_socket.close()
            print("Połączenie zamknięte")

if __name__ == "__main__":
    main()