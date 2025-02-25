import numpy as np
import matplotlib.pyplot as plt

# Parametry wejściowe
current_position = np.array([1, 1])  # Bieżąca pozycja (X, Y)
I = 0  # Przesunięcie do środka w osi X
J = 200  # Przesunięcie do środka w osi Y
x_end = 134  # Współrzędna X punktu końcowego
y_end = 350  # Współrzędna Y punktu końcowego
direction = 'G02'  # Kierunek ruchu ('G02' dla ruchu zgodnego z ruchem wskazówek zegara, 'G03' dla przeciwnego)

# Obliczanie środka okręgu
center_x = current_position[0] + I
center_y = current_position[1] + J

# Obliczanie promienia łuku
radius = np.sqrt(I**2 + J**2)

# Obliczanie kątów początkowego i końcowego
start_angle = np.arctan2(current_position[1] - center_y, current_position[0] - center_x)
end_angle = np.arctan2(y_end - center_y, x_end - center_x)

# Korekta kątów w zależności od kierunku ruchu
if direction == 'G02':  # Ruch zgodny z ruchem wskazówek zegara
    if end_angle > start_angle:
        end_angle -= 2 * np.pi
elif direction == 'G03':  # Ruch przeciwny do ruchu wskazówek zegara
    if end_angle < start_angle:
        end_angle += 2 * np.pi

# Generowanie punktów łuku (przybliżenie łuku wieloma punktami)
num_points = 100  # Liczba punktów do przybliżenia łuku
angles = np.linspace(start_angle, end_angle, num=num_points)
arc_x = center_x + radius * np.cos(angles)
arc_y = center_y + radius * np.sin(angles)

# Wykres
plt.figure(figsize=(8, 8))
plt.rcParams['font.family'] = 'Times New Roman'

# Rysowanie łuku
plt.plot(arc_x, arc_y, color='gray', linestyle='-', marker='.', label='Punkty łuku')

# Rysowanie punktów początkowego, końcowego i środka
plt.plot(current_position[0], current_position[1], marker='o', color='black', markersize=8, linestyle='None', label='Punkt początkowy')
plt.plot(x_end, y_end, marker='s', color='black', markersize=8, linestyle='None', label='Punkt końcowy')
plt.plot(center_x, center_y, marker='x', color='black', markersize=8, linestyle='None', label='Środek okręgu')

# Dodanie linii R od punktu początkowego do środka okręgu
plt.plot([current_position[0], center_x], [current_position[1], center_y],
         color='black', linestyle=(0, (5, 10)), linewidth=2, label='Promień R')

# Dodanie etykiety dla linii R, przesuniętej nieco w prawo
mid_x = (current_position[0] + center_x) / 2
mid_y = (current_position[1] + center_y) / 2
offset = 5  # Przesunięcie etykiety w prawo (możesz dostosować wartość)
plt.text(mid_x + offset, mid_y, 'R', fontsize=12, color='black', ha='left', va='bottom')

# Ustawienia wykresu
plt.axis('equal')
plt.legend()
plt.xlabel('X [mm]')
plt.ylabel('Y [mm]')
plt.grid(True, linestyle='--', color='gray', linewidth=0.5)
plt.show()
