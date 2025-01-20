import cv2
import numpy as np
import os

def process_image(image_path, output_path, new_gray_value=200):
  # Wczytaj obraz w skali szarości
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  if image is None:
    print(f"Nie udało się wczytać obrazu: {image_path}")
    return

  # Zamiana czarnych pikseli (0) na jasno szare (np. 200)
  image[image == 0] = new_gray_value

  # Zapisz zmodyfikowany obraz
  cv2.imwrite(output_path, image)
  print(f"Obraz zapisano jako {output_path}")


if __name__ == "__main__":
  photos = ["zdjecia_inzynierka_1.jpg", "zdjecia_plus_1.jpg"]
  output_dir = "output_images"
  os.makedirs(output_dir, exist_ok=True)  # Upewnij się, że folder na wyniki istnieje

  for img in photos:
    output_path = os.path.join(output_dir, f"{os.path.splitext(img)[0]}_gray.jpg")
    process_image(img, output_path)


