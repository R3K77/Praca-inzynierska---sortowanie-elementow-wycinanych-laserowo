from _functions_computer_vision import *
import keyboard
import cv2
import os

cv_data = {}
SHEET_CAMERA_ID = 1 #TODO zmienic na 1
SHEET_SIZE = 570
json_name = "zdjecia_plus"
path = "C:/Users/rafal/PycharmProjects/Praca-inzynierska---sortowanie-elementow-wycinanych-laserowo/Image preprocessing/Gcode to image conversion/zdjecia_obrot"

if __name__ == "__main__":

  crop_values_sheet = {'bottom': 499, 'left': 0, 'right': 380, 'top': 0}  # right 23 down 47
  print("zbieranie informacji o stanowisku roboczym")
  BgrSubstractor_Sheet = capture_median_frame(crop_values_sheet, SHEET_CAMERA_ID)
  print("Umieść blachę w stanowisku roboczym (spacja)")
  keyboard.wait('space')
  print("Zbieranie informacji o położeniu blachy")
  angle_sheet, translation_mm, sheetData = sheetRotationTranslation(BgrSubstractor_Sheet, SHEET_CAMERA_ID,
                                                                    crop_values_sheet, SHEET_SIZE)
  print(f"angle:{angle_sheet} ")
  print(f"translation: {translation_mm}")
  # photos1 = []
  # for img in sheetData[4]:
  #   _, buf = cv2.imencode('.jpg', img)
  #   imgbase64 = base64.b64encode(buf).decode('utf-8')
  #   photos1.append(imgbase64)
  # cv_data['sheet'] = {
  #   "right_down_point": [int(sheetData[2][0]), int(sheetData[2][1])],
  #   "right_up_point": [int(sheetData[1][0]), int(sheetData[1][1])],
  #   "left_down_point": [int(sheetData[0][0]), int(sheetData[0][1])],
  #   "right_side_linear_fcn": [int(sheetData[3][0]), int(sheetData[3][1]), int(sheetData[3][2])],
  #   "rotation": angle_sheet,
  #   "translation": translation_mm,
  #   "bonusImages": {
  #     "camera_image": photos1[2],
  #     "MOG2_image": photos1[3],
  #     "object_full_image": photos1[1],
  #   }
  # }
  #
  # with open(f'{json_name}.json', 'w', encoding='utf8') as f:
  #   json.dump(cv_data, f, ensure_ascii=False)

  for idx, img in enumerate(sheetData[4]):
    cv2.imwrite(os.path.join(path , f'{json_name}_{idx}.jpg'), img)
    cv2.imshow(f'Image {idx + 1}', img)

  print("Naciśnij dowolny klawisz, aby zamknąć wszystkie okna ze zdjęciami.")
  cv2.waitKey(0)
  cv2.destroyAllWindows()

