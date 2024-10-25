import socket
from _functions_computer_vision import *

def communication_loop(sheet_name):
    HOST = '0.0.0.0'
    PORT = 59152

    #TODO zmienić po ustawieniu kamery przy stole ;p
    crop_values = {'bottom': 0, 'left': 127, 'right': 76, 'top': 152}
    # crop_values = get_crop_values()

    computer_vision_data = []
    with open(f'elements_data_json/{sheet_name}.json','r') as f:
        data = json.load(f)

    elements = data['elements']
    sheet_size = data['sheet_size']
    curve_data = data['curveCircleData']
    linear_data = data['linearPointsData']
    print('siema')

    #
    with open('element_details.csv','r') as f:
        line = f.readline()
        while line:
            ...



if __name__ == "__main__":
    communication_loop("blacha8")
