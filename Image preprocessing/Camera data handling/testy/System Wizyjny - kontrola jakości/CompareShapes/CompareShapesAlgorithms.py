import math
import cv2
import numpy as np
import pandas as pd
import cmath
#TODO do dokonczenia obydwa fouriery
def fourierDescriptor(contours):

    kontury = contours[0]
    #TODO dorzucić ograniczenie ilości konturów (prawdopodobnie niemożliwe XD)
    buf = [] # tutaj są zespolone
    for kontur in kontury:
        buf.append(complex(kontur[0],kontur[1]))

    N = len(buf)
    descriptors = []
    for k in range(N):
        Z_k = 0 #deskryptor czestotliwosci
        for i in range(N):
            Z_k += buf[i]*math.exp(-1j*(2*math.pi*k*i)/N)
        descriptors.append(abs(Z_k))
    return descriptors,N
def fourierContourDifference(contour1,contour2):
    Fourier1 = fourierDescriptor(contour1)
    Fourier2 = fourierDescriptor(contour2)
    #TODO dokonczyć funkcję porównawczą dla Fouriera
    pass

#Dla zestawów krawędzi obrazu sortujemy krawędzie, tak aby punkty były jak najbliżej siebie
#Sortowanie listy krawędzi, tak aby były odpowiadające sobie punkty dla dwóch obrazów.
# TODO, do wywalenia albo przebudowania
def sortByDistance(contour1, contour2, ReturnMetric = True):
    ctr1 = contour1[0]
    ctr2 = contour2[0]
    len_ctr1 = len(ctr1)
    len_ctr2 = len(ctr2)
    #Funkcja działa poprawnie jeżeli len_ctr2 > len_ctr1, wiec obsluguje różnice konturów do pętli
    if len_ctr1 != len_ctr2:
        if len_ctr1 > len_ctr2:
            diff = len_ctr1 - len_ctr2
            print(f'len_cnt1 > len_cnt2 o {diff}')
            kontury1 = ctr2
            kontury2 = ctr1
            print(f'Zamiana miejsc: \n contour1 = contour2, \n contour2 = contour1')
        else:
            diff = len_ctr2 - len_ctr1
            print(f'len_cnt1 < len_cnt2 o {diff}')
            kontury1 = ctr1
            kontury2 = ctr2
            print('kontury zostaną zrównane')
    else:
        print('kontury są równe')
        kontury1=ctr1
        kontury2=ctr2

    match_index = []
    new_cntr1 = []
    new_cntr2 = []
    distance_list = []
    for i in range(len(kontury1)):
        # print(f'i: {i}, ctr: {kontury1[i][0][0]}')
        x1 = kontury1[i][0][0]
        y1 = kontury1[i][0][1]
        index = None
        new_distance = 1000000
        for j in range(len(kontury2)):
            if j in match_index:
                continue
            else:
                x2 = kontury2[j][0][0]
                y2 = kontury2[j][0][1]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if new_distance > distance:
                    new_distance = distance
                    index = j
        distance_list.append(new_distance)
        new_cntr1.append(kontury1[i])
        match_index.append(index)
        new_cntr2.append(kontury2[index])

    print(f'dlugosc list1: {len(new_cntr1)}')
    print(f'dlugosc list2: {len(new_cntr2)}')
    if ReturnMetric:
        avg_distance = sum(distance_list)/len(distance_list)
        min_error = min(distance_list)
        normalized_err = (avg_distance - min_error)/(max(distance_list) - min_error)
        return new_cntr1,new_cntr2,avg_distance
    else:
        return new_cntr1, new_cntr2

## Przeskalowanie obrazu
#TODO
# dodanie resize obrazu bezstratnego
# dodanie parametrów resize'u aby mieć równość wielkości
#
def normalizeImage(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    kontury = contours[0].reshape(-1,2)
    max_x = max(kontury[:, 0])
    max_y = max(kontury[:, 1])
    min_x = min(kontury[:, 0])
    min_y = min(kontury[:, 1])
    print(f'max_x = {max_x}')
    print(f'min_x = {min_x}')
    print(f'max_y = {max_y}')
    print(f'min_y = {min_y}')
    cropped_image = image[min_y:max_y, min_x:max_x]
    resized_image = cv2.resize(cropped_image,(500,500))
    return resized_image

##
def compareElements(gcodeImages, gcodePoints, sheetSize, opencvImage):

    for key, value in gcodeImages.items():
        pass
        #Part 1. Porównanie kształtu elementów:
        #TODO wykorzystać algorytm

        #Part 2.1 - Porównanie długości

        #Part 2.2 - Zmierzenie odległości "błędów wycięcia"
    pass

if __name__ == "__main__":
    ## load figures
    testfig1 = cv2.imread('testfig1.png')
    testfig2 = cv2.imread('testfig1.png')
    normalized_fig1 = normalizeImage(testfig1)
    normalized_fig2 = normalizeImage(testfig2)
    ## image conversion to get contours
    gr1 = cv2.cvtColor(normalized_fig1,cv2.COLOR_BGR2GRAY)
    gr2 = cv2.cvtColor(normalized_fig2,cv2.COLOR_BGR2GRAY)
    contours1, _ = cv2.findContours(gr1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(gr2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt_list1, cnt_list2, err = SortByDistance(contours1,contours2)
    print(f'\n ---------------------------------------- \n')
    print(f'znormalizowany błąd odległości konturów: {err}')
    cv2.drawContours(normalized_fig1,contours1, -1, (0,0,255),3)
    cv2.drawContours(normalized_fig2,contours2,-1 ,(0,0,255),3)

    combined = cv2.vconcat([normalized_fig1, normalized_fig2])
    cv2.imshow("combined" , combined)
    cv2.waitKey(0)
