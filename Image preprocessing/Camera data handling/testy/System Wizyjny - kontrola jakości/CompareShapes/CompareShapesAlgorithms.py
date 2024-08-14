import math
import cv2
import numpy as np
import pandas as pd
import cmath

def FourierDescriptor(contours):

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

def FourierContourDifference(contour1,contour2):
    Fourier1 = FourierDescriptor(contour1)
    Fourier2 = FourierDescriptor(contour2)
    #TODO dokonczyć funkcję porównawczą dla Fouriera
    pass



## load figures
testfig1 = cv2.imread('testfig1.png')
testfig2 = cv2.imread('testfig2.png')

## image conversion to get contours
gr1 = cv2.cvtColor(testfig1,cv2.COLOR_BGR2GRAY)
gr2 = cv2.cvtColor(testfig2,cv2.COLOR_BGR2GRAY)

contours1, _ = cv2.findContours(gr1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(gr2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(f'kontury 1: {contours1}')
print(f'kontury 2: {contours2}')
print(f'konturylen 1: {len(contours1[0])}')
cv2.drawContours(testfig1,contours1, -1, (0,0,255),3)
cv2.drawContours(testfig2,contours2,-1 ,(0,0,255),3)

combined = cv2.vconcat([testfig1, testfig2])
cv2.imshow("combined" , combined)
cv2.waitKey(0)
