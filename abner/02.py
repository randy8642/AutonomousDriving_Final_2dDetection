import numpy as np
import cv2
import os
from os import listdir

def _img(L):
    nL = []
    nsL = []
    for i in L:
        t = i.split('.')[-1]
        if t == 'jpg':
            nL.append(i)
    for j in range(len(nL)):
        f = str(j) + '.jpg'
        nsL.append(f)
    return nsL



path = './valid_00'
L = listdir(path)
nL = _img(L)

