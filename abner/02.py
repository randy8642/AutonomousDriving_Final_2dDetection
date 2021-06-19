import numpy as np
import cv2
import os
from os import listdir

def _img(L):
    nL = []
    for i in L:
        t = i.split('.')[-1]
        if t == 'jpg':
            nL.append(i)
    return nL

path = './valid_0'
L = listdir(path)
nL = _img(L)
nsL = sorted(nL)
