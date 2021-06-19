import numpy as np
import cv2
import os
from os import listdir

def _img(L):
    nL = []
    ML = []
    RL = []
    LL = []
    for i in L:
        t = i.split('.')[-1]
        if t == 'jpg':
            nL.append(i)
            
    Mc = np.arange(0, len(nL), 3)
    Lc = np.arange(1, len(nL), 3)
    Rc = np.arange(2, len(nL), 3)
    
    for m, l, r in zip(Mc, Lc, Rc):
        ML.append(str(m) + '.jpg')
        LL.append(str(l) + '.jpg')
        RL.append(str(r) + '.jpg')
    return ML, LL, RL

path = './valid_0'
L = listdir(path)
ML, LL, RL = _img(L)
sV = 'test.mp4'
size = (960, 640)
fps = 1

OUT_m = []
OUT_l = []
OUT_r = []

for mi, li, ri in zip(ML, LL, RL):
    fm = os.path.join(path, mi)
    fl = os.path.join(path, li)
    fr = os.path.join(path, ri)
    
    img_m = cv2.imread(fm)
    img_l = cv2.imread(fl)
    img_r = cv2.imread(fr)
    
    OUT_m.append(img_m)
    OUT_l.append(img_l)
    OUT_r.append(img_r)
    


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vd_m = cv2.VideoWriter(os.path.join(path, 'test_m.mp4'), fourcc, fps, size)
vd_l = cv2.VideoWriter(os.path.join(path, 'test_l.mp4'), fourcc, fps, size)
vd_r = cv2.VideoWriter(os.path.join(path, 'test_r.mp4'), fourcc, fps, size)

for i_m in range(len(OUT_m)):
    vd_m.write(OUT_m[i_m])
for i_l in range(len(OUT_l)):
    vd_l.write(OUT_l[i_l])
for i_r in range(len(OUT_r)):
    vd_r.write(OUT_r[i_r])    

vd_m.release()
vd_l.release()
vd_r.release()
