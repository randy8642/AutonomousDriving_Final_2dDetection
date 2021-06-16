import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from os import listdir
import matplotlib.pyplot as plt
import matplotlib as matplt

#%% Path
img = './train_1_img'
label = './train_1_label'

data_list_img = listdir(img)
data_list_label = listdir(label)

#%% Para
HEIGHT, WIDTH = (640, 960)

#%% Func
def imageDecode(image):
    image = tf.image.decode_jpeg(image)   
    h, w = image.shape[:2]
    image = tf.image.resize(image, (HEIGHT, WIDTH)) 
    images = tf.expand_dims(image, axis=0) / 255.0
    return images, h, w

#%% Load 
IMG = []
LAB = []
for idx in range(len(data_list_img)):
    Lf = os.path.join(label, data_list_label[idx])
    fil = np.loadtxt(Lf)
    if fil.shape[0]!=0:
        Li = os.path.join(img, data_list_img[idx])
        img_org = matplt.image.imread(Li)
        IMG.append(img_org)
        LAB.append(fil)

sP = 'Data_1.npz'
np.savez_compressed(sP, IMG=IMG, LAB=LAB)