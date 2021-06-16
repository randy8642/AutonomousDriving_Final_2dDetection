import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from os import listdir
import cv2

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

def box2yoloFormat(center_x, center_y, length, width, ORIGIN_WIDTH, ORIGIN_HEIGHT):
    box_center_x = center_x / ORIGIN_WIDTH
    box_center_y = center_y / ORIGIN_HEIGHT
    box_width = length / ORIGIN_WIDTH
    box_height = width / ORIGIN_HEIGHT
    return (box_center_x, box_center_y, box_width, box_height) 

#%% Load 
Lf = os.path.join(label, data_list_label[0])
f = np.loadtxt(Lf)
Li = os.path.join(img, data_list_img[0])




  