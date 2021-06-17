import os
from os import listdir
import shutil

#%% Func
def _split(path):
    JP = []
    L = listdir(path)
    datatype = path.split('./')[-1]
    for i in range(len(L)):
        f = L[i]
        ff = f.split('.')
        if ff[-1]=='jpg':
            JP.append('../'+ datatype + '/' + f)
    with open('./waymo_object_detection/cfg/' + datatype + '.txt', 'w') as output:
        for row in JP:
            output.write(str(row) + '\n')            
    return JP


#%% 建立資料夾
if not os.path.exists('waymo_object_detection'):
    os.mkdir('waymo_object_detection')

if not os.path.exists('waymo_object_detection/cfg'):
    os.mkdir('waymo_object_detection/cfg') 
    os.mkdir('waymo_object_detection/weights')

if not os.path.exists('waymo_object_detection/cfg/face.data'):
    shutil.copyfile('darknet/cfg/coco.data', 'waymo_object_detection/cfg/object.data')

if not os.path.exists('waymo_object_detection/cfg/face.names'):
    shutil.copyfile('darknet/cfg/coco.names', 'waymo_object_detection/cfg/object.names')

#%% DataList
tra = './train'
val = './valid'

tra_list = _split(tra)
val_list = _split(val)


#%% NAMES
classes = [
    'TYPE_UNKNOWN',
    'TYPE_VEHICLE',
    'TYPE_PEDESTRIAN',
    'TYPE_SIGN',
    'TYPE_CYCLIST'
]

with open('./waymo_object_detection/cfg/object.names',mode='w') as f:   
    for c in classes:
        f.write(c)
        f.write('\n')

#%% DATA
with open('./waymo_object_detection/cfg/object.data',mode='w') as f:   
    f.write(f'classes = {len(classes)}\n')
    f.write(f'train = ../waymo_object_detection/cfg/train.txt\n')
    f.write(f'valid = ../waymo_object_detection/cfg/val.txt\n')
    f.write(f'names = ../waymo_object_detection/cfg/object.names\n')
    f.write(f'backup = ../waymo_object_detection/cfg/weights\n')

