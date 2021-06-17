import os
from os import listdir
import shutil

#%% Func
def _split2(dic):
    TRA = dic['train']
    VAL = dic['valid']   
    # train
    TRAL = []
    for idx_tra in TRA:
        L = listdir(idx_tra)
        for i in range(len(L)):
            f = L[i]
            ff = f.split('.')
            if ff[-1]=='jpg':
                TRAL.append('.'+ idx_tra + '/' + f) 
    with open('./waymo_object_detection/cfg/train.txt', 'w') as output:
        for row in TRAL:
            output.write(str(row) + '\n')        

    # valid
    VALL = []
    for idx_val in VAL:
        L = listdir(idx_val)
        for i in range(len(L)):
            f = L[i]
            ff = f.split('.')
            if ff[-1]=='jpg':
                VALL.append('.'+ idx_val + '/' + f) 
    with open('./waymo_object_detection/cfg/valid.txt', 'w') as output:
        for row in VALL:
            output.write(str(row) + '\n') 
    return TRAL, VALL          


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
data_dict = {
    'train':['./train_0', './train_1'],
    'valid':['./valid_0']
}
# tra = './train'
# val = './valid'

# tra_list_0 = _split(tra)
# val_list_0 = _split(val)

tra, val = _split2(data_dict)

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
    f.write(f'valid = ../waymo_object_detection/cfg/valid.txt\n')
    f.write(f'names = ../waymo_object_detection/cfg/object.names\n')
    f.write(f'backup = backup\n')

