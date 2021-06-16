import os
from os import listdir
import shutil

# #%% 建立資料夾
# if not os.path.exists('waymo_object_detection'):
#     os.mkdir('waymo_object_detection')

# if not os.path.exists('waymo_object_detection/cfg'):
#     os.mkdir('waymo_object_detection/cfg') 
#     os.mkdir('waymo_object_detection/weights')

# if not os.path.exists('waymo_object_detection/cfg/face.data'):
#     shutil.copyfile('darknet/cfg/coco.data', 'waymo_object_detection/cfg/object.data')

# if not os.path.exists('waymo_object_detection/cfg/face.names'):
#     shutil.copyfile('darknet/cfg/coco.names', 'waymo_object_detection/cfg/object.names')

#%% DataList
tra = './train'
val = './valid'

tra_list = listdir(tra)
val_list = listdir(val)