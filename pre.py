import tensorflow as tf
import os
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
import matplotlib.pyplot as plt

FILEPATH = {
    'train': './training',
    'test': './testing',
    'val': './validation',
}

def getdataSet(filepath):
    dirs = os.listdir(filepath)
    dirs = [os.path.join(filepath,dir) for dir in dirs]

    raw_dataset = tf.data.TFRecordDataset(dirs)

    return raw_dataset

dataSets = {}
for key in FILEPATH.keys():
    dataSets[key] = getdataSet(FILEPATH[key])

HEIGHT, WIDTH = (640, 960)

folder_path = {
    'train':'./train',
    'val':'./val'
}

for key in folder_path.keys():
    if not os.path.exists(folder_path[key]):
        os.mkdir(folder_path[key])

def box2yoloFormat(center_x, center_y, length, width, ORIGIN_WIDTH, ORIGIN_HEIGHT):
    box_center_x = center_x / ORIGIN_WIDTH
    box_center_y = center_y / ORIGIN_HEIGHT
    box_width = length / ORIGIN_WIDTH
    box_height = width / ORIGIN_HEIGHT

    return (box_center_x, box_center_y, box_width, box_height)

def imageDecode(image):
    
    image = tf.image.decode_jpeg(image)   

    h, w = image.shape[:2]

    image = tf.image.resize(image, (HEIGHT, WIDTH)) 
    images = tf.expand_dims(image, axis=0) / 255.0

    return images, h, w

'''
enum Name {
    UNKNOWN = 0;
    FRONT = 1;
    FRONT_LEFT = 2;
    FRONT_RIGHT = 3;
    SIDE_LEFT = 4;
    SIDE_RIGHT = 5;
  }
'''
def createImgLabel(dataSetType):
    # SAVE FILE
    n = 0
    for record in dataSets[dataSetType]:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(record.numpy()))
        
        
        
        for i in range(5):
            camera_image = frame.images[i]
            if camera_image.name in [1,2,3]:
                # IMAGE
                images, ORIGIN_HEIGHT, ORIGIN_WIDTH = imageDecode(frame.images[i].image)
                plt.imsave( f'{folder_path[dataSetType]}/{n}.jpg', images[0].numpy())    
               
                # BOUNDING BOX
                for camera_labels in frame.camera_labels:
                    # Ignore camera labels that do not correspond to this camera.
                    if camera_labels.name != camera_image.name:
                        continue
                      
                    boxs = []
                    for label in camera_labels.labels:        
                        # BOX
                        box = label.box        
                        box_bound = box2yoloFormat(box.center_x, box.center_y, box.length, box.width, ORIGIN_WIDTH, ORIGIN_HEIGHT)
                        
                        # LABEL
                        box_label = label.type

                        boxs.append((box_label, *box_bound))
                    
                    with open(f'{folder_path[dataSetType]}/{n}.txt','w') as f:
                        box_str = []
                        for box in boxs:         
                            box_str.append(' '.join(str(x) for x in box))
                        f.write('\n'.join(box_str))
                
                n = n + 1
           
    

createImgLabel('val')
