```
waymo_open_dataset_v_1_2_0
│   
└───domain_adaptation
│   
└───testing
│
└───validation
│   │   validation_0000.tar
│   │   ...
│   
└───training
    │   training_0000.tar
    │   training_0001.tar
    │   ...
```
```
train_0
│   0.jpg
|   0.txt
|   1.jpg
|   1.txt
|   ...

train_1
│   0.jpg
|   0.txt
|   ...

valid_0
│   0.jpg
|   0.txt
|   ...
```
```
Ubuntu 18.04
Nvidia Driver 440.82
Docker 19.03.8
CUDA 10.2
cudnn 7

GeForce RTX 2070 SUPER
```
```
Ubuntu 20.04 LTS
Nvidia Driver 450.80.02
Tensorflow 2.5
CUDA 11.2
cudnn 8
GPU: Tesla V100-SXM2-32GB
```
```sh
!git clone https://github.com/AlexeyAB/darknet
!apt install cmake
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
!cp ./darknet/cfg/yolov4-tiny.cfg ../waymo_object_detection/cfg/yolo-obj.cfg
```
```
Workspace
└───train_0
└───train_1
└───valid_0
└───video
|   |   valid_0_frontCenter.mp4
|   |   valid_0_frontLeft.mp4
|   |   valid_0_frontRight.mp4
|
└───darknet
|   │   chart_yolo-obj.png
|   |   Makefile
|   |   yolo_frontCenter.avi
|   |   yolo_frontLeft.avi
|   |   yolo_frontRight.avi
|   │   ...
|   |
|   └───backup
|       │   yolo-obj_best.weights
|       │   ...
|
└───waymo_object_detection
    │   
    └───visualization
    │   │   train_yolov4.log
    │   │   ...
    │   
    └───cfg
        │   object.data
        │   object.names
        │   train.txt
        │   valid.txt
        │   yolo-obj.cfg
        │   yolov4-tiny.conv.29          
        │   ...
```
```sh
!./darknet detector train ../waymo_object_detection/cfg/object.data ../waymo_object_detection/cfg/yolo-obj.cfg ../waymo_object_detection/cfg/yolov4-tiny.conv.29 >> ../waymo_object_detection/visualization/train_yolov4.log 2>&1

# !./darknet detector train <.data路徑> <模型路徑(.cfg)> <預訓練權重路徑> >> <.log路徑> 2>&1
```
```sh
!./darknet detector demo ../waymo_object_detection/cfg/object.data ../waymo_object_detection/cfg/yolo-obj.cfg ./backup/yolo-obj_best.weights ../video/valid_0_frontCenter.mp4 -out_filename yolo_frontCenter.avi -dont_show

# !./darknet detector demo <.data路徑> <模型路徑(.cfg)> <最好的權重路徑> <測試影片路徑> -out_filename <輸出影片路徑> -dont_show
```
```
Workspace
└───model_main_tf2.py
└───data
|   |   label_map.pbtxt
|   |   train_0.record
|   |   valid_0.record
|
└───darknet
|   │   pipeline.congif
|   |
|   └───pre_checkpoint
|       |   checkpoint
|       |   ckpt-0.data-00000-of-00001
|       |   ckpt-0.index

```
