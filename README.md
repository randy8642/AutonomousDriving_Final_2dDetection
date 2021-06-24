# 自動駕駛實務 期末專案 2D物件辨識 AutonomousDriving_Final_2dDetection
NCKU Practices of Autonomous Driving course homework

## 環境安裝
需先安裝 [Tensorflow Object detection API - install](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

## 訓練
請參考 [Tensorflow Object detection API - training the model](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model)

1. 至 [TF2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) 下載想要的模型設定檔以及預訓練參數檔
2. 將檔案放置如下
- `models`
    - `pre_checkpoint`
        - 所下載的預訓練檔(ckpt-0系列檔案)
    - `pipeline.config` (所下載的模型設定檔案)
3. 修改 `create_config.py` 並執行，以產生所需要的設定檔
4. 從安裝Object Detection API的資料夾複製出訓練檔案\
若是按照上面安裝教學，其預設位置為：\
`./TensorFlow/models/research/object_detection/model_main_tf2.py`\
複製到本專案根目錄 `./`
5. 執行檔案\
`cd /d <本專案目錄>`
    ``` 
    python model_main_tf3.py \
        --model_dir=./models \
        --pipeline_config_path=./models/pipeline.config \
        --alsologtostderr
    ```
6. 備註
    - 訓練完成後會在 `./models` 內產生 `train` 資料夾，\
    可以透過執行 `tensorboard --logdir ./models` 來查看訓練過程記錄
    
## 檔案說明
- `waymoTFrecord_decode.py`\
用於將waymo open dataset的tfrecord檔案拆解
    - 圖片：`*.jpg`
    - 標籤：`*.txt`\
    格式為bounding box的 `[label, x_center, y_center, width, height]`
- `create_config.py`\
用來產生訓練用的 `pipline.config` 設定檔
- `predict.py`\
透過 `pipline.config` 建立模型，並讀取模型權重，輸出圖片及bounding box
- `preProcess.py`\
將資料(`*.jpg`, `*.txt`)檔案選換為Tensorflow Object Detection API所適用的tfrecord檔案
- `TF Object detect API.ipynb`\
完整之jupyter notebook檔案，內有完整訓練/驗證/預測功能