'''
https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
'''

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

from util import get_keypoint_tuples, load_image_into_numpy_array, plot_detections

def main():
    pipeline_config = 'workspace/models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config'
    model_dir = 'workspace/models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-26')).expect_partial()
    
    # Detect func
    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    # LABEL MAP
    label_map_path = 'workspace/data/label_map.pbtxt'
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)

    # PREDICT
    image_path = 'D:/Downloads/Waymo/data/train_0/0.jpg'
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)


    detections, predictions_dict, shapes = detect_fn(input_tensor)

    # VIZ
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # Use keypoints if available in detections
    # keypoints, keypoint_scores = None, None
    # if 'detection_keypoints' in detections:
    #     keypoints = detections['detection_keypoints'][0].numpy()
    #     keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
    print(detections['detection_boxes'][0].numpy() * [960,960,640,640])
    plot_detections(image_np_with_detections,
                    detections['detection_boxes'][0].numpy(),
                    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                    detections['detection_scores'][0].numpy(),
                    category_index,
                    figsize=(12, 16),
                    image_name='0.jpg')
    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #     image_np_with_detections,
    #     detections['detection_boxes'][0].numpy(),
    #     (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
    #     detections['detection_scores'][0].numpy(),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     max_boxes_to_draw=200,
    #     min_score_thresh=.30,
    #     agnostic_mode=False,
    #     keypoints=keypoints,
    #     keypoint_scores=keypoint_scores,
    #     keypoint_edges=get_keypoint_tuples(configs['eval_config']))

 


if __name__ == '__main__':
    main()
