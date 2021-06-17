'''
TF Object Detect APIs
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/
https://github.com/tensorflow/models/tree/master/research/object_detection

Finetune Tutorial
https://colab.research.google.com/drive/1cfEU53yWrVXiY8yoCXZtx_LdmnNT9t1G?authuser=1#scrollTo=SQy3ND7EpFQM

https://neptune.ai/blog/how-to-train-your-own-object-detector-using-tensorflow-object-detection-api
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

from util import load_image_into_numpy_array, plot_detections


def getImage(path):
    train_images_np = []

    n = 0
    for img in [f for f in os.listdir(path) if f.endswith('jpg')]:
        fullPath = os.path.join(path, img)
        train_images_np.append(load_image_into_numpy_array(fullPath))

        n = n + 1
        if n >= 100:
            break

    return train_images_np


def getLabel(path):
    gt_boxes = []
    box_classes = []

    n = 0
    for txt in [f for f in os.listdir(path) if f.endswith('txt')]:
        fullPath = os.path.join(path, txt)

        fs = open(fullPath, mode='r')
        line = fs.readline()

        boxes = []
        cates = []
        while line:
            label, x_center, y_center, w, h = line.split(' ')
            label, x_center, y_center, w, h = int(label), float(
                x_center), float(y_center), float(w), float(h)
            x_min = x_center - w/2
            y_min = y_center - h/2
            x_max = x_center + w/2
            y_max = y_center + h/2

            box = [y_min, x_min, y_max, x_max]
            boxes.append(box)

            cates.append(label)

            line = fs.readline()

        gt_boxes.append(np.array(boxes, dtype=np.float))
        box_classes.append(np.array(cates, dtype=np.int))

        n = n + 1
        if n >= 100:
            break
    return gt_boxes, box_classes


def getCategory_index():
    classes = ['TYPE_UNKNOWN', 'TYPE_VEHICLE',
               'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']

    cate_dict = {}
    for n, cate in enumerate(classes):
        cate_dict[n] = {
            'id': n,
            'name': cate
        }

    return cate_dict


def main():
    #
    trainPath = 'D:/Downloads/Waymo/data/train_0'

    #
    train_images_np = getImage(trainPath)
    gt_boxes, labels = getLabel(trainPath)
    category_index = getCategory_index()

    #
    # give boxes a score of 100%

    # plt.figure(figsize=(30, 15))
    # for idx in range(5):
    #     dummy_scores = np.array([1.0]*box[idx].shape[0], dtype=np.float32)
    #     plt.subplot(2, 3, idx+1)
    #     plot_detections(
    #         image[idx],
    #         box[idx],
    #         label[idx],
    #         dummy_scores, category_index, image_name=f'{idx}.jpg')
    # plt.show()

    #############################################################################################################
    num_classes = 5
    label_id_offset = 1
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    for (train_image_np, gt_box_np, label) in zip(train_images_np, gt_boxes, labels):
        train_image_tensors.append(tf.expand_dims(
            tf.convert_to_tensor(train_image_np, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(
            gt_box_np, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor( label - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(
            zero_indexed_groundtruth_classes, num_classes))
    print('Done prepping data.')

    #############################################################################################################
    tf.keras.backend.clear_session()

    print('Building model and restoring weights for fine-tuning...', flush=True)
    pipeline_config = './Tensorflow/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config'
    checkpoint_path = './Tensorflow/models/research/object_detection/test_data/checkpoint/ckpt-0'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)

    #############################################################################################################
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')

    ######################################################################################
    tf.keras.backend.set_learning_phase(True)

    # These parameters can be tuned; since our training set has 5 images
    # it doesn't make sense to have a much larger batch size, though we could
    # fit more examples in memory if we wanted to.
    batch_size = 4
    learning_rate = 0.01
    num_batches = 100

    # Select variables in top layers to fine-tune.
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    # Set up forward + backward pass for a single train step.
    def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
        """Get a tf.function for training step."""

        # Use tf.function for a bit of speed.
        # Comment out the tf.function decorator if you want the inside of the
        # function to run eagerly.
        @tf.function
        def train_step_fn(image_tensors,
                            groundtruth_boxes_list,
                            groundtruth_classes_list):
            """A single training iteration.

            Args:
            image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
            groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
            groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.

            Returns:
            A scalar tensor representing the total loss for the input batch.
            """
            shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
            model.provide_groundtruth(
                groundtruth_boxes_list=groundtruth_boxes_list,
                groundtruth_classes_list=groundtruth_classes_list)
            with tf.GradientTape() as tape:
                preprocessed_images = tf.concat([detection_model.preprocess(image_tensor)[0] for image_tensor in image_tensors], axis=0)
                prediction_dict = model.predict(preprocessed_images, shapes)
                losses_dict = model.loss(prediction_dict, shapes)
                total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
                gradients = tape.gradient(total_loss, vars_to_fine_tune)
                optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
            return total_loss

        return train_step_fn

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(
        detection_model, optimizer, to_fine_tune)

    print('Start fine-tuning!', flush=True)
    for idx in range(num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_images_np)))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]

        # Note that we do not do data augmentation in this demo.  If you want a
        # a fun exercise, we recommend experimenting with random horizontal flipping
        # and random cropping :)
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors = [train_image_tensors[key] for key in example_keys]

        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)

        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(num_batches)
            + ', loss=' +  str(total_loss.numpy()), flush=True)

    print('Done fine-tuning!')


if __name__ == '__main__':
    main()
