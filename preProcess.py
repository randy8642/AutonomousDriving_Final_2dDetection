import os
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import namedtuple
from object_detection.utils import dataset_util, label_map_util
from PIL import Image
import io

from util import load_image_into_numpy_array, plot_detections


label_map_dict = ''


def getExamples(folderPath):
    classes = ['TYPE_UNKNOWN', 'TYPE_VEHICLE',
               'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']
    row = []

    for txt in [f for f in os.listdir(folderPath) if f.endswith('txt')]:
        fullPath = os.path.join(folderPath, txt)

        fs = open(fullPath, mode='r')
        line = fs.readline()

        while line:

            label, x_center, y_center, w, h = line.split(' ')
            label, x_center, y_center, w, h = int(label), float(
                x_center), float(y_center), float(w), float(h)
            x_min = x_center - w/2
            y_min = y_center - h/2
            x_max = x_center + w/2
            y_max = y_center + h/2

            label = classes[label]

            filename = os.path.splitext(txt)[0] + '.jpg'
            row.append([filename, w, h, label, y_min, x_min, y_max, x_max])

            line = fs.readline()

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(row, columns=column_name)

    return df


def split(df: pd.DataFrame, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def class_text_to_int(row_label):
    global label_map_dict
    return label_map_dict[row_label]


def create_tf_example(group, folderPath: str):
    fullPath = os.path.join(folderPath, f'{group.filename}')

    with tf.io.gfile.GFile(fullPath, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    
    dataPath = 'D:/Downloads/waymo/data/valid_0'
    outputPath = './data/valid_0.record'
    labelmapPath = './data/label_map.pbtxt'

    global label_map_dict
    label_map = label_map_util.load_labelmap(labelmapPath)
    label_map_dict = label_map_util.get_label_map_dict(label_map)

    writer = tf.io.TFRecordWriter(outputPath)
    examples = getExamples(dataPath)
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, dataPath)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f'Successfully created the TFRecord file: {outputPath}')


if __name__ == '__main__':
    main()
