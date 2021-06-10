import tensorflow as tf
import tensorflow_datasets as tfds

traD = tfds.load('waymo_open_dataset/v1.2', split='train')