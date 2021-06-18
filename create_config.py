import os

from object_detection.utils import config_util


# PARAM
num_classes = 5
model_name = 'centernet_resnet50_v1_fpn_512x512_coco17_tpu-8'
model_path = 'C:/Users/randy/Documents/GitHub/AutonomousDriving_Final_2dDetection/models/'
pipeline_config = model_path + '/pipeline.config'
checkpoint_path = 'C:/Users/randy/Documents/GitHub/AutonomousDriving_Final_2dDetection/models/pre_checkpoint/ckpt-0'
labelmap_path = 'C:/Users/randy/Documents/GitHub/AutonomousDriving_Final_2dDetection/data/label_map.pbtxt'
input_path =  'C:/Users/randy/Documents/GitHub/AutonomousDriving_Final_2dDetection/data/train_0.record'
eval_path = 'C:/Users/randy/Documents/GitHub/AutonomousDriving_Final_2dDetection/data/valid_0.record'


configs = config_util.get_configs_from_pipeline_file(pipeline_config)
configs['model'].center_net.num_classes = num_classes
configs['train_config'].batch_size = 8
configs['train_config'].fine_tune_checkpoint = checkpoint_path
configs['train_config'].fine_tune_checkpoint_type = 'detection'
configs['train_config'].use_bfloat16 = False
#configs['train_config'].max_number_of_boxes = 50
#configs['train_config'].num_steps = 20000
configs['train_input_config'].label_map_path = labelmap_path
configs['train_input_config'].tf_record_input_reader.input_path[:] = [input_path]
configs['eval_input_config'].label_map_path = labelmap_path
configs['eval_input_config'].tf_record_input_reader.input_path[:] = [eval_path]

pipeline_proto = config_util.create_pipeline_proto_from_configs(configs)
config_util.save_pipeline_config(pipeline_proto, model_path)
