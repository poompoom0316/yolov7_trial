#!/usr/bin/env bash

data_yaml_file="../../analysis/yolov7/.yaml"
cfg_yml_path="cfg/training/yolov7-e6e.yaml"
cfg_yml_path2="cfg/training/yolov7.yaml"


#python train.py --weights yolov7-e6e.pt --data ${data_yaml_file}  \
#--workers 1 --batch-size 4 --img 416 --cfg ${cfg_yml_path}  \
#--name yolov7_plantdoc --epochs 50

python train.py --weights yolov7_training.pt --data ${data_yaml_file}  \
--workers 15 --batch-size 4 --img 640 640 \
--name yolov7_custom --epochs 300 --hyp data/hyp.scratch.custom.yaml
