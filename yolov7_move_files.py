import pandas as pd
import numpy as np
import seaborn as sns
import os
import shutil
import xml.etree.ElementTree as ET
import glob
import pandas as pd

import json


# %% Function for conversion XML to YOLO
# based on https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5
def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


# %% create folders
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

out_dir = "analysis/yolov7"

create_folder(f'{out_dir}/train/images')
create_folder(f'{out_dir}/train/labels')
create_folder(f'{out_dir}/val/images')
create_folder(f'{out_dir}/val/labels')
create_folder(f'{out_dir}/test/images')
create_folder(f'{out_dir}/test/labels')

df_train = pd.read_csv("data/xxxx/train_labels.csv")

classes = list(df_train["class"].unique())


# %% get all image files
def move_train():
    img_folder = 'data/xxxx/TRAIN'
    files = glob.glob(f"{img_folder}/*.jpg")
    pos = 0
    for f in files:
        source_img = f
        if pos < 2200:
            dest_folder = f'{out_dir}/train'
        elif (pos >= 2200):
            dest_folder = f'{out_dir}/val'
        f2 = f.split("/")[-1]

        destination_img = os.path.join(dest_folder, 'images', f2)
        shutil.copy(source_img, destination_img)

        # check for corresponding label
        label_file_basename = os.path.splitext(f)[0]
        label_source_file = f"{label_file_basename}.xml"
        label_dest_file = f"{label_file_basename.split('/')[-1]}.txt"

        # label_source_path = os.path.join('annotations', label_source_file)
        label_source_path = label_source_file
        label_dest_path = os.path.join(dest_folder, 'labels', label_dest_file)
        # if file exists, copy it to target folder
        if os.path.exists(label_source_path):
            # parse the content of the xml file
            tree = ET.parse(label_source_path)
            root = tree.getroot()
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            result = []
            for obj in root.findall('object'):
                label = obj.find("name").text
                # check for new classes and append to list
                index = classes.index(label)
                pil_bbox = [int(x.text) for x in obj.find("bndbox")]

                if (width>0)&(height>0):
                    yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
                    # convert data to string
                    bbox_string = " ".join([str(x) for x in yolo_bbox])
                    result.append(f"{index} {bbox_string}")
                    if result:
                        # generate a YOLO format text file for each xml file
                        with open(label_dest_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(result))
                else:
                    pass

        pos += 1


if __name__ == '__main__':
    move_train()
