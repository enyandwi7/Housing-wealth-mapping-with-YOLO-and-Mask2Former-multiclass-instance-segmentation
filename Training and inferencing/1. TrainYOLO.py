# -*- coding: utf-8 -*-
"""
Created on Sat May  3 23:33:49 2025
@author: enyandwi7@gmail.com

In this code, we train our yolo model using K-Fold cross-validation instead of a train-validation split. 
"""

#K-Fold cross validation

import datetime
import shutil
from pathlib import Path
from collections import Counter

import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import KFold
import glob, os
from PIL import Image


# Move data in one folder, e.g., "annotation"

import os
import shutil
import glob

source_folder_path = "./TrainValid"
splits = ["train", "valid"]
types = ["images/*.tif", "labels/*.txt"]

TARGET_IMAGES_PATH = "./annotation/images"
TARGET_LABELS_PATH = "./annotation/labels"

os.makedirs(TARGET_IMAGES_PATH, exist_ok=True)
os.makedirs(TARGET_LABELS_PATH, exist_ok=True)

image_paths = list()
label_paths = list()

for split in splits:
    for data_type in types:
        # Use a complete path structure for glob and split
        files = glob.glob(os.path.join(source_folder_path, split, data_type))
        for file_ in files:
            try:
                if "image" in data_type:
                    shutil.copy(file_, TARGET_IMAGES_PATH)
                else:
                    shutil.copy(file_, TARGET_LABELS_PATH)
            except PermissionError as e:
                print(f"PermissionError: {e} when accessing {file_}")
            except Exception as e:
                print(f"Error: {e} when accessing {file_}")

# Store image and label paths for future use
image_paths = glob.glob(os.path.join(TARGET_IMAGES_PATH, "*.tif"))
label_paths = glob.glob(os.path.join(TARGET_LABELS_PATH, "*.txt"))
print(image_paths)

#Get classes from existing YAML file
dataset_path = Path('./TrainValid') # replace with 'path/to/dataset' for your custom data
labels = sorted(dataset_path.rglob("*labels/*.txt")) # all data in 'labels'

yaml_file = './TrainValid/data.yaml'  # your data YAML with data directories and names dictionary
with open(yaml_file, 'r', encoding="utf8") as y:
    classes = yaml.safe_load(y)['names']
cls_idx = list(range(len(classes)))
print(list(zip(classes, cls_idx)))

#Generate a DataFrame to calculate Label Distribution
#For each image, label_df contains the number of objects of each class.

indx = [l.stem for l in labels] # uses base filename as ID (no extension)
labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

for label in labels:
    lbl_counter = Counter()

    with open(label,'r') as lf:
        lines = lf.readlines()

    for l in lines:
        # classes for YOLO label uses integer at first position of each line
        lbl_counter[int(l.split(' ')[0])] += 1

    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0) # replace `nan` values with `0.0`
labels_df


#Using sklearn to create different train-val splits

ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=7)   # setting random_state for repeatable results
kfolds = list(kf.split(labels_df))

#Get label distribution in each split
folds = [f'split_{n}' for n in range(1, ksplit + 1)]
fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1E-7)
    fold_lbl_distrb.loc[f'split_{n}'] = ratio
fold_lbl_distrb

#Generating YAML and text files for K-Fold Cross-Validation
kfold_base_path = Path('./kfold')
shutil.rmtree(kfold_base_path) if kfold_base_path.is_dir() else None # Remove existing folder
os.makedirs(str(kfold_base_path)) # Create nww folder
yaml_paths = list()
train_txt_paths = list()
val_txt_paths = list()
for i, (train_idx, val_idx) in enumerate(kfolds):
    # Get image paths for train-val split
    train_paths = [image_paths[j] for j in train_idx]
    val_paths = [image_paths[j] for j in val_idx]
    # Create text files to store image paths
    train_txt = kfold_base_path / f"train_{i}.txt"
    val_txt =  kfold_base_path / f"val_{i}.txt"

    # Write images paths for training and validation in split i
    with open(str(train_txt), 'w') as f:
        f.writelines(s + '\n' for s in train_paths)
    with open(str(val_txt), 'w') as f:
        f.writelines(s + '\n' for s in val_paths)

    train_txt_paths.append(str(train_txt))
    val_txt_paths.append(str(val_txt))

    # Create yaml file
    yaml_path = kfold_base_path / f'data_{i}.yaml'
    with open(yaml_path, 'w') as ds_y:
        yaml.safe_dump({
            'train': str(train_txt.name),
            'val': str(val_txt.name),
            'names': classes
        }, ds_y)
    yaml_paths.append(str(yaml_path))
print("Yaml Paths")
print(yaml_paths)

#Show a YAML file and corresponding train image paths

yaml_path = './kfold/data_0.yaml'  # Update with your file path
train_txt_path = './kfold/train_0.txt' 
# Print the content of the YAML file
print(f"{yaml_path} File:\n")
with open(yaml_path, 'r', encoding='utf-8') as file:
    content = file.read()
    print(content)

# Print the first two lines of the text file
print("\ntrain_0.txt first two lines: \n")
with open(train_txt_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    print(''.join(lines[:2]))  # Print the first two lines

save_path = Path('./kfold/')
os.makedirs(str(save_path), exist_ok=True)
fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

##########################Now train your model ############################################################

from IPython.display import clear_output
import os
from ultralytics import YOLO

# Set working directory where training results will be stored
os.chdir('./YOLOtrained')  

# Load pretrained YOLO model
model = YOLO('yolov8n-seg.yaml').load('yolov8n-seg.pt')

batch = 4
project = 'kfold_yolo_training'
epochs = 100
results = []

# Loop through each fold
for i in range(ksplit):
    # Construct full path using os.path.join
    dataset_yaml = os.path.join('./kfold', yaml_paths[i])
    
    # Print current training fold information
    #print(f"Training for fold={i} using {dataset_yaml}")
    
    # Check if the file exists before training
    if os.path.isfile(dataset_yaml):
        try:
            # Train the model
            model.train(data=dataset_yaml, batch=batch, project=project, epochs=epochs, verbose=False)
            
            # Retrieve and store metrics
            result = model.metrics
            results.append(result)
            
        except Exception as e:
            print(f"Error during training for fold {i}: {e}")
    else:
        print(f"File not found: {dataset_yaml}")
    
    # Clear output for better readability (works in Jupyter notebooks)
    clear_output()

# Print collected results
print("Training completed. Results:", results)

metric_values = dict()

for result in results:
    for metric, metric_val in result.results_dict.items():
        if metric not in metric_values:
            metric_values[metric] = []
        metric_values[metric].append(metric_val)

metric_df = pd.DataFrame.from_dict(metric_values)
visualize_metric = ['mean', 'std', 'min', 'max']
metric_df.describe().loc[visualize_metric]

