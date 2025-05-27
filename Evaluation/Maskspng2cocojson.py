# -*- coding: utf-8 -*-
"""
Created on Mon May 26 10:26:09 2025

@author: enyandwi7@gmail.com

The script read png prediction masks (DT)  and check if both prediction (DT)  and ground truth image (GT) exist,
create coco json file of GT and DT.
"""

#GT
import os
import cv2
import numpy as np
import json
import pycocotools.mask as maskUtils

def get_mask_paths(mask_dir):
    return sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

def filter_mask_paths(mask_paths, valid_names):
    filtered_paths = []
    for path in mask_paths:
        if os.path.basename(path) in valid_names:
            filtered_paths.append(path)
    return filtered_paths

def generate_gt_format(mask_paths, output_file):
    annotations = []
    images = []
    ann_id = 0
    image_id = 0
    # check which value correspond to which class. You may have your value like 125, 255. they need to be set to class 1, 2
    value_to_category = {
        1: 1,  # Low
        2: 2   # High
    }

    for mask_path in mask_paths:
        image_id += 1
        file_name = os.path.basename(mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"Warning: Failed to read {mask_path}")
            continue

        images.append({
            "id": image_id,
            "file_name": file_name,
            "height": mask.shape[0],
            "width": mask.shape[1]
        })

        unique_values = np.unique(mask)

        for value in unique_values:
            if value not in value_to_category:
                continue

            category_id = value_to_category[value]
            binary_mask = (mask == value).astype(np.uint8)

            if np.sum(binary_mask) == 0:
                continue

            rle = maskUtils.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')  # JSON serializable

            bbox = maskUtils.toBbox(rle).tolist()
            area = maskUtils.area(rle).item()

            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [
           # {"id": 1, "name": "Low"},
            {"id": 2, "name": "High"}
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(coco_output, f)

def main():
    mask_dir = './mask2former/prediction'
    det_dir = './mask2former/reference'
    output_file = './mask2former/COCO_GT_JSON.json'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get base filenames from prediction folder to filter GT masks
    det_mask_names = set(os.listdir(det_dir))

    mask_paths = get_mask_paths(mask_dir)
    filtered_mask_paths = filter_mask_paths(mask_paths, det_mask_names)

    print(f"Found {len(filtered_mask_paths)} valid masks out of {len(mask_paths)} in reference.")

    generate_gt_format(filtered_mask_paths, output_file)

if __name__ == '__main__':
    main()

####prediction

import os
import cv2
import numpy as np
import json
import pycocotools.mask as maskUtils

def get_mask_paths(mask_dir):
    return sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

def generate_dt_format(mask_paths, output_file, value_to_category, default_score=0.95):
    annotations = []
    ann_id = 0

    for image_id, mask_path in enumerate(mask_paths, start=1):
        file_name = os.path.basename(mask_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"Warning: Failed to read {mask_path}")
            continue

        unique_values = np.unique(mask)

        for value in unique_values:
            if value not in value_to_category:
                continue

            category_id = value_to_category[value]
            binary_mask = (mask == value).astype(np.uint8)

            if np.sum(binary_mask) == 0:
                continue

            rle = maskUtils.encode(np.asfortranarray(binary_mask))
            rle['counts'] = rle['counts'].decode('utf-8')  # Ensure JSON serializable

            bbox = maskUtils.toBbox(rle).tolist()
            area = maskUtils.area(rle).item()

            ann_id += 1
            annotations.append({
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "score": default_score,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0
            })

    with open(output_file, 'w') as f:
        json.dump(annotations, f)

def main():
    mask_dir = './mask2former/prediction'
    output_file = './mask2former/COCO_DT_JSON.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    value_to_category = {
        1: 1,  # Low
        2: 2   # High
    }

    mask_paths = get_mask_paths(mask_dir)
    generate_dt_format(mask_paths, output_file, value_to_category)

if __name__ == '__main__':
    main()
