# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:07:39 2025

@author: enyan
"""


#Make prediction 
import cv2
import pandas as pd
import numpy as np
# Folder containing images
import os
os.chdir('C:/dl/viz/data/model3')
folder_path = 'C:/dl/viz/data/images'

from ultralytics import YOLO

#model= YOLO('C:/dl/yolo/final/MT3/weights/best.pt')
#model= YOLO('C:/Users/enyan/Documents/trainset/resultsyolov8batch4dataset2008/trained1/weights/best.pt')
model= YOLO('C:/dl/viz/model/combined.pt')
# Define class names according to your dataset (reversed order)
class_names = ["Low", "High"]

# Define class values and corresponding colors for the mask
class_values = {
    "Low": 1,
    "High": 2
   
}

class_colors = {
    1: (179, 179, 255),       # Dark blue for wealthier
    2: (200, 0, 0)  # Light blue for medium
       # Light pink for poor
}

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif') or filename.endswith('.TIF') or filename.endswith('.PNG') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        
        try:
            # Read the image
            orig_img = cv2.imread(image_path)
            orig_img = orig_img[:, :, :3]
            orig_img=cv2.resize(orig_img, (640, 640))
            # Perform detection
            results = model(orig_img, conf=0.4)[0]
            
            # Initialize a blank mask to store all instances
            combined_mask = np.zeros(orig_img.shape[:2], np.uint8)
            colored_mask = np.zeros_like(orig_img)
            
            # Iterate over detected instances
            for i, (mask, cls) in enumerate(zip(results.masks.data, results.boxes.cls)):
                class_id = int(cls)
                class_name = class_names[class_id]
                class_value = class_values[class_name]
                class_color = class_colors[class_value]
                
                # Debugging information
                print(f"Class ID: {class_id}, Class Name: {class_name}, Class Value: {class_value}, Class Color: {class_color}")
                
                # Create a binary mask from detection
                binary_mask = mask.cpu().numpy().astype(np.uint8)
                
                # Update the combined mask with the class value
                combined_mask[binary_mask == 1] = class_value
                
                # Apply the color to the colored mask
                colored_mask[binary_mask == 1] = class_color
                
                # Find contours
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Calculate the center of the contour to place the class name
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = contour[0][0][0], contour[0][0][1]
                    
                    # Draw the class name on the image
                    cv2.putText(orig_img, class_name, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the combined mask and colored mask for all instances
            image_name = os.path.splitext(filename)[0]
            cv2.imwrite(f'{image_name}_mask.tif', combined_mask)
            cv2.imwrite(f'{image_name}_colored_mask.png', colored_mask)
            #cv2.imwrite(f'{image_name}_labeled.png', orig_img)
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue  # Skip to the next image file if an error occurs

print("Processing complete.")
