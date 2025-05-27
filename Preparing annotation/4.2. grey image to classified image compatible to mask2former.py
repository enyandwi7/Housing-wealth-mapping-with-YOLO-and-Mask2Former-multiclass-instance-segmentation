# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:48:52 2024

@author: enyan
Remap classes: convert grey image to classified RGB images for training mask2former
Hint: check the data used here: https://debuggercafe.com/multi-class-segmentation-using-mask2former/
"""

#Import necessary libraries 
import numpy as np
from PIL import Image
import os

# Define the color mapping according to the LABEL_COLORS_LIST (feel free to choose any color code)
LABEL_COLORS_LIST = [
                    [0, 0, 0],       # Class 0: black for background 
                    [205, 0, 0],     # Class 1: red for visibly poor house  
                    [0, 0, 205]  # Class 2: Blue for high wealth      
                    ]
def convert_tif_to_png(tif_path, png_path):
    """Convert a single TIFF image to PNG with colored labels."""
    # Read the TIFF image (assuming it's 16-bit)
    tiff_image = Image.open(tif_path)
    
    # Convert it to a NumPy array
    image_array = np.array(tiff_image)
    
    # Ensure the image has the right shape
    if image_array.ndim != 2:
        raise ValueError(f"Input TIFF image {tif_path} must be a single-channel (grayscale) image.")
    
    # Create a blank RGB image
    height, width = image_array.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Map each pixel value to its corresponding color
    for idx, color in enumerate(LABEL_COLORS_LIST):
        colored_image[image_array == idx] = color
    
    # Convert to PIL image and save as PNG
    result_image = Image.fromarray(colored_image)
    result_image.save(png_path)

def process_batch_images(input_folder, output_folder):
    """Process all TIFF images in the input folder and save as PNG in the output folder."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all TIFF files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            tif_path = os.path.join(input_folder, filename)
            png_filename = f"{os.path.splitext(filename)[0]}.png"  # Change extension to PNG
            png_path = os.path.join(output_folder, png_filename)

            try:
                convert_tif_to_png(tif_path, png_path)
                print(f"Converted {tif_path} to {png_path}")
            except Exception as e:
                print(f"Error processing {tif_path}: {e}")

# Example usage
input_folder = './input/labels'  # Path to your input TIFF images folder
output_folder = './input/masks'  # Path where you want to save the PNG images

process_batch_images(input_folder, output_folder)
