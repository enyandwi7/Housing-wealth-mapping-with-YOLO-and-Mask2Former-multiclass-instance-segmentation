# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 12:15:36 2024

@author: enyandwi7@gmail.com

Traditional building detection methods typically focus on identifying and extracting building shapes. 
However, this process can be extended to multi-class building detection, where each building is not only localised and delineated but also assigned a specific category.
In this study, we aim to classify buildings based on perceived wealth levels using aerial imagery. 
To train such a model, multi-class labels are required.
Given a shapefile containing building footprints, our objective is to create a classified map for training. 
The approach involves extracting cropped images of rooftops, which are then categorised according to visually perceived indicators of wealth by the expert group. 
Each rooftop is assigned a unique identifier (bID) to ensure accurate tracking throughout the process.
Finally, a dataframe is generated containing each rooftop’s bID and its corresponding class label. 
This ID is then used to link class information back to the original building shapes in the shapefile. 
The resulting shapefile includes both the geometry of each building and its assigned wealth category.   

"""
""" Ensure you install ina import the necessary libraries, mainly rasterio and geopandas. 
There are many tutorials on this, such as https://www.youtube.com/watch?v=orRBc2i1joQ """

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import os
import numpy as np
# Paths to input files
shapefile_path = "./roofshapefile.shp"  # Path to the shapefile
raster_path = "./image.tif"         # Path to the raster image
output_folder = "./roofPatches"  # Folder to save the output images

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the shapefile
buildings = gpd.read_file(shapefile_path)

# Open the raster image
with rasterio.open(raster_path) as src:
    for idx, building in buildings.iterrows():
        try:
            # Ensure the 'bID' field exists. each building need to be uniquely identified
            if 'bID' not in building:
                print(f"Skipping building {idx}: 'bID' field not found.")
                continue

            # Retrieve the unique imageID
            image_id = building['bID']
            
            # Get the bounding box of the building
            minx, miny, maxx, maxy = building.geometry.bounds

            # Generate a window for cropping based on the bounding box
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            
            # Read the data within the window
            out_image = src.read(window=window)
            
            # Normalize the pixel values to 8-bit (0–255)
            out_image = np.clip(out_image, a_min=0, a_max=255)  # Ensure values are within range
            out_image = out_image.astype(np.uint8)  # Convert to 8-bit
            
            # Update metadata for the cropped image
            out_transform = src.window_transform(window)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": "uint8"  # Ensure data type is 8-bit
            })
            
            # Define the output path using the imageID
            output_path = os.path.join(output_folder, f"{image_id}.tif")
            
            # Save the cropped image
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to process building {idx} (imageID: {building.get('imageID', 'N/A')}): {e}")
