# -*- coding: utf-8 -*-

"""
Created on Tue Apr  2 11:54:06 2024

@author: enyandwi7@gmail.com

THIS CODE IS USED TO CONVERT SHAPEFILE TO A MULTI-CLASS GRAY IMAGE. 
AFTER WE USE GDAL RETILE CLI TO GENERATE PATCHES.    
"""

#Prepare data 
###################### Rasterise############################################ 
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import os

# Paths to your files

os.chdir('rootDir') # add the folder that contains your image and shapefile here 
image_path = "./image.tif"  # Path to your raster image
shapefile_path = "./shapefile.shp"  # Path to your shapefile
output_raster_path = "./label.tif"  # Path to save the output raster

# Load the image to get the geotransform and dimensions. The label should have the same height as the image
with rasterio.open(image_path) as src:
    image_meta = src.meta.copy()
    transform = src.transform
    width = src.width
    height = src.height

# Load the shapefile
gdf = gpd.read_file(shapefile_path)

# Create a generator for shapes and values (class attributes)
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['classvalue']))

# Rasterize the shapefile
rasterized = rasterize(
    shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,  # Background class
    dtype=np.uint8
)

# Save the rasterized data to a new file
image_meta.update({
    "count": 1,  # Single band
    "dtype": "uint8"
})

with rasterio.open(output_raster_path, 'w', **image_meta) as dst:
    dst.write(rasterized, 1)

print(f"Rasterized shapefile saved to {output_raster_path}")

#################retile#####################################################

'''
Use the GDAL command line interface (CLI) to generate a patch from the rasterised image.
Simply download and install OSGEO4W.https://trac.osgeo.org 

Locate the OSGeo4W installation folder on C and run the Windows Batch File (.bat)
Use the cd shell command to change the working directory, where your image and label are stored

Now create patches for the image and label:
gdal_retile.py -ot Byte -ps 640 640 -of PNG -targetDir "DirImage" image.tif
gdal_retile.py -ot Byte -ps 640 640 -of PNG -targetDir "DirLabel" label.tif. Here ensure both image and label are named the same

You may need to clean up the dataset by deleting patches that do not meet the standard size
'''
