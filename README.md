# Housing wealth mapping using multiclass-building-detection -with-YOLO and-Mask2Former

Buildings can exhibit significant variation in size, geometry, construction materials, function, and economic value. Here we present a Multi-class building detection pipeline in which state-of-the-art deep learning instance segmentation models are trained not only to identify the presence of buildings in imagery or spatial data but also to classify them into distinct categories based on the above attributes. This goes beyond traditional binary building detection, which merely distinguishes between building and non-building pixels or regions to enable more granular analysis and supports a wide range of real-world applications, including urban planning, infrastructure management, disaster response, and socio-economic assessment.

***Pipeline Overview***  
This repository implements a housing wealth mapping pipeline using a combination of a few expert annotations based on visual interpretation,self-training techniques, and multiclass instance segmentation with YOLO and Mask2Former. The project encompasses three major stages: annotation preparation, model training and inference, and result visualization.

1. [PREPARING ANNOTATIONS](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/tree/main/Preparing%20annotation)  
This stage includes a multi-step process:
- **Expert Annotation via Google Forms**
  - Collection of training set, images, and shapefile of buildings. We leveraged data from our previous [work](https://link.springer.com/article/10.1007/s41064-024-00297-9) 
  - [Generating roof crops](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Preparing%20annotation/1.%20Get%20roof%20crops.py), of which a few samples were sent to experts via an online Google form for annotation. Each building polygon was assigned a unique identifier (bID) to track responses and merge class labels back into the original shapefile.
  - Experts were asked to classify each building's wealth level = Low and 2 = High.
   
 - **Self-Training Using Pseudo-Labeling**    
Due to scalability limitations in manual annotation:
  - A [self-training approach was implemented using a YOLO classifier pre-trained on ImageNet](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Preparing%20annotation/2.%20Self-Training.py).You can install YOLO using the [Ultralytics](https://docs.ultralytics.com/quickstart/) library
  - Pseudo-labels were generated for unlabeled samples with a confidence threshold of 0.9, iterated over 3 cycles.
  - Final predictions (bID, class) were saved to a CSV and merged back with the original shapefile.
    
- **Rasterisation and Label Conversion**  
  - Vector building polygons were rasterised using this [script](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Preparing%20annotation/3.%20Shapefile2Multiclass%20grey%20image%20patches.py) to create label rasters aligned with input imagery (e.g., UAV or aerial images).
  - The Retile GDAL CLI tool was used to generate image and label patches. To install GDAL just use the OSGeo4W Network Installer from [this website](https://trac.osgeo.org/osgeo4w/)
  - Annotations were converted to formats compatible with yolo and Mask2Former using scripts that [convert grey raster images to YoloFormat](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Preparing%20annotation/4.1.%20LabeledMaskImageAnnotation2YoloFormat.py) and [convert grey raster image to classified rgb image](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Preparing%20annotation/4.2.%20grey%20image%20to%20classified%20image%20compatible%20to%20mask2former.py)
  
2. [TRAINING AND INFERENCING](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/tree/main/Training%20and%20inferencing):

   
   This stage involves training and evaluating deep learning models for detection and segmentation:  
- **Environment Setup**  
  - Installed YOLO [Ultralytics](https://docs.ultralytics.com/de/quickstart/) and Mask2Former following this [tutorial](https://debuggercafe.com/multi-class-segmentation-using-mask2former/) by Sovit Ranjan Rath (2024).
  - [Training YOLO](https://github.com/Geodateningenieur-Emma/multiclass-building-detection-with-yolov8-Mask-RCNN-and-Transformers/blob/main/Training/TrainYOLO.py). The mask2former is trained by running the training script as described [here](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Training%20and%20inferencing/Train%20and%20predict%20with%20Mask2Former).
- **Inferencing**
  -Predictions were generated using:
    - [Predict-YOLO.py](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Training%20and%20inferencing/TrainYOLO.py)
    - Running the inferencing script as described  [here](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Training%20and%20inferencing/Train%20and%20predict%20with%20Mask2Former).
- **Post-Processing**
  - Predicted patches were georeferenced and mosaicked using this [notebook](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Training%20and%20inferencing/Batch%20georeferencing%20rasters.ipynb).   
  
3. [VISUALISATION](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/tree/main/Visualisation)  
   In the third stage, model outputs were visualised and spatially aggregated to assess urban wealth patterns:
In this [notebook](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Visualisation/Visualise%20with%20100-meter%20gridded%20housing%20wealth.ipynb), predictions are binned into 100x100 meter grids. Each grid is assigned a wealth score based on a weighted average of the predicted class area coverage, using weights derived from the average house values of the corresponding classes. The result is [a highly detailed spatial heatmap representing housing wealth distribution in Kigali and Musanze cities in Rwanda](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Visualisation/Housing%20wealth%202008-2022%20pattern.jpg), information that is critically needed by planners and other decision-makers of localised interventions.

4. [EVALUATION OF PREDICTED HOUSING WEALTH MAPS](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/tree/main/Evaluation).  
We report mean Average Precision (mAP) as implemented in the Ultralytics YOLOv8 framework, which corresponds to the COCO-style Average Precision (AP) across IoU thresholds from 0.50 to 0.95 used to assess mask2former. In addition using over 10 thousand residential house value samples, from official cadastre based property taxation data, we assessed the model’s effectiveness in capturing variations in housing wealth. Predicted to belong to the high-wealth class exhibited mean property values 2.5 times higher in Kigali and 1.7 times higher in Musanze than those classified as low-wealth, thereby demonstrating the model’s ability to capture spatial patterns of housing wealth. 
