* For YOLOv8, we report mAP@0.5, mAP@0.5:0.95 metrics as computed by the Ultralytics framework.
* For Mask2Former, these metrics were computed using the Boundary IoU API from https://github.com/bowenc0221/boundary-iou-api/blob/master/tools/coco_instance_evaluation.py was used

  To use the tool just do the following: 

  - Create a conda environment.
  - Activate the environment
  - Install Boundary IoU API following instruction given from source author.
    
    structure of evaluation folder:  
    boundary_iou_API/  
                                    ├── boundary_iou/  
                                    └── tools/  
  - Ensure you have both ground truth and predicted multi-class PNG masks in COCO JSON file format. use this [script](https://github.com/enyandwi7/Housing-wealth-mapping-with-YOLO-and-Mask2Former-multiclass-instance-segmentation/blob/main/Evaluation/MasksPNG2COOJSON.py)
  - Run the evaluation:  
    python tools/coco_instance_evaluation.py --gt-json-file ./COCO_GT_JSON/COCO_GT_JSON.json --dt-json-file ./COCO_DT_JSON/COCO_DT_JSON.json --iou-type bbox
