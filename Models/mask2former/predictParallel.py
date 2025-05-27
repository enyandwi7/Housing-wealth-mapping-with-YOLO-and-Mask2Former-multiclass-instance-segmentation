import argparse
import cv2
import os
import glob
import torch
import multiprocessing
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from config import VIS_LABEL_MAP as LABEL_COLORS_LIST
from utils import draw_segmentation_map, image_overlay, predict

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to the input image directory', default='input/inference_data/images')
parser.add_argument('--device', default='cuda:0', help='Compute device, cpu or cuda')
parser.add_argument('--imgsz', default=None, type=int, nargs='+', help='Width, height')
parser.add_argument('--model', default='outputs/model_iou')
args = parser.parse_args()

# Output directory
out_dir = 'C:/dl/mtformer/validation/prediction'
os.makedirs(out_dir, exist_ok=True)

# Load model and processor once
processor = Mask2FormerImageProcessor()
model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model)
model.to(args.device).eval()

# Get all image paths
image_paths = glob.glob(os.path.join(args.input, '*'))

def process_image(image_path):
    """Processes a single image and saves the segmentation result."""
    image = cv2.imread(image_path)
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz[0], args.imgsz[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get segmentation labels
    labels = predict(model, processor, image, args.device)

    # Generate segmentation map
    seg_map = draw_segmentation_map(labels.cpu(), LABEL_COLORS_LIST)

    # Save output image
    image_name = os.path.basename(image_path)
    save_path = os.path.join(out_dir, image_name)
    cv2.imwrite(save_path, seg_map)

    #print(f"Processed: {image_name}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Ensures proper multiprocessing behavior
    num_workers = min(4, os.cpu_count())  # Use up to 4 workers (or max CPU cores available)
    
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_image, image_paths)

    print("All images processed successfully!")
