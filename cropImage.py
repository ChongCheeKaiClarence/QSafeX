import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

weights_person = "weights/braniv4_100epoch.pt"
source = "input_media\Hoistlift6.mp4"
weights_boots = "weights\safety_shoe_3Jun_3.pt"

# Load the YOLO model
model = YOLO(weights_person)

# Predict with the model
results = model(source, imgsz=960, show=True, classes=[3])  # predict on an image

# Load the original image using OpenCV
original_image = cv2.imread(source)

# Create a directory to save cropped images if it doesn't exist
cropped_dir = "cropped_images"
os.makedirs(cropped_dir, exist_ok=True)

# Process each result
for i, result in enumerate(results):
    # Iterate through each detected object in the result
    for j, (bbox, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
        cropped_image = original_image[y1 + int((y2 - y1) * 0):y2, x1:x2]  # Crop the image

        # Save the cropped image
        crop_filename = f"{cropped_dir}/crop_{i}_{j}.jpg"
        cv2.imwrite(crop_filename, cropped_image)
        
print("Cropping complete.") 
