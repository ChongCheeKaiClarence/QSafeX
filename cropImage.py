import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('weights/Goodweights/humans/humanv11.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('input_media/humanDistance2.jpg', stream=True, show=True)  # return a generator of Results objects

# Create a directory to store the cropped images
if not os.path.exists('cropped_images'):
    os.makedirs('cropped_images')

# Process results generator
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    for j, box in enumerate(boxes):
        box_coords = box.xyxy.int().squeeze().tolist()
        x1, y1, x2, y2 = box_coords
        img = cv2.imread('input_media/humanDistance2.jpg')
        cropped_img = img[y1:y2, x1:x2]
        cv2.imwrite(f'cropped_images/cropped_{i}_{j}.jpg', cropped_img)