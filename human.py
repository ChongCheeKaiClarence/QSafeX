import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('weights\ppe(16May_3).pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('input_media/humanDistance1.jpg', stream=True, show=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename='output_media/humanDistance1.jpg')  # save to disk
