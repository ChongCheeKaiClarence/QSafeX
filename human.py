import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('weights\ppe.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('cropped_images\cropped_0_3.jpg', stream=True, show=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename='output_media/cropped_0_3_test.jpg')  # save to disk
