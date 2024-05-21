import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-oiv7.pt")  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model
results = model("input_media\ppetest.png", imgsz=640)  # predict on an image

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    # result.save(filename='output_media/cropped_0_3_test.jpg')  # save to disk