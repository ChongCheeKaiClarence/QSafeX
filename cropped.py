import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
from ultralytics import YOLOv10
import cv2

weights = "weights\safety_shoe_3Jun_2.pt"
source = "cropped_images\crop_0_0.jpg"

# Load a model
model = YOLO(weights)  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model
results = model(source, imgsz=160, show=True)  # predict on an image

for result in results:
    result.show()
    for box in result.boxes:
        conf = box.conf  # Confidence score
        print(f"Confidence: {conf.item()}")  # Print the class name and confidence score
