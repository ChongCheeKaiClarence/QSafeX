import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

weights1 = "weights\safety_shoe_3Jun_3.pt"
weights2 = "weights/shoe_2_23July.pt"
source = "input_media\photo_2024-07-24_11-24-22.jpg"

# Load a model
model = YOLO(weights2)  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model
results = model(source, imgsz=320, classes=[2], conf=0.1, show=True, save=True)  # predict on an image

for result in results:
    result.show()
    for box in result.boxes:
        conf = box.conf  # Confidence score
        class_id = int(box.cls)
        print(f"Confidence: {conf}, Class ID: {class_id}")