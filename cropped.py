import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
from ultralytics import YOLOv10
import cv2

weights = "weights/footwear_cls_18June.pt"
source = "cropped_images\crop_0_1.jpg"

# Load a model
model = YOLO(weights)  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model
results = model(source, imgsz=128, show=True)  # predict on an image

for result in results:
    result.show()
    for box in result.boxes:
        conf = box.conf  # Confidence score
        class_id = int(box.cls)
        print(f"Confidence: {conf}, Class ID: {class_id}")
