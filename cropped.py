import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

weights = "weights\safety_shoe_3Jun_3.pt"
source = "cropped_images\crop_frame1_obj0_1.jpg"

# Load a model
model = YOLO(weights)  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model
results = model(source, imgsz=160, show=True, classes=[1])  # predict on an image

for result in results:
    result.show()
    for box in result.boxes:
        conf = box.conf  # Confidence score
        class_id = int(box.cls)
        print(f"Confidence: {conf}, Class ID: {class_id}")
