import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
from ultralytics import YOLOv10
import cv2

weight = 'weights/ppev1.pt'
source = "input_media\humanDistance4.jpg"

# Load a model
model = YOLO(weight)  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model

results = model(source, imgsz=640, classes=[1])  # predict on an image

# using weights/snehilsanyal-constructionn-site-safety-ppe.pt,, classes=[0, 5, 7], hardhat person vest

# Iterate through results and print class names and confidence scores
for result in results:
    result.show()  # Display the results if needed
    for box in result.boxes:
        class_idx = box.cls  # Get the class index
        conf = box.conf  # Get the confidence score
        class_name = model.names[int(class_idx)]  # Map class index to class name
        print(f"Class: {class_name}, Confidence: {conf.item()}")

# # Load a model
# model = YOLOv10("weights\yolov10_28May_2.pt")  # load an official model
# # model("input_media/black_screen.png")s
# # model = YOLO("weights\mnist_cls.pt")  # load a custom model

# # Predict with the mode
# results = model("input_media\humanDistance1.jpg", imgsz=1280)  # predict on an image

# for result in results:
#     result.show()
