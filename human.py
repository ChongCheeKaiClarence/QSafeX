import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
from ultralytics import YOLOv10
import cv2

weight = "weights\snehilsanyal-constructionn-site-safety-ppe.pt"
source = "input_media\humanDistance2.jpg"

# Load a model
model = YOLO(weight)  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model

results = model("input_media\humanDistance2.jpg", imgsz=960, show=True, classes=[5])  # predict on an image

for result in results:
    result.show()
    for box in result.boxes:
        conf = box.conf  # confidence score
        print(f"Confidence: {conf.item()}")  # Print the confidence score9420a29c46d2d438f08cc36f634caf717e6253b6

# # Load a model
# model = YOLOv10("weights\yolov10_28May_2.pt")  # load an official model
# # model("input_media/black_screen.png")s
# # model = YOLO("weights\mnist_cls.pt")  # load a custom model

# # Predict with the mode
# results = model("input_media\humanDistance1.jpg", imgsz=1280)  # predict on an image

# for result in results:
#     result.show()
