import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
from ultralytics import YOLOv10
import cv2

# Load a model
model = YOLO("weights\snehilsanyal-constructionn-site-safety-ppe.pt")  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model
results = model("input_media\humanDistance1.jpg", imgsz=1280, show=True, classes=[0, 5, 7 ])  # predict on an image

for result in results:
    result.show()


# # Load a model
# model = YOLOv10("weights\yolov10-ppe-roboflow-28May.pt")  # load an official model
# # model("input_media/black_screen.png")
# # model = YOLO("weights\mnist_cls.pt")  # load a custom model

# # Predict with the model
# results = model("input_media\humanDistance1.jpg", imgsz=640)  # predict on an image

# for result in results:
#     result.show()
