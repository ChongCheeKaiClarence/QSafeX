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
results = model("input_media\humanDistance4.jpg", imgsz=1280, show=True)  # predict on an image

for result in results:
    result.show()
