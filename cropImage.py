import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("weights\safety_shoe_30May.pt")  # load an official model
# model("input_media/black_screen.png")
# model = YOLO("weights\mnist_cls.pt")  # load a custom model

# Predict with the model
results = model("cropped_images\crop_0_1.jpg", imgsz=640, show=True, classes=[1])  # predict on an image

for result in results:
    result.show()