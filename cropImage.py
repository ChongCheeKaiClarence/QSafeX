import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('weights/Goodweights/humans/humanv11.pt')  # pretrained YOLOv8n model

# Open the video file
cap = cv2.VideoCapture('test\human_video.mp4')

# Create a directory to store the cropped images
if not os.path.exists('cropped_images'):
    os.makedirs('cropped_images')

# Process video frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every 30th frame (1 frame per second)
    if frame_count % 30 == 0:
        # Run inference on the current frame
        results = model(frame, stream=True, show=True)

        # Process results generator
        for i, result in enumerate(results):
            boxes = result.boxes  # Boxes object for bounding box outputs
            for j, box in enumerate(boxes):
                box_coords = box.xyxy.int().squeeze().tolist()
                x1, y1, x2, y2 = box_coords
                cropped_img = frame[y1:y2, x1:x2]
                cv2.imwrite(f'cropped_images/cropped_{frame_count}_{j}.jpg', cropped_img)

    frame_count += 1

# Release the video capture
cap.release()