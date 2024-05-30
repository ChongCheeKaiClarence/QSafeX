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
results = model(source, imgsz=1280, show=True, classes=[5])  # predict on an image

# for result in results:
#     result.show()

original_image = cv2.imread(source)

cropped_dir = "cropped_images"
os.makedirs(cropped_dir, exist_ok=True)

# Process each result
for i, result in enumerate(results):
    # Iterate through each detected object in the result
    for j, (bbox, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
        cropped_image = original_image[y1:y2, x1:x2]  # Crop the image

        # Save the cropped image
        crop_filename = f"{cropped_dir}/crop_{i}_{j}.jpg"
        cv2.imwrite(crop_filename, cropped_image)

        # Optionally, display the cropped image using OpenCV
        # cv2.imshow(f"Cropped Image {i}_{j}", cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# # Load a model
# model = YOLOv10("weights\yolov10-ppe-roboflow-28May.pt")  # load an official model
# # model("input_media/black_screen.png")
# # model = YOLO("weights\mnist_cls.pt")  # load a custom model

# # Predict with the model
# results = model("input_media\humanDistance1.jpg", imgsz=640)  # predict on an image

# for result in results:
#     result.show()
