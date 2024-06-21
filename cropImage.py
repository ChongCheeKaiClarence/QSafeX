import copy
import time
import os
import threading
import torch
import calibrateregions
from ultralytics import YOLO
import cv2

weights_person = "weights\snehilsanyal-constructionn-site-safety-ppe.pt"
source = "input_media\HoistLift15.jpg"
weights_boots = "weights\safety_shoe_3Jun_3.pt"

# Load the YOLO model
model = YOLO(weights_person)

# Predict with the model
results = model(source, imgsz=960, show=True, classes=[5])  # predict on an image

# Load the original image using OpenCV
original_image = cv2.imread(source)

# Create a directory to save cropped images if it doesn't exist
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
        
#         model_ppe = YOLO(weights_person)
#         model_boots = YOLO(weights_boots)

#         results_ppe = model_ppe(cropped_image, imgsz=160, show=True, classes=[0, 7])
#         results_boots = model_boots(cropped_image, imgsz=192, show=True, classes=[1])

#         classes = []

#         for result_ppe in results_ppe:
#             for box in result_ppe.boxes:
#                 classes.append(box.cls.item())

#         for result_boots in results_boots:
#             for box in result_boots.boxes:
#                 classes.append(box.cls.item())
# #
#         print(crop_filename, classes)


        # # Optionally, display the cropped image using OpenCV
        # cv2.imshow(f"Cropped Image {i}_{j}", cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

print("Cropping complete.") 
