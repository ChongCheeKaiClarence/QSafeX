import os
import cv2
from ultralytics import YOLO

# Paths to weights and input folder
weights_person = "weights/HumanV3Dataset_18July.pt"
input_folder = "pre_cropped"
output_folder = "post_cropped"

# Load the YOLO model
model = YOLO(weights_person)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load the original image using OpenCV
        image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(image_path)

        # Predict with the model
        results = model(image_path, imgsz=640)  # predict on an image

        # Process each result
        for i, result in enumerate(results):
            # Iterate through each detected object in the result
            for j, (bbox, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
                if (x2 - x1) * (y2 - y1) < 3500:
                    cropped_image = original_image[y1 - 5:y2 + 5, x1 - 5:x2 + 5]  # Crop the image

                    # Save the cropped image
                    crop_filename = os.path.join(output_folder, f"crop_{filename}_{i}_{j}.jpg")
                    cv2.imwrite(crop_filename, cropped_image)

print("Cropping complete.")
