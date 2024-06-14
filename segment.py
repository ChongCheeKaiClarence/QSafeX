from ultralytics import YOLO

from PIL import Image

image_path = "cropped_images\crop_0_0.jpg"

# Open the image
img = Image.open(image_path)

# Get dimensions
width, height = img.size

# Calculate the crop box
crop_box = (0, int(height * 0.8), width, height)

# Crop the image
cropped_img = img.crop(crop_box)

# Load a model
model = YOLO("weights\shoe_seg_12June.pt")  # load a custom model

# Predict with the model
results = model(cropped_img, imgsz=64)  # predict on an image

for result in results:
     result.show()
     for box in result.boxes:
         conf = box.conf  # Confidence score
         class_id = int(box.cls)
         print(f"Confidence: {conf}, Class ID: {class_id}")