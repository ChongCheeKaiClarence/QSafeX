import cv2
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("weights\snehilsanyal-constructionn-site-safety-ppe.pt")  # load your model

# Open the video file
source = "test\human_video.mp4"

vcap = cv2.VideoCapture(source)

ret = True

while ret:
    ret, image = vcap.read()

    # Predict with the model
    results = model(image, imgsz=2560, show=True, classes=[5], conf=0.4)  # predict on an image


# Release the video capture and writer objects
vcap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()