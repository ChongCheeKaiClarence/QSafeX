import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

SOURCE_IMAGE_PATH = "input_media/humanDistance4.jpg"

image = cv2.imread(SOURCE_IMAGE_PATH)
model = YOLO("weights\ppe_roboflow_23May.pt")

def callback(image_slice: np.ndarray) -> sv.Detections:
    result = model(image_slice)[0]
    return sv.Detections.from_ultralytics(result)

slicer = sv.InferenceSlicer(callback = callback)

detections = slicer(image)

# Create a copy of the original image
image_with_boxes = image.copy()

# Loop through the detections
for detection in detections:
    x, y, w, h = detection[0]
    cv2.rectangle(image_with_boxes, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow("Detections", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()