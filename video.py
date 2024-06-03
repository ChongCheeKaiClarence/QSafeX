import cv2
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("weights\snehilsanyal-constructionn-site-safety-ppe.pt")  # load your model

# Open the video file or stream
source = "test\human_video.mp4"
vcap = cv2.VideoCapture(source)

# Check if the video capture opened successfully
if not vcap.isOpened():
    print("Error: Could not open video source.")
    exit()


while True:
    ret, frame = vcap.read()
    
    # If the frame was not grabbed, break from the loop
    if not ret:
        print("Error: Could not read frame from video source.")
        break

    # Predict with the model
    results = model(frame, imgsz=960, stream=True, classes=[5])

    # Loop through the results and draw bounding boxes
    for result in results:
        for box in result.boxes:
            # Extract box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # Extract confidence and class label
            conf = box.conf.item()
            cls = box.cls.item()
            label = f'{model.names[int(cls)]} {conf:.2f}'

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('YOLO Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
vcap.release()
cv2.destroyAllWindows()
