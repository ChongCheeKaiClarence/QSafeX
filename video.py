import cv2
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("weights/snehilsanyal-constructionn-site-safety-ppe.pt")  # load your model

# Open the video file or stream
source = "input_media/Hoistlift18.mp4"

# Initialize video capture
cap = cv2.VideoCapture(source)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_media/Hoistlift18_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process the video
for results in model(source, stream=True, classes=[0, 5, 7], imgsz=3200):
    for result in results:
        # Get the original frame
        frame = result.orig_img

        # Draw bounding boxes on the frame
        frame_with_boxes = result.plot()

        # Write the frame with bounding boxes to the output video
        out.write(frame_with_boxes)

        # Display the resulting frame
        cv2.imshow('Frame', frame_with_boxes)

        # Press Q on keyboard to exit the video early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
