import os
import cv2
from ultralytics import YOLO

# Path configurations
weights_person = "weights/braniv4_100epoch.pt"
source_video = "input_media/Hoistlift29.mp4"
weights_boots = "weights/safety_shoe_3Jun_3.pt"

# Output directory for cropped images
cropped_dir = "cropped_images"
os.makedirs(cropped_dir, exist_ok=True)

# Load the YOLO model
model = YOLO(weights_person)

# Open the video file
cap = cv2.VideoCapture(source_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Loop through each frame
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_num += 1
    print(f"Processing frame {frame_num}/{total_frames}")

    # Predict with the model on the frame
    results = model(frame, imgsz=640, classes=[3])  # Adjust imgsz as needed

    # Process each result in the frame
    for i, result in enumerate(results):
        # Iterate through each detected object in the result
        for j, (bbox, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)  # Convert to integers

            # Crop the image frame
            cropped_image = frame[y1:y2, x1:x2]

            # Save the cropped image
            crop_filename = f"{cropped_dir}/crop_frame{frame_num}_obj{i}_{j}.jpg"
            cv2.imwrite(crop_filename, cropped_image)

# Release the video capture and print completion message
cap.release()
cv2.destroyAllWindows()
print("Cropping complete.")
