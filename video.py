import cv2
import os

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("weights/HumanV3Dataset_1Aug.pt")

# Open the video file
video_path = "input_media\human_290724_1244.mp4"
cap = cv2.VideoCapture(video_path)

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Output video writer
output_dir = 'output_Media'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_video_path = os.path.join(output_dir, 'annotated_detect_video.avi')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, imgsz=640, classes=[1])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
            # Display the frame (optional)
        cv2.namedWindow("Annotated Frame", cv2.WINDOW_NORMAL)
        imS = cv2.resize(annotated_frame, (960, 540))  
        cv2.imshow('Annotated Frame', imS)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break
    
    out.write(annotated_frame)

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
