from ultralytics import YOLO
import cv2
import os
import time 

# Load your models
detector_human = YOLO('weights/ppev1.pt')
detector_shoes = YOLO('weights/safety_shoe_3Jun_3.pt')

# Open the video file
video_path = 'input_media\Human_090724_1321.mp4'
cap = cv2.VideoCapture(video_path)

# Get video information
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video writer
output_dir = 'output_media'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_video_path = os.path.join(output_dir, 'annotated_detect_video.avi')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

start_time = time.time()  # Capture the start time
# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run object detection on the frame for humans
    human_results = detector_human(frame, imgsz=1280, classes=[1])  
    
    for human_result in human_results:
        for bbox1 in human_result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox1)

            offset = 10

            if (x1 - offset) < 0:
                x1 = 0
            else:
                x1 = x1 - offset
            if (y1 - offset) < 0:
                y1 = 0
            else:
                y1 = y1 - offset
            if (x2 + offset) > frame_width:
                x2 = frame_width
            else:
                x2 = x2 + offset
            if (y2 + offset) > frame_height:
                y2 = frame_height
            else:
                y2 = y2 + offset
            
            # Crop the region of interest (human)
            roi = frame[y1:y2, x1:x2].copy()
            
            # Run shoe detection on the ROI
            shoe_results = detector_shoes(roi, imgsz=320, classes=[1], conf=0.1) 
            
            for shoe_result in shoe_results:
                for bbox2 in shoe_result.boxes.xyxy:
                    x1_shoe, y1_shoe, x2_shoe, y2_shoe = map(int, bbox2)
                    
                    # Adjust shoe bounding box coordinates relative to the whole frame
                    cv2.rectangle(frame, (x1 + x1_shoe, y1 + y1_shoe), (x1 + x2_shoe, y1 + y2_shoe), (0, 255, 0), 2)
                    cv2.putText(frame, 'shoe', (x1 + x1_shoe, y1 + y1_shoe - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # # Draw bounding box and label for the human
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, 'human', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Write the annotated frame to the output video
    out.write(frame)
    
    # Display the frame (optional)
    cv2.namedWindow("Annotated Frame", cv2.WINDOW_NORMAL)
    imS = cv2.resize(frame, (960, 540))  
    cv2.imshow('Annotated Frame', imS)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
end_time = time.time()  # Capture the end time

print(f"Processing complete. Annotated video saved at {output_video_path}")
print(end_time - start_time)  # Print the total time taken for processing
