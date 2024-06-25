from ultralytics import YOLO
import cv2
import os

# Load your models
detector = YOLO('weights/braniv4_100epoch.pt')
classifier = YOLO('weights/footwear_cls_18June_2.pt')

# Open the video file
video_path = 'input_media\Hoistlift6.mp4'
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
output_video_path = os.path.join(output_dir, 'annotated_video.avi')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run object detection on the frame
    detection_results = detector(frame, imgsz=640, classes=[3])
    
    # Iterate over detections
    for result in detection_results:
        for bbox in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1 + int((y2 - y1) * 0.75):y2, x1:x2]

            # Run classification on the RoI (or on the entire image)
            classification_results = classifier(roi, imgsz=128)
            
            for result2 in classification_results:
                class_name = classifier.names[result2.probs.top1]
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Write the annotated frame to the output video
    out.write(frame)
    
    # Display the frame (optional)
    cv2.imshow('Annotated Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Annotated video saved at {output_video_path}")
