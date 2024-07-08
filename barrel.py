import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('weights/barrel_v2.pt')

# Open video capture
video_path = 'input_media\Barrel1.MOV'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video capture is opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create video writer
output_path = 'output_media/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Function to classify object based on average HSV color
def classify_object_hsv(roi):
    # Convert ROI to HSV color space
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate average HSV values
    h, s, v = cv2.split(roi_hsv)
    average_hue = cv2.mean(h)[0]
    average_saturation = cv2.mean(s)[0]
    average_value = cv2.mean(v)[0]
    
    # Define thresholds for red and blue in HSV space (adjust as needed)
    # Red thresholds
    lower_red = (0, 100, 100)
    upper_red = (10, 255, 255)
    
    # Blue thresholds
    lower_blue = (100, 100, 100)
    upper_blue = (130, 255, 255)
    
    # Calculate distances from average HSV values to red and blue thresholds
    dist_to_red = ((average_hue - lower_red[0]) ** 2 + (average_saturation - lower_red[1]) ** 2 + (average_value - lower_red[2]) ** 2) ** 0.5
    dist_to_blue = ((average_hue - lower_blue[0]) ** 2 + (average_saturation - lower_blue[1]) ** 2 + (average_value - lower_blue[2]) ** 2) ** 0.5
    
    # Classify based on distance
    if dist_to_red < dist_to_blue:
        return "red"
    else:
        return "blue"

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get detections from YOLO model
    results = model(frame, imgsz=640, conf=0.4)
    
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            alpha = 1
            beta = 1
            
            # Crop region of interest (ROI)
            roi = frame[y1:y2, x1:x2]  # Slightly smaller ROI to avoid edges
            
            # adjusted_roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)
            # # Calculate average color of ROI
            average_color = cv2.mean(roi)[:3]  # Extract only BGR channels
            
            # Classify based on average color
            b, g, r = average_color
            if r > b:  # Simple heuristic: classify as "red" if red component is stronger than blue
                classification = "red"
            else:
                classification = "blue"

            # classification = classify_object_hsv(roi)
            
            # Print classification result
            print(f"Object at ({x1}, {y1}) to ({x2}, {y2}) is classified as: {classification}")
            
            # Optionally, draw classification label on frame
            cv2.putText(frame, classification, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 0), 2)
    
    # Write processed frame to output video
    out.write(frame)
    
    # Display the frame (optional, for visualization during processing)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Processed video saved to: {output_path}")
