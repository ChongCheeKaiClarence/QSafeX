import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('weights/barrel_v2.pt')

# Load the input image
image_path = 'input_media/barrel4.jpg'  # Replace with your image file path
frame = cv2.imread(image_path)

if frame is None:
    print("Error: Could not read the image.")
    exit()

# Get detections from YOLO model
results = model(frame, imgsz=640, conf=0.4)

# Process each detection in the image
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop region of interest (ROI)
        roi = frame[y1:y2, x1:x2]  # Slightly smaller ROI to avoid edges

        # Classify based on average color (using HSV classification for example)
        average_color = cv2.mean(roi)[:3]
        b, g, r = average_color
        if r > b:
            classification = "red"
        else:
            classification = "blue"
        
        # Print classification result
        print(f"Object at ({x1}, {y1}) to ({x2}, {y2}) is classified as: {classification}")
        
        # Draw classification label on frame
        cv2.putText(frame, classification, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 0), 2)

# Display the processed image
cv2.imshow('Processed Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the processed image
output_path = 'output_media/output_image.jpg'
cv2.imwrite(output_path, frame)
print(f"Processed image saved to: {output_path}")
