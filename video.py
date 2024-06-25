import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("weights\safety_shoe_3Jun_3.pt")

# Open the video file
video_path = "input_media\Hoistlift6.mp4"
cap = cv2.VideoCapture(video_path)

display_width = 800
display_height = 600

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, imgsz=1280, classes=[0, 5, 7])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Resize the annotated frame to fit the display window
        resized_frame = cv2.resize(annotated_frame, (display_width, display_height))

        # Display the resized frame
        cv2.imshow("YOLOv8 Tracking", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        print("Failed to read frame")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()