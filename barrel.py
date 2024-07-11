import cv2

import calibrateregions
import unauthorized_access
import ultralytics
from ultralytics import YOLO
import userSelectionHandler
import telegram
import asyncio
import signalTelegram
from IPython import display
from ultralytics import YOLO
from PIL import Image
import ultralytics

import matplotlib.pyplot as plt
import numpy as np
def lab_to_lch(lab_image):
    # Split the LAB image into L, A, and B channels
    L, A, B = cv2.split(lab_image)
    L = L.astype(np.float32)
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    # Calculate Chroma (C) and Hue (H)
    C = np.sqrt(A + B)
    H = np.arctan2(B, A)
    H = np.degrees(H)  # Convert to degrees
    H[H < 0] += 360  # Ensure hue is in the range [0, 360]
    
    # Merge L, C, and H channels back together
    lch_image = cv2.merge([L, C, H])
    return lch_image

def main():
    
    model = YOLO('weights/barrel_v2.pt')
    video_path = ("input_media/IMG_0211.MOV")
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Loop until the end of the video
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # If the frame was not successfully read, break the loop
        if not ret:
            break
        
        # Process the frame 
        
        results = model(frame , imgsz = 640, show = False, show_labels = False, show_conf = False, conf = 0.4 )
        for result in results:
            for box in result.boxes.xyxy:
                x1 , y1 , x2 ,y2  = map(int,box)
                cv2.rectangle(frame , (x1 ,y1 ) , (x2,y2) , (0,255,0), 1)
                roi = frame[y1:y2, x1:x2]
                hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # lab_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                # lch_frame = lab_to_lch(lab_frame)
                hsv_image_np = np.array(hsv_image)
                
                #If saturation Above 15, it is a good pixel
                hsv_condition =  hsv_image_np[:,:,1] > 18
                hsv_impt_values = hsv_image_np [ hsv_condition]

    
                
                hue = hsv_impt_values[:,0]
                hue = np.mod(hue , 150)
                
                mean_hue = np.mean(hue)
                
                if  mean_hue < 60 :
                    color = "red"
                else:
                    np.set_printoptions(threshold= np.inf)
                    print("hsv:",hsv_impt_values)
                    print("hue",mean_hue)
                    color = "blue"
                    

                cv2.putText(frame, color , (x1,y1) , cv2.FONT_HERSHEY_COMPLEX , 0.5 , (0,255,0) , 1)
        
        # Display the frame (optional)
        cv2.imshow('Frame', frame)
        
        # Wait for 1 ms and check if the user has pressed the 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    


    
    # model = YOLO("yolov8s.pt")
    # model("input_media/black_screen.png")
    # model = YOLO("weights/Goodweights/humans/humanv11.pt")
    # human_results = model(source="input_media/Hoistlift1.mp4" ,imgsz = 1280, save = True)
    # for human_result in human_results:
    #     # model = YOLO("weights/barrel_v2.pt")
    #     model = YOLO("weights/braniv4_100epoch.pt")
    #     orig_img = human_result.orig_img
    #     for x1, y1, x2, y2 in human_result.boxes.xyxy:
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         cropped_img = orig_img[y1:y2, x1:x2]
            
    #         ppe_results = model(cropped_img,imgsz = 128, show=True)
        #     print(model.model_name)

    # model = YOLO('weights/humanv12.pt')
    # results = model("input_media/brani_ppe_test.jpg")
    # model = YOLO('weights/braniv4_100epoch.pt')
    # results1 = model(results[0].orig_img, save = True)