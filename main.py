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
from IPython.display import display, Image
import ultralytics
from roboflow import Roboflow

def main():


    # model = YOLO('weights/humanv12.pt')
    # results = model("input_media/video_2024-03-19_16-01-58.mp4", save=True)
    #
    print("main.py: main() called")
    userSelectionHandler.userSelectionHandler()

    #signalTelegram.sendMessage()
    # switcher ={
    #     1: detectionModes.UnauthorizedAccess(source=rtsp_url, model=model),
    #     2: detectionModes.counting(source=rtsp_url, model=model)
    # }
    # switcher.get(m)
if __name__ == '__main__':

    main()
# Replace the below URL with your RTSP stream URL including the username and password
# rtsp_url = 'rtsp://admin:admin@10.10.9.177:6968/h264.sdp'
# model = YOLO("weights/best(2feb).pt")
# unauthorized_access.UnauthorizedAccess(rtsp_url, model)
# mode = int(input("What detection mode would you like? \n (1) Unauthorized Access \n (2) Speeding"))
# switch(mode)

# detectionModes.UnauthorizedAccess(rtsp_url, model)




#
# detectionModes.speeding(results, line=0, names=model.names)

# # Start capturing video from the URL
# cap = cv2.VideoCapture(rtsp_url)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     results = model(frame)
#     annotated_frame = results[0].plot()
#     # Display the resulting frame
#     cv2.imshow('Frame', annotated_frame)
#
#     # Press Q on keyboard to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
