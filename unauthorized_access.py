from ultralytics import YOLO
import cv2
import calibrateregions
import weights
from ultralytics.solutions import speed_estimation

def UnauthorizedAccess(source,region_of_interest):
    ## this part is to set up the yolo library with official weights, after that can run normally using custom weights
    ## run this on if u are encounterring the following : WARNING ⚠️ weights/barrel.pt appears to require 'ultralytics.yolo', which is not in ultralytics requirements.
    model = YOLO("yolov8n.pt")
    model("input_media/black_screen.png")
    #
    #

    model = YOLO("yolov8x-oiv7.pt")
    results = model(source=source, show=True, stream=True)
    breach_inAccess = False
    for result in results:
        boxes = result.boxes
        xyxy, classes = calibrateregions.boxes_in_region(boxes, region_of_interest)
        unsafe_people = 0
        boatxyxy = []
        othersxyxy = []
        for i in range(len(classes)):
            if (classes[i] == 0):
                boatxyxy.append(xyxy[i])
            else:
                othersxyxy.append(xyxy[i])
        for boats in boatxyxy:
            for others in othersxyxy:
                if others[0] > boats[0] and others[1] > boats[1] and others[2] < boats[2]:
                    unsafe_people += 1
                    breach_inAccess = True
        print(f"DANGER, {unsafe_people} of people are unsafe")
    return unsafe_people

    # if line == 0: ## horizontal speed detection
    #     line_pts = [(0, 360), (1280, 360)]
    # else: ## vertical detection
    #     line_pts = [(640, 0), (640, 720)]
    # # Init speed-estimation obj
    # speed_obj = speed_estimation.SpeedEstimator()
    # names = model.names
    # speed_obj.set_args(reg_pts=line_pts,
    #                    names=names,
    #                    view_img=True)
    # results = model.track(source, persist=True, show=True, stream = True)
    # for result in results:
    #     result = speed_obj.estimate_speed(result,results)

    # while cap.isOpened():
    #
    #     success, im0 = cap.read()
    #     if not success:
    #         print("Video frame is empty or video processing has been successfully completed.")
    #         break
    #
    #     tracks = model.track(im0, persist=True, show=False)
    #
    #     im0 = speed_obj.estimate_speed(im0, tracks)
    #     video_writer.write(im0)
