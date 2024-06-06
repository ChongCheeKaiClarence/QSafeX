import copy
import time
import torch
import calibrateregions
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

def ppe(source,region_of_interest):
    ## this part is to set up the yolo library with official weights, after that can run normally using custom weights
    ## run this on if u are encounterring the following : WARNING ⚠️ weights/barrel.pt appears to require 'ultralytics.yolo', which is not in ultralytics requirements.
    model = YOLO("yolov8s.pt")
    model("input_media/black_screen.png")
    #
    #

    model = YOLO("weights\snehilsanyal-constructionn-site-safety-ppe.pt")

    results = model(source=source, stream=True, show=True, imgsz=3200, classes=[0, 5, 7])
    num_ppe_class = 2
    last_saved_time = time.perf_counter()
    for result in results:

        ##box of humans
        humans_xyxy = []

        ##centreofmass of ppe
        ppe_xy = {}
        boxes = result.boxes
        xyxy, classes = calibrateregions.boxes_in_region(boxes, region_of_interest)
        ##{0: 'boots', 1: 'gloves', 2: 'helmet', 3: 'human', 4: 'vest'}
        ## filling up information for every frame
        for i in range(len(classes)):
            if int(classes[i]) == 3:
                humans_xyxy.append(xyxy[i])
            else:
                if(int(classes[i])) not in ppe_xy.keys():
                    ppe_xy[int(classes[i])]= [[(xyxy[i][0]+xyxy[i][2])/2, (xyxy[i][1]+xyxy[i][3])/2]]
                else:
                    ppe_xy[int(classes[i])].append([(xyxy[i][0]+xyxy[i][2])/2, (xyxy[i][1]+xyxy[i][3])/2])
        humans_unsafe = len(humans_xyxy)
        humans_unsafe_xyxy = copy.deepcopy(humans_xyxy)
        for xyxy in humans_xyxy:
            num_ppe_detected =0
            for k in ppe_xy.keys():
                for v in ppe_xy[k]:
                    if ( xyxy[0] <v[0] < xyxy[2] and xyxy[1] <v[1] < xyxy[3] ):
                        num_ppe_detected += 1
                        continue
            if num_ppe_detected == num_ppe_class:
                humans_unsafe -= 1
                # humans_unsafe_xyxy.remove(xyxy) guessis xyxy object type got prob
        print("number of humans not in Full PPE: ", humans_unsafe)

    return humans_unsafe
    ####### BELOW IS ARCHIVED FOR MP4 FORMAT #################
    # model = YOLO("weights/ppe.pt")
    # cap = cv2.VideoCapture(source)
    # assert cap.isOpened(), "Error reading video file"
    # w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    #
    # # Define region points
    # region_points = [(20, 0), (1280, 0), (1280, 640), (20, 640)]
    #
    # # Video writer
    # video_writer = cv2.VideoWriter("output_media/ppe_output.avi",
    #                        cv2.VideoWriter_fourcc(*'mp4v'),
    #                        fps,
    #                        (w, h))
    #
    # # Init Object Counter
    # counter = object_counter.ObjectCounter()
    # counter.set_args(view_img=True,
    #                  reg_pts=region_points,
    #                  classes_names=model.names,
    #                  draw_tracks=True)
    #
    # while cap.isOpened():
    #     success, im0 = cap.read()
    #     if not success:
    #         print("Video frame is empty or video processing has been successfully completed.")
    #         break
    #     tracks = model.track(im0, persist=True, show=False)
    #
    #     im0 = counter.start_counting(im0, tracks)
    #     video_writer.write(im0)
    #
    # cap.release()
    # video_writer.release()
    # cv2.destroyAllWindows()
