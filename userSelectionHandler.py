import counting
import ppe
import speedestimation
import unauthorized_access
import threading
from ultralytics import YOLO
import cv2
import calibrateregions
# def rtsp_required(mode):
#     if(mode==1):
#         return True
#     else:
#         return False

# def load_model(mode):
#     if(mode == 1):
#         return "weights/best(2feb).pt"
#     elif mode == 2:
#         return "weights/barrel.pt"
#     elif mode==3:
#         return "ppe.pt"
#     elif mode == 4:
#         return "best.pt" ##tobeupdated


def userSelectionHandler():


    print("userSelectionHandler.py: userSelectionHandler() called")
    source = input("What is the rtsp link? format is rtsp://admin:admin@IP:Port/h264.sdp: ")

    # source = input("What is the file path of your media?")
    if source=='test':
        source = "test\human_video.mp4"
    elif source=='rtsptest':
        source = 'rtsp://admin:admin@192.168.50.135:8080/h264.sdp' ##developer testing rtsp://admin:admin@10.10.9.177:6968/h264.sdp
    vcap = cv2.VideoCapture(source)
    ret, image = vcap.read()
    points_of_interest = calibrateregions.define_rect_all(image, 3)
    vcap.release()
    cv2.destroyAllWindows()
    print(points_of_interest)

    
    detection_thread1 = threading.Thread(target=counting.counting, args=(source, points_of_interest[0],))
    detection_thread2 = threading.Thread(target=unauthorized_access.UnauthorizedAccess, args=(source,points_of_interest[2],))
    detection_thread3 = threading.Thread(target=ppe.ppe, args=(source, points_of_interest[1],))

    # # Start the detection threads
    # detection_thread1.start()
    detection_thread2.start()
    # detection_thread3.start()


    # # Wait for the detection threads to finish
    # detection_thread1.join()
    detection_thread2.join()
    # detection_thread3.join()

    # ## call the model
    # if(mode == 1):
    #     return unauthorized_access.UnauthorizedAccess(source)
    # elif mode == 2:
    #     return counting.counting(source)
    # elif mode==3:
    #     return ppe.ppe(source)
    # elif mode == 4:
    #     return speedestimation.speedestimation(source)

