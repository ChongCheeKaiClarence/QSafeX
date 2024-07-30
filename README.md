# Handover notes

## Resources

1. Github: https://github.com/ChongCheeKaiClarence/QSafeX/tree/master
   (if using git clone, remember to git checkout into the master branch)

2. Roboflow: All datasets are annotated using Roboflow. Log in using HTX team 2 google account.
   Other videos can be found in the HTX team 1 google drive New_Videos

3. Ultralytics: Currently, all the models deployed are using Yolov8 under the Ultralytics library.

4. Google colab notebooks: Cannot be found in HTX team 1 google drive

   1. Copy of train-yolov8-object-detection-on-custom-dataset.ipynb

      Used to train yolov8 object detection models.

   2. QSafeX video inference.ipynb

      Used to run codes locally in colab to speed up process. Generally used to create videos to test the accuracy of the object detection.

   3. Copy of zero-shot-object-detection-with-grounding-dino.ipynb

      Still in experimental phase. Currently it is able take in video input and output an annotated video. No tracking has been implemented yet.

   4. Copy of zero-shot-object-detection-with-yolo-world.ipynb

      Still in experimental phase.

   5. For more Roboflow notebooks tutorial: https://github.com/roboflow/notebooks

## Weights

### Previous weight sources (Not in Use)

#### Shoe

1. weights/safety_shoe_3Jun_2.pt -> https://universe.roboflow.com/huiyao-hu-sj18e/construction-ppe-detection/dataset/1

2. weights/shoe_cls_3Jun.pt -> https://universe.roboflow.com/newstep30/shoe-classication-kaggle

3. weights\safety_shoe_3Jun_3.pt -> https://universe.roboflow.com/ppe-detection-ik50q/alert-for-safety-violation-xzjtj/dataset/9#

4. weights\shoe_seg_10June.pt -> https://universe.roboflow.com/minor-project-i54la/shoe-segmentation-0kxvd/dataset/1#

### Current best weights(as of 29/07/24)

1. Shoe -> weights/shoe_2_23July.pt, class 2 represents boots

   Trained using dataset from our own videos and [boots and shoe images from outside sources](https://universe.roboflow.com/ppe-detection-ik50q/alert-for-safety-violation-xzjtj/dataset/9#)

   This dataset also includes safety vest and helment for better feature segregation, so that it can detect boots better.

   The actual dataset can be found [here](https://app.roboflow.com/q-team-2/shoe-7yduj/2)

2. Human -> weights/HumanV3Dataset_18July.pt

   Trained using dataset from our own videos. Actual dataset can be found [here](https://app.roboflow.com/q-team-2/humanv3dataset/9)

## Findings

1. In Roboflow, default training image size if 640x640. When handling small objects in large pictures eg. our video frames, using a larger image size like 1280x1280 can yield better accuracy.

   However, if the image is very small, applying a large image size will cause the images to upscale. This can cause important features to be distorted in the process if the upscaling is not ideal. Consider using a smaller image size like 320x320 to maintain the feature structure.

   I have tried [SAHI](https://docs.ultralytics.com/guides/sahi-tiled-inference/) for detecting small objects, but at this moment, there is no significant improvement and it does take quite a while for the detection to be finished.

2. When working on inference, you can refer to this [guide](https://docs.ultralytics.com/usage/cfg/#predict-settings)
3. When working with shoe inference, try using confidence of 0.1 or lower. The shoe in our images takes up few pixels, thus the detection will not have a high confidence in its detection. Currently, we perform human detection, crop the images, before performing shoe detection on each human. This strategy allows us to freely control the image size while reducing false positive shoe detection in the video frame.

4. SRGAN is a possible strategy to upscale images for better shoe detection. I have attempted it using [Fast-SRGAN](https://github.com/HasnainRaz/Fast-SRGAN). The attempted implementation performs human detection, followed up with Fast-SRGAN for image upscale, then performs shoe detection. This process has been tested and proven to take 10x slower as compared to without using SRGAN

   Speed test on t4 gpu, example can be found [here](https://drive.google.com/drive/folders/1eDqdsIwv8P-qbczkzeA_eALQVb4qd4g8):

   1. output*media\Hoistlift29_320_offset_xyxy*.avi -> 4848s

   2. output_media\Hoistlift29_320_offset_xyxy_SR.avi -> 43630s

5. I have done some testings with zero-shot detection using Grounding Dino and Yolov8-World. I have not done any speed test, but based on online sources, they should have similar detection speeds.

   Generally, I would prefer using Grounding Dino because it had a wider vocabulary than Yolov8-World.

   Eg. When inputting 'crane' as a class name, Grounding Dino is able to detect the Hoist lift while Yolov8-World cannot. There is a possibility that I may not have found the right class name.

   Yolov8-World can use Roboflow's Supervision library for object tracking. Need to look for trakcing library for Grounding Dino.

   As of 30/07/2024, `model.set_classes(classes)` not working in `Copy of zero-shot-object-detection-with-yolo-world.ipynb`

## Things to possibly work on

1. Experiment onn yolov8-World class names. Find keywords that can detect what the client wants.
2. Implement trackinng with Grounding Dino.
3. Try [RT-DETR](https://docs.ultralytics.com/models/rtdetr/#overview) models instead. Code structure will differ greatly from Ultralytics.
