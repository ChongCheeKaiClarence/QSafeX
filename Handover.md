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

### Previous weight sources

#### shoe

    1. weights/safety_shoe_3Jun_2.pt -> https://universe.roboflow.com/huiyao-hu-sj18e/construction-ppe-detection/dataset/1

    2. weights/shoe_cls_3Jun.pt -> https://universe.roboflow.com/newstep30/shoe-classication-kaggle

    3. weights\safety_shoe_3Jun_3.pt -> https://universe.roboflow.com/ppe-detection-ik50q/alert-for-safety-violation-xzjtj/dataset/9#

    4. weights\shoe_seg_10June.pt -> https://universe.roboflow.com/minor-project-i54la/shoe-segmentation-0kxvd/dataset/1#

### Current best weights(as of 29/07/24)

    1. Shoe -> weights/shoe_2_23July.pt 

        Trained using dataset from our own videos and images from 
        [here](https://universe.roboflow.com/ppe-detection-ik50q/alert-for-safety-violation-xzjtj/dataset/9#)


    
    2. Human -> weights/HumanV3Dataset_18July.pt



