# from ultralytics import YOLO
# import cv2
# import os

# # Load your models
# detector = YOLO('weights\Goodweights\humans\humanv11.pt')
# classifier = YOLO('weights\shoe_cls_3Jun.pt')

# # Run object detection
# detection_results = detector('input_media\humanDistance1.jpg', imgsz=640)

# original_image = cv2.imread('input_media\humanDistance1.jpg')

# # Ensure the output directory exists
# output_dir = 'output_media'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# # Iterate over detections
# for i, result in enumerate(detection_results):
#     # Iterate through each detected object in the result
#     for j, bbox in enumerate(result.boxes.xyxy):
#         # Crop the RoI
#         x1, y1, x2, y2 = map(int, bbox)
#         roi = original_image[y1:(y1 + int((y2 - y1) * 0.8)), x1:x2]

#         # Run classification on the RoI (or on the entire image)
#         classification_results = classifier(roi, imgsz=160)
#         top1_value = classification_results.xyxy[0].probs[0].item() if len(classification_results) > 0 else None

#         # Use top1_value as needed
#         print("Top1 value:", top1_value)

#         for k, result2 in enumerate(classification_results):
#             class_name = classifier.names[result2.probs.top1]
#             # Combine results (example: draw bounding box and label)
#             cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(original_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#             if class_name in ['flip_flops', 'sandals', 'slippers']:
#                 output_path = os.path.join(output_dir, f'shoe_{i}_{j}.jpg')
#                 cv2.imwrite(output_path, roi)

# # Save the annotated image with bounding boxes and labels
# annotated_image_path = os.path.join(output_dir, 'annotated_image.jpg')
# cv2.imwrite(annotated_image_path, original_image)

                

# # Display or save the combined result
# cv2.imshow('Combined', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from ultralytics import YOLO

from PIL import Image

image_path = "cropped_images\crop_0_2.jpg"

# Open the image
img = Image.open(image_path)

# Get dimensions
width, height = img.size

# Calculate the crop box
crop_box = (0, int(height * 0.8), width, height)

# Crop the image
cropped_img = img.crop(crop_box)

# Load a model
model = YOLO("weights/footwear_cls_18June.pt")  # load a custom model

# Predict with the model
results = model(cropped_img, imgsz=128)  # predict on an image

for result in results:
    result.show()
