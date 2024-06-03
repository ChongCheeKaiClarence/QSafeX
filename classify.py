from ultralytics import YOLO

model = YOLO("weights\shoe_cls_3Jun.pt")  # load a custom model

# Predict with the model
results = model("cropped_images\crop_0_1.jpg")  # predict on an image

for result in results:
    index = result.probs.top1
    # print(index)
    print(f"Class: {model.names[index]}")  # Print the class name and confidence score
