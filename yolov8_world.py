from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld('weights/yolov8s-worldv2.pt')  # or select yolov8m/l-world.pt for different sizes

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data='datasets/lvis.yaml', epochs=100, imgsz=640)

# Execute inference with the YOLOv8s-world model on the specified image
results = model.predict('test/barrel.jpg')

# Show results
results[0].show()