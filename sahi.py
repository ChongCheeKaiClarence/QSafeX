from pathlib import Path

from IPython.display import Image
import sahi

# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
sahi.utils.file.download_yolov8s_model(yolov8_model_path)

# Download test images
sahi.utils.file.download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)
sahi.utils.file.download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png",
    "demo_data/terrain2.png",
)

detection_model = sahi.AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cpu",  # or 'cuda:0'
)