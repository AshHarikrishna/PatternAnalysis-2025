# recognition/train.py
from modules import ISICDetector
from dataset import get_data_yaml

# Create data YAML for YOLO
data_yaml = get_data_yaml(
    train_path="/Users/ashwinharikrishna/Desktop/PatternAnalysis-2025/data/images/train",
    val_path="/Users/ashwinharikrishna/Desktop/PatternAnalysis-2025/data/images/val",
    num_classes=2,
    class_names=["nevus", "melanoma"]
)


# Initialize YOLO detector
detector = ISICDetector(model_path="yolov8n.pt")

# Train (short epochs for working submission)
detector.train(data_yaml, epochs=5, imgsz=640, batch=16)
