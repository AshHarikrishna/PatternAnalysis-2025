from modules import ISICDetector
from dataset import get_data_yaml

# Corrected paths
data_yaml = get_data_yaml(
    train_path="/content/PatternRecognition/images/train",
    val_path="/content/PatternRecognition/images/val",
    num_classes=3,
    class_names=["pigment_network", "negative_network", "milia_like_cyst"]
)

detector = ISICDetector(model_path="yolov8n.pt")

# Run training
detector.train(data_yaml, epochs=50, imgsz=640, batch=16)
