# train.py
from modules import ISICDetector
from dataset import get_data_yaml

# Paths updated to Colab
data_yaml = get_data_yaml(
    train_path="/content/PatternRecognition/ISIC-2017_Training_Data/train",
    val_path="/content/PatternRecognition/ISIC-2017_Training_Data/val",
    num_classes=3,
    class_names=["pigment_network", "negative_network", "milia_like_cyst"]
)

detector = ISICDetector(model_path="yolov8n.pt")

# Short epochs for testing
detector.train(data_yaml, epochs=5, imgsz=640, batch=16)
