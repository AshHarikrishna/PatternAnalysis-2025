
from modules import ISICDetector
from dataset import get_data_yaml

# paths
data_yaml = get_data_yaml(
    train_path="/content/PatternRecognition/images/train",
    val_path="/content/PatternRecognition/images/val",
    num_classes=4,
    class_names=["pigment_network", "negative_network", "milia_like_cyst", "streaks"]
)


detector = ISICDetector(model_path="yolov8m.pt")  # was yolov8n.pt

#optimal parameters
detector.train(
    data_yaml,
    epochs=150,       
    imgsz=768,       
    batch=8,          
    augment=True,     
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    lr0=0.001,      
    lrf=0.01,        
    patience=20,
    freeze=10         
)

