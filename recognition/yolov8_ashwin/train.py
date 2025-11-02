# from modules import ISICDetector
# from dataset import get_data_yaml

# # Corrected paths
# data_yaml = get_data_yaml(
#     train_path="/content/PatternRecognition/images/train",
#     val_path="/content/PatternRecognition/images/val",
#     num_classes=4,
#     class_names=["pigment_network", "negative_network", "milia_like_cyst", "streaks"]
# )

# detector = ISICDetector(model_path="yolov8n.pt")

# # Run training
# detector.train(data_yaml, epochs=50, imgsz=640, batch=16)

from modules import ISICDetector
from dataset import get_data_yaml

# Corrected paths
data_yaml = get_data_yaml(
    train_path="/content/PatternRecognition/images/train",
    val_path="/content/PatternRecognition/images/val",
    num_classes=4,
    class_names=["pigment_network", "negative_network", "milia_like_cyst", "streaks"]
)

# --- TUNING CHANGES ---
# Upgrade YOLO model for better feature extraction
detector = ISICDetector(model_path="yolov8m.pt")  # was yolov8n.pt

# Run training with:
# - longer epochs
# - larger images
# - strong augmentations
# - cosine LR scheduler
# - optional freeze backbone for stability
detector.train(
    data_yaml,
    epochs=150,       # increased from 50
    imgsz=768,        # increased from 640
    batch=8,          # smaller batch due to larger image
    augment=True,     # strong augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    lr0=0.001,        # initial learning rate
    lrf=0.01,         # final LR factor for cosine scheduler
    patience=20,
    freeze=10         # optional, freeze first 10 layers
)

