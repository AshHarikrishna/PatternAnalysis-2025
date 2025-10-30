# recognition/predict.py
from modules import ISICDetector

# Load the trained model
detector = ISICDetector(model_path="runs/detect/train/weights/last.pt")

# Run prediction on a test image
results = detector.predict("data/images/test/ISIC_0015158.jpg", save=True)
results.show()
