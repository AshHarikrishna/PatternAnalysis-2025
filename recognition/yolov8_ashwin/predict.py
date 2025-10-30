# predict.py
from modules import ISICDetector

# Load trained model
detector = ISICDetector(model_path="/content/PatternRecognition/runs/detect/train/weights/last.pt")

# Example prediction
results = detector.predict("/content/PatternRecognition/ISIC-2017_Training_Data/test/ISIC_0015158.jpg", save=True)
results.show()
