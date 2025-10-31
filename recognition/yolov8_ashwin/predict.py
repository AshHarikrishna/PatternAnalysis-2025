from modules import ISICDetector

# Load trained model
detector = ISICDetector(model_path="/content/PatternRecognition/runs/detect/train/weights/last.pt")

# Example prediction on a test image
results = detector.predict(
    "/content/PatternRecognition/images/test/ISIC_0015158.jpg",  # 
    save=True
)

results.show()
