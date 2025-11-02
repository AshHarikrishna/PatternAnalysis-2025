# from modules import ISICDetector

# # Load trained model
# detector = ISICDetector(model_path="/content/PatternRecognition/runs/detect/train/weights/last.pt")

# # Example prediction on a test image
# results = detector.predict(
#     "/content/PatternRecognition/images/test/ISIC_0015158.jpg",  
#     save=True
# )

# results.show()

from modules import ISICDetector
import os

detector = ISICDetector(model_path="/content/PatternRecognition/runs/detect/train/weights/last.pt")

input_dir = "/content/PatternRecognition/ISIC-2017_Training_Data/ISIC-2017_Training_Data"
output_dir = "/content/PatternRecognition/yolo_detections"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    if not img_name.endswith(".jpg"):
        continue
    img_path = os.path.join(input_dir, img_name)
    results = detector.predict(img_path, save_crop=True, save_txt=False)
