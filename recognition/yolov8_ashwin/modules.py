# # recognition/modules.py
# from ultralytics import YOLO

# class ISICDetector:
#     """
#     Wrapper for YOLOv8 model for ISIC lesion detection.
#     """

#     def __init__(self, model_path="yolov8n.pt"):
#         """
#         Initialize with pretrained YOLOv8 model.
#         """
#         self.model = YOLO(model_path)

#     def train(self, data_yaml, epochs=5, imgsz=640, batch=16):
#         """
#         Train the model on the dataset.
#         """
#         self.model.train(
#             data=data_yaml,
#             epochs=epochs,
#             imgsz=imgsz,
#             batch=batch
#         )

#     def predict(self, image_path, save=False):
#         """
#         Run inference on a single image.
#         """
#         results = self.model.predict(image_path)
#         if save:
#             results.save()
#         return results
# modules.py
from ultralytics import YOLO

class ISICDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            self.model = YOLO("yolov8n.pt")  # default
        else:
            self.model = YOLO(model_path)

    def train(self, data_yaml, 
              epochs=150, 
              imgsz=768, 
              batch=16, 
              augment=True, 
              lr0=0.001, 
              lrf=0.01, 
              freeze=None):
        """
        Train the YOLO model with final optimal tuning.
        """
        kwargs = {
            "data": data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "augment": augment,
            "lr0": lr0,
            "lrf": lrf
        }
        if freeze is not None:
            kwargs["freeze"] = freeze

        self.model.train(**kwargs)

    def predict(self, image_path, save=False):
        results = self.model.predict(image_path, save=save)
        return results


