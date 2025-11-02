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

        Train the YOLO model with optimal specified hyperparameters.
        
        Args:
            data_yaml (str): Path to YOLO data YAML file.
            epochs (int): Number of training epochs.
            imgsz (int): Image size for training.
            batch (int): Batch size.
            augment (bool): Whether to use data augmentation.
            lr0 (float): Initial learning rate.
            lrf (float): Final learning rate factor.
            freeze (list or None): List of layers to freeze during training.
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


