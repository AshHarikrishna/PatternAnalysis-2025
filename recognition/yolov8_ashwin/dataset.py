import os
from torch.utils.data import Dataset
from PIL import Image

class LesionDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        label_path = os.path.join(self.label_dir, self.images[idx].replace(".jpg", ".txt"))
        if self.transform:
            image = self.transform(image)
        return image, label_path
