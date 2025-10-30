# prepare_yolo.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile

BASE_DIR = "/content/PatternRecognition"
IMAGES_DIR = os.path.join(BASE_DIR, "ISIC-2017_Training_Data")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
BINARY_LABELS_FILE = os.path.join(LABELS_DIR, "binary_labels.npy")

# Load labels
labels = np.load(BINARY_LABELS_FILE)

# Make folders if they don't exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)

# List all images (png/jpg)
all_images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith((".jpg", ".png"))])

# Split into train/val/test
train_imgs, test_imgs = train_test_split(all_images, test_size=0.15, random_state=42)
train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.15, random_state=42)

splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
num_classes = labels.shape[1]

for split, files in splits.items():
    for f in files:
        idx = all_images.index(f)
        img_labels = labels[idx]
        yolo_lines = [f"{class_id} 0.5 0.5 1 1" for class_id, present in enumerate(img_labels) if present]

        # Save YOLO label
        label_file = os.path.join(LABELS_DIR, split, f.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(label_file, "w") as lf:
            lf.write("\n".join(yolo_lines))

        # Copy image to split folder
        src_img = os.path.join(IMAGES_DIR, f)
        dst_img = os.path.join(IMAGES_DIR, split, f)
        if not os.path.exists(dst_img):
            copyfile(src_img, dst_img)

print("YOLO labels prepared and images split!")
