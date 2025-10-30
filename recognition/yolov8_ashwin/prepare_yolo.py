# prepare_yolo.py
import os
import numpy as np
from shutil import copyfile

BASE_DIR = "/content/PatternRecognition"
IMAGES_DIR = os.path.join(BASE_DIR, "ISIC-2017_Training_Data")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
BINARY_LABELS_FILE = os.path.join(LABELS_DIR, "binary_labels.npy")

# Load labels
labels = np.load(BINARY_LABELS_FILE)

# Make train/val/test label folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

# Process each split folder
num_classes = labels.shape[1]
for split in ["train", "val", "test"]:
    split_folder = os.path.join(IMAGES_DIR, split)
    all_images = sorted([f for f in os.listdir(split_folder) if f.endswith((".jpg", ".png"))])

    for f in all_images:
        # Extract index from filename (assuming original filenames match label order)
        idx = int(f.split("_")[-1].split(".")[0]) - 1  # adjust if filenames are ISIC_000001.jpg etc.
        img_labels = labels[idx]
        yolo_lines = [f"{class_id} 0.5 0.5 1 1" for class_id, present in enumerate(img_labels) if present]

        # Save YOLO label
        label_file = os.path.join(LABELS_DIR, split, f.replace(".jpg", ".txt").replace(".png", ".txt"))
        with open(label_file, "w") as lf:
            lf.write("\n".join(yolo_lines))

        # Copy image to a flat folder (optional)
        dst_img_folder = os.path.join(BASE_DIR, "images", split)
        os.makedirs(dst_img_folder, exist_ok=True)
        dst_img = os.path.join(dst_img_folder, f)
        if not os.path.exists(dst_img):
            copyfile(os.path.join(split_folder, f), dst_img)

print("YOLO labels prepared and images organized!")
