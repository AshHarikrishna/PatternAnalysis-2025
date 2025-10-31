# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from shutil import copyfile

# BASE_DIR = "/content/PatternRecognition"
# RAW_IMAGES_DIR = os.path.join(BASE_DIR, "ISIC-2017_Training_Data", "ISIC-2017_Training_Data")
# LABELS_DIR = os.path.join(BASE_DIR, "labels")
# BINARY_LABELS_FILE = os.path.join(LABELS_DIR, "binary_labels.npy")

# # Load labels
# labels = np.load(BINARY_LABELS_FILE)
# if labels.ndim == 1:
#     labels = np.expand_dims(labels, axis=1)

# num_classes = labels.shape[1]

# # Create folders for organized images
# for split in ["train", "val", "test"]:
#     os.makedirs(os.path.join(BASE_DIR, "images", split), exist_ok=True)
#     os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

# # Get all image filenames (ignore *_superpixels.png)
# all_images = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(".jpg")])

# # Split into train/val/test (70/15/15)
# train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)
# val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

# # Write YOLO labels + copy images
# for split, img_files in splits.items():
#     for f in img_files:
#         idx = int(f.split("_")[-1].split(".")[0]) - 1  # e.g. ISIC_000001.jpg → 0
#         img_labels = labels[idx]

#         yolo_lines = [f"{class_id} 0.5 0.5 1 1" for class_id, present in enumerate(img_labels) if present]

#         # Save YOLO label
#         label_file = os.path.join(LABELS_DIR, split, f.replace(".jpg", ".txt"))
#         with open(label_file, "w") as lf:
#             lf.write("\n".join(yolo_lines))

#         # Copy image
#         src = os.path.join(RAW_IMAGES_DIR, f)
#         dst = os.path.join(BASE_DIR, "images", split, f)
#         if not os.path.exists(dst):
#             copyfile(src, dst)

# print("✅ YOLO labels prepared and images split successfully!")

# prepare_yolo.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile

BASE_DIR = "/content/PatternRecognition"
RAW_IMAGES_DIR = os.path.join(BASE_DIR, "ISIC-2017_Training_Data", "ISIC-2017_Training_Data")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
BINARY_LABELS_FILE = os.path.join(LABELS_DIR, "binary_labels.npy")
ANNOTATIONS_FOLDER = os.path.join(BASE_DIR, "ISIC-2017_Training_Part2_GroundTruth/ISIC-2017_Training_Part2_GroundTruth")

# Load labels
labels = np.load(BINARY_LABELS_FILE)
if labels.ndim == 1:
    labels = np.expand_dims(labels, axis=1)

num_classes = labels.shape[1]

# Get image filenames corresponding to labels.npy
label_files = sorted([f.split("_features")[0] + ".jpg" for f in os.listdir(ANNOTATIONS_FOLDER) if f.endswith(".json")])

# Filter only images that have labels
all_images = sorted([f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(".jpg")])
all_images = [f for f in all_images if f in label_files]

print(f"Total images with labels: {len(all_images)}")

# Split into train/val/test (70/15/15)
train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

splits = {"train": train_imgs, "val": val_imgs, "test": test_imgs}

# Create folders for organized images and YOLO labels
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(BASE_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

# Write YOLO labels + copy images
for split, img_files in splits.items():
    for f in img_files:
        idx = label_files.index(f)  # find index in labels.npy
        img_labels = labels[idx]

        # YOLO format: class_id x_center y_center width height
        # Using dummy values for full-image labels
        yolo_lines = [f"{class_id} 0.5 0.5 1 1" for class_id, present in enumerate(img_labels) if present]

        # Save YOLO label
        label_file = os.path.join(LABELS_DIR, split, f.replace(".jpg", ".txt"))
        with open(label_file, "w") as lf:
            lf.write("\n".join(yolo_lines))

        # Copy image
        src = os.path.join(RAW_IMAGES_DIR, f)
        dst = os.path.join(BASE_DIR, "images", split, f)
        if not os.path.exists(dst):
            copyfile(src, dst)

print("✅ YOLO labels prepared and images split successfully!")
