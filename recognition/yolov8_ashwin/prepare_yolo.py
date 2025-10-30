import os
import numpy as np
from sklearn.model_selection import train_test_split

# Paths
images_dir = "images"  # all 2000 images here
labels_dir = "labels"  # folder where YOLO labels will go
binary_labels_file = "labels/binary_labels.npy"

# Load labels
labels = np.load(binary_labels_file)  # shape: (2000, num_classes)

# Make folders if they don't exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)

# List all images (sorted so they match labels)
all_images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

# Split into train/val/test
train_imgs, test_imgs = train_test_split(all_images, test_size=0.15, random_state=42)
train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.15, random_state=42)

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}

num_classes = labels.shape[1]

for split, files in splits.items():
    for f in files:
        idx = all_images.index(f)
        img_labels = labels[idx]

        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
        # For now, we'll just mark the class with a dummy full-image bbox (x_center=0.5, y_center=0.5, width=1, height=1)
        # Later you can replace with actual lesion bbox if available
        yolo_lines = []
        for class_id, present in enumerate(img_labels):
            if present:
                yolo_lines.append(f"{class_id} 0.5 0.5 1 1")

        # Save label file
        label_file = os.path.join(labels_dir, split, f.replace(".jpg", ".txt"))
        with open(label_file, "w") as lf:
            lf.write("\n".join(yolo_lines))

        # Copy image to split folder
        src_img = os.path.join(images_dir, f)
        dst_img = os.path.join(images_dir, split, f)
        if not os.path.exists(dst_img):
            from shutil import copyfile
            copyfile(src_img, dst_img)

print("YOLO labels prepared and images split!")
