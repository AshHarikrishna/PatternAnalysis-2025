# convert.py
import os
import json
import numpy as np

# Use Colab-friendly paths for annotations and output labels
ANNOTATIONS_FOLDER = ANNOTATIONS_FOLDER = "/content/PatternRecognition/ISIC-2017_Training_Part2_GroundTruth/ISIC-2017_Training_Part2_GroundTruth"
LABELS_FOLDER = "/content/PatternRecognition/labels"
OUTPUT_FILE = "binary_labels.npy"

os.makedirs(LABELS_FOLDER, exist_ok=True)

# Keys corresponding to the features in the JSON annotations
label_keys = ["pigment_network", "negative_network", "milia_like_cyst", "streaks"]

all_labels = []

for filename in os.listdir(ANNOTATIONS_FOLDER):
    if not filename.endswith(".json"):
        continue
    path = os.path.join(ANNOTATIONS_FOLDER, filename)
    with open(path) as f:
        data = json.load(f)

    labels = []
    # Convert each key into a binary label (1 if feature exists, 0 otherwise)
    for key in label_keys:
        count = sum(data.get(key, [])) if isinstance(data.get(key, []), list) else data.get(key, 0)
        labels.append(1 if count > 0 else 0)

    all_labels.append(labels)

# Convert list of labels to a numpy array for efficient storage
all_labels = np.array(all_labels, dtype=np.int8)
np.save(os.path.join(LABELS_FOLDER, OUTPUT_FILE), all_labels)
print(f"Saved binary labels to {os.path.join(LABELS_FOLDER, OUTPUT_FILE)}")
