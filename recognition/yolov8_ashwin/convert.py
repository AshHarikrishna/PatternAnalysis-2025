import os
import json
import numpy as np

# Paths
ANNOTATIONS_FOLDER = "annotations"
LABELS_FOLDER = "labels"
OUTPUT_FILE = "binary_labels.npy"

# Make sure labels folder exists
os.makedirs(LABELS_FOLDER, exist_ok=True)

# Define your classes (adjust to your dataset)
label_keys = ["pigment_network", "negative_network", "milia_like_cyst"]

all_labels = []

# Loop through each JSON annotation
for filename in os.listdir(ANNOTATIONS_FOLDER):
    if not filename.endswith(".json"):
        continue

    path = os.path.join(ANNOTATIONS_FOLDER, filename)
    with open(path) as f:
        data = json.load(f)

    # Extract label counts and convert to binary (0/1)
    labels = []
    for key in label_keys:
        count = sum(data.get(key, [])) if isinstance(data.get(key, []), list) else data.get(key, 0)
        labels.append(1 if count > 0 else 0)
    
    all_labels.append(labels)

# Convert to NumPy array
all_labels = np.array(all_labels, dtype=np.int8)

print("Shape of labels:", all_labels.shape)
print("First 5 labels:\n", all_labels[:5])

# Save to labels folder
np.save(os.path.join(LABELS_FOLDER, OUTPUT_FILE), all_labels)
print(f"Saved binary labels to {os.path.join(LABELS_FOLDER, OUTPUT_FILE)}")
