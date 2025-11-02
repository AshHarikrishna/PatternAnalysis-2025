# classify_part3.py
import os
import pandas as pd
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms


# Directory containing cropped lesion images (from YOLO detection)
IMAGES_DIR = "/content/PatternRecognition/yolo_detections"  # Cropped lesion images
PART3_CSV = "/content/PatternRecognition/ISIC-2017_Training_Part3_GroundTruth.csv"


df = pd.read_csv(PART3_CSV)

# Convert multi-label to single label
df["label"] = df[["melanoma", "seborrheic_keratosis"]].idxmax(axis=1)
df.loc[(df["melanoma"] == 0) & (df["seborrheic_keratosis"] == 0), "label"] = "benign_nevus"

LABEL_MAP = {"melanoma": 0, "seborrheic_keratosis": 1, "benign_nevus": 2}



# Custom Dataset class for loading lesion images
class LesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['image_id']}.jpg"
        img_path = os.path.join(IMAGES_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        label = LABEL_MAP[row["label"]]
        return image, label

# Define transformations for images: resize, convert to tensor, normalize
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = LesionDataset(df, transform=transform)

# Train/val split (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pretrained ResNet18 model and replace the final fully connected layer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


EPOCHS = 30

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(train_loader.dataset)

    # # Validation loop
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
    val_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")


torch.save(model.state_dict(), "lesion_classifier_resnet18.pth")
print(" Model saved as lesion_classifier_resnet18.pth")
