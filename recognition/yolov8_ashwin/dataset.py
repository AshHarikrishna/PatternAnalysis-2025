# dataset.py
def get_data_yaml(
    train_path="/content/PatternRecognition/ISIC-2017_Training_Data/train",
    val_path="/content/PatternRecognition/ISIC-2017_Training_Data/val",
    num_classes=3,  # 3 classes: pigment_network, negative_network, milia_like_cyst
    class_names=None,
):
    if class_names is None:
        class_names = ["pigment_network", "negative_network", "milia_like_cyst"]

    yaml_content = f"""
train: {train_path}
val: {val_path}

nc: {num_classes}
names: {class_names}
"""
    yaml_file = "/content/PatternRecognition/isic_data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    return yaml_file
