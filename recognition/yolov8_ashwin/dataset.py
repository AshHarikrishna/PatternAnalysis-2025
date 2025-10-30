def get_data_yaml(
    train_path="/Users/ashwinharikrishna/Desktop/PatternAnalysis-2025/data/images/train",
    val_path="/Users/ashwinharikrishna/Desktop/PatternAnalysis-2025/data/images/val",
    num_classes=2,
    class_names=None,
):
    """
    Create YOLOv8 data YAML file dynamically.
    Returns path to YAML.
    """
    if class_names is None:
        class_names = ["nevus", "melanoma"]

    yaml_content = f"""
train: {train_path}
val: {val_path}

nc: {num_classes}
names: {class_names}
"""
    yaml_file = "isic_data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    return yaml_file
