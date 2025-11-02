import json

"""
    Generates a YOLO YAML file pointing to training and validation image folders
    and defining the number of classes and their names.

    Args:
        train_path (str): Path to training images.
        val_path (str): Path to validation images.
        num_classes (int): Number of classes.
        class_names (list or None): list of class names

    Returns:
        str: path ofcreated YAML file.
    """
    
def get_data_yaml(
    train_path="/content/PatternRecognition/images/train",
    val_path="/content/PatternRecognition/images/val",
    num_classes=4,
    class_names=None,
):
    if class_names is None:
        class_names = ["pigment_network", "negative_network", "milia_like_cyst", "streaks"]

    yaml_content = f"""
train: {train_path}
val: {val_path}

nc: {num_classes}
names: {json.dumps(class_names)}
"""
    yaml_file = "/content/PatternRecognition/isic_data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    return yaml_file
