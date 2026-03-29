import os
import random
import shutil

dataset_path = r'C:\Users\NST37\AI Project\Kaggle'
# Define the class names and number of images in each class
class_names = ["Big Truck", "City Car", "Multi Purpose Vehicle", "Sedan", "Sport Utility Vehicle","Truck", "van"]

# Define the percentage of images for training, validation, and testing sets
train_percent = 0.7
val_percent = 0.15
test_percent = 0.15

# Create the directories for the training, validation, and testing sets
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Iterate over the class names and create the directories for each class in the training, validation, and testing sets
for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Get the list of images for the current class
    images_path = os.path.join(dataset_path, class_name)
    images_list = os.listdir(images_path)

    # Shuffle the images list randomly
    random.shuffle(images_list)

    # Split the images list into training, validation, and testing sets
    train_index = int(len(images_list) * train_percent)
    val_index = int(len(images_list) * (train_percent + val_percent))

    train_list = images_list[:train_index]
    val_list = images_list[train_index:val_index]
    test_list = images_list[val_index:]

    # Move the images to the corresponding directories in the training, validation, and testing sets
    for image_name in train_list:
        shutil.move(os.path.join(images_path, image_name), os.path.join(train_dir, class_name, image_name))
    for image_name in val_list:
        shutil.move(os.path.join(images_path, image_name), os.path.join(val_dir, class_name, image_name))
    for image_name in test_list:
        shutil.move(os.path.join(images_path, image_name), os.path.join(test_dir, class_name, image_name))
