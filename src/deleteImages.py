import os
import random
import shutil

# Set up paths
train_dir = 'data2/train'
test_dir = 'data2/test'
val_dir = 'data2/validation'


# Function to randomly delete files until the desired limit is reached
def limit_images(directory, max_images):
    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            images = [os.path.join(category_path, img) for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]
            if len(images) > max_images:
                images_to_delete = random.sample(images, len(images) - max_images)
                for img_path in images_to_delete:
                    os.remove(img_path)

# Limit train images to 1000 per category
limit_images(train_dir, 1000)

# Calculate max images for test and validation (30% of train folder)
max_images_test_val = int(0.3 * 1000)

# Limit test images to 30% of train per category
limit_images(test_dir, max_images_test_val)

# Limit validation images to 30% of train per category
limit_images(val_dir, max_images_test_val)

print("Images have been limited successfully.")