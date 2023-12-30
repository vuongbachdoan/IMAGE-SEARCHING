import cv2
import glob
import os
from detection import Detection
from training import train


detection = Detection()

# Get all directories within 'datasets/train'
folder_paths = os.listdir("datasets/train")

images = []
labels = []

# Iterate through each directory
for folder_path in folder_paths:
    # Build full path
    folder_path = folder_path.replace(' ', '_')
    category_path = os.path.join("datasets/train", folder_path)
    
    # Check if it's a directory
    if os.path.isdir(category_path):
        # Collect images and labels
        category_images = glob.glob(os.path.join(category_path, "*.jpg"))
        images.extend(category_images)
        labels.extend([folder_path] * len(category_images))

train(images, labels)  

detection.load_model('model.pkl')
test_image = cv2.imread("test.jpg")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
predicted_category = detection.classify(test_image)

# Print the predicted category
print(f"Predicted category for 'test.jpg': {detection.labels[predicted_category]}")
