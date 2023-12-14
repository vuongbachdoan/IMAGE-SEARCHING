import cv2
from cv2 import xfeatures2d
import numpy as np
import pickle

def train(images, labels, save_path="model.pkl"):
    """
    Trains a model to classify images using SURF features and K-Nearest Neighbors.

    Args:
        images: A list of N images represented as NumPy arrays.
        labels: A list of N labels corresponding to image categories.
        save_path: Optional path to save the trained model and labels.

    Returns:
        None
    """
    print(images, labels)
    # Feature extraction with SURF
    features = []
    for image in images:
        surf = xfeatures2d.SURF.create()
        kp, des = surf.detectAndCompute(image, None)
        features.append(des)

    # Feature normalization
    features = normalize_features(features)

    # Train K-Nearest Neighbors model
    model = cv2.ml.KNearest.create()
    model.train(np.asarray(features), np.asarray(labels), cv2.ml.ROW_SAMPLE)

    # Save model and labels
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
        pickle.dump(labels, f)

def normalize_features(features):
    """
    Normalizes a list of SURF feature descriptors using standardization.

    Args:
        features: A list of NumPy arrays representing SURF descriptors.

    Returns:
        A list of normalized feature descriptors.
    """

    features_norm = []
    for feature in features:
        feature_norm = (feature - np.mean(feature)) / np.std(feature)
        features_norm.append(feature_norm)
    return features_norm