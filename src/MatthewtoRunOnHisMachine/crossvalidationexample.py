import cv2
import numpy as np
import os
import json
import random
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Define HOG parameters
HOG_PARAMS = {
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True  
}

# Load dataset with HOG features
def load_hog_features(image_dict, directory):
    X = []
    y = []

    for label, image_paths in image_dict.items():
        for img_path in image_paths:
            full_path = os.path.join(directory, img_path)

            # Read image and convert to grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip if image not found
            gsimg = img
            #convert to int np array
            if gsimg.dtype == np.uint8:
                gsimg = gsimg.astype(float) / 255
            #assert dimensions else skip
            if gsimg.shape[0] != 350 or gsimg.shape[1] != 350:
                continue
            #grayscale
            if(len(img.shape) == 3):
                gsimg = np.mean(img, axis=2)
            
            # Resize for consistency
            img = cv2.resize(img, (64, 64))

            # Extract HOG features
            features = hog(img, **HOG_PARAMS)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# Load features
directory = "c:\\Users\\mukun\\CS 178\\Emotions CNN Project\\images"
X, y = load_hog_features(cumulative_images, directory)

# Define the SVM classifier with best parameters
classifier = SVC(
    C=10, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
    shrinking=True, probability=False, tol=0.001, cache_size=200,
    class_weight="balanced", verbose=False, max_iter=-1, decision_function_shape="ovr",
    random_state=None, break_ties=False
)

# Perform k-fold cross-validation
kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)  # 5-Fold Cross Validation
cv_scores = cross_val_score(classifier, X, y, cv=kf, scoring="accuracy")

# Print cross-validation results
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}")
