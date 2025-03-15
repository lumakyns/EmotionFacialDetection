import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Define the dataset loading function
def load_hog_features(image_dict, directory, pixels_per_cell, cells_per_block):
    """
    Extracts HOG features from images using specified parameters.
    """
    X, y = [], []

    for label, image_paths in image_dict.items():
        for img_path in image_paths:
            full_path = os.path.join(directory, img_path)

            # Read image in grayscale
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # Skip missing images

            # Resize to 64x64 for consistency
            img = cv2.resize(img, (64, 64))

            # Extract HOG features
            features = hog(img, pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block, block_norm="L2-Hys", feature_vector=True)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# Directory containing images
directory = "c:\\Users\\mukun\\CS 178\\Emotions CNN Project\\images"

# Define parameter grid for HOG & SVM
hog_params_grid = {
    "pixels_per_cell": [(8, 8), (16, 16)],  # Different pixel sizes
    "cells_per_block": [(2, 2), (3, 3)]  # Different block sizes
}

svm_params_grid = {
    "C": [0.1, 1, 10, 100],  # SVM regularization parameter
    "kernel": ["linear", "rbf"],  # SVM kernel types
}

# Perform Grid Search over HOG parameters
best_hog_params = None
best_svm_model = None
best_score = 0

for pixels_per_cell in hog_params_grid["pixels_per_cell"]:
    for cells_per_block in hog_params_grid["cells_per_block"]:
        print(f"\nTesting HOG: pixels_per_cell={pixels_per_cell}, cells_per_block={cells_per_block}")

        # Extract features with current HOG parameters
        X, y = load_hog_features(cumulative_images, directory, pixels_per_cell, cells_per_block)

        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Grid Search for best SVM parameters
        svm = SVC()
        grid_search = GridSearchCV(svm, svm_params_grid, cv=3, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Evaluate best model from grid search
        best_svm = grid_search.best_estimator_
        y_pred = best_svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Best SVM Params: {grid_search.best_params_}")
        print(f"Test Accuracy: {accuracy:.2f}")

        # Track the best overall model
        if accuracy > best_score:
            best_score = accuracy
            best_hog_params = (pixels_per_cell, cells_per_block)
            best_svm_model = best_svm

# Print the best overall parameters and accuracy
print("\nBest Overall HOG Parameters:", best_hog_params)
print("Best Overall SVM Parameters:", best_svm_model.get_params())
print(f"Best Overall Accuracy: {best_score:.2f}")
