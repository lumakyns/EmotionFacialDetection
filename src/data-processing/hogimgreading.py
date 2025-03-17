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