import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict

def process_and_split_data_json(csv_path, deposit_path):
    # Step 1: Load CSV into NumPy array
    data = np.loadtxt(csv_path, delimiter=',', dtype=str)

    # Step 2: Remove the first column (file titles)
    data = data[:, 1:]

    # Step 3: Convert last row to lowercase
    data[:, -1] = np.char.lower(data[:, -1])

    # Step 4: Remove rows where the last column is "contempt" or "fear"
    mask = ~np.isin(data[:, -1], ["contempt", "fear"])
    data = data[mask]

    # Step 5: Train-test split
    X = data[:, 0]  # File titles (first column)
    y = data[:, -1]  # Labels (last column)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Recombine features and labels into a list of dictionaries for training data only
    train_data = {"POSITIVE.json": defaultdict(list), "NEGATIVE.json": defaultdict(list)}

    # Step 7: Organize data for POSITIVE and NEGATIVE
    # Build POSITIVE JSON: Images by label
    for i in range(len(X_train)):
        label = y_train[i]
        file_title = X_train[i]
        train_data["POSITIVE.json"][label].append(file_title)

    # Build NEGATIVE JSON: Images that do not have a certain label
    all_labels = set(y_train)  # Get all labels in the training data
    for label in all_labels:
        # Get all images that do not have this label
        negative_images = [X_train[i] for i in range(len(X_train)) if y_train[i] != label]
        train_data["NEGATIVE.json"][label] = negative_images

    # Step 8: Save POSITIVE.json and NEGATIVE.json
    with open(f"{deposit_path}/POSITIVE.json", 'w', encoding='utf-8') as positive_file:
        json.dump(train_data["POSITIVE.json"], positive_file, ensure_ascii=False, indent=4)

    with open(f"{deposit_path}/NEGATIVE.json", 'w', encoding='utf-8') as negative_file:
        json.dump(train_data["NEGATIVE.json"], negative_file, ensure_ascii=False, indent=4)

# Example usage
csv_path = "C:/Users/1frew/Desktop/Code/Classes/178Project/rsrc/facial_expressions/data/legend.csv"
deposit_path = "C:/Users/1frew/Desktop/Code/Classes/178Project/src/luca-split/pos-neg-train"
process_and_split_data_json(csv_path, deposit_path)
