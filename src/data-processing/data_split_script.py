import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_and_split_data(csv_path, deposit_path):
    # Step 1: Load CSV into NumPy array
    data = np.loadtxt(csv_path, delimiter=',', dtype=str)

    # Step 2: Remove the first column
    data = data[:, 1:]

    # Step 3: Convert last row to lowercase
    data[:, -1] = np.char.lower(data[:, -1])

    # Step 4: Remove rows where the last column is "contempt" or "fear"
    mask = ~np.isin(data[:, -1], ["contempt", "fear"])
    data = data[mask]

    # Step 5: Train-test split
    X = data[:, :-1]  # All columns except the last (features)
    y = data[:, -1]   # Last column (labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 6: Recombine features and labels
    train_data = np.column_stack((X_train, y_train))
    test_data = np.column_stack((X_test, y_test))

    # Step 7: Save to CSV
    np.savetxt(f"{deposit_path}/train.csv", train_data, delimiter=',', fmt='%s')
    np.savetxt(f"{deposit_path}/test.csv", test_data, delimiter=',', fmt='%s')


csv_path = "C:/Users/1frew/Desktop/Code/Classes/178Project/rsrc/facial_expressions/data/legend.csv"
deposit_path = "C:/Users/1frew/Desktop/Code/Classes/178Project/src/luca-split/split-csv"
process_and_split_data(csv_path, deposit_path)
