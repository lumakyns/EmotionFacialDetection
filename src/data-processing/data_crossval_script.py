import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

## Function to read a CSV and convert it into a NumPy array
def csv_to_numpy(csv_file):
    data = pd.read_csv(csv_file)
    numpy_array = data.values

    return numpy_array

## Paths
csv_path = os.path.join(os.path.dirname(__file__), "../luca-split/split-csv/train.csv")
deposit_path = os.path.join(os.path.dirname(__file__), "../luca-split/7fold-splits")

full_train_set = csv_to_numpy(csv_path)


## Splitting
kf = KFold(n_splits=7, shuffle=True, random_state=42)

# Perform k-fold cross-validation
i = 1
for train_index, test_index in kf.split(full_train_set):
    k_train, k_test = full_train_set[train_index, :], full_train_set[test_index, :]
    
    np.savetxt(os.path.join(deposit_path, f"train_{str(i)}.csv"), k_train, delimiter=",", fmt="%s")
    np.savetxt(os.path.join(deposit_path, f"test_{str(i)}.csv"), k_test, delimiter=",", fmt="%s")
    i += 1
