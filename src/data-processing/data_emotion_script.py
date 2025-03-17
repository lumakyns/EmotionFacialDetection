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
csv_path = os.path.join(os.path.dirname(__file__), "../luca-split/split-csv/test.csv")
deposit_path = os.path.join(os.path.dirname(__file__), "../luca-split/per-emotion/")

test_set = csv_to_numpy(csv_path)


## Put each emotion into a directory
emotions = np.unique(test_set[:, 1])

for emote in emotions:
    this_emotion = test_set[test_set[:, 1] == emote]

    np.savetxt(os.path.join(deposit_path, f"{emote}.csv"), this_emotion, delimiter=",", fmt="%s")
    
print(emotions)