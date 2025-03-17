import os
import numpy as np
import matplotlib.pyplot as plt


## CSV Processing Function
def files_to_numpy(directory_path, file_extension='', skip_header=True):
    
    # List all files in the directory with the given extension
    all_files = [f for f in os.listdir(directory_path) 
                if os.path.isfile(os.path.join(directory_path, f)) and f.endswith(file_extension)]
    
    # Exit if no files found
    if not all_files:
        print(f"No{' ' + file_extension if file_extension else ''} files found in the directory.")
        return None, []
    
    # Sort files for consistent ordering
    all_files.sort()
    
    # Process each file
    file_data = []
    processed_files = []
    
    for file_name in all_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            # Skip header row if specified
            skiprows = 1 if skip_header else 0
            data = np.loadtxt(file_path, delimiter=',', skiprows=skiprows)
            file_data.append(data)
            processed_files.append(file_name)
            print(f"Processed: {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    # Combine all data into a single array if files were processed
    if file_data:
        data_array = np.array(file_data)
        print(f"\nSuccessfully processed {len(file_data)} files")
        print(f"Final array shape: {data_array.shape}")
        print(f"Array dtype: {data_array.dtype}")
        return data_array, processed_files
    else:
        print("Failed to process any files into a NumPy array")
        return None, []

## Plotting when Epochs reached optimal value
def plot_histogram(data, bins=10, red_bins=[]):
    plt.figure(figsize=(8, 6))
    counts, bin_edges, patches = plt.hist(data, bins=bins, alpha=0.75, edgecolor='black', label='')
    
    # Make chosen model(s) red.
    for i, patch in enumerate(patches):
        if i in red_bins:
            patch.set_facecolor("red")
            patch.set_label("Best model")
    
    # Placeholder for customizations
    plt.title('First Epoch Where Optimal Test Error Was Reached.')
    plt.xlabel('Epoch')
    plt.ylabel('No. of occurrences')
    plt.legend

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(os.path.dirname(__file__), "../results/graphs-CNN/epochHistogram.png"))

## Plotting the error rate accross Epochs for the best model
def plot_error_rate(error_values):

    plt.figure(figsize=(8, 6))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    plt.plot(range(1, 33), error_values[:, 0], marker='o', linestyle='-', color='b', label='Training Error')
    plt.plot(range(1, 33), error_values[:, 1], marker='o', linestyle='-', color='r', label='Testing Error')
    
    # Placeholder for customizations
    plt.title('Error Rate over Epochs for Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Error rate')
    plt.figtext(0.5, 0.01, 'Learning Rate: 0.001 - Batch Size: 64 - Conolution filter sizes: [16, 64] - Dropout rate: 0.3', ha='center', fontsize=10)
    plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(os.path.dirname(__file__), "../results/graphs-CNN/bestModelError.png"))

## Main
if __name__ == "__main__":

    directory_path = os.path.join(os.path.dirname(__file__), "../results/grid-search-CNN")
    
    # All data
    data_array, processed_files = files_to_numpy(directory_path, file_extension='', skip_header=True)
    zipped_data = list(zip(data_array, processed_files))
    
    # Best model
    best_model_index = min(range(len(zipped_data)), key=lambda i: np.min(zipped_data[i][0][:, 1]))
    best_data_array, best_processed_file = zipped_data[best_model_index]
    
    # Epoch histogram
    lowest_epoch = [np.argmin(result[:, 1])+1 for result in data_array]
    plot_histogram(lowest_epoch, bins=32, red_bins=[best_model_index])
    
    # Error rate for best model
    plot_error_rate(best_data_array)