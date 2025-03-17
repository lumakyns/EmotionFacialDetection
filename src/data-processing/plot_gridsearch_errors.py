import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set the directory path here
data_directory    = os.path.join(os.path.dirname(__file__), "../results/CNN/grid-search-CNN/")
deposit_directory = os.path.join(os.path.dirname(__file__), "../../visualizations/graphs-CNN")

def process_csv_files(directory, deposit):

    if not os.path.isdir(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Get all CSV files in the specified directory
    csv_files = [f for f in os.listdir(directory)]

    # Create a figure for plotting
    plt.figure(figsize=(10, 6))
    
    # Generate color ranges for blue and red
    num_files = len(csv_files)
    blues = plt.cm.Blues(np.linspace(0.5, 0.9, num_files))
    reds  = plt.cm.Reds(np.linspace(0.5, 0.9, num_files))
    
    # Process each CSV file
    for idx, csv_file in enumerate(csv_files):
        
        file_path = os.path.join(directory, csv_file)
        df = pd.read_csv(file_path, header=0)
        
        # Extract columns (assuming first column is training error, second is testing error)
        train_errors = df.iloc[:, 0].values
        test_errors = df.iloc[:, 1].values
        
        # Create x-axis values (0 to 31)
        x_values = range(len(train_errors))
        
        # Plot the data with different shades of blue for training and red for testing
        plt.plot(x_values, train_errors, color=blues[idx], label="", linewidth=2)
        plt.plot(x_values, test_errors, color=reds[idx], label="", linewidth=2)
    
    # Set up plot attributes (left empty for customization)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title("Grid Search Models' Error over Epochs")
    plt.grid(True)
    
    # Save the plot
    output_path = os.path.join(deposit, "combinedErrorPlot.png")
    plt.savefig(output_path)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    import numpy as np  # Added numpy import for linspace
    process_csv_files(data_directory, deposit_directory)