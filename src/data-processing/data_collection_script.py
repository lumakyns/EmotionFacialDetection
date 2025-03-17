import csv

# Function to read multiple CSV files and create a cumulative dictionary
def read_images_from_csv(files):
    images_by_label = {}

    for csv_file in files:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Get image path and label
                _, image_name, label = row
                # Standardize titles in case of capitlization
                label = label.lower()
                if label not in images_by_label and label != 'emotion':
                    # Merge sad into sadness label
                    if label == 'sad':
                        images_by_label['sadness'] = []
                    elif label == 'contempt':
                        continue
                    else:
                        images_by_label[label] = []
                        
                # Exclude header line
                if label != 'emotion':
                    #merge cont.
                    if label == 'sad':
                        images_by_label['sadness'].append(image_name)
                    elif label == 'contempt':
                        continue
                    else:
                        images_by_label[label].append(image_name)

    return images_by_label

csv_files = ["C:\\Users\\mukun\\CS 178\\Emotions CNN Project\\data\\500_picts_satz.csv", "C:\\Users\\mukun\\CS 178\\Emotions CNN Project\\data\\legend.csv"]
cumulative_images = read_images_from_csv(csv_files)

# Print the merged dictionary
for label, images in cumulative_images.items():
    print(f"{label}: {images}")
