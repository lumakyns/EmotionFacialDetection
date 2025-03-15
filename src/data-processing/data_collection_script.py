import csv

# Read multiple CSV files and create a cumulative dictionary
def read_images_from_csv(files):
    images_by_label = {}

    for csv_file in files:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Get image path and label
                _, image_name, label = row

                # Standardize titles in case of capitalization
                label = label.lower()
                if label not in images_by_label and label != 'emotion':
                    # Merge sad into sadness label
                    if label == 'sad':
                        images_by_label['sadness'] = []
                    else:
                        images_by_label[label] = []

                # Exclude header line
                if label != 'emotion':
                    if label == 'sad':
                        images_by_label['sadness'].append(image_name)
                    else:
                        images_by_label[label].append(image_name)

    return images_by_label

csv_files = ["/home/kyns/Desktop/Code/Classes/178/ClassProject/rsrc/facial_expressions/data/500_picts_satz.csv", "/home/kyns/Desktop/Code/Classes/178/ClassProject/rsrc/facial_expressions/data/legend.csv"]
cumulative_images = read_images_from_csv(csv_files)

# Print the merged dictionary
for label, images in cumulative_images.items():
    print(f"{label}: {images}")