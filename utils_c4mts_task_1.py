import os
import shutil
from PIL import Image
import cv2
import random



def convert_pixel_to_yolo(image_width, image_height, xmin, ymin, width, height):
    center_x = (xmin + (width / 2)) / image_width
    center_y = (ymin + (height / 2)) / image_height
    yolo_width = width / image_width
    yolo_height = height / image_height

    return center_x, center_y, yolo_width, yolo_height


# Define the path to the folder containing .png and .txt files
folder_path = './dataset/task1train540p'  # Replace with the actual path to your folder

# Define the output folder to save modified .txt files and corresponding .png files
output_folder = './output/'  # Replace with the desired path for the output folder
os.makedirs(output_folder, exist_ok=True)

# Define the class labels and corresponding IDs
class_labels = ['gap-in-median', 'left-hand-curve', 'right-hand-curve', 'side-road-left']  # Replace with your class labels
class_ids = [0, 1, 2, 3]  # Replace with your corresponding class IDs

# Iterate over files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Process .txt files
    if file_name.endswith('.txt'):
        # Read the class label and coordinates from the .txt file
        with open(file_path, 'r') as txt_file:
            line = txt_file.readline().strip()
            class_label, coordinates = line.split(': ')
            coordinates = [int(coord) for coord in coordinates.split(',')]

        # Process corresponding .png files
        corresponding_png_file = os.path.join(folder_path, file_name[:-4] + '.png')
        if os.path.isfile(corresponding_png_file):
            # Get the image dimensions
            image = Image.open(corresponding_png_file)
            image_width, image_height = image.size

            # Convert the coordinates to YOLO format
            yolo_coordinates = convert_pixel_to_yolo(image_width, image_height, *coordinates)

            # Get the corresponding class ID
            class_id = class_ids[class_labels.index(class_label)]

            # Create a new .txt file with YOLO format in the output folder
            output_file_path = os.path.join(output_folder, file_name)
            with open(output_file_path, 'w') as output_txt_file:
                output_txt_file.write(f"{class_id} {' '.join(str(coord) for coord in yolo_coordinates)}")

            # Copy the corresponding .png file to the output folder
            output_png_file_path = os.path.join(output_folder, file_name[:-4] + '.png')
            shutil.copy(corresponding_png_file, output_png_file_path)

print("Conversion complete. Modified files saved in the output folder.")

folder_path = './dataset/'  # Replace with the actual path to your folder

# Initialize counters
png_count = 0
txt_count = 0

# Iterate over files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Count .png files
    if file_name.endswith('.png'):
        png_count += 1

    # Count .txt files
    elif file_name.endswith('.txt'):
        txt_count += 1

# Print the counts
print("Total number of .png files:", png_count)
print("Total number of .txt files:", txt_count)

folder_path = './output/'  # Replace with the actual path to your folder

# Initialize variables to track PNG and TXT files
png_files = []
txt_files = []

# Iterate over files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Check if file is a PNG file
    if file_name.endswith('.png'):
        png_files.append(file_name)

    # Check if file is a TXT file
    elif file_name.endswith('.txt'):
        txt_files.append(file_name)

# Check if each PNG file has a corresponding TXT file
for png_file in png_files:
    corresponding_txt_file = png_file[:-4] + '.txt'
    if corresponding_txt_file not in txt_files:
        print(f"Missing TXT file for PNG file: {png_file}")

print("Folder check complete.")

def convert_yolo_to_pixel(image_width, image_height, yolo_x, yolo_y, yolo_width, yolo_height):
    box_x = yolo_x * image_width
    box_y = yolo_y * image_height
    box_width = yolo_width * image_width
    box_height = yolo_height * image_height

    box_xmin = int(box_x - (box_width / 2))
    box_ymin = int(box_y - (box_height / 2))
    box_xmax = int(box_x + (box_width / 2))
    box_ymax = int(box_y + (box_height / 2))

    return box_xmin, box_ymin, box_xmax, box_ymax

# Load the image
image_path = "./output/1012_frame_703.png"
image = cv2.imread(image_path)

# Get image dimensions
image_height, image_width, _ = image.shape

# Define YOLO coordinates
yolo_x = 0.3098958333333333
yolo_y = 0.5824074074074074
yolo_width = 0.017708333333333333
yolo_height = 0.027777777777777776

# Convert YOLO to pixel coordinates
box_xmin, box_ymin, box_xmax, box_ymax = convert_yolo_to_pixel(image_width, image_height, yolo_x, yolo_y, yolo_width, yolo_height)

# Draw bounding box rectangle on the image
cv2.rectangle(image, (box_xmin, box_ymin), (box_xmax, box_ymax), (0, 255, 0), 2)

# Show the image with the bounding box
cv2.imwrite('image_with_bounding_box.png', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Set the path to the original folder containing PNG and TXT files
data_folder = "./output/"

# Set the path to the folder where the training and validation data will be saved
train_folder = "./dataset/train8"
val_folder = "./dataset/val8"

# Set the desired split ratio
split_ratio = 0.9  # 90% for training, 10% for validation

# Create the train and validation folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get the list of PNG files
png_files = [file for file in os.listdir(data_folder) if file.endswith('.png')]

# Shuffle the list of PNG files
random.shuffle(png_files)

# Calculate the index to split the data
split_index = int(len(png_files) * split_ratio)

# Split the PNG files into training and validation sets
train_png_files = png_files[:split_index]
val_png_files = png_files[split_index:]

# Move the corresponding TXT files for training set
for png_file in train_png_files:
    txt_file = os.path.splitext(png_file)[0] + '.txt'
    source_png_path = os.path.join(data_folder, png_file)
    source_txt_path = os.path.join(data_folder, txt_file)
    destination_png_path = os.path.join(train_folder, png_file)
    destination_txt_path = os.path.join(train_folder, txt_file)

    print(f"Copying {png_file} and {txt_file} to the training folder.")
    try:
        shutil.copy(source_png_path, destination_png_path)
        shutil.copy(source_txt_path, destination_txt_path)
        print(f"Successfully copied {png_file} and {txt_file} to the training folder.")
    except Exception as e:
        print(f"Error copying {png_file} and {txt_file}: {str(e)}")

# Move the corresponding TXT files for validation set
for png_file in val_png_files:
    txt_file = os.path.splitext(png_file)[0] + '.txt'
    source_png_path = os.path.join(data_folder, png_file)
    source_txt_path = os.path.join(data_folder, txt_file)
    destination_png_path = os.path.join(val_folder, png_file)
    destination_txt_path = os.path.join(val_folder, txt_file)

    print(f"Copying {png_file} and {txt_file} to the validation folder.")
    try:
        shutil.copy(source_png_path, destination_png_path)
        shutil.copy(source_txt_path, destination_txt_path)
        print(f"Successfully copied {png_file} and {txt_file} to the validation folder.")
    except Exception as e:
        print(f"Error copying {png_file} and {txt_file}: {str(e)}")
import os

folder_path = './dataset/train8'  # Replace with the actual path to your folder

# Initialize counters
png_count = 0
txt_count = 0

# Iterate over files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Count .png files
    if file_name.endswith('.png'):
        png_count += 1

    # Count .txt files
    elif file_name.endswith('.txt'):
        txt_count += 1

# Print the counts
print("Total number of .png files:", png_count)
print("Total number of .txt files:", txt_count)

folder_path = './dataset/val8'  # Replace with the actual path to your folder

# Initialize counters
png_count = 0
txt_count = 0

# Iterate over files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Count .png files
    if file_name.endswith('.png'):
        png_count += 1

    # Count .txt files
    elif file_name.endswith('.txt'):
        txt_count += 1

# Print the counts
print("Total number of .png files:", png_count)
print("Total number of .txt files:", txt_count)

folder_path = './dataset/train8'  # Replace with the actual path to your folder

# Initialize variables to track PNG and TXT files
png_files = []
txt_files = []

# Iterate over files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Check if file is a PNG file
    if file_name.endswith('.png'):
        png_files.append(file_name)

    # Check if file is a TXT file
    elif file_name.endswith('.txt'):
        txt_files.append(file_name)

# Check if each PNG file has a corresponding TXT file
for png_file in png_files:
    corresponding_txt_file = png_file[:-4] + '.txt'
    if corresponding_txt_file not in txt_files:
        print(f"Missing TXT file for PNG file: {png_file}")

print("Folder check complete.")

folder_path = './dataset/val8'  # Replace with the actual path to your folder

# Initialize variables to track PNG and TXT files
png_files = []
txt_files = []

# Iterate over files in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Check if file is a PNG file
    if file_name.endswith('.png'):
        png_files.append(file_name)

    # Check if file is a TXT file
    elif file_name.endswith('.txt'):
        txt_files.append(file_name)

# Check if each PNG file has a corresponding TXT file
for png_file in png_files:
    corresponding_txt_file = png_file[:-4] + '.txt'
    if corresponding_txt_file not in txt_files:
        print(f"Missing TXT file for PNG file: {png_file}")

print("Folder check complete.")
