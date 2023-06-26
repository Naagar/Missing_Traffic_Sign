import os
import shutil
from PIL import Image
import cv2
import random



# def convert_pixel_to_yolo(image_width, image_height, xmin, ymin, width, height):
#     center_x = (xmin + (width / 2)) / image_width
#     center_y = (ymin + (height / 2)) / image_height
#     yolo_width = width / image_width
#     yolo_height = height / image_height

#     return center_x, center_y, yolo_width, yolo_height


# Define the path to the folder containing .png and .txt files
folder_path = './datasets/task1train540p/'  # Replace with the actual path to your folder

# Define the output folder to save modified .txt files and corresponding .png files
output_folder = './train_cls_from_task_1_4000/'  # Replace with the desired path for the output folder
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
            # image_width, image_height = image.size

            # Convert the coordinates to YOLO format
            # yolo_coordinates = convert_pixel_to_yolo(image_width, image_height, *coordinates)

            # Get the corresponding class ID
            class_id = class_ids[class_labels.index(class_label)]

            # Create a new .txt file with YOLO format in the output folder
            output_file_path = os.path.join(output_folder, file_name)
            # with open(output_file_path, 'w') as output_txt_file:
            #     output_txt_file.write(f"{class_id} {' '.join(str(coord) for coord in yolo_coordinates)}")

            # Copy the corresponding .png file to the output class folder folder
            output_png_file_path = os.path.join(output_folder + class_label + '/' + file_name[:-4] + '.png')
            print(output_png_file_path)
            shutil.copy(corresponding_png_file, output_png_file_path)

print("Conversion complete. Modified files saved in the output folder.")