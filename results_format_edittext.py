# import csv
# import os

# def csv_columns_to_text_files(input_file, output_folder):
#     with open(input_file, 'r') as file:
#         reader = csv.reader(file)

#         # Skip the header row if present
#         header = next(reader, None)

#         # Create the output folder if it doesn't exist
#         os.makedirs(output_folder, exist_ok=True)

#         # Iterate through the rows
#         for index, row in enumerate(reader, start=1):
#             # Generate the output text file path
#             output_file = os.path.join(output_folder, f'row_{index}.txt')
#             # print(row[0])
#             # print(row[1])

#             # Switch the columns in the row
#             switched_row = [row[1], row[0]]  # Assuming switching the first and second columns

#             # Write the switched row to the text file
#             with open(output_file, 'w') as text_file:
#                 text_file.write(', '.join(switched_row))

#     print(f"Text files have been created in '{output_folder}'.")

# # Example usage
# input_file = 'output_01.csv'
# output_folder = './runs/classify/predict/labels_edit/'
# csv_columns_to_text_files(input_file, output_folder)

import os
import csv

def first_line_of_text_files_to_csv(folder_path, output_file):
    # files = os.listdir(folder_path)
    files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    sorted_files = sorted(files)
    print(sorted_files)
    # sorted
    text_files = [file for file in files if file.endswith('.txt')]

    rows = []  # Store the rows from the first line of each text file

    for file in text_files:
        file_path = os.path.join(folder_path, file)

        # Read the first line of the text file
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            rows.append([line])

    # Write the CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"CSV file '{output_file}' has been created.")

# Example usage
# folder_path = 'folder_path'  # Specify the folder path
folder_path = './task_1_results/labels/'  # Specify the folder path
output_file = 'output_task_1_.csv'
first_line_of_text_files_to_csv(folder_path, output_file)

# import os
# import csv

# def text_files_to_csv(folder_path, output_file):
#     files = os.listdir(folder_path)
#     text_files = [file for file in files if file.endswith('.txt')]

#     rows = []  # Store the rows from all text files

#     for file in text_files:
#         file_path = os.path.join(folder_path, file)
#         row = []
#         # Read the text file
#         with open(file_path, 'r') as f:
#             lines = f.readlines()

#         # Remove newline characters and split into columns
#         rows.extend([line.strip().split() for line in lines])
#         # rows = rows.append(row[0])
#         # rows.extend(lines[0])
#         # Write the CSV file
#     with open(output_file, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows(rows)

#     print(f"CSV file '{output_file}' has been created.")

# # Example usage
# folder_path = './runs/classify/predict/labels/'  # Specify the folder path
# output_file = 'output.csv'
# text_files_to_csv(folder_path, output_file)
