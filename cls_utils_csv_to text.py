import csv

def csv_to_text_files(input_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)

        # Skip the header row if present
        header = next(reader, None)

        for row in reader:
            # file_name = row[0] + '.txt'
            file_name = 'results/' + row[0] + '.txt'

            file_content = ', '.join(row[1:])

            # Save the row content as a text file
            with open(file_name, 'w') as text_file:
                text_file.write(file_content)

    print("Text files have been created.")

# Example usage
input_file = 'testtask1.csv'
csv_to_text_files(input_file)
