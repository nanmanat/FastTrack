import os
import shutil

def move_files(file_list_path, source_dir, destination_dir):
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    with open(file_list_path, 'r') as file_list:
        for line in file_list:
            file_name = line.strip().split()[0].split(".")[0] + ".jpg"
            source_path = os.path.join(source_dir, file_name)
            destination_path = os.path.join(destination_dir, file_name)

            # Check if the file exists before attempting to move it
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                print(f"Moved: {file_name}")
            else:
                print(f"File not found: {file_name}")

# Example usage
file_list_path = 'NNEW_test_0.txt'  # Path to the .txt file containing file names
source_dir = 'vgg/dataset/JPEGImages'
destination_dir = 'vgg/dataset/test'

move_files(file_list_path, source_dir, destination_dir)
