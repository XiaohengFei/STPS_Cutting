import os
import shutil

# Define the folder path and its corresponding prefix
folders_with_prefixes = {
    r"E:\\Project\STPS\\1723458596.99": "folder10",
    r"E:\\Project\\STPS\\1723458687.18": "folder11",
    r"E:\\Project\\STPS\\1723458929.70": "folder12",
    r"E:\\Project\\STPS\\1723459117.17": "folder13",
    r"E:\\Project\\STPS\\1723459276.74": "folder14",
    r"E:\\Project\\STPS\\1723459367.35": "folder15",
    r"E:\\Project\\STPS\\1723459420.66": "folder16"
}

# Destination folder for renamed images
target_folder = r"E:\\Project\\STPS\\combined_folder"

# Creates the destination folder, or creates it if it doesn't exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Iterate through each folder and process files
for folder_path, prefix in folders_with_prefixes.items():

    if os.path.exists(folder_path):

        for filename in os.listdir(folder_path):

            if filename.endswith(('.png', '.npy')):

                new_filename = f"{prefix}_{filename}"
                

                src_file = os.path.join(folder_path, filename)
                dest_file = os.path.join(target_folder, new_filename)
                

                shutil.copy(src_file, dest_file)

print("All images have been processed and moved to the destination folder.")
