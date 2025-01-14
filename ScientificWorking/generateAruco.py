import os
import shutil
import numpy as np
import cv2
import random
import csv

# Define the available ArUco dictionaries and their max IDs
ARUCO_DICT = {
    "DICT_4X4_50": (cv2.aruco.DICT_4X4_50, 50),
    "DICT_4X4_100": (cv2.aruco.DICT_4X4_100, 100),
    "DICT_4X4_250": (cv2.aruco.DICT_4X4_250, 250),
    "DICT_4X4_1000": (cv2.aruco.DICT_4X4_1000, 1000),
}

# Define the output folder
output_folder = "markers"

# Overwrite the folder if it exists, otherwise create it
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # Delete the existing folder
os.makedirs(output_folder)  # Create a new folder

# Create a CSV file to log marker information
csv_file = os.path.join(output_folder, "marker_info.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Dictionary", "Marker ID", "File Path"])  # Write the header

    # Generate 10 random markers
    for i in range(10):
        # Select a random dictionary
        random_dict_key = random.choice(list(ARUCO_DICT.keys()))
        arucoDict, max_id = ARUCO_DICT[random_dict_key]

        # Generate a random marker ID within the valid range
        random_id = random.randint(0, max_id - 1)

        # Load the dictionary
        dictionary = cv2.aruco.getPredefinedDictionary(arucoDict)

        # Generate the marker
        tag_size = 250
        marker = np.zeros((tag_size, tag_size), dtype="uint8")
        cv2.aruco.generateImageMarker(dictionary, random_id, tag_size, marker, 1)

        # Add a white border around the marker
        border_size = 20  # Size of the white border
        bordered_marker = cv2.copyMakeBorder(marker, border_size, border_size, border_size, border_size, 
                                            cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Save the bordered marker
        tag_name = os.path.join(output_folder, f"{random_dict_key}_{random_id}.png")
        cv2.imwrite(tag_name, bordered_marker)
        print(f"Generated marker with border saved as: {tag_name}")

        # Log marker information in the CSV file
        writer.writerow([random_dict_key, random_id, tag_name])

print("All 10 markers have been generated and saved.")
print(f"Marker information logged in: {csv_file}")