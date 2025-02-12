import os
import shutil
import numpy as np
import cv2
import random
import csv

ARUCO_DICT = {
    "DICT_4X4_50": (cv2.aruco.DICT_4X4_50, 50),
    "DICT_4X4_100": (cv2.aruco.DICT_4X4_100, 100),
    "DICT_4X4_250": (cv2.aruco.DICT_4X4_250, 250),
    "DICT_4X4_1000": (cv2.aruco.DICT_4X4_1000, 1000),
}

output_folder = "markers"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

csv_file = os.path.join(output_folder, "marker_info.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Dictionary", "Marker ID", "File Path"])

    for i in range(10):
        random_dict_key = random.choice(list(ARUCO_DICT.keys()))
        arucoDict, max_id = ARUCO_DICT[random_dict_key]

        random_id = random.randint(0, max_id - 1)

        dictionary = cv2.aruco.getPredefinedDictionary(arucoDict)

        tag_size = 250
        marker = np.zeros((tag_size, tag_size), dtype="uint8")
        cv2.aruco.generateImageMarker(dictionary, random_id, tag_size, marker, 1)

        bordered_marker = cv2.copyMakeBorder(marker, border_size, border_size, border_size, border_size, 
                                            cv2.BORDER_CONSTANT, value=(255, 255, 255))

        tag_name = os.path.join(output_folder, f"{random_dict_key}_{random_id}.png")
        cv2.imwrite(tag_name, bordered_marker)
        print(f"Generated marker with border saved as: {tag_name}")

        writer.writerow([random_dict_key, random_id, tag_name])

print("All 10 markers have been generated and saved.")
print(f"Marker information logged in: {csv_file}")