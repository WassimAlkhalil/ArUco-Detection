import numpy as np
import cv2 as cv
import glob
import pickle
import os
import pandas as pd

chessboardSize = (9, 6)
frameSize = (720, 1080) 

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp *= size_of_chessboard_squares_mm

objpoints = []
imgpoints = []

images = glob.glob('data/captured_images/*.png')

if not images:
    raise FileNotFoundError("No images found in the 'images' folder. Please ensure the images are in the correct location.")

output_dir = "data/calibration_output"
os.makedirs(output_dir, exist_ok=True)

for image in images:
    img = cv.imread(image)
    if img is None:
        print(f"Warning: Unable to read image {image}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret:
        corners = cv.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        cv.drawChessboardCorners(img, chessboardSize, corners, ret)
        image_name = os.path.basename(image)
        cv.imwrite(os.path.join(output_dir, image_name), img)
        cv.imshow('Refined Chessboard Corners', img)
        cv.waitKey(500)

        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f"Chessboard not found in image: {image}")

cv.destroyAllWindows()

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

with open("calibration.pkl", "wb") as f:
    pickle.dump((cameraMatrix, dist), f)

with open("cameraMatrix.pkl", "wb") as f:
    pickle.dump(cameraMatrix, f)

with open("dist.pkl", "wb") as f:
    pickle.dump(dist, f)

test_image_path = 'data/captured_images/image_0.png'
img = cv.imread(test_image_path)
if img is None:
    raise FileNotFoundError(f"Test image '{test_image_path}' not found or unreadable.")

h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)

csv_output_dir = "data/csv_data"
os.makedirs(csv_output_dir, exist_ok=True)

calibration_data = []

for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    rvec_array = np.squeeze(rvec).tolist()
    tvec_array = np.squeeze(tvec).tolist()
    calibration_data.append({
        "Image Index": i,
        "rvec_x": rvec_array[0], "rvec_y": rvec_array[1], "rvec_z": rvec_array[2],
        "tvec_x": tvec_array[0], "tvec_y": tvec_array[1], "tvec_z": tvec_array[2],
    })

calibration_df = pd.DataFrame(calibration_data)

csv_output_path = os.path.join(csv_output_dir, "calibration_data.csv")
calibration_df.to_csv(csv_output_path, index=False)

print(f"Calibration vectors saved to: {csv_output_path}")

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
    
    calibration_data[i].update({"Reprojection Error": error})

calibration_df = pd.DataFrame(calibration_data)

csv_output_path = os.path.join(csv_output_dir, "calibration_data_with_errors.csv")
calibration_df.to_csv(csv_output_path, index=False)

print(f"Calibration data with reprojection errors saved to: {csv_output_path}")

print("\n--- Camera Calibration Results ---")
print("Total error: {:.4f}".format(mean_error / len(objpoints)))
print("Camera Matrix:\n", np.round(cameraMatrix, 3))
print("\nDistortion Coefficients:\n", np.round(dist, 3))
print("\nNew Camera Matrix:\n", np.round(newCameraMatrix, 3))
print("\nRegion of Interest (ROI):", roi)
print("\nUndistorted Image Shape:", dst.shape)