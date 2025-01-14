import numpy as np
import cv2 as cv
import glob
import pickle

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9, 6)
frameSize = (640, 480)

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Size of each square in millimeters
size_of_chessboard_squares_mm = 20
objp *= size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Read images from the 'images' folder
images = glob.glob('images/*.png')

if not images:
    raise FileNotFoundError("No images found in the 'images' folder. Please ensure the images are in the correct location.")

for image in images:
    img = cv.imread(image)
    if img is None:
        print(f"Warning: Unable to read image {image}")
        continue
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points and image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

cv.destroyAllWindows()

############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the calibration results for later use
with open("calibration.pkl", "wb") as f:
    pickle.dump((cameraMatrix, dist), f)

with open("cameraMatrix.pkl", "wb") as f:
    pickle.dump(cameraMatrix, f)

with open("dist.pkl", "wb") as f:
    pickle.dump(dist, f)

############## UNDISTORTION ######################################################

# Test with a specific image (replace with a valid image path)
test_image_path = 'images/img0.png'
img = cv.imread(test_image_path)
if img is None:
    raise FileNotFoundError(f"Test image '{test_image_path}' not found or unreadable.")

h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.png', dst)

# Undistort with remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# Crop the image
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.png', dst)

############## REPROJECTION ERROR ################################################

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("Total error: {:.4f}".format(mean_error / len(objpoints)))
# print the camera matrix and distortion coefficients with 3 decimal places
print("Calibration Matrix :\n")
print(np.round(cameraMatrix, 3))

print("\nDistortion array :\n")
print(np.round(dist))