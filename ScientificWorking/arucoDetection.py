import numpy as np
import cv2

# Define the ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}

# Define marker size (in meters)
MARKER_SIZE = 0.05  # 5 cm

# Load camera calibration data

camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])
dist_coeffs = np.zeros((5,))

def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the bounding box
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Draw the center
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            # Pose estimation
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers([markerCorner], MARKER_SIZE, camera_matrix, dist_coeffs)
            distance = np.linalg.norm(tvec[0][0])  # Euclidean distance
            cv2.putText(image, f"ID: {markerID} Dist: {distance:.2f}m", 
                        (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"[INFO] Detected marker ID: {markerID}, Distance: {distance:.2f} m")
    return image

# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (1000, int(frame.shape[0] * (1000 / frame.shape[1]))))

    for dict_name, dict_id in ARUCO_DICT.items():
        arucoDict = cv2.aruco.getPredefinedDictionary(dict_id)
        arucoParams = cv2.aruco.DetectorParameters()

        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(frame_resized, arucoDict, parameters=arucoParams)

        if ids is not None:
            print(f"[INFO] Detected markers in {dict_name}")
            frame_resized = aruco_display(corners, ids, rejected, frame_resized)
            break  # Stop checking other dictionaries once markers are detected

    cv2.imshow("ArUco Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()