import numpy as np
import pickle
import cv2
import cv2.aruco as aruco

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}

MARKER_SIZE = 0.05

with open("cameraMatrix.pkl", "rb") as f:
    camera_matrix = pickle.load(f)

with open("dist.pkl", "rb") as f:
    dist_coeffs = pickle.load(f)

def calculate_real_distance(tvec1, tvec2):
    """Calculate the real-world distance between two markers using their translation vectors."""
    return np.linalg.norm(tvec1 - tvec2)

def aruco_display(corners, ids, rvecs, tvecs, image):
    marker_centers = []
    min_distance_threshold = 0.001

    if len(corners) > 0:
        ids = ids.flatten()

        for i, (markerCorner, markerID) in enumerate(zip(corners, ids)):
            corners_reshaped = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners_reshaped
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            marker_centers.append((cX, cY))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(
                image,
                f"ID: {markerID}",
                (topLeft[0], topLeft[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            text_rvec = f"r:[{rvec[0]:.2f}, {rvec[1]:.2f}, {rvec[2]:.2f}]"
            text_tvec = f"t:[{tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f}]"

            cv2.putText(
                image,
                text_rvec,
                (topLeft[0], topLeft[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            cv2.putText(
                image,
                text_tvec,
                (topLeft[0], topLeft[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

        if len(marker_centers) > 1:
            for i in range(len(marker_centers)):
                for j in range(i + 1, len(marker_centers)):
                    dist_m = calculate_real_distance(tvecs[i][0], tvecs[j][0])
                    if dist_m >= min_distance_threshold:
                        cv2.line(image, marker_centers[i], marker_centers[j], (255, 0, 0), 2)
                        mid_point = (
                            (marker_centers[i][0] + marker_centers[j][0]) // 2,
                            (marker_centers[i][1] + marker_centers[j][1]) // 2
                        )
                        cv2.putText(
                            image,
                            f"{dist_m:.2f} m",
                            mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )

    return image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (1000, int(frame.shape[0] * (1000 / frame.shape[1]))))

    all_corners = []
    all_ids = []
    all_rvecs = []
    all_tvecs = []

    for dict_name, dict_id in ARUCO_DICT.items():
        arucoDict = cv2.aruco.getPredefinedDictionary(dict_id)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

        corners, ids, rejected = detector.detectMarkers(frame_resized)
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                MARKER_SIZE,
                camera_matrix,
                dist_coeffs
            )
            all_corners.extend(corners)
            all_ids.extend(ids)
            all_rvecs.extend(rvecs)
            all_tvecs.extend(tvecs)

    if len(all_ids) > 0:
        all_ids = np.array(all_ids).reshape(-1, 1)
        all_rvecs = np.array(all_rvecs)
        all_tvecs = np.array(all_tvecs)

        frame_resized = aruco_display(all_corners, all_ids, all_rvecs, all_tvecs, frame_resized)

    cv2.imshow("ArUco Detection with Pose Axes", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()