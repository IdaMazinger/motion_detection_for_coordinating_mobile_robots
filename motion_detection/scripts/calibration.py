import cv2
import json
import numpy as np

# Absolute path to motion_detection package
ABSOLUTE_PATH = '<path_to_home_repo>/catkin_ws/src/motion_detection/'
# Camera to use (0: default, 1: usb on windows, 2: usb on ubuntu)
CAMERA_INDEX = 2


def detect_markers() -> None:
    """Detects markers from camera image. Stores calibration data in calibration.json and saves calibration images."""
    markers = []

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise Exception("Could not open camera")

    success, frame = cap.read()
    if not success:
        raise Exception("Could not read camera frame")

    # Detect markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(frame)

    # Error if no markers are found
    if not isinstance(marker_ids, np.ndarray):
        raise Exception("No markers found.")

    markers_to_identify = [1, 2, 3, 4]
    for i, marker_id_array in enumerate(marker_ids):
        marker_id = int(marker_id_array[0])

        # Skip marker if not corner marker
        if marker_id not in markers_to_identify:
            print("Detected marker", marker_id, "not in markers list.")
            continue

        # First corner in list is top-left of marker
        top_left_marker_corner = marker_corners[i][0][0].tolist()
        markers.append((marker_id, top_left_marker_corner))

    # Error if not all four corners are detected
    if len(markers) != 4:
        raise Exception("Could not detect all markers.", "Markers detected:", markers)

    # Sort makers
    markers.sort(key=lambda x: x[0])
    print("Markers:", markers)

    # Calculate transformation matrix
    marker_points = np.float32([markers[0][1], markers[1][1], markers[2][1], markers[3][1]])
    x_max = 400
    y_max = 400
    plane_corners = np.float32([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]])
    perspective_matrix = cv2.getPerspectiveTransform(marker_points, plane_corners)
    print("Perspective matrix:", perspective_matrix)

    # Save calibration data (markers and transformation matrix)
    file_path = ABSOLUTE_PATH + 'calibration/'
    with open(file_path + 'calibration.json', 'w') as calibration_file:
        calibration_dict = {
            "markers": markers,
            "perspective_transformation_matrix": perspective_matrix.tolist(),
            "x_max": x_max,
            "y_max": y_max
        }
        json.dump(calibration_dict, calibration_file)

    # Save calibration image to file
    output_image = cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
    cv2.imwrite(file_path + 'calibration_image.jpg', output_image)
    output_image_perspective = cv2.warpPerspective(frame, perspective_matrix, (x_max, y_max))
    cv2.imwrite(file_path + 'calibration_image_perspective.jpg', output_image_perspective)


if __name__ == "__main__":
    detect_markers()
