import cv2
import numpy as np


def create_markers() -> None:
    """Create markers to print for calibrating the area of interest for the camera"""
    # Define marker dictionary
    marker_image = np.zeros((256, 256, 1), dtype="uint8")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Create markers for each corner (ids: 1, 2, 3, 4)
    marker_image = cv2.aruco.generateImageMarker(dictionary, 1, 250, marker_image)
    cv2.imwrite("markers/marker_top_left.png", marker_image)
    marker_image = cv2.aruco.generateImageMarker(dictionary, 2, 250, marker_image)
    cv2.imwrite("markers/marker_top_right.png", marker_image)
    marker_image = cv2.aruco.generateImageMarker(dictionary, 3, 250, marker_image)
    cv2.imwrite("markers/marker_bottom_right.png", marker_image)
    marker_image = cv2.aruco.generateImageMarker(dictionary, 4, 250, marker_image)
    cv2.imwrite("markers/marker_bottom_left.png", marker_image)


if __name__ == "__main__":
    create_markers()

