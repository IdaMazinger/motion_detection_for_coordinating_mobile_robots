#!/usr/bin/env python
import datetime
import cv2
from typing import Tuple
from ultralytics import YOLO
import numpy as np
import json
import rospy
from motion_detection.msg import CoordinatesAndVector

# Absolute path to motion_detection package
ABSOLUTE_PATH = '<path_to_home_repo>/catkin_ws/src/motion_detection/'
# Camera to use (0: default, 1: usb on windows, 2: usb on ubuntu)
CAMERA_INDEX: int = 2
# Class id of object to track tracking_old(3: tennis ball)
CLASS_TO_DETECT: int = 3
VISUALIZE: bool = True
SAVE_VIDEO: bool = True

# Declare global variables type
PUBLISHER: rospy.Publisher
X_MAX: int
Y_MAX: int
PERSPECTIVE_TRANSFORMATION_MATRIX: np.array


def create_kalman_filter(noise_tolerance: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2, 0)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32) * noise_tolerance
    kf.statePre = np.array([[0], [0], [0], [0]], np.float32)
    kf.statePost = np.array([[0], [0], [0], [0]], np.float32)

    return kf


def create_video_writer(width, height) -> cv2.VideoWriter:
    frame_fps = 20
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        ABSOLUTE_PATH + 'tracking_videos/tracking' + str(datetime.datetime.now()) + '.mp4',
        fourcc, frame_fps, (width, height))
    return video_writer


def init_ros_node() -> rospy.Publisher:
    """Initializes the ROS node and the publisher"""
    rospy.init_node('object_tracker', anonymous=True)
    p = rospy.Publisher('motion_detection', CoordinatesAndVector, queue_size=10)
    rospy.sleep(2)
    return p


def get_calibration_dict() -> dict:
    """Initializes the rectangle points and area of interest with data from calibration.json"""
    with open(ABSOLUTE_PATH + 'calibration/calibration.json', 'r') as calibration_file:
        cdict: dict = json.load(calibration_file)
        point_list = cdict['markers']
        # Check if calibration is valid
        if len(point_list) != 4:
            raise Exception("Invalid rectangle points. Please rerun calibration.")

        return cdict


def set_parameters() -> None:
    """Sets edge parameters for camera coordinate system (coordinates will be published between 0 and these values)"""
    # print("Setting parameters for area of interest edges: x_edge=%f, y_edge=%f" % (x_edge, y_edge))
    rospy.loginfo("Setting parameters for area of interests ends at x=%f, y=%f" % (X_MAX, Y_MAX))
    rospy.set_param('x_edge_camera', X_MAX)
    rospy.set_param('y_edge_camera', Y_MAX)
    rospy.sleep(2)


def publish_coordinates(coordinates: Tuple[np.float32, np.float32],
                        motion_vector: Tuple[np.float32, np.float32]) -> None:
    """Publishes coordinates and vector to topic 'motion_detection'"""
    # print('Detected object at: coordinates =', coordinates, ", vector =", motion_vector)
    rospy.loginfo("Object detected at: %s, moving to: %s" % (coordinates, motion_vector))
    PUBLISHER.publish(coordinates, motion_vector)


def track_object() -> None:
    """Track objects of camera input in the area transformed with the calibration data"""
    video_writer = None
    if SAVE_VIDEO:
        video_writer = create_video_writer(X_MAX, Y_MAX)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # Create Kalman Filter with 0.02 tolerance
    kf = create_kalman_filter(0.02)
    # Load trained YOLO model
    model = YOLO(ABSOLUTE_PATH + 'trained_yolo_model.pt')

    last_prediction = (np.float32(0), np.float32(0))
    try:
        # Loop through the video frames
        while cap.isOpened() and not rospy.is_shutdown():
            success, original_frame = cap.read()

            # End if unable to read frame
            if not success:
                raise Exception("Unable to read from frame")

            # Transform frame perspective
            frame = cv2.warpPerspective(
                original_frame, PERSPECTIVE_TRANSFORMATION_MATRIX, (X_MAX, Y_MAX))

            # Use trained YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(source=frame, persist=True, classes=[CLASS_TO_DETECT],
                                  max_det=1, conf=0.01, verbose=False)

            # Ignore frame if no objects are detected
            if not results[0].boxes.id is None:
                # Get the boxes and track IDs
                box = results[0].boxes.xywh.cpu().tolist()[0]
                track_id = results[0].boxes.id.int().cpu().tolist()[0]
                x, y, w, h = box
                # Save center coordinates (x, y): (0, 0) -> top left corner from user perspective
                coordinates = (np.float32(x), np.float32(y))

                # Predict next coordinate
                kf.correct(np.array([[coordinates[0]], [coordinates[1]]]))
                predicted = kf.predict()
                px, py = np.float32(predicted[0][0]), np.float32(predicted[1][0])

                # Calculate motion vector
                motion_vector = (px - x, py - y)

                if (abs(motion_vector[0] - last_prediction[0]) > 1 and
                        abs(motion_vector[1] - last_prediction[1]) > 1):
                    publish_coordinates(coordinates, motion_vector)
                last_prediction = motion_vector

                # Show tracking results on frame
                cv2.circle(frame, (int(px), int(py)), 15, (20, 220, 0), 4)
                cv2.circle(frame, (int(x), int(y)), 15, (0, 20, 220), 1)
                frame = results[0].plot()

            # Visualize tracking result
            if VISUALIZE:
                cv2.imshow("Tracking result", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # Save video
            if SAVE_VIDEO:
                video_writer.write(frame)

    except Exception as ex:
        print("Exception occurred", ex)
    finally:
        # Release the video capture object and close the display window
        cap.release()
        if SAVE_VIDEO:
            video_writer.release()
        cv2.destroyAllWindows()
        rospy.loginfo("Safely terminated tracker node")


if __name__ == '__main__':
    PUBLISHER = init_ros_node()
    calibration_dict = get_calibration_dict()
    PERSPECTIVE_TRANSFORMATION_MATRIX = np.array(
        calibration_dict['perspective_transformation_matrix'], dtype=np.float32)
    X_MAX, Y_MAX = int(calibration_dict['x_max']), int(calibration_dict['y_max'])
    set_parameters()
    track_object()
