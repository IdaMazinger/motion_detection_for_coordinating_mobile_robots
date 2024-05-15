#!/usr/bin/env python
import rospy
import tf
import cv2
import numpy as np
from motion_detection.msg import CoordinatesAndVector
from geometry_msgs.msg import PoseStamped, Quaternion
from typing import Tuple

# Calibration points
AFFINE_TRANSFORMATION_MATRIX: np.array

# Corners from robot map
TOP_LEFT = np.array([-0.2930329442024231, 0.20282116532325745], dtype=np.float32)
TOP_RIGHT = np.array([1.0359686613082886, 0.2613855302333832], dtype=np.float32)
BOTTOM_LEFT = np.array([-0.29444265365600586, -1.1416414976119995], dtype=np.float32)
BOTTOM_RIGHT = np.array([1.1025389432907104, -1.1351553201675415], dtype=np.float32)

# North corner in RViz for calculating orientation
MAP_NORTH_VECTOR = (TOP_RIGHT[0] - TOP_LEFT[0], TOP_RIGHT[1] - TOP_LEFT[1])

# ROS Publisher
PUBLISHER: rospy.Publisher


def init_coordinator() -> None:
    """Initializes the coordinator node and sets global environments from parameters"""
    # Initialize ROS node
    rospy.init_node('coordinator', anonymous=True)
    # Read params
    if rospy.has_param('x_edge_camera') and rospy.has_param('y_edge_camera'):
        x_edge_camera = rospy.get_param('x_edge_camera')
        y_edge_camera = rospy.get_param('y_edge_camera')
    else:
        rospy.logerr("Unable to read parameters. Run tracking node first.")
        raise Exception("Unable to read parameters. Run tracking node first.")
    rospy.loginfo(rospy.get_caller_id() + "Read Params: x_edge: %s, y_edge: %s" % (x_edge_camera, y_edge_camera))
    # Calculate affine transformation matrix (take top left, top right and bottom right corners)
    camera_corners = np.float32([[0, 0], [x_edge_camera, 0], [x_edge_camera, y_edge_camera]])
    robot_corners = np.float32([TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT])

    # Set affine transformation matrix
    global AFFINE_TRANSFORMATION_MATRIX
    AFFINE_TRANSFORMATION_MATRIX = cv2.getAffineTransform(camera_corners, robot_corners)
    rospy.loginfo(rospy.get_caller_id() + "Setting transformation matrix to to: %s" % AFFINE_TRANSFORMATION_MATRIX)
    rospy.sleep(2)


def run_coordinator() -> None:
    """Creates subscriber and publisher.
    Subscriber listens to motion_detection topic.
    Publisher sends next coordinate to moveit."""
    # Init subscriber
    rospy.Subscriber('motion_detection', CoordinatesAndVector, motion_detection_callback)
    # Init publisher
    global PUBLISHER
    PUBLISHER = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=5)
    # Keep node alive
    rospy.loginfo(rospy.get_caller_id() + "Subscriber and Publisher started")
    rospy.spin()


def motion_detection_callback(data: CoordinatesAndVector) -> None:
    """Callback function for motion_detection Subscriber.
    Converts incoming coordinates and vector to coordinate system of robot and publishes next coordinates."""
    rospy.loginfo(rospy.get_caller_id() + "I heard coordinate: %s, vector: %s" % (data.coordinates, data.motion_vector))
    # Make cure it doesn't bump into objects
    buffer_size = 30  # real size around 17-19 from camera perspective
    buffer = [buffer_size, buffer_size]
    if data.motion_vector[0] < 0:
        buffer[0] *= -1
    if data.motion_vector[1] < 0:
        buffer[1] *= -1

    # Calculate next coordinates
    next_coordinates = np.float32([[data.coordinates[0] + data.motion_vector[0] - buffer[0],
                                    data.coordinates[1] + data.motion_vector[1] - buffer[1]]])
    next_coordinates = cv2.transform(np.array([next_coordinates]), AFFINE_TRANSFORMATION_MATRIX)[0][0]
    # rospy.loginfo(next_coordinates.tolist())
    orientation = convert_to_robot_orientation(data.motion_vector)
    # Create PoseStamped goal
    goal = create_goal(next_coordinates.tolist(), orientation)
    # Log and publish next coordinates
    rospy.loginfo("Sending goal to robot: {0}".format(goal))
    PUBLISHER.publish(goal)


def convert_to_robot_orientation(motion_vector: Tuple[float, float]) -> Quaternion:
    """Convert motion_vector to orientation quaternion (from Euler angles)."""
    # Calculate angle in radians between North vector and motion vertex
    x = np.array(MAP_NORTH_VECTOR)
    y = np.array(motion_vector)
    unit_x = x / np.linalg.norm(x)
    unit_y = y / np.linalg.norm(y)
    angle = np.arccos(np.dot(unit_x, unit_y))
    # Create and return Quaternion from Euler angles (only y set)
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))


def create_goal(coordinates: Tuple[float, float], orientation: Quaternion) -> PoseStamped:
    """Create PoseStamped object from next_coordinates and orientation."""
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "map"
    goal.pose.position.x = coordinates[0]
    goal.pose.position.y = coordinates[1]
    goal.pose.position.z = 0.0  # no z position
    goal.pose.orientation = orientation
    return goal


if __name__ == '__main__':
    init_coordinator()
    run_coordinator()

