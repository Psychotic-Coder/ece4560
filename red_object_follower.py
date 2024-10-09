#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2, PointField
import math

class RedObjectFollower:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('red_object_follower', anonymous=True)
        
        # Subscribers
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.rgb_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_image_callback)
        
        # Publisher for TurtleBot velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # OpenCV bridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Variables to store the red object position and distance
        self.red_object_center = None
        self.red_object_distance = None

        # Flag to control the robot's movement
        self.is_moving = False

    def rgb_image_callback(self, data):
        try:
            # Convert the ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Detect the red object in the image
        self.detect_red_object(cv_image)

    def depth_image_callback(self, data):
        if self.red_object_center is None:
            return

        try:
            # Convert the ROS depth image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Get the distance to the red object
        self.get_object_distance(depth_image)

    def detect_red_object(self, frame):
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV
        red_lower = np.array([0, 120, 70])
        red_upper = np.array([10, 255, 255])

        # Create a mask for red color
        red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

        # Find contours in the mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (most likely the red object)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 500:
                # Get the bounding rectangle for the largest red object
                x, y, w, h = cv2.boundingRect(largest_contour)
                self.red_object_center = (x + w // 2, y + h // 2)

                # Draw a rectangle and center point on the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, self.red_object_center, 5, (0, 255, 0), -1)

                # Show the processed image
                cv2.imshow("Red Object Detection", frame)
                cv2.waitKey(1)
            else:
                self.red_object_center = None

    def get_object_distance(self, depth_image):
        if self.red_object_center is not None:
            x, y = self.red_object_center
            self.red_object_distance = depth_image[y, x]

            # If the distance is valid, start moving towards the object
            if self.red_object_distance > 0 and self.red_object_distance < 10:
                self.move_towards_red_object()

    def move_towards_red_object(self):
        twist = Twist()

        if self.red_object_distance is not None:
            if self.red_object_distance > 0.1:  # If the object is farther than 10cm
                # Move towards the object
                twist.linear.x = 0.2  # Forward speed
                twist.angular.z = 0  # No rotation

                self.cmd_vel_pub.publish(twist)
                rospy.loginfo(f"Moving towards the red object. Distance: {self.red_object_distance}")
            else:
                # Stop the robot when it's close enough (within 10cm)
                twist.linear.x = 0
                twist.angular.z = 0
                self.cmd_vel_pub.publish(twist)
                rospy.loginfo("Red object is within 10cm. Stopping.")
                rospy.signal_shutdown("Reached the target!")

if __name__ == '__main__':
    try:
        # Instantiate the class and keep the program running
        RedObjectFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()