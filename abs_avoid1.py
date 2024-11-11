#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

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
        self.image_width = None

        # Motion control parameters
        self.max_speed = 0.5  # Maximum speed
        self.min_speed = 0.05  # Minimum speed for smooth start/stop
        self.acceleration = 0.02  # Rate of speed increase
        self.current_speed = 0.0  # Start with zero speed

    def rgb_image_callback(self, data):
        try:
            # Convert the ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Set image width for angular control
        if self.image_width is None:
            self.image_width = cv_image.shape[1]

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
        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])
        red_upper2 = np.array([180, 255, 255])

        # Create a mask for red color
        red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

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
            else:
                self.red_object_center = None

        cv2.imshow("Red Object Detection", frame)
        cv2.waitKey(1)

    def get_object_distance(self, depth_image):
        if self.red_object_center is not None:
            x, y = self.red_object_center
            self.red_object_distance = depth_image[y, x]

            # Move towards the red object if it's farther than 10 cm
            if self.red_object_distance > 0.1:
                self.move_towards_red_object()

    def move_towards_red_object(self):
        twist = Twist()

        # Smooth speed control
        if self.red_object_distance > 1.0:
            target_speed = self.max_speed
        elif self.red_object_distance > 0.1:
            target_speed = max(self.min_speed, self.red_object_distance * 0.5)
        else:
            target_speed = 0.0

        # Smooth acceleration or deceleration
        if self.current_speed < target_speed:
            self.current_speed = min(self.current_speed + self.acceleration, target_speed)
        elif self.current_speed > target_speed:
            self.current_speed = max(self.current_speed - self.acceleration, target_speed)

        twist.linear.x = self.current_speed

        # Angular control
        error = (self.red_object_center[0] - self.image_width / 2) / float(self.image_width / 2)
        twist.angular.z = -error * 0.5  # Proportional control

        self.cmd_vel_pub.publish(twist)
        rospy.loginfo(f"Speed: {self.current_speed:.2f}, Distance: {self.red_object_distance:.2f}m, Angular Error: {error:.2f}")

if __name__ == '__main__':
    try:
        # Instantiate the class and keep the program running
        follower = RedObjectFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
