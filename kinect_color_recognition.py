#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class ColorRecognition:
    def __init__(self):
        # Initialize the node
        rospy.init_node('color_recognition_node', anonymous=True)

        # Subscribe to the Kinect's RGB image topic
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.image_callback)

        # Subscribe to the Kinect's depth image topic
        self.depth_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_callback)

        # Use CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Store the most recent RGB and depth images
        self.rgb_image = None
        self.depth_image = None

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # If both RGB and depth images are available, process them
        if self.rgb_image is not None and self.depth_image is not None:
            self.detect_colors(self.rgb_image)

    def depth_callback(self, data):
        try:
            # Convert the ROS depth image to OpenCV format
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

    def detect_colors(self, frame):
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges in HSV space
        red_lower = np.array([0, 120, 70])
        red_upper = np.array([10, 255, 255])

        green_lower = np.array([40, 40, 40])
        green_upper = np.array([70, 255, 255])

        blue_lower = np.array([100, 150, 0])
        blue_upper = np.array([140, 255, 255])

        # Create masks for each color
        red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)

        # Find contours for each color mask
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Display the detected colors and calculate the distance
        self.display_color_regions(frame, contours_red, (0, 0, 255), "Red")
        self.display_color_regions(frame, contours_green, (0, 255, 0), "Green")
        self.display_color_regions(frame, contours_blue, (255, 0, 0), "Blue")

        # Show the original image with detected colors
        cv2.imshow("Color Recognition", frame)
        cv2.waitKey(1)

    def display_color_regions(self, frame, contours, color, label):
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Only consider larger areas
                # Get the bounding rectangle for the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate the center of the contour
                center_x, center_y = x + w // 2, y + h // 2

                # If the depth image is available, get the depth at the contour's center
                if self.depth_image is not None:
                    # Extract the depth value at the center of the contour
                    distance = self.depth_image[center_y, center_x]

                    # Draw the contour, label, and distance on the image
                    cv2.drawContours(frame, [contour], -1, color, 2)
                    cv2.putText(frame, f"{label} - {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

if __name__ == '__main__':
    try:
        # Instantiate the ColorRecognition class
        ColorRecognition()
        # Keep the program alive
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
