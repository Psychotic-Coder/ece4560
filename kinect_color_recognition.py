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

        # Use CvBridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Process the image to recognize colors
        self.detect_colors(cv_image)

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

        # Display the detected colors
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
                # Draw the contour and label the color
                cv2.drawContours(frame, [contour], -1, color, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
