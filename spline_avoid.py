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

        # Obstacle detection
        self.obstacle_detected = False
        self.obstacle_threshold = 0.5  # Distance in meters to consider something an obstacle

        # Motion control parameters
        self.max_speed = 0.5
        self.min_speed = 0.05
        self.acceleration = 0.02
        self.current_speed = 0.0

    def rgb_image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        if self.image_width is None:
            self.image_width = cv_image.shape[1]

        self.detect_red_object(cv_image)

    def depth_image_callback(self, data):
        if self.red_object_center is None:
            return

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self.get_object_distance_and_check_obstacle(depth_image)

    def detect_red_object(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])
        red_upper2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                self.red_object_center = (x + w // 2, y + h // 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, self.red_object_center, 5, (0, 255, 0), -1)
            else:
                self.red_object_center = None

        cv2.imshow("Red Object Detection", frame)
        cv2.waitKey(1)

    def get_object_distance_and_check_obstacle(self, depth_image):
        if self.red_object_center is not None:
            x, y = self.red_object_center
            self.red_object_distance = depth_image[y, x]

            if self.red_object_distance > 0 and not np.isnan(self.red_object_distance):
                self.check_for_obstacles(depth_image)

                if not self.obstacle_detected:
                    self.move_towards_red_object()

    def check_for_obstacles(self, depth_image):
        x, y = self.red_object_center
        obstacle_region = depth_image[y - 20:y + 20, x - 20:x + 20]  # Region around the object

        obstacle_distances = obstacle_region[~np.isnan(obstacle_region)]

        if obstacle_distances.size > 0 and np.min(obstacle_distances) < self.obstacle_threshold:
            self.obstacle_detected = True
            self.navigate_around_obstacle()
        else:
            self.obstacle_detected = False

    def navigate_around_obstacle(self):
        twist = Twist()

        twist.linear.x = 0.0
        twist.angular.z = 0.5  # Rotate to find a clear path
        self.cmd_vel_pub.publish(twist)
        rospy.loginfo("Obstacle detected. Navigating around...")

    def move_towards_red_object(self):
        twist = Twist()

        if self.red_object_distance > 1.0:
            target_speed = self.max_speed
        elif self.red_object_distance > 0.1:
            target_speed = max(self.min_speed, self.red_object_distance * 0.5)
        else:
            target_speed = 0.0

        if self.current_speed < target_speed:
            self.current_speed = min(self.current_speed + self.acceleration, target_speed)
        elif self.current_speed > target_speed:
            self.current_speed = max(self.current_speed - self.acceleration, target_speed)

        twist.linear.x = self.current_speed

        error = (self.red_object_center[0] - self.image_width / 2) / float(self.image_width / 2)
        twist.angular.z = -error * 0.5

        self.cmd_vel_pub.publish(twist)
        rospy.loginfo(f"Speed: {self.current_speed:.2f}, Distance: {self.red_object_distance:.2f}m, Angular Error: {error:.2f}")

if __name__ == '__main__':
    try:
        follower = RedObjectFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
