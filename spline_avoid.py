#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import time

class RedObjectFollowerWithDynamicSpeed:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('red_object_follower_with_dynamic_speed', anonymous=True)
        
        # Subscribers for RGB and Depth images
        self.rgb_image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.rgb_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/depth_registered/image_raw", Image, self.depth_image_callback)
        
        # Publisher for TurtleBot velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # OpenCV bridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Variables for red object detection and obstacle avoidance
        self.red_object_center = None
        self.red_object_distance = None
        self.image_width = None
        self.obstacle_distance = float('inf')
        self.obstacle_threshold = 0.5  # Threshold for obstacle detection in meters

        # Velocity parameters
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.min_linear_velocity = 0.05
        self.max_linear_velocity = 0.4
        self.max_angular_velocity = 0.5
        self.velocity_step = 0.01  # Increment for smooth ramping

        # Distance thresholds for dynamic velocity adjustment
        self.distance_slow_threshold = 0.3  # Slow down if within 30cm
        self.distance_fast_threshold = 1.0  # Max speed if farther than 1m

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

        self.get_object_distance_and_check_obstacles(depth_image)

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

    def get_object_distance_and_check_obstacles(self, depth_image):
        if self.red_object_center is not None:
            x, y = self.red_object_center
            self.red_object_distance = depth_image[y, x]

            center_region = depth_image[240:300, 310:330]
            self.obstacle_distance = np.nanmin(center_region)

            if self.red_object_distance > 0 and not np.isnan(self.red_object_distance):
                self.move_towards_red_object()

    def move_towards_red_object(self):
        twist = Twist()

        if self.red_object_distance > 0.1:  # Object farther than 10 cm
            if self.obstacle_distance < self.obstacle_threshold:
                rospy.logwarn("Obstacle detected! Stopping.")
                self.smooth_stop()
                twist.angular.z = 0.3  # Turn away from the obstacle
            else:
                # Adjust velocity dynamically based on distance
                target_linear_velocity = self.compute_dynamic_velocity(self.red_object_distance)
                self.current_linear_velocity = self.smooth_velocity(self.current_linear_velocity, target_linear_velocity)
                twist.linear.x = self.current_linear_velocity

                # Adjust angular velocity to align with the object
                error = (self.red_object_center[0] - self.image_width / 2) / float(self.image_width / 2)
                self.current_angular_velocity = max(min(-error * self.max_angular_velocity, self.max_angular_velocity), -self.max_angular_velocity)
                twist.angular.z = self.current_angular_velocity
        else:
            self.smooth_stop()

        self.cmd_vel_pub.publish(twist)

    def compute_dynamic_velocity(self, distance):
        if distance < self.distance_slow_threshold:
            return self.min_linear_velocity
        elif distance > self.distance_fast_threshold:
            return self.max_linear_velocity
        else:
            return self.min_linear_velocity + (self.max_linear_velocity - self.min_linear_velocity) * \
                   ((distance - self.distance_slow_threshold) / (self.distance_fast_threshold - self.distance_slow_threshold))

    def smooth_velocity(self, current, target):
        if current < target:
            return min(current + self.velocity_step, target)
        elif current > target:
            return max(current - self.velocity_step, target)
        return target

    def smooth_stop(self):
        while self.current_linear_velocity > 0:
            self.current_linear_velocity = max(self.current_linear_velocity - self.velocity_step, 0)
            twist = Twist()
            twist.linear.x = self.current_linear_velocity
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)

if __name__ == '__main__':
    try:
        RedObjectFollowerWithDynamicSpeed()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
