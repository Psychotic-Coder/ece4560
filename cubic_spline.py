import numpy as np
import scipy.interpolate as spi
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# Initialize global variables for position and orientation
current_x = 0.0
current_y = 0.0
current_yaw = 0.0

def odom_callback(data):
    global current_x, current_y, current_yaw
    # Extract the current position
    current_x = data.pose.pose.position.x
    current_y = data.pose.pose.position.y
    # Extract the current orientation in yaw
    orientation_q = data.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (_, _, current_yaw) = euler_from_quaternion(orientation_list)

def create_cubic_spline(waypoints_x, waypoints_y):
    # Create a parameter array for the waypoints
    t = np.linspace(0, 1, len(waypoints_x))
    # Generate cubic splines for x and y
    spline_x = spi.CubicSpline(t, waypoints_x)
    spline_y = spi.CubicSpline(t, waypoints_y)
    return spline_x, spline_y

def track_trajectory(spline_x, spline_y, duration=10.0):
    rospy.init_node('turtlebot_cubic_spline_tracker', anonymous=True)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/odom', Odometry, odom_callback)
    rate = rospy.Rate(10)  # 10 Hz control loop

    start_time = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        current_time = rospy.Time.now().to_sec() - start_time
        t = current_time / duration  # Normalize time to [0, 1]

        if t > 1.0:  # End of trajectory
            break

        # Get desired (x, y) positions from the spline
        desired_x = spline_x(t)
        desired_y = spline_y(t)

        # Compute errors
        error_x = desired_x - current_x
        error_y = desired_y - current_y
        desired_yaw = np.arctan2(error_y, error_x)
        yaw_error = desired_yaw - current_yaw

        # Normalize yaw error to [-pi, pi]
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # Control gains
        k_linear = 0.5
        k_angular = 2.0

        # Compute control inputs
        linear_velocity = k_linear * np.hypot(error_x, error_y)
        angular_velocity = k_angular * yaw_error

        # Publish velocity command
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_velocity
        cmd_vel.angular.z = angular_velocity
        velocity_publisher.publish(cmd_vel)

        rate.sleep()

if __name__ == "__main__":
    # Define waypoints for the cubic spline
    waypoints_x = [0, 1, 2, 3, 4]
    waypoints_y = [0, 1, 0, -1, 0]

    # Create the cubic spline
    spline_x, spline_y = create_cubic_spline(waypoints_x, waypoints_y)

    # Track the trajectory
    try:
        track_trajectory(spline_x, spline_y)
    except rospy.ROSInterruptException:
        pass
