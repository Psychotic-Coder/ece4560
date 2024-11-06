import numpy as np
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

def cubic_spline_coefficients(t, y):
    """
    Computes the cubic spline coefficients for a set of waypoints.
    :param t: Array of parameter values for each waypoint (usually normalized time).
    :param y: Array of y-coordinates for each waypoint.
    :return: Arrays a, b, c, d representing the spline coefficients for each segment.
    """
    n = len(t) - 1
    h = np.diff(t)
    
    # Solve for a (the original points) and setup the linear system for c
    a = y
    alpha = [3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1] for i in range(1, n)]
    alpha.insert(0, 0)  # No second derivative constraint at first point
    
    # Tridiagonal matrix setup
    l = np.ones(n + 1)
    mu = np.zeros(n)
    z = np.zeros(n + 1)
    
    for i in range(1, n):
        l[i] = 2 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    
    l[n] = 1
    c = np.zeros(n + 1)
    b = np.zeros(n)
    d = np.zeros(n)
    
    # Back substitution
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    
    return a, b, c, d, t

def evaluate_spline(a, b, c, d, t, x):
    """
    Evaluates the cubic spline at a given x using the coefficients.
    :param a, b, c, d: Spline coefficients for each segment.
    :param t: Array of parameter values for each waypoint.
    :param x: The input parameter (normalized time).
    :return: The interpolated y-value.
    """
    # Find the right interval
    for i in range(len(t) - 1):
        if t[i] <= x <= t[i + 1]:
            dx = x - t[i]
            return a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
    return a[-1]  # Return the last point if x is out of bounds

def create_cubic_spline(waypoints_x, waypoints_y):
    t = np.linspace(0, 1, len(waypoints_x))
    a_x, b_x, c_x, d_x, t_x = cubic_spline_coefficients(t, waypoints_x)
    a_y, b_y, c_y, d_y, t_y = cubic_spline_coefficients(t, waypoints_y)
    
    def spline_x(x):
        return evaluate_spline(a_x, b_x, c_x, d_x, t_x, x)

    def spline_y(x):
        return evaluate_spline(a_y, b_y, c_y, d_y, t_y, x)
    
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
