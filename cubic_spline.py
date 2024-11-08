import rospy
from geometry_msgs.msg import Twist
import numpy as np

def compute_cubic_spline(x_points, y_points, num_samples=100):
    """
    Compute cubic spline trajectory.

    Parameters:
    - x_points: List of x-coordinates.
    - y_points: List of y-coordinates.
    - num_samples: Number of samples for interpolation.

    Returns:
    - trajectory: Array of (x, y) coordinates along the spline.
    """
    def cubic_coeffs(x0, x1, y0, y1, dy0, dy1):
        A = np.array([
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [3, 2, 1, 0]
        ])
        b = np.array([y0, y1, dy0, dy1])
        return np.linalg.solve(A, b)

    trajectory = []
    for i in range(len(x_points) - 1):
        t = np.linspace(0, 1, num_samples // (len(x_points) - 1))
        dx = x_points[i + 1] - x_points[i]
        dy = y_points[i + 1] - y_points[i]
        slope = dy / dx if dx != 0 else 0
        coeffs = cubic_coeffs(0, 1, y_points[i], y_points[i + 1], 0, slope)

        x_i = np.linspace(x_points[i], x_points[i + 1], num_samples // (len(x_points) - 1))
        y_i = coeffs[0] * t**3 + coeffs[1] * t**2 + coeffs[2] * t + coeffs[3]

        trajectory.extend(np.column_stack((x_i, y_i)))

    return np.array(trajectory)

def offset_trajectory(trajectory, offset_distance):
    """
    Apply an offset to the trajectory perpendicular to its direction.

    Parameters:
    - trajectory: Array of (x, y) waypoints.
    - offset_distance: Distance to offset perpendicular to the trajectory.

    Returns:
    - offset_trajectory: Array of (x, y) waypoints with offset.
    """
    offset_trajectory = []
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]
        norm = np.sqrt(dx**2 + dy**2)

        # Perpendicular offset
        if norm > 0:
            offset_x = -dy * offset_distance / norm
            offset_y = dx * offset_distance / norm
        else:
            offset_x = offset_y = 0

        offset_trajectory.append([trajectory[i][0] + offset_x, trajectory[i][1] + offset_y])

    return np.array(offset_trajectory)

def track_trajectory(trajectory, pub, rate):
    """
    Track a given trajectory.

    Parameters:
    - trajectory: List of (x, y) waypoints to track.
    - pub: ROS Publisher to send Twist messages.
    - rate: ROS Rate for the publishing loop.
    """
    twist = Twist()
    for i in range(1, len(trajectory)):
        dx = trajectory[i][0] - trajectory[i - 1][0]
        dy = trajectory[i][1] - trajectory[i - 1][1]

        linear_velocity = np.sqrt(dx**2 + dy**2) / rate.sleep_dur.to_sec()
        angular_velocity = np.arctan2(dy, dx)

        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        pub.publish(twist)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('offset_cubic_spline_tracker')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rate = rospy.Rate(10)

    # Define waypoints for cubic spline
    x_points = [0, 0.5, 1.0, 1.5]
    y_points = [0, 0.2, 0.0, -0.2]

    # Compute cubic spline
    trajectory = compute_cubic_spline(x_points, y_points, num_samples=100)
    
    # Apply 10cm offset
    offset_trajectory = offset_trajectory(trajectory, offset_distance=0.1)

    # Track the offset trajectory
    rospy.loginfo("Tracking trajectory with offset...")
    track_trajectory(offset_trajectory, pub, rate)

    # Stop the robot
    twist = Twist()
    pub.publish(twist)
    rospy.loginfo("Tracking complete.")
