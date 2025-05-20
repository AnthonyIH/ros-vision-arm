import rclpy  # Import ROS client library for Python
from rclpy.node import Node  # Import Node class from ROS Python library
from std_msgs.msg import Float32MultiArray  # Import message type for ROS
from dynamixel_sdk import *  # Import Dynamixel SDK for servo control
import numpy as np  # Import numpy for numerical operations
from rclpy.logging import LoggingSeverity  # Import LoggingSeverity for setting log levels

# Defining the SimplePublisher class which inherits from Node
class SimplePublisher(Node):

    def __init__(self):
        super().__init__('publisher_mcontrol')
        # Publisher for the joint angles on the 'joint_pos' topic
        self.publisher_ = self.create_publisher(Float32MultiArray, 'joint_pos', 10)
        # Publisher for the Cartesian position on the 'cartesian_pos' topic
        self.fk_publisher = self.create_publisher(Float32MultiArray, 'cartesian_pos', 10)
        # Subscriber to receive the joint state (expected to contain at least 2 values)
        self.subscription = self.create_subscription(
            Float32MultiArray, 'joint_state', self.send_command, 10)
        # Initialize current joint angles (in degrees)
        self.theta = np.zeros(2)
        # Link lengths for the 2-DOF arm
        self.l = [100, 70]
        # Desired end-effector Cartesian target 
        self.des_pos = [0.0, 170.0]
        self.q = [0.0, 0.0]
        # Maximum Cartesian step allowed (same units as l)
        self.step = 10.0
        # Damping factor for the Damped Least Squares (DLS) method
        self.damping_factor = 0.05
        # Gain for the Jacobian Transpose method (unused here)
        self.gain = 0.5

    def forward_kinematics(self, l, theta):
        """
        Compute the end-effector (x,y) position given the link lengths and joint angles.
        The joint angles (theta) should be provided in radians.
        """
        x = l[0] * np.cos(theta[0]) + l[1] * np.cos(theta[0] + theta[1])
        y = l[0] * np.sin(theta[0]) + l[1] * np.sin(theta[0] + theta[1])
        return np.array([x, y])

    def jacobian_IK(self, l, theta):
        """
        Compute the 2x2 Jacobian matrix for the 2-DOF arm.
        theta should be provided in radians.
        """
        J = np.array([
            [-l[0] * np.sin(theta[0]) - l[1] * np.sin(theta[0] + theta[1]),
             -l[1] * np.sin(theta[0] + theta[1])],
            [ l[0] * np.cos(theta[0]) + l[1] * np.cos(theta[0] + theta[1]),
              l[1] * np.cos(theta[0] + theta[1])]
        ])
        return J

    def anal_IK(self, l, P):
        D = (P[0]**2 + P[1]**2 - l[0]**2 - l[1]**2) / (2 * l[0] * l[1])
        self.q[1] = np.arctan2(np.sqrt(1 - D**2) / (D + 0.00001))
        self.q[0] = np.arctan2(P[1] / P[0]) - np.arctan2(l[1] * np.sin(self.q[1]) / (l[0] + l[1] * np.cos(self.q[1])))
        return self.q 

    def send_command(self, msg):
        # Create a message for joint position updates
        pos = Float32MultiArray()
        # Update current joint angles from the incoming joint state (assumed in degrees)
        self.theta = np.array([msg.data[0], msg.data[1]])
        # Compute the current Cartesian position from the joint angles
        current_pos = self.forward_kinematics(self.l, np.deg2rad(self.theta))
        # Compute error relative to a desired position (if used)
        error = np.array(self.des_pos) - current_pos
        # Clamp the error to avoid large jumps
        if np.linalg.norm(error) > self.step:
            error = self.step * (error / np.linalg.norm(error))
        # Compute the Jacobian based on the current joint angles (converted to radians)
        J = self.jacobian_IK(self.l, np.deg2rad(self.theta))

        # Compute the damped pseudo-inverse of the Jacobian (DLS method)
        inv_J_damped = np.linalg.inv(J.T @ J + self.damping_factor**2 * np.eye(J.shape[1])) @ J.T
        delta_q_dls = inv_J_damped @ error

        # Update the joint angles using the computed delta (convert delta from radians to degrees)
        self.theta = self.theta + np.rad2deg(delta_q_dls)

        # Publish the updated joint angles (this is the command sent to the robot)
        pos.data = [float(self.theta[0]), float(self.theta[1])]
        self.publisher_.publish(pos)
        # Optionally log the updated joint angles:
        # self.get_logger().info(f"Updated Theta: {self.theta}")

        ## Here was the issue with your code ##
        
        # Now, compute the updated Cartesian position from the updated joint angles
        updated_cartesian = self.forward_kinematics(self.l, np.deg2rad(self.theta))
        # Create a message for the Cartesian position
        cartesian_msg = Float32MultiArray()
        cartesian_msg.data = [float(updated_cartesian[0]), float(updated_cartesian[1])]
        # Publish the computed Cartesian position
        self.fk_publisher.publish(cartesian_msg)
        # Optionally log the Cartesian position:
        # self.get_logger().info(f"Published Cartesian pos: {cartesian_msg.data}")

# The main function which serves as the entry point for the program
def main(args=None):
    rclpy.init(args=args)  # Initialize the ROS2 Python client library
    simple_publisher = SimplePublisher()  # Create an instance of the SimplePublisher
    try:
        rclpy.spin(simple_publisher)  # Keep the node alive and listening for messages
    except KeyboardInterrupt:  # Allow the program to exit on a keyboard interrupt (Ctrl+C)
        pass
    simple_publisher.destroy_node()  # Properly destroy the node
    rclpy.shutdown()  # Shutdown the ROS2 Python client library

# This condition checks if the script is executed directly (not imported)
if __name__ == '__main__':
    main()  # Execute the main function
