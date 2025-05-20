import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import re
import threading

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('publisher_mcontrol')
        # Publishers for joint angles, Cartesian positions, and gripper commands
        self.publisher_ = self.create_publisher(Float32MultiArray, 'joint_pos', 10)
        self.fk_publisher = self.create_publisher(Float32MultiArray, 'cartesian_pos', 10)
        self.gripper_pub = self.create_publisher(Float32MultiArray, 'gripper_command', 10)
        
        # Subscriber for current joint state (in degrees)
        self.subscription = self.create_subscription(
            Float32MultiArray, 'joint_state', self.send_command, 10)
        
        # Dictionary for dynamic subscriptions to marker topics
        self.dynamic_subs = {}
        
        # Robot kinematics parameters
        self.theta = np.zeros(2)  # Current joint angles (degrees)
        self.l = [100, 160]       # Link lengths
        self.damping_factor = 0.05
        self.speed_factor = 0.5
        
        # Stage 1: Record desired positions (from markers, expected even IDs)
        # Stage 2: Object positions (expected odd IDs)
        self.desired_positions = {}  # { marker_id: {'pos': [x, y], 'yaw': value, 'last_update': time} }
        self.object_targets = {}     # { marker_id: {'pos': [x, y], 'yaw': value, 'last_update': time} }
        
        # For moving stage: list of object marker IDs (odd numbers) that have matching desired positions (even = object_id + 1)
        self.moving_ids = []
        self.current_object_index = 0
        # State machine steps: "open_pick", "move_to_object", "close_pick", "move_to_goal", "open_release"
        self.move_step = None
        self.state_start_time = self.get_clock().now()
        
        # Timeout for marker updates (in seconds)
        self.marker_timeout = 3.0
        
        # Operation stage: "recording" (Stage 1) or "moving" (Stage 2)
        self.operation_stage = "recording"
        
        # Timer to scan for marker topics (e.g. /desired_pos_{ID})
        self.create_timer(1.0, self.scan_for_topics)
        # Timer to remove stale markers
        self.create_timer(0.5, self.cleanup_markers)
        
        # Start a thread to wait for user input to transition stages
        threading.Thread(target=self.wait_for_start, daemon=True).start()

    def wait_for_start(self):
        input("Stage 1: Desired positions are being recorded (even marker IDs). Press Enter when ready to start moving stage...")
        self.get_logger().info("Switching to moving stage.")
        self.operation_stage = "moving"
        # Initialize the state machine for moving objects:
        self.move_step = "open_pick"
        self.current_object_index = 0
        self.state_start_time = self.get_clock().now()
        # For clarity, log the final desired positions recorded.
        self.get_logger().info(f"Final desired positions: {self.desired_positions}")

    def scan_for_topics(self):
        """
        Scan for topics matching desired_pos_{ID} and subscribe if not already done.
        """
        topics = self.get_topic_names_and_types()
        for topic, types in topics:
            m = re.match(r'^(?:/)?desired_pos_(\d+)$', topic)
            if m and topic not in self.dynamic_subs:
                marker_id = int(m.group(1))
                sub = self.create_subscription(
                    Float32MultiArray,
                    topic,
                    lambda msg, marker_id=marker_id: self.object_callback(msg, marker_id),
                    10)
                self.dynamic_subs[topic] = sub
                self.get_logger().info(f"Subscribed to topic: {topic} with marker id {marker_id}")

    def cleanup_markers(self):
        """
        Remove markers that have not been updated recently.
        """
        now = self.get_clock().now()
        if self.operation_stage == "recording":
            stale = [m_id for m_id, info in self.desired_positions.items()
                     if (now - info['last_update']).nanoseconds / 1e9 > self.marker_timeout]
            for m_id in stale:
                del self.desired_positions[m_id]
        else:  # moving stage
            stale = [m_id for m_id, info in self.object_targets.items()
                     if (now - info['last_update']).nanoseconds / 1e9 > self.marker_timeout]
            for m_id in stale:
                del self.object_targets[m_id]

    def object_callback(self, msg, marker_id):
        """
        Callback for marker messages.
        Expected message layout: [x, y, z, roll, pitch, yaw]
        During the recording stage, we only record desired positions (even marker IDs).
        During the moving stage, we record object positions (odd marker IDs).
        """
        if len(msg.data) < 6:
            self.get_logger().warn(f"Received insufficient data for marker {marker_id}")
            return
        
        pos = [float(msg.data[0]), float(msg.data[1])]
        yaw = float(msg.data[5])
        
        if self.operation_stage == "recording":
            if marker_id % 2 == 0:  # even IDs for desired positions
                self.desired_positions[marker_id] = {
                    'pos': pos,
                    'yaw': yaw,
                    'last_update': self.get_clock().now()
                }
                self.get_logger().info(f"Recorded desired position for marker {marker_id}: pos={pos}, yaw={yaw}")
            else:
                self.get_logger().info(f"Ignoring marker {marker_id} in recording stage. Expected desired markers to have even IDs.")
        else:  # moving stage
            if marker_id % 2 == 1:  # odd IDs for object positions
                self.object_targets[marker_id] = {
                    'pos': pos,
                    'yaw': yaw,
                    'last_update': self.get_clock().now()
                }
                self.get_logger().info(f"Updated object position for marker {marker_id}: pos={pos}, yaw={yaw}")
            else:
                self.get_logger().info(f"Ignoring marker {marker_id} in moving stage. Expected object markers to have odd IDs.")

    def forward_kinematics(self, l, theta):
        """
        Compute the end-effector (x, y) position from joint angles (radians).
        """
        x = l[0] * np.cos(theta[0]) + l[1] * np.cos(theta[0] + theta[1])
        y = l[0] * np.sin(theta[0]) + l[1] * np.sin(theta[0] + theta[1])
        return np.array([x, y])

    def jacobian_IK(self, l, theta):
        """
        Compute the 2x2 Jacobian for a 2-DOF arm.
        """
        J = np.array([
            [-l[0] * np.sin(theta[0]) - l[1] * np.sin(theta[0] + theta[1]),
             -l[1] * np.sin(theta[0] + theta[1])],
            [l[0] * np.cos(theta[0]) + l[1] * np.cos(theta[0] + theta[1]),
             l[1] * np.cos(theta[0] + theta[1])]
        ])
        return J

    def send_command(self, msg):
        """
        Callback triggered by joint state updates.
        In the recording stage, the node is only recording desired positions.
        In the moving stage the state machine controls the movement sequence:
          1. Open gripper for picking,
          2. Move to the object's current position,
          3. Close gripper to grasp,
          4. Move to the recorded goal position (from desired marker, which is object_id+1),
          5. Open gripper to release.
        """
        # Update current joint angles (in degrees)
        self.theta = np.array([msg.data[0], msg.data[1]])
        # Publish current Cartesian position for feedback
        current_pos = self.forward_kinematics(self.l, np.deg2rad(self.theta))
        cartesian_msg = Float32MultiArray()
        cartesian_msg.data = [float(current_pos[0]), float(current_pos[1])]
        self.fk_publisher.publish(cartesian_msg)
        
        # In recording stage, no movement commands are sent.
        if self.operation_stage == "recording":
            return
        
        # In moving stage, build list of object marker IDs (odd) that have a matching desired marker (object_id+1)
        self.moving_ids = sorted([obj_id for obj_id in self.object_targets.keys() if (obj_id + 1) in self.desired_positions])
        if len(self.moving_ids) == 0 or len(self.moving_ids) != len(self.desired_positions):
            self.get_logger().warn("Mismatch between recorded desired positions and detected object markers. Waiting...")
            return
        if self.current_object_index >= len(self.moving_ids):
            self.get_logger().info("All objects have been processed.")
            return
        
        current_id = self.moving_ids[self.current_object_index]
        threshold = 5.0  # distance threshold

        # Helper function to perform an IK update toward a target position.
        def move_toward(target_pos):
            pos_error = np.array(target_pos) - current_pos
            error_norm = np.linalg.norm(pos_error)
            error_vector = np.array([pos_error[0], pos_error[1], 0.0])
            J = self.jacobian_IK(self.l, np.deg2rad(self.theta))
            # Extend Jacobian for orientation control (dummy row)
            J_extended = np.vstack((J, np.array([1.0, 1.0])))
            inv_J_damped = np.linalg.inv(J_extended.T @ J_extended + self.damping_factor**2 * np.eye(2)) @ J_extended.T
            delta_q = inv_J_damped @ error_vector
            self.theta = self.theta + np.rad2deg(self.speed_factor * delta_q)
            pos_msg = Float32MultiArray()
            pos_msg.data = [float(self.theta[0]), float(self.theta[1])]
            self.publisher_.publish(pos_msg)
            return error_norm

        now = self.get_clock().now()
        elapsed = (now - self.state_start_time).nanoseconds / 1e9
        
        if self.move_step == "open_pick":
            # Step 1: Open gripper for picking
            gripper_msg = Float32MultiArray()
            gripper_msg.data = [1.0]  # 1.0 means open
            self.gripper_pub.publish(gripper_msg)
            self.get_logger().info("Opening gripper for picking object...")
            if elapsed > 1.0:
                self.move_step = "move_to_object"
                self.state_start_time = now

        elif self.move_step == "move_to_object":
            # Step 2: Move toward the object's current position (from object marker, odd ID)
            target = self.object_targets[current_id]['pos']
            error_norm = move_toward(target)
            self.get_logger().info(f"Moving to object {current_id} at {target}, error: {error_norm:.2f}")
            if error_norm < threshold:
                self.move_step = "close_pick"
                self.state_start_time = now
                self.get_logger().info("Reached object position. Closing gripper to pick object.")

        elif self.move_step == "close_pick":
            # Step 3: Close gripper to grasp the object
            gripper_msg = Float32MultiArray()
            gripper_msg.data = [0.0]  # 0.0 means closed
            self.gripper_pub.publish(gripper_msg)
            self.get_logger().info("Closing gripper to grasp object...")
            if elapsed > 1.0:
                self.move_step = "move_to_goal"
                self.state_start_time = now

        elif self.move_step == "move_to_goal":
            # Step 4: Move toward the recorded desired (goal) position.
            # For object marker ID X, its desired goal is recorded in desired_positions with key (X + 1)
            target = self.desired_positions[current_id + 1]['pos']
            error_norm = move_toward(target)
            self.get_logger().info(f"Moving to goal for object {current_id} at {target}, error: {error_norm:.2f}")
            if error_norm < threshold:
                self.move_step = "open_release"
                self.state_start_time = now
                self.get_logger().info("Reached goal position. Opening gripper to release object.")

        elif self.move_step == "open_release":
            # Step 5: Open gripper to release the object
            gripper_msg = Float32MultiArray()
            gripper_msg.data = [1.0]
            self.gripper_pub.publish(gripper_msg)
            self.get_logger().info("Opening gripper to release object...")
            if elapsed > 1.0:
                self.get_logger().info(f"Object {current_id} moved to desired position.")
                # Prepare for the next object
                self.current_object_index += 1
                self.move_step = "open_pick"
                self.state_start_time = now

def main(args=None):
    rclpy.init(args=args)
    node = SimplePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
