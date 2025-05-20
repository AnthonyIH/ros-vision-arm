import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from dynamixel_sdk import *
import numpy as np
import time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy


class HardwareInterfaceNode(Node):
    def __init__(self):
        super().__init__('hardware_interface_ros')
        self.get_logger().info('node is alive')

        # 40 Hz update rate
        self.timer_period = 1.0 / 40.0

        # Recovery flag to disable normal commands during recovery
        self.in_recovery = False

        # Dynamixel motor control parameters
        self.ADDR_TORQUE_ENABLE    = 64
        self.ADDR_OPERATING_MODE   = 11
        self.ADDR_GOAL_CURRENT     = 102
        self.ADDR_GOAL_VELOCITY    = 104
        self.ADDR_GOAL_POSTION     = 116
        self.ADDR_PRESENT_POSITION = 132
        self.ADDR_PRESENT_VELOCITY = 128
        self.ADDR_PRESENT_CURRENT  = 126

        self.MODE_CUR = 0
        self.MODE_VEL = 1
        self.MODE_POS = 5

        self.POS_SCALING = 0.087891
        self.VEL_SCALING = 0.087891 * 0.22888 * 6
        self.CUR_SCALING = 1

        # Limits for arm joints (motors 1 and 2)
        self.LIMIT_POS = 100
        self.LIMIT_VEL = 10
        self.LIMIT_CURRENT = 500
        self.limit_pos_tol = 1

        self.BAUDRATE = 1000000
        self.DEVICENAME = '/dev/ttyACM0'
        self.ACTIVATE_MOTORS = True

        # Define motor IDs:
        # Arm motors (1 and 2) are controlled via /joint_pos, etc.
        # Motor 3 is the end effector (gripper) and will be controlled separately.
        self.arm_ids = [1, 2]
        self.ee_id   = 3
        self.DXL_IDs = self.arm_ids + [self.ee_id]

        # State arrays (for all motors)
        self.joint_pos_all = np.zeros(len(self.DXL_IDs))
        self.joint_vel_all = np.zeros(len(self.DXL_IDs))
        self.joint_cur_all = np.zeros(len(self.DXL_IDs))

        # Motor 3 calibration parameters (gripper)
        self.calibration_velocity = 20.0  # deg/s
        self.current_threshold    = 1000   # mA
        self.open_angle  = 0.0
        self.closed_angle = 0.0

        # Collision variables
        self.collision_thresholds = {1: 750, 2: 500, 3: 2500}
        self.velocity_threshold = 0.1
        # Emergency stop
        self.emergency_publisher = self.create_publisher(Float32MultiArray, '/emergency_stop', 10)

        # Initialise Dynamixel communication
        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(2.0)

        if not self.portHandler.openPort():
            self.get_logger().error('Failed to open port')
            quit()

        if not self.portHandler.setBaudRate(self.BAUDRATE):
            self.get_logger().error('Failed to set baudrate')
            quit()

        self.operating_mode = -1
        self.set_operating_mode(self.MODE_POS)

        # ROS Subscribers for arm control (motors 1 and 2)
        self.sub_angle   = self.create_subscription(Float32MultiArray, '/joint_pos', self.desired_pos_callback, 10)
        self.sub_vel     = self.create_subscription(Float32MultiArray, '/joint_vel', self.desired_vel_callback, 10)
        self.sub_cur     = self.create_subscription(Float32MultiArray, '/joint_cur', self.desired_cur_callback, 10)
        self.sub_pos_rel = self.create_subscription(Float32MultiArray, '/joint_pos_rel', self.joint_pos_rel_callback, 10)
        self.fk_publisher = self.create_publisher(Float32MultiArray, '/cartesian_pos', 10)


        # Subscriber for gripper (motor 3) command.
        # Expected message: [ratio] where 0.0 = closed and 1.0 = open.
        self.sub_gripper = self.create_subscription(Float32MultiArray, '/gripper_command', self.gripper_command_callback, 10)

        # Publishers for joint state and calibration feedback
        self.publisher = self.create_publisher(Float32MultiArray, '/joint_state', 10)
        qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.ee_pub = self.create_publisher(Float32MultiArray, '/ee_min_max', qos_profile)

        # Timer for periodic joint state updates
        self.timer = self.create_timer(self.timer_period, self.joint_state_callback)

        self.motor_modes = {id: None for id in self.DXL_IDs}


        # Initialise all motors to 0 position
        initial_positions = {
            1: -90.0,  # Initial angle for joint 1
            2: 90.0, # Initial angle for joint 2
            3: 0.0    # Gripper - gets calibrated later anyway
        }

        for id, angle in initial_positions.items():
            self.set_pos(id, angle)


        # Allow time for initialisation and then calibrate motor 3 (gripper)
        time.sleep(2.0)
        self.calibrate_motor3()

    def carrying_callback(self, msg):
        """
        Callback to update whether the robot is carrying an object.
        Expected message: [state] where 0 = not carrying, 1 = carrying.
        """
        if len(msg.data) > 0:
            self.carrying_object = bool(msg.data[0])
            self.get_logger().info(f"Updated carrying state: {self.carrying_object}")


    def calibrate_motor3(self):
        """Calibrate end effector motor (ID 3) with retry logic"""
        max_attempts = 3
        min_separation = 5.0  # deg, minimum difference between open and closed
        for attempt in range(1, max_attempts + 1):
            try:
                self.get_logger().info(f"Starting motor 3 calibration (Attempt {attempt}/{max_attempts})...")

                # Switch to velocity mode for calibration
                self.set_operating_mode_for_motor(self.ee_id, self.MODE_VEL)

                # Find closed position
                self.move_until_obstruction(direction=1)
                closed = self.get_pos(self.ee_id)

                # Find open position
                self.move_until_obstruction(direction=-1)
                open_pos = self.get_pos(self.ee_id)

                separation = abs(open_pos - closed)
                if separation < min_separation:
                    self.get_logger().warn(
                        f"Calibration difference too small ({separation:.2f}° < {min_separation}°). Retrying..."
                    )
                    if attempt < max_attempts:
                        time.sleep(0.5)
                        continue
                    else:
                        self.get_logger().error("Max calibration attempts reached. Calibration failed.")
                        break

                # Successful calibration
                self.open_angle = open_pos
                self.closed_angle = closed

                # Publish calibration results (order: [open, closed])
                ee_msg = Float32MultiArray()
                ee_msg.data = [self.open_angle, self.closed_angle]
                self.ee_pub.publish(ee_msg)
                self.get_logger().info(
                    f"Calibration complete: Open={self.open_angle:.2f}°, Closed={self.closed_angle:.2f}°"
                )

                # Return motor 3 to position mode and set it to the closed position
                self.set_operating_mode_for_motor(self.ee_id, self.MODE_POS)
                self.set_pos(self.ee_id, self.closed_angle)
                break

            except Exception as e:
                self.get_logger().error(f"Calibration attempt {attempt} failed: {str(e)}")
                if attempt < max_attempts:
                    time.sleep(0.5)
                else:
                    self.get_logger().error("Max calibration attempts reached. Calibration failed.")


    def set_operating_mode_for_motor(self, motor_id, mode):
        if self.motor_modes[motor_id] == mode:
            return  # Skip if already in the correct mode

        self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 0)
        self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, self.ADDR_OPERATING_MODE, mode)
        if self.ACTIVATE_MOTORS:
            self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 1)

        self.motor_modes[motor_id] = mode
        time.sleep(0.1)


    def move_until_obstruction(self, direction=1):
        """Move motor 3 until current spike is detected"""
        vel_cmd = int(direction * self.calibration_velocity / self.VEL_SCALING)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, vel_cmd)
        
        while True:
            current = abs(self.get_cur(self.ee_id))
            if current >= self.current_threshold:
                # Stop the motor
                self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, 0)
                time.sleep(0.5)  # Wait for motor to stop
                
                # If moving in the open direction, add a small nudge
                if direction == -1:
                    nudge_vel = int(direction * (self.calibration_velocity * 0.2) / self.VEL_SCALING)
                    self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, nudge_vel)
                    time.sleep(0.2)  # Nudge for a short time
                    self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, 0)
                break
            time.sleep(0.01)


    def set_pos(self, id, pos):
        """Set the position of a motor in position control mode.
           For arm motors, enforce joint limits. Motor 3 uses calibrated values."""
        
        self.set_operating_mode_for_motor(id, self.MODE_POS)

        if id in self.arm_ids:
            if pos > self.LIMIT_POS:
                pos = self.LIMIT_POS
            elif pos < -self.LIMIT_POS:
                pos = -self.LIMIT_POS

        # Convert degrees to motor command value and send command.
        pos_cmd = int((pos + 180.0) / self.POS_SCALING)
        self.packetHandler.write4ByteTxRx(self.portHandler, id, self.ADDR_GOAL_POSTION, pos_cmd)
        return

    def set_vel(self, id, vel):
        """Set the velocity of a motor in velocity control mode.
           For arm motors, enforce velocity limits."""
        self.set_operating_mode(self.MODE_VEL)
        if id in self.arm_ids:
            if vel > self.LIMIT_VEL:
                vel = self.LIMIT_VEL
            elif vel < -self.LIMIT_VEL:
                vel = -self.LIMIT_VEL

        vel_cmd = int(vel / self.VEL_SCALING)
        self.packetHandler.write4ByteTxRx(self.portHandler, id, self.ADDR_GOAL_VELOCITY, vel_cmd)
        return

    def set_cur(self, id, cur):
        """Set the current of a motor in current control mode.
           For arm motors, enforce current limits."""
        self.set_operating_mode(self.MODE_CUR)
        if id in self.arm_ids:
            if cur > self.LIMIT_CURRENT:
                cur = self.LIMIT_CURRENT
            elif cur < -self.LIMIT_CURRENT:
                cur = -self.LIMIT_CURRENT

        self.packetHandler.write2ByteTxRx(self.portHandler, id, self.ADDR_GOAL_CURRENT, int(cur))
        return

    def set_operating_mode(self, mode):
        """Change the operating mode for all motors."""
        if mode == self.operating_mode:
            return 1

        for id in self.DXL_IDs:
            # Disable torque, change mode, and re-enable torque
            self.packetHandler.write1ByteTxRx(self.portHandler, id, self.ADDR_TORQUE_ENABLE, 0)
            self.packetHandler.write1ByteTxRx(self.portHandler, id, self.ADDR_OPERATING_MODE, mode)
            if self.ACTIVATE_MOTORS:
                self.packetHandler.write1ByteTxRx(self.portHandler, id, self.ADDR_TORQUE_ENABLE, 1)

            mode_actual, _, _ = self.packetHandler.read1ByteTxRx(self.portHandler, id, self.ADDR_OPERATING_MODE)
        if mode_actual == mode:
            self.operating_mode = mode_actual
            self.get_logger().info("Updated mode to %d" % self.operating_mode)
            return 1
        else:
            self.get_logger().info("Failed to update mode")
            return 0

    # ----------------------------
    # Arm Control Callbacks (motors 1 and 2)
    # ----------------------------

    def desired_pos_callback(self, pos_msg):
        """Callback for desired arm joint positions.
           Expects a Float32MultiArray with 2 values."""
        targets = pos_msg.data
        if len(targets) != len(self.arm_ids):
            self.get_logger().info("Number of given angles doesn't match number of arm motors")
            return
        for idx, id in enumerate(self.arm_ids):
            self.set_pos(id, targets[idx])
        return

    def desired_vel_callback(self, vel_msg):
        """Callback for desired arm joint velocities.
           Expects a Float32MultiArray with 2 values."""
        targets = vel_msg.data
        if len(targets) != len(self.arm_ids):
            self.get_logger().info("Number of given velocities doesn't match number of arm motors")
            return
        for idx, id in enumerate(self.arm_ids):
            predicted_pos = self.get_pos(id) + self.timer_period * self.get_vel(id)
            if (predicted_pos > self.LIMIT_POS and targets[idx] > 0.0) or \
               (predicted_pos < -self.LIMIT_POS and targets[idx] < 0.0):
                self.set_vel(id, 0.0)
            else:
                self.set_vel(id, targets[idx])
        return
    
    def forward_kinematics(self, angles):
        """Calculate forward kinematics for the 2-DOF arm"""
        # Convert angles from degrees to radians
        theta = np.radians(angles)
        # Link lengths (same as in your control_dls script)
        l = [100, 160]  # Using the lengths from your hardware interface
        
        # Calculate end effector position
        x = l[0] * np.cos(theta[0]) + l[1] * np.cos(theta[0] + theta[1])
        y = l[0] * np.sin(theta[0]) + l[1] * np.sin(theta[0] + theta[1])
        
        return [x, y]

    def desired_cur_callback(self, cur_msg):
        """Callback for desired arm joint currents.
           Expects a Float32MultiArray with 2 values."""
        targets = cur_msg.data
        if len(targets) != len(self.arm_ids):
            self.get_logger().info("Number of given currents doesn't match number of arm motors")
            return
        for idx, id in enumerate(self.arm_ids):
            self.set_cur(id, targets[idx])
        return

    def joint_pos_rel_callback(self, pos_rel_msg):
        """Callback for relative position changes for arm joints.
           Expects a Float32MultiArray with 2 values."""
        rel_positions = pos_rel_msg.data
        if len(rel_positions) != len(self.arm_ids):
            self.get_logger().info("Mismatch in the number of arm joints and relative position commands")
            return
        for idx, id in enumerate(self.arm_ids):
            current_pos = self.get_pos(id)
            target_pos = current_pos + rel_positions[idx]
            if target_pos > self.LIMIT_POS:
                target_pos = self.LIMIT_POS
            elif target_pos < -self.LIMIT_POS:
                target_pos = -self.LIMIT_POS
            self.set_pos(id, target_pos)
        return

    # ----------------------------
    # Gripper Control Callback (motor 3)
    # ----------------------------
    def gripper_command_callback(self, msg):
        """
        Callback for gripper commands.
        Expected message: [ratio] where 0.0 = closed, 1.0 = open.
        """
        if len(msg.data) < 1:
            self.get_logger().warn("Received empty gripper command")
            return
        ratio = msg.data[0]
        # Clamp ratio between 0 and 1
        ratio = max(0.0, min(1.0, ratio))
        # Map ratio to target position between the closed and open calibration values
        target_pos = self.closed_angle + (self.open_angle - self.closed_angle) * ratio
        self.get_logger().info(f"Gripper command: ratio={ratio}, target_pos={target_pos}")
        self.set_pos(self.ee_id, target_pos)
        tolerance = 15.0  # degrees tolerance
        current_gripper_pos = self.get_pos(self.ee_id)
        if abs(current_gripper_pos - self.closed_angle) < tolerance:
            self.carrying_object = True
            self.get_logger().info("Gripper is closed: Object is being carried.")
        else:
            self.carrying_object = False
            self.get_logger().info("Gripper is open: No object is being carried.")


    # ----------------------------
    # Reading Motor States
    # ----------------------------
    def get_pos(self, id):
        pos, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, id, self.ADDR_PRESENT_POSITION)
        return float(pos) * self.POS_SCALING - 180.0

    def get_vel(self, id):
        vel, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, id, self.ADDR_PRESENT_VELOCITY)
        return float(self.s16(vel)) * self.VEL_SCALING

    def get_cur(self, id):
        cur, _, _ = self.packetHandler.read2ByteTxRx(self.portHandler, id, self.ADDR_PRESENT_CURRENT)
        if cur > 1750:
            cur -= 2**16        
        return cur

    def s16(self, value):
        return -(value & 0x8000) | (value & 0x7fff)

    def joint_state_callback(self):
        """
        Periodic callback to update and publish the state of all motors.
        Publishes positions, velocities, and currents.
        """
        state_msg = Float32MultiArray()
        for idx, id in enumerate(self.DXL_IDs):
            self.joint_pos_all[idx] = self.get_pos(id)
            self.joint_vel_all[idx] = self.get_vel(id)
            self.joint_cur_all[idx] = self.get_cur(id)    
            # Log the current for each motor for debugging
            self.get_logger().info(
                f"Motor {id}: Current = {self.joint_cur_all[idx]:.2f} mA, Velocity = {self.joint_vel_all[idx]:.2f}"
            )
        
        
        # For arm motors, log if they are at or beyond limits
        for id in self.arm_ids:
            index = self.DXL_IDs.index(id)
            if self.joint_pos_all[index] >= self.LIMIT_POS or self.joint_pos_all[index] <= -self.LIMIT_POS:
                self.get_logger().info(f"Arm Joint {id} at or beyond limit: Position = {self.joint_pos_all[index]}")

        # Enforce hard joint limits in velocity mode for arm motors
        if self.operating_mode == self.MODE_VEL:
            for id in self.arm_ids:
                index = self.DXL_IDs.index(id)
                predicted_pos = self.joint_pos_all[index] + self.timer_period * self.joint_vel_all[index]
                if (predicted_pos > (self.LIMIT_POS + self.limit_pos_tol)) and (self.joint_vel_all[index] > 0.0):
                    self.set_vel(id, 0.0)
                elif (predicted_pos < (-self.LIMIT_POS + self.limit_pos_tol)) and (self.joint_vel_all[index] < 0.0):
                    self.set_vel(id, 0.0)

        # Check collision for each motor individually
        for motor_id in self.DXL_IDs:
            idx = self.DXL_IDs.index(motor_id)
            current = abs(self.joint_cur_all[idx])
            velocity = abs(self.joint_vel_all[idx])
            threshold = self.collision_thresholds.get(motor_id, 1000)
            if current > threshold and velocity > self.velocity_threshold:
                self.get_logger().warn(
                    f"Collision detected on Motor {motor_id}! Current = {current:.2f} mA, Velocity = {velocity:.2f}"
                )
                self.set_vel(motor_id, 0.0)
                stop_msg = Float32MultiArray()
                stop_msg.data = [1.0]
                self.emergency_publisher.publish(stop_msg)
                # Begin recovery: open gripper and return arm to home positions
                self.get_logger().info("Opening gripper and returning to home positions due to collision.")
                self.set_pos(self.ee_id, self.open_angle)
                time.sleep(0.2)
                home_positions = {1: 90.0, 2: -90.0, 3: self.open_angle}
                # Set in recovery mode to avoid interference from other callbacks
                self.in_recovery = True
                for id, angle in home_positions.items():
                    self.set_pos(id, angle)
                # Optionally, wait for a short period to ensure recovery is complete
                time.sleep(0.5)
                self.in_recovery = False
                break

        # Populate state message with all motor states
        state_msg.data.extend(self.joint_pos_all)
        state_msg.data.extend(self.joint_vel_all)
        state_msg.data.extend(self.joint_cur_all)

        # Calculate and publish forward kinematics for arm joints
        arm_angles = [self.joint_pos_all[self.DXL_IDs.index(id)] for id in self.arm_ids]
        cartesian_pos = self.forward_kinematics(arm_angles)
                
        # Create and publish cartesian position message
        cartesian_msg = Float32MultiArray()
        cartesian_msg.data = [float(cartesian_pos[0]), float(cartesian_pos[1])]
        self.fk_publisher.publish(cartesian_msg)

        self.publisher.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HardwareInterfaceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
