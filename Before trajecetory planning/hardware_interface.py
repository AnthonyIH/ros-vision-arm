#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from dynamixel_sdk import *
import numpy as np
import time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import os
import signal
import csv

class HardwareInterfaceNode(Node):
    def __init__(self):
        super().__init__('hardware_interface_ros')
        self.get_logger().info('Hardware interface node is alive.')

        # 40 Hz update rate
        self.timer_period = 1.0 / 40.0

        # Recovery flags
        self.in_recovery = False
        self.ignore_collision_until = 0  # temporary disable collision detection

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
        self.LIMIT_VEL = 5
        self.LIMIT_CURRENT = 500
        self.limit_pos_tol = 1

        self.BAUDRATE = 1000000
        self.DEVICENAME = '/dev/ttyACM0'
        self.ACTIVATE_MOTORS = True

        # Define motor IDs:
        self.arm_ids = [1, 2]
        self.ee_id   = 3
        self.DXL_IDs = self.arm_ids + [self.ee_id]

        # State arrays
        self.joint_pos_all = np.zeros(len(self.DXL_IDs))
        self.joint_vel_all = np.zeros(len(self.DXL_IDs))
        self.joint_cur_all = np.zeros(len(self.DXL_IDs))

        # Motor 3 calibration parameters (gripper)
        self.calibration_velocity = 20.0  # deg/s
        self.current_threshold    = 1000   # mA
        # For a tool, the gripper should remain closed initially.
        self.open_angle  = 0.0   # open angle command
        self.closed_angle = 0.0  # closed angle command

        # Collision thresholds (time-average limits)
        # For this simulation, the thresholds for motors 1 & 2 are set to trigger collision detection.
        self.collision_thresholds = {1: 150, 2: 100, 3: 2000}
        self.velocity_threshold = 0.01

        # Emergency stop publisher
        self.emergency_publisher = self.create_publisher(Float32MultiArray, '/emergency_stop', 10)

        # Initialize Dynamixel communication
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

        # ROS Subscribers for arm control
        self.sub_angle   = self.create_subscription(Float32MultiArray, '/joint_pos', self.desired_pos_callback, 10)
        self.sub_vel     = self.create_subscription(Float32MultiArray, '/joint_vel', self.desired_vel_callback, 10)
        self.sub_cur     = self.create_subscription(Float32MultiArray, '/joint_cur', self.desired_cur_callback, 10)
        self.sub_pos_rel = self.create_subscription(Float32MultiArray, '/joint_pos_rel', self.joint_pos_rel_callback, 10)
        self.fk_publisher = self.create_publisher(Float32MultiArray, '/cartesian_pos', 10)

        # Subscriber for gripper command
        self.sub_gripper = self.create_subscription(Float32MultiArray, '/gripper_command', self.gripper_command_callback, 10)

        # Publishers for joint state and calibration feedback
        self.publisher = self.create_publisher(Float32MultiArray, '/joint_state', 10)
        qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.ee_pub = self.create_publisher(Float32MultiArray, '/ee_min_max', qos_profile)

        # Timer for periodic joint state updates
        self.timer = self.create_timer(self.timer_period, self.joint_state_callback)
        self.motor_modes = {id: None for id in self.DXL_IDs}

        # Initialize motors to initial positions
        # For arm motors, home position is (90, -90). For the gripper, we start in the closed state.
        initial_positions = {1: 90.0, 2: -90.0, 3: self.closed_angle}
        for id, angle in initial_positions.items():
            self.set_pos(id, angle)

        time.sleep(2.0)
        self.calibrate_motor3()

        # Initialize current history dictionary (1-second window)
        self.current_history = {id: [] for id in self.DXL_IDs}

        # CSV Logging Setup: Save to /home/pi/ws/motor_thresholds_log_force_3.csv
        save_dir = "/home/pi/ws"
        os.makedirs(save_dir, exist_ok=True)
        log_file_path = os.path.join(save_dir, "motor_thresholds_log_force_3.csv")
        self.log_file = open(log_file_path, "w", newline="")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(["timestamp", "motor_id", "current_mA", "velocity"])
        self.get_logger().info(f"Logging motor thresholds to {log_file_path}")

    def shutdown_procedure(self):
        self.get_logger().info("Shutdown procedure initiated.")
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()
            self.get_logger().info("CSV log file closed successfully.")
        self.destroy_node()
        rclpy.shutdown()

    def calibrate_motor3(self):
        while True:
            self.get_logger().info("Starting motor 3 calibration...")
            self.set_operating_mode_for_motor(self.ee_id, self.MODE_VEL)
            self.move_until_obstruction(direction=1)
            self.closed_angle = self.get_pos(self.ee_id)
            self.move_until_obstruction(direction=-1)
            self.open_angle = self.get_pos(self.ee_id)
            angle_diff = abs(self.open_angle - self.closed_angle)
            self.get_logger().info(f"Calibration result: Open={self.open_angle:.2f}°, Closed={self.closed_angle:.2f}°, Difference={angle_diff:.2f}°")
            if angle_diff >= 5.0:
                break
            self.get_logger().warn("Calibration failed: Difference between open and closed angle is less than 5 degrees. Re-calibrating...")
            time.sleep(1.0)  # Optional delay before retrying

        # Publish calibration result
        ee_msg = Float32MultiArray()
        ee_msg.data = [self.open_angle, self.closed_angle]
        self.ee_pub.publish(ee_msg)
        self.get_logger().info(f"Calibration complete: Open={self.open_angle:.2f}°, Closed={self.closed_angle:.2f}°")
        self.set_operating_mode_for_motor(self.ee_id, self.MODE_POS)
        self.set_pos(self.ee_id, self.closed_angle)

    def set_operating_mode_for_motor(self, motor_id, mode):
        if self.motor_modes[motor_id] == mode:
            return
        self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 0)
        self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, self.ADDR_OPERATING_MODE, mode)
        if self.ACTIVATE_MOTORS:
            self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, self.ADDR_TORQUE_ENABLE, 1)
        self.motor_modes[motor_id] = mode
        time.sleep(0.1)

    def move_until_obstruction(self, direction=1):
        vel_cmd = int(direction * self.calibration_velocity / self.VEL_SCALING)
        self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, vel_cmd)
        while True:
            current = abs(self.get_cur(self.ee_id))
            if current >= self.current_threshold:
                self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, 0)
                time.sleep(0.5)
                if direction == -1:
                    nudge_vel = int(direction * (self.calibration_velocity * 0.2) / self.VEL_SCALING)
                    self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, nudge_vel)
                    time.sleep(0.2)
                    self.packetHandler.write4ByteTxRx(self.portHandler, self.ee_id, self.ADDR_GOAL_VELOCITY, 0)
                break
            time.sleep(0.01)

    def set_pos(self, id, pos):
        self.set_operating_mode_for_motor(id, self.MODE_POS)
        if id in self.arm_ids:
            if pos > self.LIMIT_POS:
                pos = self.LIMIT_POS
            elif pos < -self.LIMIT_POS:
                pos = -self.LIMIT_POS
        pos_cmd = int((pos + 180.0) / self.POS_SCALING)
        self.packetHandler.write4ByteTxRx(self.portHandler, id, self.ADDR_GOAL_POSTION, pos_cmd)

    def set_vel(self, id, vel):
        self.set_operating_mode(self.MODE_VEL)
        if id in self.arm_ids:
            if vel > self.LIMIT_VEL:
                vel = self.LIMIT_VEL
            elif vel < -self.LIMIT_VEL:
                vel = -self.LIMIT_VEL
        vel_cmd = int(vel / self.VEL_SCALING)
        self.packetHandler.write4ByteTxRx(self.portHandler, id, self.ADDR_GOAL_VELOCITY, vel_cmd)

    def set_cur(self, id, cur):
        self.set_operating_mode(self.MODE_CUR)
        if id in self.arm_ids:
            if cur > self.LIMIT_CURRENT:
                cur = self.LIMIT_CURRENT
            elif cur < -self.LIMIT_CURRENT:
                cur = -self.LIMIT_CURRENT
        self.packetHandler.write2ByteTxRx(self.portHandler, id, self.ADDR_GOAL_CURRENT, int(cur))

    def set_operating_mode(self, mode):
        if mode == self.operating_mode:
            return 1
        for id in self.DXL_IDs:
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

    # ---------------------------
    # Arm Control Callbacks
    # ---------------------------
    def desired_pos_callback(self, pos_msg):
        targets = pos_msg.data
        if len(targets) != len(self.arm_ids):
            self.get_logger().info("Number of given angles doesn't match number of arm motors")
            return
        for idx, id in enumerate(self.arm_ids):
            self.set_pos(id, targets[idx])

    def desired_vel_callback(self, vel_msg):
        targets = vel_msg.data
        if len(targets) != len(self.arm_ids):
            self.get_logger().info("Number of given velocities doesn't match number of arm motors")
            return
        for idx, id in enumerate(self.arm_ids):
            predicted_pos = self.get_pos(id) + self.timer_period * self.get_vel(id)
            if (predicted_pos > self.LIMIT_POS and targets[idx] > 0.0) or (predicted_pos < -self.LIMIT_POS and targets[idx] < 0.0):
                self.set_vel(id, 0.0)
            else:
                self.set_vel(id, targets[idx])

    def forward_kinematics(self, angles):
        theta = np.radians(angles)
        l = [100, 155]
        x = l[0] * np.cos(theta[0]) + l[1] * np.cos(theta[0] + theta[1])
        y = l[0] * np.sin(theta[0]) + l[1] * np.sin(theta[0] + theta[1])
        return [x, y]

    def desired_cur_callback(self, cur_msg):
        targets = cur_msg.data
        if len(targets) != len(self.arm_ids):
            self.get_logger().info("Number of given currents doesn't match number of arm motors")
            return
        for idx, id in enumerate(self.arm_ids):
            self.set_cur(id, targets[idx])

    def joint_pos_rel_callback(self, pos_rel_msg):
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

    # ---------------------------
    # Gripper Control Callback
    # ---------------------------
    def gripper_command_callback(self, msg):
        if len(msg.data) < 1:
            self.get_logger().warn("Received empty gripper command")
            return
        ratio = max(0.0, min(1.0, msg.data[0]))
        target_pos = self.closed_angle + (self.open_angle - self.closed_angle) * ratio
        self.get_logger().info(f"Gripper command: ratio={ratio}, target_pos={target_pos}")
        self.set_pos(self.ee_id, target_pos)
        tolerance = 20.0
        current_gripper_pos = self.get_pos(self.ee_id)
        if abs(current_gripper_pos - self.closed_angle) < tolerance:
            self.carrying_object = True
            self.get_logger().info("Gripper is closed: Object is being carried.")
        else:
            self.carrying_object = False
            self.get_logger().info("Gripper is open: No object is being carried.")

    # ---------------------------
    # Motor State Reading Methods
    # ---------------------------
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

    # ---------------------------
    # Recovery Method: Moves arm to home then opens gripper
    # ---------------------------
    def recover_home(self):
        self.get_logger().info("Recovering: moving to home positions.")
        # Disable collision detection for 2 seconds
        self.ignore_collision_until = time.time() + 2.0
        # Clear current history
        self.current_history = {motor_id: [] for motor_id in self.DXL_IDs}

        # Phase 1: Move arm joints to home positions (using angles, not Cartesian points)
        # Here, motor 1 is set to 90° and motor 2 is set to -90°.
        home_positions_arm = {1: 90.0, 2: -90.0}
        for motor_id, angle in home_positions_arm.items():
            self.set_pos(motor_id, angle)
        # Keep the gripper closed initially
        self.set_pos(self.ee_id, self.closed_angle)
        self.get_logger().info("Arm joints reached home position. Now opening gripper.")

        # Phase 2: Open the gripper (motor 3)
        self.set_pos(self.ee_id, self.open_angle)
        self.get_logger().info("Recovery complete: Home positions commanded and gripper opened.")

    # ---------------------------
    # Periodic Joint State Callback (with CSV logging and collision check)
    # ---------------------------
    def joint_state_callback(self):
        if self.in_recovery:
            return

        current_time = time.time()
        state_msg = Float32MultiArray()
        for idx, id in enumerate(self.DXL_IDs):
            self.joint_pos_all[idx] = self.get_pos(id)
            self.joint_vel_all[idx] = self.get_vel(id)
            self.joint_cur_all[idx] = self.get_cur(id)
            self.get_logger().info(
                f"Motor {id}: Current = {self.joint_cur_all[idx]:.2f} mA, Velocity = {self.joint_vel_all[idx]:.2f}"
            )
            self.current_history[id].append((current_time, abs(self.joint_cur_all[idx])))
            self.current_history[id] = [(t, c) for (t, c) in self.current_history[id] if current_time - t <= 2.0]

        for id in self.arm_ids:
            index = self.DXL_IDs.index(id)
            if self.joint_pos_all[index] >= self.LIMIT_POS or self.joint_pos_all[index] <= -self.LIMIT_POS:
                self.get_logger().info(f"Arm Joint {id} at or beyond limit: Position = {self.joint_pos_all[index]}")

        if self.operating_mode == self.MODE_VEL:
            for id in self.arm_ids:
                index = self.DXL_IDs.index(id)
                predicted_pos = self.joint_pos_all[index] + self.timer_period * self.joint_vel_all[index]
                if (predicted_pos > (self.LIMIT_POS + self.limit_pos_tol)) and (self.joint_vel_all[index] > 0.0):
                    self.set_vel(id, 0.0)
                elif (predicted_pos < (-self.LIMIT_POS + self.limit_pos_tol)) and (self.joint_vel_all[index] < 0.0):
                    self.set_vel(id, 0.0)

        # Modified collision detection: For arm motors, trigger recovery when average current exceeds threshold (ignoring velocity)
        if current_time >= self.ignore_collision_until:
            for motor_id in self.DXL_IDs:
                idx = self.DXL_IDs.index(motor_id)
                history = self.current_history[motor_id]
                if history:
                    avg_current = sum(c for t, c in history) / len(history)
                    threshold = self.collision_thresholds.get(motor_id, 1000)
                    if motor_id in self.arm_ids:
                        velocity = abs(self.joint_vel_all[idx])
                        if avg_current > threshold and velocity < self.velocity_threshold:
                            self.get_logger().warn(
                                f"Collision detected on Arm Motor {motor_id}! Avg Current = {avg_current:.2f} mA, Velocity = {velocity:.2f}"
                            )
                            self.set_vel(motor_id, 0.0)
                            self.get_logger().info("Collision detected on arm. Initiating recovery (rehoming).")
                            self.in_recovery = True
                            self.recover_home()
                            self.in_recovery = False
                            return
                    else:
                        velocity = abs(self.joint_vel_all[idx])
                        if avg_current > threshold and velocity > self.velocity_threshold:
                            self.get_logger().warn(
                                f"Collision detected on Motor {motor_id}! Avg Current = {avg_current:.2f} mA, Velocity = {velocity:.2f}"
                            )
                            self.set_vel(motor_id, 0.0)
                            self.get_logger().info("Collision detected. Initiating recovery.")
                            self.in_recovery = True
                            self.recover_home()
                            self.in_recovery = False
                            return

        state_msg.data.extend(self.joint_pos_all)
        state_msg.data.extend(self.joint_vel_all)
        state_msg.data.extend(self.joint_cur_all)

        arm_angles = [self.joint_pos_all[self.DXL_IDs.index(id)] for id in self.arm_ids]
        cartesian_pos = self.forward_kinematics(arm_angles)
        cartesian_msg = Float32MultiArray()
        cartesian_msg.data = [float(cartesian_pos[0]), float(cartesian_pos[1])]
        self.fk_publisher.publish(cartesian_msg)

        self.publisher.publish(state_msg)

        timestamp = time.time()
        for id in self.DXL_IDs:
            idx = self.DXL_IDs.index(id)
            current_val = self.joint_cur_all[idx]
            velocity_val = self.joint_vel_all[idx]
            self.csv_writer.writerow([f"{timestamp:.3f}", id, f"{current_val:.2f}", f"{velocity_val:.2f}"])
        self.log_file.flush()

def main(args=None):
    rclpy.init(args=args)
    node = HardwareInterfaceNode()

    def signal_handler(sig, frame):
        node.get_logger().info(f"Signal {sig} received, shutting down.")
        node.shutdown_procedure()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Exception in spin: {e}")
    finally:
        if rclpy.ok():
            node.shutdown_procedure()

if __name__ == '__main__':
    main()
