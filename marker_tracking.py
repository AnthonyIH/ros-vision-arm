#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from scipy.interpolate import CubicSpline
from std_msgs.msg import Float32MultiArray
import numpy as np


class SimplePublisher(Node):
    def __init__(self):
        super().__init__('publisher_mcontrol')
        # Publishers
        self.joint_pub = self.create_publisher(Float32MultiArray, 'joint_pos', 10)
        self.fk_pub    = self.create_publisher(Float32MultiArray, 'cartesian_pos', 10)

        # State
        self.path_targets   = []       # list of (x_mm, y_mm)
        self.current_wp     = 0
        self._last_msg_data = None     # flat list of floats
        self.goal_xy        = None     # final marker-1 position

        # Subs
        self.create_subscription(
            Float32MultiArray, 'next_step', self.next_step_callback, 10
        )
        self.theta = np.zeros(2)
        self.create_subscription(
            Float32MultiArray, 'joint_state', self.joint_state_callback, 10
        )

        # Kinematics / IK params
        self.link_lengths = [100.0, 160.0]  # mm
        self.damping      = 0.05            # DLS factor
        self.speed        = 3.0             # step‐size
        self.threshold    = 45.0            # mm

    def interpolate_spline(self, raw_targets, points_per_segment=10):
        t      = np.arange(len(raw_targets))
        xs     = np.array([p[0] for p in raw_targets])
        ys     = np.array([p[1] for p in raw_targets])
        cs_x   = CubicSpline(t, xs)
        cs_y   = CubicSpline(t, ys)

        t_new  = np.linspace(0, len(raw_targets)-1,
                            num=(len(raw_targets)-1)*points_per_segment + 1)
        xs_new = cs_x(t_new)
        ys_new = cs_y(t_new)
        return list(zip(xs_new.tolist(), ys_new.tolist()))

    def next_step_callback(self, msg: Float32MultiArray):
        d = list(msg.data)
        # 1) empty → clear everything
        if not d:
            self.get_logger().info("Received empty path → stopping")
            self.path_targets   = []
            self.current_wp     = 0
            self._last_msg_data = []
            self.goal_xy        = None
            return

        # 2) only on new, even-length data
        if d != self._last_msg_data and len(d) >= 2 and len(d) % 2 == 0:
            raw_targets = [(d[i], d[i+1]) for i in range(0, len(d), 2)]
            self.current_wp     = 0
            self._last_msg_data = d

            # drop first if already at or within threshold
            cur_xy = self.forward_kinematics(self.theta)
            if np.linalg.norm(np.array(raw_targets[0]) - cur_xy) < self.threshold:
                raw_targets.pop(0)

            # 3) try spline smoothing
            try:
                self.path_targets = self.interpolate_spline(raw_targets,
                                                            points_per_segment=2)
            except ImportError:
                self.get_logger().warn("SciPy not found, using raw waypoints")
                self.path_targets = raw_targets
            except Exception as e:
                self.get_logger().warn(f"Spline failed ({e}), using raw waypoints")
                self.path_targets = raw_targets

            # 4) record the **final** spline point as our marker₁ goal
            if self.path_targets:
                self.goal_xy = np.array(self.path_targets[-1])
                self.get_logger().info(
                    f"Loaded path: {len(raw_targets)} → {len(self.path_targets)} points; "
                    f"goal = {self.goal_xy.tolist()}"
                )

    def forward_kinematics(self, theta_deg: np.ndarray) -> np.ndarray:
        t = np.radians(theta_deg)
        x = self.link_lengths[0]*np.cos(t[0]) \
          + self.link_lengths[1]*np.cos(t[0]+t[1])
        y = self.link_lengths[0]*np.sin(t[0]) \
          + self.link_lengths[1]*np.sin(t[0]+t[1])
        return np.array([x, y])

    def jacobian(self, theta_deg: np.ndarray) -> np.ndarray:
        t = np.radians(theta_deg)
        l1, l2 = self.link_lengths
        return np.array([
            [-l1*np.sin(t[0]) - l2*np.sin(t[0]+t[1]),
             -l2*np.sin(t[0]+t[1])],
            [ l1*np.cos(t[0]) + l2*np.cos(t[0]+t[1]),
              l2*np.cos(t[0]+t[1])]
        ])

    def joint_state_callback(self, msg: Float32MultiArray):
        # 1) Update joint angles
        self.theta = np.array([msg.data[0], msg.data[1]])

        # 2) Publish FK for debug
        pos = self.forward_kinematics(self.theta)
        fk  = Float32MultiArray(data=[float(pos[0]), float(pos[1])])
        self.fk_pub.publish(fk)

        # 3) Check if we've reached the marker₁ goal
        if self.goal_xy is not None and np.linalg.norm(pos - self.goal_xy) < self.threshold:
            self.get_logger().info("Reached marker 1 location → HARD stopping motion")

            # ——— HARD STOP: publish current pose as the new set-point ———
            stop_cmd = Float32MultiArray(data=[float(self.theta[0]), float(self.theta[1])])
            self.joint_pub.publish(stop_cmd)

            # clear everything so no further IK steps
            self.path_targets   = []
            self.current_wp     = 0
            self.goal_xy        = None
            return

        # 4) Otherwise, proceed with IK toward the next waypoint
        if self.current_wp < len(self.path_targets):
            target = np.array(self.path_targets[self.current_wp])
            err_vec = target - pos
            err     = np.linalg.norm(err_vec)

            J   = self.jacobian(self.theta)
            JJ  = J.T @ J + (self.damping**2)*np.eye(2)
            dq  = np.linalg.inv(JJ) @ J.T @ err_vec
            self.theta += np.degrees(self.speed * dq)

            cmd = Float32MultiArray(data=[float(self.theta[0]), float(self.theta[1])])
            self.joint_pub.publish(cmd)

            self.get_logger().info(f"WP {self.current_wp+1}/{len(self.path_targets)}  err={err:.1f}mm")
            if err < self.threshold:
                self.get_logger().info(f"Reached WP {self.current_wp+1}")
                self.current_wp += 1

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
