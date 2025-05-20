import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray  # Message type for Control DLS
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from cv_bridge import CvBridge

class MarkerRelativePositionNode(Node):
    __slots__ = ['publishers']
    
    def __init__(self):
        super().__init__('marker_relative_position')
        self.publishers = {}  # Now allowed because 'publishers' is declared in __slots__
        # ... rest of your initialization code ...


        # Subscription to Camera Images
        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            10
        )

        # Use object.__setattr__ to create a publishers attribute
        object.__setattr__(self, 'publishers', {})

        # OpenCV Bridge for converting ROS Image messages
        self.bridge = CvBridge()

        # Declare marker size parameter
        self.declare_parameter("marker_size", 0.025)
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value

        # Load Camera Calibration Data
        try:
            with np.load('calibration.npz') as f:
                self.camera_matrix = f['camMatrix']
                self.dist_coeffs = f['distCoef']
            self.get_logger().info("Calibration file loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration.npz: {e}")
            self.camera_matrix = None
            self.dist_coeffs = None

        # ArUco marker dictionary and detector parameters (using AprilTag dictionary)
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self.aruco_params = aruco.DetectorParameters_create()

    def image_callback(self, msg):
        """Process incoming ROS2 Image messages and detect ArUco markers."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and self.camera_matrix is not None and self.dist_coeffs is not None:
            ids_flat = ids.flatten()

            # Check if Marker 0 (reference) is present
            if 0 not in ids_flat:
                self.get_logger().warn("❌ Marker 0 (reference) not detected! Cannot compute relative positions.")
                return

            # Get index and pose of Marker 0
            idx0 = np.where(ids_flat == 0)[0][0]
            # Compute pose for all markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            tvec0 = tvecs[idx0][0]  # Pose of Marker 0

            # Iterate over all detected markers (skip Marker 0)
            for i, marker_id in enumerate(ids_flat):
                if marker_id == 0:
                    continue

                # Extract the translation vector for this marker
                tvec_marker = tvecs[i][0]
                # Compute the relative position (in meters) w.r.t Marker 0
                relative_x = tvec_marker[0] - tvec0[0]
                relative_y = tvec_marker[1] - tvec0[1]
                relative_z = tvec_marker[2] - tvec0[2]

                # Convert positions from meters to millimeters
                relative_x_mm = relative_x * 1000.0
                relative_y_mm = relative_y * 1000.0
                relative_z_mm = relative_z * 1000.0

                # Compute the orientation of the marker (wrt camera)
                rvec_marker = rvecs[i][0]
                rot_matrix, _ = cv2.Rodrigues(rvec_marker)
                sy = math.sqrt(rot_matrix[0, 0] ** 2 + rot_matrix[1, 0] ** 2)
                singular = sy < 1e-6
                if not singular:
                    roll = math.atan2(rot_matrix[2, 1], rot_matrix[2, 2])
                    pitch = math.atan2(-rot_matrix[2, 0], sy)
                    yaw = math.atan2(rot_matrix[1, 0], rot_matrix[0, 0])
                else:
                    roll = math.atan2(-rot_matrix[1, 2], rot_matrix[1, 1])
                    pitch = math.atan2(-rot_matrix[2, 0], sy)
                    yaw = 0

                # Convert orientation from radians to degrees
                roll_deg = math.degrees(roll)
                pitch_deg = math.degrees(pitch)
                yaw_deg = math.degrees(yaw)

                # Create the message for Control DLS
                dls_msg = Float32MultiArray()
                # Format: [ -relative_x_mm, relative_y_mm, relative_z_mm, roll_deg, pitch_deg, yaw_deg ]
                dls_msg.data = [
                    float(relative_y_mm),
                    float(relative_x_mm),
                    float(relative_z_mm),
                    float(roll_deg),
                    float(pitch_deg),
                    float(yaw_deg)
                ]

                # Check if a publisher for this marker id exists; if not, create one.
                topic_name = f"desired_pos_{marker_id}"
                if marker_id not in self.publishers:
                    self.publishers[marker_id] = self.create_publisher(Float32MultiArray, topic_name, 10)

                # Publish the message on the corresponding topic
                self.publishers[marker_id].publish(dls_msg)

                # Log the computed relative position and orientation
                self.get_logger().info(
                    f"✅ Marker {marker_id} Relative Position wrt Marker 0: X={(relative_x_mm):.2f}mm, "
                    f"Y={relative_y_mm:.2f}mm, Z={relative_z_mm:.2f}mm | Orientation (cam frame): "
                    f"roll={roll_deg:.2f}°, pitch={pitch_deg:.2f}°, yaw={yaw_deg:.2f}°"
                )
        else:
            self.get_logger().warn("⚠️ No markers detected or calibration data is missing.")

def main(args=None):
    rclpy.init(args=args)
    node = MarkerRelativePositionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
