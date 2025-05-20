import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher = self.create_publisher(Image, 'camera/image', 10)
        self.timer = self.create_timer(0.5, self.capture_frame)  # Capture at 10Hz
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)  # Adjust if using multiple cameras

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher.publish(img_msg)
            self.get_logger().info("Published camera frame")
        else:
            self.get_logger().warn("Failed to capture frame")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

