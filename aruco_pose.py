#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
import cv2.aruco as aruco
import numpy as np
import heapq
from cv_bridge import CvBridge
from math import atan2, acos, sin, cos, degrees

class MarkerRelativePositionNode(Node):
    __slots__ = ['bridge', 'ee_position', 'grid', 'marker_positions',
                 'cells_per_m','grid_cell_px','aruco_padding','ee_padding',
                 'marker_size','_has_published','current_path_cells',
                 'grid_pub','next_step_pub','goal_visible_prev']

    def __init__(self):
        super().__init__('marker_relative_position')
        # Core state
        self.bridge            = CvBridge()
        self.ee_position       = None
        self.grid              = None
        self.marker_positions  = {}
        self._has_published    = False
        self.current_path_cells= []
        # Track whether marker 1 was visible in the previous frame
        self.goal_visible_prev = False

        # Subscriptions & Publishers
        self.create_subscription(Float32MultiArray, '/cartesian_pos',
                                 self.ee_position_callback, 10)
        self.create_subscription(Image, 'camera/image',
                                 self.image_callback, 10)
        self.grid_pub      = self.create_publisher(Float32MultiArray,
                                                   '/occupancy_grid', 10)
        self.next_step_pub = self.create_publisher(Float32MultiArray,
                                                   '/next_step', 10)

        # Parameters
        # Grid resolution: number of cells representing 1 meter (used to convert mm to grid)
        self.declare_parameter('cells_per_meter', 440.0)
        # Size of each grid cell in pixels (used for visualization and image-based grid mapping)     
        self.declare_parameter('grid_cell_px', 5) 
        # Radius (in grid cells) to mark around ArUco markers as obstacles (except marker 1)           
        self.declare_parameter('aruco_padding', 30) 
        # Padding (in grid cells) around end-effector to carve out free space in the occupancy grid         
        self.declare_parameter('ee_padding', 0)
        # Physical size of ArUco markers in meters (used in pose estimation from the camera)              
        self.declare_parameter('marker_size', 0.025)         


        self.cells_per_m   = self.get_parameter('cells_per_meter')\
                                 .get_parameter_value().double_value
        self.grid_cell_px  = self.get_parameter('grid_cell_px')\
                                 .get_parameter_value().integer_value
        self.aruco_padding = self.get_parameter('aruco_padding')\
                                 .get_parameter_value().integer_value
        self.ee_padding    = self.get_parameter('ee_padding')\
                                 .get_parameter_value().integer_value
        self.marker_size   = self.get_parameter('marker_size')\
                                 .get_parameter_value().double_value

        # Arm parameters
        self.link_lengths = [100.0, 160.0]
        self.joint_limits = [100.0, 100.0]

        # ArUco setup
        self.aruco_dict   = aruco.getPredefinedDictionary(
                              cv2.aruco.DICT_APRILTAG_16h5)
        self.aruco_params = aruco.DetectorParameters_create()

        # Load camera calibration
        try:
            with np.load('calibration.npz') as f:
                self.camera_matrix = f['camMatrix']
                self.dist_coeffs   = f['distCoef']
            self.get_logger().info('Calibration loaded.')
        except Exception as e:
            self.get_logger().error(f'Calib load failed: {e}')
            self.camera_matrix = None
            self.dist_coeffs   = None

    def ee_position_callback(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self.ee_position = (msg.data[0], msg.data[1])

    def _is_reachable(self, x_mm, y_mm):
        l1, l2 = self.link_lengths
        d2 = x_mm**2 + y_mm**2
        cos_q2 = (d2 - l1*l1 - l2*l2) / (2*l1*l2)
        if abs(cos_q2) > 1.0:
            return False
        for sign in (+1.0, -1.0):
            q2 = sign * acos(cos_q2)
            k1 = l1 + l2*cos(q2)
            k2 = l2*sin(q2)
            q1 = atan2(y_mm, x_mm) - atan2(k2, k1)
            if abs(degrees(q1)) <= self.joint_limits[0] and \
               abs(degrees(q2)) <= self.joint_limits[1]:
                return True
        return False

    def image_callback(self, msg: Image):
        # 1) Convert to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params)

        # Build occupancy grid
        h, w = gray.shape
        cell_px = self.grid_cell_px
        num_cells_x = w // cell_px
        num_cells_y = h // cell_px
        self.grid = np.zeros((num_cells_y, num_cells_x), dtype=np.uint8)
        self.marker_positions.clear()
        marker0_center_px = None

        # Detect markers and record their grid‐cells
        if ids is not None:
            for corner, mid in zip(corners, ids.flatten()):
                pts = corner[0]
                cx, cy = int(pts[:,0].mean()), int(pts[:,1].mean())
                gx, gy = cx//cell_px, cy//cell_px
                self.marker_positions[mid] = (gx, gy)
                if mid == 0:
                    marker0_center_px = (cx, cy)

        # Reset path when marker 1 reappears
        goal_visible = (1 in self.marker_positions)
        if goal_visible and not self.goal_visible_prev:
            self.get_logger().info('Marker 1 re-detected → resetting path.')
            self._has_published = False
            self.current_path_cells = []
        self.goal_visible_prev = goal_visible

        # Mark ArUco obstacles (all except marker 1)
        r = self.aruco_padding
        for mid, (gx, gy) in self.marker_positions.items():
            if mid != 1 and r > 0:
                for yy in range(max(0,gy-r), min(num_cells_y, gy+r+1)):
                    for xx in range(max(0,gx-r), min(num_cells_x, gx+r+1)):
                        if (xx-gx)**2 + (yy-gy)**2 <= r*r:
                            self.grid[yy, xx] = 1

        # Mask unreachable workspace
        workspace_mask = np.zeros_like(self.grid, dtype=bool)
        if marker0_center_px:
            ogx, ogy = marker0_center_px[0]//cell_px, marker0_center_px[1]//cell_px
            mm_per_cell = 1000.0 / self.cells_per_m
            for yy in range(num_cells_y):
                for xx in range(num_cells_x):
                    x_mm = (yy-ogy)*mm_per_cell
                    y_mm = (xx-ogx)*mm_per_cell
                    if not self._is_reachable(x_mm, y_mm):
                        self.grid[yy, xx] = 1
                        workspace_mask[yy, xx] = True

        # Carve out end-effector region
        ee_cell = None
        if self.ee_position and marker0_center_px:
            ogx, ogy = marker0_center_px[0]//cell_px, marker0_center_px[1]//cell_px
            mm_per_cell = 1000.0 / self.cells_per_m
            dx = int(round(self.ee_position[1]/mm_per_cell))
            dy = int(round(self.ee_position[0]/mm_per_cell))
            gx, gy = ogx+dx, ogy+dy
            ee_cell = (gx, gy)
            for yy in range(max(0,gy-self.ee_padding), min(num_cells_y, gy+self.ee_padding+1)):
                for xx in range(max(0,gx-self.ee_padding), min(num_cells_x, gx+self.ee_padding+1)):
                    if (xx-gx)**2 + (yy-gy)**2 <= self.ee_padding**2:
                        self.grid[yy, xx] = 0
                        workspace_mask[yy, xx] = False

        # 3) Plan path with Dijkstra
        goal_cell = self.marker_positions.get(1)
        path_cells = []
        if ee_cell and goal_cell:
            path_cells = self.dijkstra(ee_cell, goal_cell, self.grid)

        # 4) Visualization
        vis = frame.copy()
        # draw grid lines
        for y in range(0, h, cell_px): cv2.line(vis, (0,y), (w,y), (200,200,200),1)
        for x in range(0, w, cell_px): cv2.line(vis, (x,0), (x,h), (200,200,200),1)
        # path overlay
        for px, py in path_cells:
            cv2.rectangle(vis, (px*cell_px, py*cell_px),
                          ((px+1)*cell_px, (py+1)*cell_px),
                          (0,255,255), -1)
            
        # obstacle & unreachable overlays
        overlay = vis.copy()
        for mid,(gx,gy) in self.marker_positions.items():
            if mid!=1 and r>0:
                cv2.circle(overlay,
                           (gx*cell_px+cell_px//2,gy*cell_px+cell_px//2),
                           r*cell_px,(0,0,255),-1)
                
        ys,xs = np.nonzero(workspace_mask)
        for yy,xx in zip(ys,xs):
            cv2.rectangle(overlay,
                          (xx*cell_px,yy*cell_px),
                          ((xx+1)*cell_px,(yy+1)*cell_px),
                          (0,0,255),-1)
        cv2.addWeighted(overlay,0.4,vis,0.6,0,vis)

        # labels & EE pixel/robot coords
        if ee_cell:
            cv2.putText(vis,'EE',(ee_cell[0]*cell_px+2,ee_cell[1]*cell_px+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        if goal_cell:
            cv2.putText(vis,'M1',(goal_cell[0]*cell_px+2,goal_cell[1]*cell_px+15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        if ee_cell:
            ex,ey = ee_cell[0]*cell_px+cell_px//2, ee_cell[1]*cell_px+cell_px//2
            cv2.circle(vis,(ex,ey),5,(255,0,0),-1)
            cv2.putText(vis,f"cam:({ex},{ey})",(ex+5,ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)
            mm_pp = (1000.0/self.cells_per_m)/cell_px
            dx_px = ey-marker0_center_px[1]
            dy_px = ex-marker0_center_px[0]
            cv2.putText(vis,f"robot: x={dx_px*mm_pp:.1f}mm, y={dy_px*mm_pp:.1f}mm",
                        (ex+5,ey+10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,255),1)
        # axes & EE text
        if marker0_center_px:
            ox,oy = marker0_center_px
            cv2.arrowedLine(vis,(ox,oy),(ox,oy+50),(0,0,255),2)
            cv2.putText(vis,'X',(ox+5,oy+55),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
            cv2.arrowedLine(vis,(ox,oy),(ox+50,oy),(0,255,0),2)
            cv2.putText(vis,'Y',(ox+55,oy+5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        if self.ee_position:
            cv2.putText(vis,f"EE: x={self.ee_position[0]:.1f}mm, y={self.ee_position[1]:.1f}mm",
                        (10,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        cv2.imshow('Planned Path',vis)
        cv2.waitKey(1)

        # 5) Publish next step
        flat = []
        if path_cells and marker0_center_px:
            ogx,ogy = marker0_center_px[0]//cell_px, marker0_center_px[1]//cell_px
            mm_pc = 1000.0/self.cells_per_m
            for (px,py) in path_cells:
                flat += [(py-ogy)*mm_pc, (px-ogx)*mm_pc]

        if not self._has_published:
            if flat:
                msg = Float32MultiArray(data=[float(x) for x in flat])
                self.next_step_pub.publish(msg)
                self.current_path_cells = path_cells.copy()
                self._has_published = True
        else:
            # replan on collision
            collision = any(self.grid[py][px] for px,py in self.current_path_cells)
            if collision:
                if flat:
                    msg = Float32MultiArray(data=[float(x) for x in flat])
                    self.next_step_pub.publish(msg)
                    self.current_path_cells = path_cells.copy()
                else:
                    # no path so reset
                    self.next_step_pub.publish(Float32MultiArray(data=[]))
                    self._has_published = False
                    self.current_path_cells = []

    def dijkstra(self, start, goal, grid):
        h,w = grid.shape
        dist,prev = {start:0}, {}
        pq = [(0,start)]
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        while pq:
            d,u = heapq.heappop(pq)
            if u==goal: break
            if d>dist.get(u,1e9): continue
            ux,uy = u
            for dx,dy in dirs:
                vx,vy = ux+dx, uy+dy
                if 0<=vx<w and 0<=vy<h and grid[vy,vx]==0:
                    nd=d+1; v=(vx,vy)
                    if nd<dist.get(v,1e9):
                        dist[v]=nd; prev[v]=u
                        heapq.heappush(pq,(nd,v))
        path=[]
        if goal in prev or start==goal:
            cur=goal
            while cur!=start:
                path.append(cur)
                cur=prev[cur]
            path.append(start)
            path.reverse()
        return path

def main(args=None):
    rclpy.init(args=args)
    node = MarkerRelativePositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
