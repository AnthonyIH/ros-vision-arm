# ros-vision-arm
This ROS 2 project implements a 2-DOF robotic arm system with camera-based ArUco marker tracking, real-time path planning, and motor control. It includes modules for visual localisation, inverse kinematics, and low-level hardware interfacing to enable autonomous object reaching and manipulation.

# marker-reach-robot

A ROS 2-based system for controlling a 2-DOF robotic arm using camera-guided ArUco marker tracking. The robot autonomously detects markers, plans a path in real time, and manipulates objects via inverse kinematics and low-level hardware control.

---

## 🛠 Features

- **Camera-based Marker Tracking**: Uses OpenCV and ArUco to detect and localise targets.
- **Real-time Path Planning**: Dynamically generates collision-free paths to target markers using an occupancy grid and Dijkstra’s algorithm.
- **Inverse Kinematics Controller**: Smoothly tracks waypoints with DLS-based IK for a 2-link arm.
- **Motor Hardware Interface**: Communicates with Dynamixel motors for joint control and gripper operation.
- **Collision Detection & Recovery**: Monitors current spikes and triggers emergency stop and recovery routines.

---

## 📦 Project Structure

```text
.
├── camera_publisher.py      # Publishes live camera feed
├── aruco_pose.py            # Marker detection and path planning
├── marker_tracking.py       # IK controller for waypoint tracking
├── hardware_interface.py    # Dynamixel motor control and state feedback
├── calibration.npz          # Camera calibration file (needed for pose estimation)
```
---

## ⚙️ Dependencies

This project requires the following:

- ROS 2 (tested with **Humble**)
- OpenCV (with ArUco module)
- NumPy
- SciPy
- Dynamixel SDK
- cv_bridge
- sensor_msgs, std_msgs

---

### 📦 Installation

Install the required dependencies using the following commands:

```bash
# ROS 2 dependencies
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-std-msgs

# Python packages
pip install opencv-python numpy scipy dynamixel-sdk

```
---

🚀 Getting Started

1) Calibrate the Camera
2) Save your calibration data to calibration.npz.

3) Launch the System - In separate terminals, run:
  
  ```bash
  ros2 run camera_publisher camera_publisher.py
  ros2 run aruco_pose aruco_pose.py
  ros2 run marker_tracking marker_tracking.py
  ros2 run hardware_interface hardware_interface.py
  ```
---

🎯 How It Works

  - The camera node streams live images.
  - The pose node detects markers and builds an occupancy grid.
  - When marker _n_ is visible, it plans a path to _n+1_ and sends waypoints.
  - The controller node uses inverse kinematics to track the path.
  - The hardware interface executes joint commands and handles motor safety.

---

4) 📍 Place Markers

  Use **ArUco markers** from the `DICT_APRILTAG_16h5` dictionary. The system interprets marker IDs as follows:
  
  - **ID 0** → Acts as the **origin** (robot base reference)
  - **ID _n_** → Represents the **object** to be moved
  - **ID _n+1_** → Represents the **goal location** for object **_n_**
  - **All other IDs (except 0, _n_, and _n+1_)** → Treated as **obstacles**
  
  For example:
  - If the robot is handling **ID 1**, then **ID 2** is its target location.
  - While doing so, **IDs 3, 4, 5, ...** will be considered obstacles.
  
  This approach allows context-aware planning where only the active object and its goal are considered, and all others are dynamically treated as barriers.

---

📸 Marker & Grid Notes

  Marker size is configurable (default: 0.025 m).
  Grid resolution, padding, and reachability limits are set via ROS parameters.
  Marker 0 is treated as the reference frame.

---

🛑 Safety

  Joint limits, current thresholds, and velocity limits are enforced.
  Collisions trigger an emergency stop and recovery to the home pose.
  Gripper is auto-calibrated at startup via soft limit detection.


