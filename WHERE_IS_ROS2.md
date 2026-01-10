# Where ROS2 Actually Lives

## TL;DR
- **ROS2 framework**: Open Robotics provides it (like React or Django)
- **stretch_ros2**: Hello Robot's code that USES ROS2
- **stretch_body**: Hello Robot's hardware library (NO ROS2)
- **stretch_ai**: Your client code (NO ROS2)

---

## What Gets Installed on the Stretch Robot

### 1. ROS2 Humble (from Open Robotics)
```bash
# This installs ROS2 framework
sudo apt install ros-humble-desktop

# Provides these Python packages:
# - rclpy (ROS Client Library Python)
# - std_msgs, geometry_msgs, sensor_msgs (message types)
# - ros2_control (generic controller framework)
```

**Files installed**:
```
/opt/ros/humble/lib/python3.10/site-packages/rclpy/
    ├─ node.py                    # Node base class
    ├─ action/
    │   ├─ server.py              # ActionServer class
    │   └─ client.py              # ActionClient class
    ├─ publisher.py               # Publisher class
    └─ subscription.py            # Subscription class

/opt/ros/humble/lib/python3.10/site-packages/geometry_msgs/
    └─ msg/
        ├─ pose.py                # Pose message type
        └─ twist.py               # Twist message type
```

These are **generic** - work for ANY robot, not just Stretch.

---

### 2. stretch_body (Hello Robot's Hardware Library)
```bash
# Install from pip or GitHub
pip install hello-robot-stretch-body
```

**Files installed**:
```
/usr/local/lib/python3.10/site-packages/stretch_body/
    ├─ robot.py                   # Main Robot class
    ├─ arm.py                     # Arm control
    ├─ lift.py                    # Lift control
    ├─ base.py                    # Base control
    └─ dynamixel_hello_XL430.py   # Servo protocol
```

**Key point**: This code has **ZERO ROS2 imports**! It's pure Python + serial I/O.

Example from `arm.py`:
```python
# NO ROS2 HERE!
import serial
import struct

class Arm:
    def startup(self):
        # Open USB serial port
        self.serial_port = serial.Serial('/dev/hello-dynamixel-arm', 115200)

    def push_command(self):
        # Create byte packet
        packet = self._create_dynamixel_packet(self._target_position)

        # Send to motor controller
        self.serial_port.write(packet)  # Pure serial I/O
```

---

### 3. stretch_ros2 (Hello Robot's ROS2 Integration)
```bash
# Clone and build from source
cd ~/ament_ws/src
git clone https://github.com/hello-robot/stretch_ros2.git
cd ~/ament_ws
colcon build
```

**Files created**:
```
~/ament_ws/src/stretch_ros2/
    └─ stretch_core/
        └─ stretch_core/
            ├─ stretch_driver.py       # Main ROS2 node
            └─ joint_trajectory_server.py  # Action server
```

**This is where ROS2 is used!**

Example from `stretch_driver.py`:
```python
# USES ROS2 FRAMEWORK
import rclpy                          # ← From ROS2
from rclpy.node import Node           # ← From ROS2
from sensor_msgs.msg import JointState  # ← From ROS2
import stretch_body.robot as rb       # ← From Hello Robot

class StretchDriverNode(Node):  # ← Inherits ROS2's Node
    def __init__(self):
        # Initialize ROS2 node
        super().__init__('stretch_driver')  # ← ROS2 call

        # Create hardware interface (NO ROS2)
        self.robot = rb.Robot()  # ← stretch_body
        self.robot.startup()

        # Create ROS2 publisher
        self.joint_state_pub = self.create_publisher(  # ← ROS2 method
            JointState,                                 # ← ROS2 message type
            '/joint_states',
            10
        )

        # Create ROS2 timer
        self.create_timer(1.0/30.0, self.timer_callback)  # ← ROS2 method

    def timer_callback(self):
        # Get hardware state (NO ROS2)
        status = self.robot.get_status()  # ← stretch_body

        # Publish to ROS2 (USES ROS2)
        msg = JointState()  # ← ROS2 message
        msg.name = ['joint_lift', 'joint_arm', ...]
        msg.position = [status['lift'], status['arm'], ...]
        self.joint_state_pub.publish(msg)  # ← ROS2 method

        # Send commands to hardware (NO ROS2)
        self.robot.push_command()  # ← stretch_body
```

---

### 4. stretch_ai (On Your GPU Computer - NO ROS2!)
```bash
# On YOUR computer
pip install stretch_ai  # or clone from GitHub
```

**Files installed**:
```
~/.local/lib/python3.10/site-packages/stretch/
    └─ agent/
        └─ zmq_client.py
```

**NO ROS2 imports here!**

Example:
```python
# NO ROS2!
import zmq  # ← Standard Python library
import numpy as np

class HomeRobotZmqClient:
    def __init__(self, robot_ip):
        # Create ZMQ socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(f"tcp://{robot_ip}:4402")

    def move_base_to(self, xyt):
        # Send dictionary over network
        command = {"xyt": xyt}
        self.socket.send_pyobj(command)  # ZMQ, not ROS2!
```

---

## The Complete Picture

```
┌─────────────────────────────────────────┐
│ GPU COMPUTER (Your laptop)              │
│                                         │
│  NO ROS2 INSTALLED                      │
│                                         │
│  stretch_ai/                            │
│  └─ zmq_client.py (Python + ZMQ)        │
│      └─ send_pyobj({"xyt": [...]})      │
│           │                             │
│           │ Network (TCP)               │
│           ↓                             │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ STRETCH ROBOT (Onboard computer)        │
│                                         │
│  ROS2 HUMBLE INSTALLED (/opt/ros/)      │
│  - rclpy                                │
│  - geometry_msgs                        │
│  - control_msgs                         │
│  - ros2_control                         │
│                                         │
│  stretch_ros2/ (Hello Robot's code)     │
│  ├─ stretch_driver.py                   │
│  │   ├─ Uses: rclpy.Node  ← ROS2       │
│  │   ├─ Uses: ActionServer  ← ROS2     │
│  │   └─ Creates: /joint_states topic   │
│  │                                      │
│  └─ joint_trajectory_server.py          │
│      ├─ Uses: ActionServer  ← ROS2     │
│      └─ Receives: FollowJointTrajectory │
│           │                             │
│           ↓                             │
│  stretch_body/ (Hello Robot's code)     │
│  └─ robot.py, arm.py, lift.py          │
│      ├─ NO ROS2!                        │
│      └─ Pure Python + serial            │
│           │                             │
│           ↓                             │
│  /dev/hello-dynamixel-arm (Linux)       │
│           │                             │
│           ↓                             │
│  Motor Controllers (Hardware)           │
└─────────────────────────────────────────┘
```

---

## Summary: Who Wrote What?

| Component | Author | Uses ROS2? | Purpose |
|-----------|--------|-----------|---------|
| **rclpy** | Open Robotics | - | ROS2 framework |
| **ros2_control** | Open Robotics | - | Controller framework |
| **stretch_body** | Hello Robot | ❌ NO | Hardware interface |
| **stretch_ros2** | Hello Robot | ✅ YES | ROS2 wrapper for hardware |
| **stretch_ai** | Hello Robot | ❌ NO | Client library (ZMQ) |
| **PERSISTEDMap_360mapping.py** | You/Your PI | ❌ NO | High-level control |

---

## Where ROS2 Actually Runs

**ROS2 code executes in**: `stretch_driver.py` on the robot

**What it does**:
1. Receives action goals via ROS2 (`/follow_joint_trajectory`)
2. Translates to `stretch_body` commands
3. Publishes joint states back via ROS2 (`/joint_states`)

**Everything else** is either:
- Your code (client side, no ROS2)
- Hardware library (stretch_body, no ROS2)
- Motor firmware (embedded C, no ROS2)

---

## The Key Insight

**ROS2 is ONLY the communication layer on the robot!**

It's like a web framework:
- Django provides request/response handling
- Your code uses Django to build an app
- Database access (like stretch_body) doesn't know Django exists

Similarly:
- ROS2 provides topics/actions/services
- stretch_ros2 uses ROS2 to build a robot driver
- stretch_body (hardware access) doesn't know ROS2 exists
