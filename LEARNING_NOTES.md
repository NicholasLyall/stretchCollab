# PERSISTEDMap_360mapping.py - Deep Dive Learning Notes

**Purpose**: Interview prep for autonomous drones - understanding ROS2, robotics concepts, and the Stretch robot codebase

**Context**:
- Code works! Robot does 360 scans, finds objects (sports ball/cup), navigates, grasps, places
- Built by modifying stretch_ai repo via high-level prompts
- File is 2401 lines, likely has dead code from vibe coding sessions
- Missing ML models (CLIP etc) - this is review-only version

---

## Topics Covered

### 0. The Stretch Robot Hardware ✅

**Robot Configuration Space (11 DOF):**
```
Index 0-2:  BASE (x, y, theta) - Robot position & rotation on ground
Index 3:    LIFT - Vertical torso extension (0 to ~1.1m)
Index 4:    ARM - Horizontal telescoping extension (0 to ~0.52m)
Index 5:    GRIPPER - Gripper opening (-0.3 closed, 0.6 open)
Index 6-8:  WRIST (roll, pitch, yaw) - 3-axis wrist rotation
Index 9-10: HEAD (pan, tilt) - Camera head direction
```

**Low-Level Command Format (Python Dictionaries over ZMQ):**

1. **Base Movement:**
   ```python
   {"xyt": [x, y, theta], "nav_relative": False, "nav_blocking": True}
   ```

2. **Arm/Manipulation (6 values, not 11!):**
   ```python
   {
       "joint": [base_x, lift, arm, wrist_roll, wrist_pitch, wrist_yaw],
       "gripper": 0.6,           # Optional
       "head_to": [pan, tilt],   # Optional
       "manip_blocking": True
   }
   ```

3. **Mode Switching:**
   ```python
   {"control_mode": "navigation"}  # or "manipulation"
   ```

4. **Predefined Postures:**
   ```python
   {"posture": "navigation"}  # Tucks arm, looks forward
   ```

**Key Files:**
- `zmq_client.py`: Command construction (YOUR GPU computer)
- `server.py`: Command execution (Stretch robot, from stretch_ai repo)
- `kinematics.py`: Joint indexing and IK solver

**Architecture:**
- GPU computer constructs Python dict commands
- Sends over ZMQ network to Stretch
- Stretch server translates to ROS2 motor commands
- Server is from stretch_ai repo, runs on the robot

### 1. ROS2 Fundamentals ✅
**What ROS2 is**: Middleware for distributed robot systems - lets multiple processes communicate in real-time

**3 Communication Patterns**:
1. **Topics** (Pub/Sub) - Fire and forget, continuous data streams (e.g., camera images)
2. **Services** (Request/Response) - Blocking API calls, get immediate response
3. **Actions** (Long Tasks) - For operations that take time, provide feedback, can be cancelled

**This Project's Architecture**:
- **NOT a ROS2 node** - Uses ZeroMQ (ZMQ) as lightweight message broker instead
- `HomeRobotZmqClient` = Client that talks to robot server over network
- Robot server (runs on Stretch) = Actual ROS2 node that controls hardware
- Your code (runs on GPU computer) = High-level commands via ZMQ

**Why ZMQ instead of ROS2 directly?**
- Lighter weight, easier to set up across networks
- Abstracts ROS2 complexity - you send commands, robot server handles ROS2 details
- Still follows similar patterns (request/response for commands, streaming for sensor data)

---

## Commands Actually Used in Production
*(To be filled in later - user will provide terminal commands)*

---

## Learning Strategy

**Your Codebase = CLIENT SIDE ONLY**
- You have: Command construction, path planning, object detection, mapping
- You DON'T have: ROS2 server that translates commands to motors (runs on robot)
- Server is from stretch_ai repo installed on the Stretch robot

**Two-Track Learning Approach:**

### Track 1: Concepts (For Interview - Breadth)
- ROS2 fundamentals ✅
- Robot kinematics & joints ✅
- Coordinate frames & transformations
- Sensor fusion & mapping
- Path planning algorithms
- Object detection pipeline

### Track 2: Your Code (For Understanding Project - Depth)
- PERSISTEDMap_360mapping.py structure
- How 360 scan works
- Voxel mapping system
- Waypoint detection & navigation
- Grab/place sequences
- Identify dead code

---

## Potential Dead Code / Bloat Areas
*(To identify during review)*

---

## Complete Command Flow: GPU → Robot Motors

### Example: `grab_object("sports ball")`

```
┌─────────────────────────────────────────────────────────────────┐
│ YOUR CODE (GPU Computer) - PERSISTEDMap_360mapping.py          │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─→ grab_object("sports ball")
    │   └─→ _smart_navigate_to_object("sports ball")
    │       └─→ HomeRobotZmqClient.move_base_to([x, y, theta])
    │           └─→ send_action({"xyt": [x,y,θ], "nav_blocking": True})
    │               └─→ send_socket.send_pyobj({...})
    │                   │
    │                   │ ⚡ ZMQ over Network (TCP)
    │                   ↓
┌─────────────────────────────────────────────────────────────────┐
│ STRETCH ROBOT - stretch_ai/stretch_ros2_bridge                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─→ recv_socket.recv_pyobj()  ← Receives {"xyt": [x,y,θ], ...}
    │   └─→ ZmqServer.handle_action(action)
    │       │
    │       ├─ if "xyt" in action:
    │       │   └─→ StretchClient.move_base_to(xyt, relative, blocking)
    │       │       └─→ StretchNavigationClient.move_base_to()
    │       │           │
    │       │           │ 🤖 ROS2 Topics/Actions
    │       │           ↓
    │       │       ROS2 Publish to: "/goto_controller/goal"
    │       │       Message Type: Pose (x, y, theta)
    │       │
    │       ├─ if "joint" in action:
    │       │   └─→ StretchClient.arm_to(q, gripper, head_pan, head_tilt)
    │       │       └─→ StretchManipulationClient.goto_joint_positions()
    │       │           │
    │       │           │ 🤖 ROS2 Actions
    │       │           ↓
    │       │       ROS2 Action: "/stretch_controller/follow_joint_trajectory"
    │       │       Message: FollowJointTrajectory.Goal
    │       │           joints: [joint_lift, joint_arm_l0, joint_arm_l1,
    │       │                    joint_arm_l2, joint_arm_l3, wrist_extension,
    │       │                    joint_wrist_yaw, joint_wrist_pitch,
    │       │                    joint_wrist_roll]
    │       │
    │       └─ if "gripper" in action:
    │           └─→ StretchClient.gripper_to(position)
    │               │
    │               │ 🤖 ROS2 Topics
    │               ↓
    │           ROS2 Publish: Gripper trajectory commands
    │
    └─→ ROS2 Controllers (ros2_control)
        └─→ Hardware Interface
            └─→ 🔧 ACTUAL MOTORS MOVE
```

### Key ROS2 Topics/Actions on the Robot:

1. **Base Navigation:**
   - Topic: `/goto_controller/goal` (Pose messages)
   - Topic: `/stretch/cmd_vel` (Twist messages for velocity control)

2. **Arm/Manipulation:**
   - Action: `/stretch_controller/follow_joint_trajectory`
   - Type: `FollowJointTrajectory` action

3. **Joint States (Feedback):**
   - Topic: `/joint_states` (robot publishes current positions)
   - Your client subscribes to this to know when movements complete

4. **Camera Images:**
   - Topic: `/camera/color/image_raw` (head camera RGB)
   - Topic: `/camera/aligned_depth_to_color/image_raw` (depth)
   - Your client receives these for perception

## Key Insights
*(Running list of "aha!" moments)*

1. **Commands are dictionaries, not arrays** - More flexible and readable
2. **Arm control uses 6 values, not 11** - Base, lift, arm, and 3 wrist angles
3. **ZMQ abstracts ROS2** - You don't need ROS2 on GPU computer
4. **Robot server does the ROS2 translation** - Converts dicts to proper ROS2 messages
5. **Blocking mode = polling** - Client keeps checking joint states until goal reached
