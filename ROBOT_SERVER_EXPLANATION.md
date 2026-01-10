# Robot Server Code Explanation

This is what runs ON THE STRETCH ROBOT to receive your commands.

## File: `stretch_ai/src/stretch_ros2_bridge/stretch_ros2_bridge/remote/server.py`

### The Server Class

```python
class ZmqServer(BaseZmqServer):
    """
    Server running on the Stretch robot.
    Receives ZMQ commands from your GPU computer.
    Translates them to ROS2 actions.
    """

    def __init__(self, ...):
        # Create ROS2 interface to control robot
        self.robot = StretchClient(...)  # This talks to ROS2

        # Check robot is ready
        assert self.robot.is_homed(), "Robot must be homed!"
        assert self.robot.is_runstopped(), "Robot emergency stop!"
```

### The Command Handler (handle_action)

This is called every time your GPU sends a command:

```python
def handle_action(self, action: Dict[str, Any]):
    """
    Process incoming command from GPU computer

    Args:
        action: Dictionary like {"xyt": [1.0, 0.5, 0.0], "nav_blocking": True}
    """

    # Case 1: NAVIGATION COMMAND
    if "xyt" in action:
        # Extract parameters
        xyt = action["xyt"]              # [x, y, theta]
        relative = action.get("nav_relative", False)
        blocking = action.get("nav_blocking", True)

        # Make sure robot is in navigation mode
        if self.robot.control_mode != "navigation":
            logger.warning("Not in navigation mode!")
            return

        # Send to ROS2 navigation system
        self.robot.move_base_to(
            xyt=xyt,
            relative=relative,
            blocking=blocking
        )
        # This publishes to ROS2 topic: /goto_controller/goal

    # Case 2: ARM/MANIPULATION COMMAND
    elif "joint" in action:
        # Extract joint angles [base_x, lift, arm, roll, pitch, yaw]
        q = action["joint"]

        # Optional parameters
        gripper = action.get("gripper", None)
        head_pan = action.get("head_to", [None, None])[0]
        head_tilt = action.get("head_to", [None, None])[1]
        blocking = action.get("manip_blocking", False)

        # Make sure robot is in manipulation mode
        if self.robot.control_mode != "manipulation":
            logger.warning("Not in manipulation mode!")
            return

        # Send to ROS2 arm controller
        self.robot.arm_to(
            q=q,
            gripper=gripper,
            head_pan=head_pan,
            head_tilt=head_tilt,
            blocking=blocking
        )
        # This calls ROS2 action: /stretch_controller/follow_joint_trajectory

    # Case 3: GRIPPER ONLY
    elif "gripper" in action:
        gripper_pos = action["gripper"]
        self.robot.move_gripper(gripper_pos)

    # Case 4: MODE SWITCH
    elif "control_mode" in action:
        mode = action["control_mode"]  # "navigation" or "manipulation"
        self.robot.switch_to_mode(mode)

    # Case 5: POSTURE (predefined poses)
    elif "posture" in action:
        posture = action["posture"]  # "navigation" or "manipulation"
        self.robot.move_to_posture(posture)

    # Case 6: SPEECH
    elif "say" in action:
        text = action["say"]
        self.robot.say(text)

    else:
        logger.warning(f"Unknown action: {action.keys()}")
```

---

## What StretchClient.move_base_to() Does

This is where it goes from Python dict → ROS2:

```python
class StretchClient:
    def move_base_to(self, xyt, relative, blocking):
        """
        Move robot base to target position

        Args:
            xyt: [x, y, theta] in meters and radians
            relative: If True, relative to current position
            blocking: If True, wait until robot arrives
        """

        # Create ROS2 message
        goal_msg = Pose()
        goal_msg.position.x = xyt[0]
        goal_msg.position.y = xyt[1]
        goal_msg.orientation.z = xyt[2]  # theta (simplified)

        # Publish to ROS2 topic
        self.goal_publisher.publish(goal_msg)
        # Topic: /goto_controller/goal

        # If blocking, wait for arrival
        if blocking:
            while not self.at_goal():
                time.sleep(0.1)
```

---

## What StretchClient.arm_to() Does

```python
class StretchClient:
    def arm_to(self, q, gripper, head_pan, head_tilt, blocking):
        """
        Move arm to target joint configuration

        Args:
            q: [base_x, lift, arm, wrist_roll, wrist_pitch, wrist_yaw]
        """

        # Create ROS2 FollowJointTrajectory goal
        goal = FollowJointTrajectory.Goal()

        # Stretch has telescoping arm = 4 prismatic joints!
        # Your "arm" value (single float) becomes 4 joints:
        arm_extension = q[2]
        goal.trajectory.joint_names = [
            "joint_lift",           # Your q[1] (lift)
            "joint_arm_l0",         # arm_extension / 4
            "joint_arm_l1",         # arm_extension / 4
            "joint_arm_l2",         # arm_extension / 4
            "joint_arm_l3",         # arm_extension / 4
            "wrist_extension",      # Your q[0] (base_x forward)
            "joint_wrist_yaw",      # Your q[5]
            "joint_wrist_pitch",    # Your q[4]
            "joint_wrist_roll",     # Your q[3]
        ]

        # Set target positions
        goal.trajectory.points[0].positions = [...]

        # Send ROS2 action
        self.trajectory_action_client.send_goal(goal)
        # Action: /stretch_controller/follow_joint_trajectory

        # If blocking, wait for completion
        if blocking:
            self.trajectory_action_client.wait_for_result()
```

---

## Key Takeaway

**Your dictionary command:**
```python
{"xyt": [1.0, 0.5, 0.0], "nav_blocking": True}
```

**Becomes ROS2 message:**
```
Topic: /goto_controller/goal
Message Type: geometry_msgs/Pose
Data:
  position:
    x: 1.0
    y: 0.5
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0  # theta encoded differently in real code
    w: 1.0
```

**Then ROS2 navigation stack:**
- Plans path around obstacles
- Generates velocity commands
- Sends to motor controllers
- Motors actually move!
