# The Actual ROS2 Motor Control Code

This shows what REALLY moves the motors at the lowest level.

---

## Complete Stack: Your Command → Physical Motors

### Layer 1: ROS2 Action Server
**File**: `stretch_ros2/stretch_core/joint_trajectory_server.py`

```python
class JointTrajectoryAction:
    """
    ROS2 Action Server that receives FollowJointTrajectory goals
    """

    def execute_cb(self, goal_handle):
        """Execute a joint trajectory"""
        trajectory = goal_handle.request.trajectory

        # For each waypoint
        for point in trajectory.points:
            # 1. SET GOALS for each joint
            for command_group in self.command_groups:
                command_group.set_goal(point)

            # 2. SEND TO HARDWARE  ← KEY LINE!
            self.node.robot.push_command()

            # 3. WAIT FOR COMPLETION
            while not all_reached_goal():
                time.sleep(0.01)
```

---

### Layer 2: Stretch Driver
**File**: `stretch_ros2/stretch_core/stretch_driver.py`

```python
class StretchDriverNode(Node):
    def __init__(self):
        # Import Hello Robot's hardware library
        import stretch_body.robot as rb
        self.robot = rb.Robot()  # Connects to motors via USB
        self.robot.startup()

    def move_to_position(self, qpos):
        """Command robot to specific joint positions"""
        # Queue commands (not sent yet)
        self.robot.arm.move_to(qpos[Idx.ARM])
        self.robot.lift.move_to(qpos[Idx.LIFT])
        self.robot.end_of_arm.move_to('wrist_yaw', qpos[Idx.WRIST_YAW])
        # ... etc

    def command_loop(self):
        """Main control loop (30 Hz)"""
        while True:
            # SEND ALL QUEUED COMMANDS
            self.robot.push_command()  ← CRITICAL LINE!

            # Publish joint states back to ROS2
            self.publish_joint_states()

            time.sleep(1.0 / 30.0)
```

---

### Layer 3: Stretch Body Library (Proprietary Python)
**Package**: `stretch_body` (Hello Robot's hardware interface)

```python
class Robot:
    """Main robot interface"""

    def __init__(self):
        self.arm = Arm()
        self.lift = Lift()
        self.base = Base()
        self.end_of_arm = EndOfArm()
        self.head = Head()

    def startup(self):
        """Connect to robot hardware over USB"""
        self.arm.startup()    # Opens /dev/hello-dynamixel-arm
        self.lift.startup()   # Opens /dev/hello-motor-lift
        self.base.startup()   # Opens /dev/hello-motor-left-wheel, etc.

    def push_command(self):
        """Send all queued commands to motor controllers"""
        self.arm.push_command()
        self.lift.push_command()
        self.base.push_command()
        self.end_of_arm.push_command()
        self.head.push_command()


class Arm:
    """Telescoping arm (4 prismatic joints)"""

    def move_to(self, position):
        """Queue position command (0.0 to 0.52 meters)"""
        self._target_position = position

    def push_command(self):
        """Send to Dynamixel motor controller"""
        # Create command packet
        command = {
            'mode': 'position',
            'position': self._target_position,
            'velocity': self.params['max_vel'],
            'acceleration': self.params['max_accel']
        }

        # Serialize to bytes
        packet = self._serialize_dynamixel_command(command)

        # SEND VIA USB SERIAL
        self.serial_port.write(packet)  ← ACTUAL HARDWARE I/O!


class Lift:
    """Vertical lift (stepper motor)"""

    def push_command(self):
        """Send to stepper motor controller"""
        # Different protocol than Dynamixel
        command_bytes = self._create_stepper_command(
            position=self._target_position,
            speed=self.params['max_speed']
        )

        # SEND VIA USB SERIAL
        self.serial_port.write(command_bytes)  ← ACTUAL HARDWARE I/O!


class Base:
    """Mobile base (2 wheel motors)"""

    def set_velocity(self, v, w):
        """v = linear m/s, w = angular rad/s"""
        # Convert to left/right wheel velocities
        wheel_sep = 0.33  # meters
        v_left = v - (w * wheel_sep / 2)
        v_right = v + (w * wheel_sep / 2)

        self._target_vel_left = v_left
        self._target_vel_right = v_right

    def push_command(self):
        """Send velocity commands to wheel controllers"""
        # Left wheel
        left_cmd = self._create_velocity_command(self._target_vel_left)
        self.left_wheel_serial.write(left_cmd)

        # Right wheel
        right_cmd = self._create_velocity_command(self._target_vel_right)
        self.right_wheel_serial.write(right_cmd)
```

---

### Layer 4: Motor Controllers (Embedded Hardware/Firmware)

The motor controllers are microcontrollers on the robot:

#### Dynamixel Servos (Arm, Wrist, Head)
- **Protocol**: Dynamixel Protocol 2.0
- **Connection**: USB → UART → Dynamixel bus
- **Controllers**: Dynamixel XM430, XL430 servos
- **Receives**: Position/velocity/current commands
- **Runs**: Internal PID loop at 1 kHz
- **Outputs**: PWM to motor coils

#### Stepper Motor (Lift)
- **Controller**: Custom PCB with stepper driver
- **Receives**: Step position + speed
- **Outputs**: Step pulses + direction signal

#### DC Motors (Base Wheels)
- **Controller**: Custom motor controller PCB
- **Receives**: Velocity commands
- **Outputs**: PWM to motor drivers

---

## Serial Communication Details

### USB Device Paths on Robot:
```
/dev/hello-dynamixel-head     → Head pan/tilt servos
/dev/hello-dynamixel-wrist    → Wrist servos
/dev/hello-motor-arm          → Arm prismatic joints
/dev/hello-motor-lift         → Lift stepper
/dev/hello-motor-left-wheel   → Left wheel motor
/dev/hello-motor-right-wheel  → Right wheel motor
```

### Example Dynamixel Command Packet:
```
[Header] [ID] [Length] [Instruction] [Params] [CRC]
  0xFF   0x01   0x09      0x03        [pos]   0xABCD

Instruction 0x03 = Write
Params = Goal Position register + 4-byte position value
```

---

## Complete Flow Diagram

```
grab("ball")  [YOUR CODE]
    ↓
{"joint": [0, 0.5, 0.3, ...]}  [ZMQ DICT]
    ↓
/stretch_controller/follow_joint_trajectory  [ROS2 ACTION]
    ↓
JointTrajectoryAction.execute_cb()  [PYTHON]
    ↓
robot.push_command()  [PYTHON]
    ↓
arm.push_command()  [PYTHON]
    ↓
self.serial_port.write(packet)  [PYTHON OS CALL]
    ↓
/dev/hello-dynamixel-arm  [LINUX DEVICE]
    ↓
USB-UART Bridge  [FTDI CHIP]
    ↓
Dynamixel Bus  [RS-485]
    ↓
Dynamixel Servo XM430  [MOTOR CONTROLLER]
    ├─ Parse command packet
    ├─ Run PID: error = target - current_position
    ├─ Calculate PWM duty cycle
    └─ Set H-bridge outputs
        ↓
Motor Coils  [BRUSHLESS DC MOTOR]
    ├─ Current flows through windings
    ├─ Magnetic field rotates
    └─ Rotor spins
        ↓
Gearbox (1:353 reduction)
    ↓
Telescoping Arm Extends! 🤖⚡
```

---

## Key Insight

**Q: Where does ROS2 end and hardware begin?**

**A: At the `serial_port.write()` call in `stretch_body`!**

Everything before that is Python/ROS2 wrappers. The serial write sends bytes over USB to actual motor controller chips, which then drive the physical motors.

This is the same architecture you'll find in drones:
- High-level: ROS2/Python (mission planning, perception)
- Mid-level: Flight controller interface
- Low-level: ESC (Electronic Speed Controller) firmware → motor PWM
