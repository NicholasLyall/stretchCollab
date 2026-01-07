# Stretch AI Commands Reference

Complete command reference for all Stretch AI programs.

## Dev Container Setup

```bash
# Enter dev container (from project root)
./scripts/run_stretch_ai_gpu_client.sh --dev
```

## Core Pickup Programs

### AI Pickup (Main)
```bash
python -m stretch.app.ai_pickup --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--target_object     Object to pick up (default: "")  
--receptacle        Where to place object (default: "")
--explore_iter      Exploration iterations (default: 5)
--dry_run           Dry run mode (flag)
--reset             Reset to starting position (flag)
--show_intermediate Show intermediate steps (flag)
--visualize         Enable visualization (flag)
```

### Table Pickup Standalone (Reference)
```bash
python -m stretch.app.table_pickup_standalone --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--target_object     Object to pick up (default: "")
--receptacle        Where to place object (default: "")  
--dry_run           Dry run mode (flag)
--visualize         Enable visualization (flag)
--explore_iter      Exploration iterations (default: 5)
```

### Table Pickup New (Enhanced)
```bash
python -m stretch.app.table_pickup_new --robot_ip 192.168.1.108

# Enhanced Three Functions approach with:
# - Random movement with 2-step sequence  
# - Speech announcements
# - Waypoint return navigation
# - Extended 6m camera range
# - 0.3m table scanning increments
```

## Persistent Mapping Programs

### 360° Mapping with Object Detection
```bash
python -m stretch.app.PERSISTEDMap_360mapping --robot_ip 192.168.1.108

python -m stretch.app.PERSISTEDMap_360mapping --robot_ip 192.168.1.108 --show-camera-points
# NOTE: Camera point clouds disabled by default for performance. Without --show-camera-points flag,
# camera RGB feeds flicker red/blue due to BGR->RGB conversion issue in direct image display.
# Point cloud processing accidentally fixes this by handling color channels correctly.

# Options:
--robot_ip          Robot IP address (default: "")
--scan_points       Number of 360° scan points (default: 12)
--debug             Enable debug output (default: True)
--dry_run           Dry run mode (flag)
--target_object     Object to detect as waypoint (default: "")
--receptacle        Receptacle to detect as waypoint (default: "")
--dynamic_memory    Enable dynamic memory system (flag)
--memory_threshold  Observations before object cleanup (default: 50)
--show_camera_points Show ALL camera point clouds in rerun (flag, disabled by default for performance)
--use_table_aware_pickup Enable height-adaptive pickup for objects at any elevation (default: True)
--enable_speech_debug Enable vocal debugging announcements for workflow phases (flag, disabled by default)
--show_object_bboxes Show 3D object bounding boxes in rerun viewer (flag, disabled by default)

# Verbose Debug Output:
# By default, detailed mask scoring during visual servoing is hidden for clean terminal output.
# To see detailed object detection scoring (Score for X is Y / 0.05), the grasp operations
# have a verbose=False parameter that would need to be modified in the TableAwarePickupExecutor
# initialization to verbose=True for debugging.

# Interactive Commands (after scan):
arrive <waypoint>   - Navigate to detected waypoint
move <distance>     - Move forward in meters
turn <degrees>      - Turn (positive=left, negative=right) 
waypoints          - List detected waypoints
status             - Show mapping status
quit               - Exit service
```

### Forward Movement with Mapping
```bash
python -m stretch.app.PERSISTEDMap_moveforward --robot_ip 192.168.1.108 --distance 2.5

# Options:
--robot_ip          Robot IP address (default: "")
--distance          Distance to move forward in meters (required)
--debug             Enable debug output (default: True)
--dry_run           Dry run mode (flag)
--keep_mapping      Keep mapping active (default: True)
```

### Turn with Mapping
```bash
python -m stretch.app.PERSISTEDMap_turn --robot_ip 192.168.1.108 --degrees -90

# Options:
--robot_ip          Robot IP address (default: "")
--degrees           Degrees to turn (required, positive=left, negative=right)
--debug             Enable debug output (default: True)
--dry_run           Dry run mode (flag)
--keep_mapping      Keep mapping active (default: True)
```

## LLM and Chat Programs

### Chat Interface
```bash
python -m stretch.app.chat --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--llm               LLM provider: openai, anthropic, google, ollama (default: "openai")
--model             Model name (default: varies by provider)
--use_voice         Enable voice input/output (flag)
--input_path        Audio input file path (default: "")
--output_path       Audio output file path (default: "")
--show_intermediate Show intermediate steps (flag)
--visualize         Enable visualization (flag)
```

### Mapping and Exploration
```bash
python -m stretch.app.mapping --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--explore_iter      Exploration iterations (default: 5)
--dry_run           Dry run mode (flag)
--visualize         Enable visualization (flag)
--random_goals      Use random goal generation (flag)
```

### OVMM (Object Navigation)
```bash
python -m stretch.app.ovmm --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--input_path        Path to input file (default: "")
--use_voice         Enable voice input (flag)
--model             LLM model (default: "gpt-4")
--dry_run           Dry run mode (flag)
--visualize         Enable visualization (flag)
```

## Utility Programs

### EQA (Embodied Question Answering)
```bash
python -m stretch.app.run_eqa --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--question          Question to answer (default: "")
--use_voice         Enable voice input (flag)
--model             LLM model (default: "gpt-4")
--explore_iter      Exploration iterations (default: 3)
```

### Camera Viewing
```bash
python -m stretch.app.view_camera --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--camera_name       Camera to view: head_rgb, ee_rgb (default: "head_rgb")
--visualize         Enable visualization (flag)
```

### Gripper Control
```bash
python -m stretch.app.gripper --robot_ip 192.168.1.108

# Options:  
--robot_ip          Robot IP address (default: "")
--open              Open gripper (flag)
--close             Close gripper (flag)
--position          Gripper position 0.0-1.0 (default: 0.5)
```

### Keyboard Teleop
```bash
python -m stretch.app.keyboard_teleop --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--step_size         Movement step size (default: 0.1)
--turn_size         Turn step size in radians (default: 0.1)
```

### Print Joint State
```bash
python -m stretch.app.print_joint_state --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--rate              Update rate in Hz (default: 10.0)
```

## Development and Testing

### Print Instances
```bash
python -m stretch.app.print_instances --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--config_path       Config file path (default: "default_planner.yaml")
```

### View Map
```bash
python -m stretch.app.view_map --robot_ip 192.168.1.108

# Options:
--robot_ip          Robot IP address (default: "")
--map_path          Path to saved map file (default: "")
--visualize         Enable visualization (flag)
```

## Usage Notes

### Common Patterns
```bash
# Basic pickup task
python -m stretch.app.ai_pickup --robot_ip 192.168.1.108

# Enhanced pickup with movements  
python -m stretch.app.table_pickup_new --robot_ip 192.168.1.108

# 360° environmental mapping
python -m stretch.app.PERSISTEDMap_360mapping --robot_ip 192.168.1.108

# Chat with robot
python -m stretch.app.chat --robot_ip 192.168.1.108 --use_voice
```

### Default Robot IP
- Most programs default to empty string `""` for robot_ip
- When empty, connects to localhost or uses ZMQ default
- Always specify `--robot_ip 192.168.1.108` for remote robots

### Dry Run Mode
- Available in most programs with `--dry_run` flag
- Tests logic without robot movements
- Useful for debugging and development

### Visualization
- Many programs support `--visualize` flag
- Opens Rerun viewer at http://localhost:9090
- Shows 3D mapping and robot state in real-time

### Interactive Commands
Only `PERSISTEDMap_360mapping` currently supports interactive command mode:
- `arrive <object>` - Navigate to detected objects
- `move <distance>` - Forward movement  
- `turn <degrees>` - Rotation
- `waypoints` - List available targets
- `status` - System status
- `quit` - Exit

## Latest Features (December 2025)

### Enhanced Persistent Mapping
- 6-meter camera range (up from 4m)
- Object detection during 360° scans
- Waypoint-based navigation system
- Interactive command interface
- Suppressed YOLO output spam
- Background continuous mapping

### Fixed Issues
- Robot turning wrong direction (atan2 coordinate fix)
- Memory limit handling in Rerun visualization  
- Improved movement reliability and error handling