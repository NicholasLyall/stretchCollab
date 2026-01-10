Key Technical Concepts for Interview
Architecture & Communication:
ZMQ (TCP) vs ROS2 (UDP) - why different protocols for commands vs sensor streams
Client-server split: GPU does planning, robot does execution
ROS2 fundamentals: Topics (pub/sub), Services (request/response), Actions (long tasks)
Perception & Mapping:
Point clouds: depth images → 3D coordinates (x,y,z) with RGB
Voxel maps: 3D grid structure storing observed geometry
CLIP/DETIC: object detection runs on every scan point
Waypoints stored separately from voxel map (dict vs 3D reconstruction)
360° Scan Implementation:
12 rotations at 30° increments ([0, 0, rotation_increment] relative movements)
agent.update() → CLIP/DETIC detection → waypoint storage
Waypoints = {object_name: (instance, confidence_score)} dictionary
Coordinate Transformations:
Point cloud translation: new_points = old_points + (new_center - old_center)
Updating placed object positions by shifting point clouds
Maintaining visual consistency in world representation
Performance & Concurrency:
Rerun websocket bottleneck: 600K points/sec causes 5+ min delays
Frame throttling/dropping to stay current vs buffering with lag
Background mapping paused during manipulation (threading without locks = race conditions)
TCP for reliability (commands) vs UDP for speed (streams)
Robot-Specific:
11 DOF config space, but commands use 6-value joint arrays
Arm 90° offset from head camera = observation gap during manipulation
Self-occlusion filtering (why you don't see arm in map)
State Management:
Waypoint updates after object placement (belief state tracking)
Detection vs expected object positions
System can't re-observe placed objects (mapping paused + camera blind spot)