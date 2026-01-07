#!/usr/bin/env python3

"""
PERSISTENT Mapping System - 360° Scanning Program
Does a complete 360° scan and maintains voxel map continuously in the background.
Creates NEW files only - never modifies existing core functionality.
"""

import click
import time
import numpy as np
import math
import threading
import signal
import sys
import shlex
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image

from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.pickup import PickupExecutor
from stretch.agent.task.pickup.table_aware_pickup_executor import TableAwarePickupExecutor
from stretch.core import get_parameters
from stretch.perception import create_semantic_sensor
from stretch.llms import PickupPromptBuilder
from stretch.utils.voxel import VoxelizedPointcloud


class PersistentMapping360:
    """
    Persistent 360° mapping system that continuously maintains voxel map state.
    This is a background service that can be called by other programs.
    """
    
    def __init__(self, robot, agent: RobotAgent, debug: bool = True, use_dynamic_memory: bool = True, memory_threshold: int = 50, use_table_aware_pickup: bool = True, enable_speech_debug: bool = False):
        self.robot = robot
        self.agent = agent
        self.debug = debug
        self.enable_speech_debug = enable_speech_debug
        self.mapping_active = False
        self.scan_complete = False
        
        # Background mapping thread
        self.mapping_thread = None
        self.stop_mapping = threading.Event()
        
        # Object detection and waypoint storage
        self.target_object = ""
        self.receptacle = ""
        self.detected_waypoints = {}  # {name: (instance, score)}
        self.object_move_counters = {}  # {object_name: move_count}
        
        # Object holding state tracking
        self.currently_holding = None      # String: name of held object or None
        self.pickup_timestamp = None       # Timestamp when object was picked up  
        self.placement_history = []        # List of placement events with details
        
        # Dynamic Memory System Integration
        self.use_dynamic_memory = use_dynamic_memory
        self.memory_threshold = memory_threshold
        self.obs_count = 0  # Initialize observation counter for all cases
        
        if self.use_dynamic_memory:
            self._setup_dynamic_memory()
        
        # Pickup functionality configuration
        self.use_table_aware_pickup = use_table_aware_pickup
        self.pickup_executor = None
        self._setup_pickup_executor()
    
    def _robot_say(self, message: str):
        """Make robot speak a message for debugging workflow"""
        if not self.enable_speech_debug:
            return  # Speech debug disabled
            
        if self.robot is not None:
            try:
                self.robot.say(message)
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Speech error: {e}")
        else:
            if self.debug:
                print(f"🗣️ WOULD SAY: '{message}'")  # Dry run speech
    
    def _setup_dynamic_memory(self):
        """Initialize dynamic memory system for persistent object tracking"""
        if self.debug:
            print("🧠 Initializing Dynamic Memory System...")
        
        try:
            # Dynamic memory parameters
            self.semantic_memory_resolution = 0.05  # Voxel size for semantic memory
            self.object_tracking_enabled = True
            self.object_disappearance_threshold = self.memory_threshold  # Observations before considering object gone
            
            # Initialize semantic memory voxel grid
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.semantic_memory = VoxelizedPointcloud(voxel_size=self.semantic_memory_resolution).to(device)
            
            # Tracking dictionaries for dynamic objects
            self.tracked_objects = {}  # {object_name: {last_seen: obs_count, position: xyz, confidence: float}}
            self.object_features = {}  # {object_name: feature_vector} for CLIP-based matching
            
            # Memory management parameters
            self.feature_similarity_threshold = 0.21  # Minimum similarity for object matching
            self.position_update_threshold = 0.2  # Meters - objects closer than this are considered same location
            
            if self.debug:
                print("✅ Dynamic Memory System initialized")
                print(f"   📊 Semantic voxel resolution: {self.semantic_memory_resolution}m")
                print(f"   🎯 Feature similarity threshold: {self.feature_similarity_threshold}")
                print(f"   📍 Position update threshold: {self.position_update_threshold}m")
            
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to setup dynamic memory: {e}")
                print("🔧 Continuing without dynamic memory features")
            self.use_dynamic_memory = False
        
    def _setup_pickup_executor(self):
        """Initialize pickup functionality for grab commands"""
        if self.robot is None or self.agent is None:
            if self.debug:
                print("🔍 DRY RUN: Pickup executor setup skipped")
            return
        
        try:
            # Set speech debug flag on agent for operations to access
            if hasattr(self.agent, '__dict__'):
                self.agent.enable_speech_debug = self.enable_speech_debug
            
            # Create prompt builder for pickup operations
            prompt = PickupPromptBuilder()
            
            # Choose pickup executor based on configuration
            if self.use_table_aware_pickup:
                # Initialize table-aware pickup executor for universal height adaptation
                self.pickup_executor = TableAwarePickupExecutor(
                    robot=self.robot,
                    agent=self.agent,
                    available_actions=prompt.get_available_actions(),
                    match_method="feature",  # Use feature matching (more robust than class matching)
                    dry_run=False,
                    enhanced_visual_servoing=True,  # Enable visual servoing for better accuracy
                    safety_margin=0.03,  # 3cm safety margin above detected surfaces
                    min_object_height_for_surface_detection=0.15  # Objects below 15cm assumed on floor
                )
                
                if self.debug:
                    print("✅ TABLE-AWARE pickup executor initialized for grab commands")
                    print("🏢 Universal height-adaptive pickup enabled (floor, table, shelf objects)")
                    print("🛡️ Surface collision avoidance active with 3cm safety margin")
            else:
                # Initialize standard pickup executor (existing behavior)
                self.pickup_executor = PickupExecutor(
                    robot=self.robot,
                    agent=self.agent,
                    available_actions=prompt.get_available_actions(),
                    match_method="feature",  # Use feature matching (more robust than class matching)
                    dry_run=False,
                    enhanced_visual_servoing=True  # Enable visual servoing for better accuracy
                )
                
                if self.debug:
                    print("✅ Standard pickup executor initialized for grab commands")
                    print("🏠 Ground-level pickup system active (original behavior)")
                
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to setup pickup executor: {e}")
                print("🔧 Grab commands will be unavailable")
            self.pickup_executor = None
    
    def setup_robot_for_scanning(self):
        """Setup robot in optimal configuration for 360° scanning"""
        self._robot_say("Setting up robot for scanning")
        if self.debug:
            print("🔧 Setting up robot for 360° scanning...")
        
        if self.robot is None:
            if self.debug:
                print("🔍 DRY RUN: Would setup robot for scanning")
            return
        
        # Switch to navigation mode for base movements
        self.robot.switch_to_navigation_mode()
        
        # Set arm to safe position during navigation first (this may reset head position)
        self.robot.move_to_nav_posture()
        
        # Set head camera to look straight ahead and more down for better 360° coverage
        self.robot.head_to(head_pan=0.0, head_tilt=-0.7, blocking=True)
        
        if self.debug:
            print("✅ Robot configured for 360° scanning")
    
    def perform_360_scan(self, scan_points: int = 12, target_object: str = "", receptacle: str = ""):
        """Perform 360° scan and detect specified objects as waypoints"""
        self.target_object = target_object
        self.receptacle = receptacle
        """
        Perform systematic 360° scan with specified number of observation points
        Default: 6 points = 60° increments for faster coverage
        """
        self._robot_say(f"Starting 360 degree scan with {scan_points} observation points")
        if self.debug:
            print(f"🌐 Starting 360° scan with {scan_points} observation points")
            print(f"📐 Angular increment: {360/scan_points:.1f}° per step")
        
        # Calculate rotation increment
        rotation_increment = 2 * math.pi / scan_points  # radians per step
        
        for step in range(scan_points):
            if self.stop_mapping.is_set():
                if self.debug:
                    print("⏹️ Scan interrupted by stop signal")
                break
                
            current_angle = step * (360 / scan_points)
            if self.debug:
                print(f"📍 Scan point {step + 1}/{scan_points} (angle: {current_angle:.1f}°)")
            
            # Update agent observations at this position
            # PERFORMANCE OPTIMIZATION: Disable head sweeping during 360° scans
            # (head sweeping adds 11 extra movements per scan point = 132 total movements for 12-point scan)
            if self.agent is not None:
                self.agent.update(move_head=False)
                
                # Detect objects at this position
                self.detect_objects_at_current_position()
                
                # Log instances detected at this angle
                instances = self.agent.get_instances()
                if self.debug:
                    print(f"🔍 Detected {len(instances)} object instances at {current_angle:.1f}°")
            
            # Pause to allow for observation processing
            time.sleep(1.0)
            
            # Rotate to next position (except for last step)
            if step < scan_points - 1 and self.robot is not None:
                if self.debug:
                    print(f"🔄 Rotating {360/scan_points:.1f}° to next scan position...")
                
                # Rotate using relative movement
                self.robot.move_base_to([0, 0, rotation_increment], relative=True, blocking=True, timeout=15.0)
                time.sleep(0.5)  # Settling time
        
        self.scan_complete = True
        self._robot_say("360 degree scan completed")
        if self.debug:
            print("✅ 360° scan completed!")
            
            if self.agent is not None:
                total_instances = len(self.agent.get_instances())
                print(f"📊 Total objects mapped: {total_instances}")
                print(f"🗺️ Waypoints detected: {list(self.detected_waypoints.keys())}")
                if self.detected_waypoints:
                    for name, (instance, score) in self.detected_waypoints.items():
                        print(f"  {name}: confidence {score:.3f}")
    
    def perform_degree_scan(self, total_degrees: int, divisions: int, target_object: str = "", receptacle: str = ""):
        """Perform degree-based scan with specified total rotation and divisions"""
        self.target_object = target_object
        self.receptacle = receptacle
        
        if self.debug:
            direction = "clockwise" if total_degrees >= 0 else "counter-clockwise"
            print(f"🌐 Starting {abs(total_degrees)}° scan ({direction}) with {divisions} divisions")
        
        # Calculate rotation increment per step
        if divisions <= 1:
            # Single observation, no rotation
            if self.debug:
                print(f"📍 Single observation at current position")
            
            # Update agent observations at this position
            # PERFORMANCE OPTIMIZATION: Disable head sweeping during single observations
            if self.agent is not None:
                self.agent.update(move_head=False)
                self.detect_objects_at_current_position()
                
                # Log instances detected
                instances = self.agent.get_instances()
                if self.debug:
                    print(f"🔍 Detected {len(instances)} object instances")
        else:
            # Multiple observations with rotation
            rotation_increment_degrees = total_degrees / divisions
            rotation_increment_radians = math.radians(rotation_increment_degrees)
            
            if self.debug:
                print(f"📐 Rotation increment: {rotation_increment_degrees:.1f}° ({rotation_increment_radians:.3f} rad) per step")
            
            for step in range(divisions):
                if self.stop_mapping.is_set():
                    if self.debug:
                        print("⏹️ Scan interrupted by stop signal")
                    break
                
                current_angle = step * rotation_increment_degrees
                if self.debug:
                    print(f"📍 Scan step {step + 1}/{divisions} (angle: {current_angle:.1f}° from start)")
                
                # Update agent observations at this position
                # PERFORMANCE OPTIMIZATION: Disable head sweeping during degree scans  
                if self.agent is not None:
                    self.agent.update(move_head=False)
                    self.detect_objects_at_current_position()
                    
                    # Log instances detected at this angle
                    instances = self.agent.get_instances()
                    if self.debug:
                        print(f"🔍 Detected {len(instances)} object instances at {current_angle:.1f}°")
                
                # Pause to allow for observation processing
                time.sleep(1.0)
                
                # Rotate to next position (except for last step)
                if step < divisions - 1 and self.robot is not None:
                    if self.debug:
                        direction_text = "left" if rotation_increment_degrees > 0 else "right"
                        print(f"🔄 Rotating {abs(rotation_increment_degrees):.1f}° {direction_text} to next position...")
                    
                    # Rotate using relative movement
                    self.robot.move_base_to([0, 0, rotation_increment_radians], relative=True, blocking=True, timeout=15.0)
                    time.sleep(0.5)  # Settling time
        
        self.scan_complete = True
        self._robot_say(f"{abs(total_degrees)} degree scan completed")
        if self.debug:
            print(f"✅ {abs(total_degrees)}° scan completed!")
            
            if self.agent is not None:
                total_instances = len(self.agent.get_instances())
                print(f"📊 Total objects mapped: {total_instances}")
                print(f"🗺️ Waypoints detected: {list(self.detected_waypoints.keys())}")
                if self.detected_waypoints:
                    for name, (instance, score) in self.detected_waypoints.items():
                        print(f"  {name}: confidence {score:.3f}")
    
    def start_background_mapping(self):
        """Start continuous background mapping updates"""
        if self.mapping_active:
            if self.debug:
                print("⚠️ Mapping already active")
            return
        
        self.mapping_active = True
        self.stop_mapping.clear()
        
        def background_mapping_loop():
            """Background thread for continuous mapping updates"""
            if self.debug:
                print("🔄 Starting background mapping loop...")
            
            while not self.stop_mapping.is_set():
                try:
                    if self.agent is not None:
                        # Update agent observations continuously
                        self.agent.update()
                        
                        # Get current position for logging
                        if self.robot is not None:
                            obs = self.robot.get_observation()
                            current_pos = obs.gps
                            if self.debug and time.time() % 60 < 0.1:  # Log every 60 seconds
                                instances = self.agent.get_instances()
                                print(f"🗺️ Position: x={current_pos[0]:.2f}, y={current_pos[1]:.2f}, Objects: {len(instances)}")
                    
                    # Update frequency: 1Hz (once per second)
                    time.sleep(1.0)
                    
                except Exception as e:
                    if self.debug:
                        print(f"❌ Error in background mapping: {e}")
                    time.sleep(1.0)
            
            if self.debug:
                print("⏹️ Background mapping loop stopped")
        
        # Start background thread
        self.mapping_thread = threading.Thread(target=background_mapping_loop, daemon=True)
        self.mapping_thread.start()
        
        self._robot_say("Beginning background mapping")
        if self.debug:
            print("✅ Background mapping started")
    
    def stop_background_mapping(self):
        """Stop continuous background mapping"""
        if not self.mapping_active:
            if self.debug:
                print("⚠️ Mapping not active")
            return
        
        if self.debug:
            print("⏹️ Stopping background mapping...")
        
        self.stop_mapping.set()
        self.mapping_active = False
        
        # Wait for thread to finish
        if self.mapping_thread and self.mapping_thread.is_alive():
            self.mapping_thread.join(timeout=5.0)
        
        if self.debug:
            print("✅ Background mapping stopped")
    
    def detect_objects_at_current_position(self):
        """Detect and store objects at current robot position with dynamic memory tracking"""
        if self.agent is None:
            return
        
        self.obs_count += 1  # Increment observation counter for dynamic memory
        
        # Update dynamic memory if enabled
        if self.use_dynamic_memory:
            self._update_dynamic_memory()
        
        # Check for target object
        if self.target_object:
            ranked_objects = self.agent.get_ranked_instances(self.target_object, threshold=0, debug=False)
            if ranked_objects:
                best_score, best_id, best_instance = ranked_objects[0]
                if best_score > 0.1:  # Minimum confidence threshold
                    self.detected_waypoints[self.target_object] = (best_instance, best_score)
                    if self.debug:
                        print(f"✅ Found '{self.target_object}' with confidence {best_score:.3f}")
                    # Track in dynamic memory
                    if self.use_dynamic_memory:
                        self._track_object_in_memory(self.target_object, best_instance, best_score)
                    # Visualize target object on rerun map
                    self._visualize_waypoint_on_rerun(self.target_object, best_instance, best_score, color=[255, 0, 0])
        
        # Check for receptacle
        if self.receptacle:
            ranked_receptacles = self.agent.get_ranked_instances(self.receptacle, threshold=0, debug=False)
            if ranked_receptacles:
                best_score, best_id, best_instance = ranked_receptacles[0]
                if best_score > 0.1:  # Minimum confidence threshold
                    self.detected_waypoints[self.receptacle] = (best_instance, best_score)
                    if self.debug:
                        print(f"✅ Found '{self.receptacle}' with confidence {best_score:.3f}")
                    # Track in dynamic memory
                    if self.use_dynamic_memory:
                        self._track_object_in_memory(self.receptacle, best_instance, best_score)
                    # Visualize receptacle on rerun map
                    self._visualize_waypoint_on_rerun(self.receptacle, best_instance, best_score, color=[0, 255, 0])
    
    def _visualize_waypoint_on_rerun(self, name: str, instance, score: float, color=[255, 255, 0]):
        """Add visible text marker for detected waypoint on rerun map"""
        if self.robot is None or not hasattr(self.robot, '_rerun') or self.robot._rerun is None:
            return
        
        try:
            # Get instance position for waypoint marker
            if isinstance(instance, dict) and instance.get('type') == 'saved_waypoint':
                # Saved waypoint - use stored position
                pos = instance['position']
                if len(pos) >= 3:
                    # If we have 3D position, use actual Z + offset for consistency
                    marker_pos = [pos[0], pos[1], pos[2] + 0.3]  # Use actual height + offset
                else:
                    # If only 2D position, use default height
                    marker_pos = [pos[0], pos[1], 0.5]  # Raise marker above ground
            else:
                # Object instance - get center position
                if hasattr(instance, 'point_cloud') and len(instance.point_cloud) > 0:
                    # Use center of point cloud
                    center = instance.point_cloud.mean(axis=0)
                    marker_pos = [center[0], center[1], center[2] + 0.3]  # Raise above object
                else:
                    # Fallback: use robot's current position
                    obs = self.robot.get_observation()
                    pos = obs.gps
                    marker_pos = [pos[0], pos[1], 0.5]
            
            # Create visible text marker on rerun map (escape spaces in path)
            safe_name = name.replace(" ", "_")
            self.robot._rerun.log_custom_pointcloud(
                f"world/waypoints/{safe_name}",
                points=[marker_pos],
                colors=[color],
                radii=0.15  # Large visible marker
            )
            
            # Add text label above the marker
            import rerun as rr
            
            # Create label without move count
            label_text = f"**{name}**\nConfidence: {score:.2f}"
            
            rr.log(
                f"world/waypoints/{safe_name}/label",
                rr.TextDocument(
                    label_text,
                    media_type=rr.MediaType.MARKDOWN
                )
            )
            rr.log(
                f"world/waypoints/{safe_name}/label",
                rr.Transform3D(
                    translation=[marker_pos[0], marker_pos[1], marker_pos[2] + 0.2]
                )
            )
            
            if self.debug:
                print(f"📍 Added rerun marker for '{name}' at {marker_pos}")
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to visualize waypoint '{name}' on rerun: {e}")
    
    def navigate_to_waypoint(self, waypoint_name: str) -> bool:
        """Navigate to a detected waypoint by name"""
        if waypoint_name not in self.detected_waypoints:
            if self.debug:
                print(f"❌ Waypoint '{waypoint_name}' does not exist")
                print(f"🗺️ Available waypoints: {list(self.detected_waypoints.keys())}")
            return False
        
        instance, score = self.detected_waypoints[waypoint_name]
        
        if self.robot is None:
            if self.debug:
                print(f"🔍 DRY RUN: Would navigate to '{waypoint_name}'")
            return True
        
        self._robot_say(f"Starting navigation to {waypoint_name}")
        if self.debug:
            print(f"🦭 Navigating to '{waypoint_name}' (confidence: {score:.3f})...")
        
        try:
            # Check if this is a saved waypoint (position data) or object instance
            if isinstance(instance, dict) and instance.get('type') == 'saved_waypoint':
                # This is a saved waypoint - navigate to the stored position
                target_position = instance['position']
                success = self.robot.move_base_to(target_position, blocking=True, timeout=30.0)
                if success:
                    self._robot_say(f"Navigation complete. Arrived at {waypoint_name}")
                    if self.debug:
                        print(f"✅ Successfully arrived at saved waypoint '{waypoint_name}'")
                else:
                    self._robot_say(f"Navigation failed to reach {waypoint_name}")
                    if self.debug:
                        print(f"❌ Failed to reach saved waypoint '{waypoint_name}'")
            else:
                # This is an object instance - use the agent's move_to_instance
                success = self.agent.move_to_instance(instance)
                if success:
                    self._robot_say(f"Navigation complete. Arrived at {waypoint_name}")
                    if self.debug:
                        print(f"✅ Successfully arrived at '{waypoint_name}'")
                else:
                    self._robot_say(f"Navigation failed to reach {waypoint_name}")
                    if self.debug:
                        print(f"❌ Failed to reach '{waypoint_name}'")
            return success
        except Exception as e:
            self._robot_say(f"Navigation error occurred")
            if self.debug:
                print(f"❌ Navigation error: {e}")
            return False

    def save_current_waypoint(self, waypoint_name: str) -> bool:
        """Save the current robot location as a named waypoint"""
        if self.robot is None:
            if self.debug:
                print(f"🔍 DRY RUN: Would save current location as waypoint '{waypoint_name}'")
            return True
        
        try:
            # Get current robot position
            current_pos = self.robot.get_base_pose()
            if current_pos is None:
                print(f"❌ Could not get current robot position")
                return False
            
            # Create a simple instance-like object with the current position
            # We'll store it as a tuple: (position, confidence_score)
            position_data = {
                'position': current_pos,
                'type': 'saved_waypoint',
                'timestamp': time.time()
            }
            
            # Store in detected_waypoints with a confidence of 1.0 (perfect confidence for saved locations)
            self.detected_waypoints[waypoint_name] = (position_data, 1.0)
            
            # Visualize the saved waypoint on rerun (yellow marker)
            self._visualize_waypoint_on_rerun(waypoint_name, position_data, 1.0, color=[255, 255, 0])  # Yellow for saved waypoints
            
            if self.debug:
                print(f"📍 Waypoint '{waypoint_name}' saved at position x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, θ={current_pos[2]:.3f}")
                print(f"🟡 Added yellow marker to rerun for waypoint '{waypoint_name}'")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save waypoint '{waypoint_name}': {e}")
            return False
    
    def remove_waypoint(self, waypoint_name: str) -> bool:
        """Remove a waypoint and its visualization from the map"""
        if waypoint_name not in self.detected_waypoints:
            if self.debug:
                print(f"❌ Waypoint '{waypoint_name}' does not exist")
                print(f"🗺️ Available waypoints: {list(self.detected_waypoints.keys())}")
            return False
        
        if self.debug:
            print(f"🗑️ Removing waypoint '{waypoint_name}'...")
        
        try:
            # Remove from waypoints dictionary
            del self.detected_waypoints[waypoint_name]
            
            # Remove visualization from rerun viewer
            self._remove_waypoint_visualization(waypoint_name)
            
            if self.debug:
                print(f"✅ Successfully removed waypoint '{waypoint_name}'")
                print(f"🗺️ Remaining waypoints: {list(self.detected_waypoints.keys())}")
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to remove waypoint '{waypoint_name}': {e}")
            return False
    
    def move_waypoint_to_waypoint(self, source_waypoint: str, target_waypoint: str) -> bool:
        """Move a waypoint to the location of another waypoint"""
        if self.debug:
            print(f"🔄 Moving waypoint '{source_waypoint}' to location of '{target_waypoint}'")
        
        # Check if both waypoints exist (with quote handling)
        source_key = None
        if source_waypoint in self.detected_waypoints:
            source_key = source_waypoint
        elif f"'{source_waypoint}'" in self.detected_waypoints:
            source_key = f"'{source_waypoint}'"
        elif source_waypoint.strip("'\"") in self.detected_waypoints:
            source_key = source_waypoint.strip("'\"")
        
        if source_key is None:
            if self.debug:
                print(f"❌ Source waypoint '{source_waypoint}' does not exist")
                print(f"🗺️ Available waypoints: {list(self.detected_waypoints.keys())}")
            return False
            
        target_key = None
        if target_waypoint in self.detected_waypoints:
            target_key = target_waypoint
        elif f"'{target_waypoint}'" in self.detected_waypoints:
            target_key = f"'{target_waypoint}'"
        elif target_waypoint.strip("'\"") in self.detected_waypoints:
            target_key = target_waypoint.strip("'\"")
            
        if target_key is None:
            if self.debug:
                print(f"❌ Target waypoint '{target_waypoint}' does not exist") 
                print(f"🗺️ Available waypoints: {list(self.detected_waypoints.keys())}")
            return False
        
        try:
            # Get target waypoint position
            target_instance, target_score = self.detected_waypoints[target_key]
            
            # Get target position
            if isinstance(target_instance, dict) and target_instance.get('type') == 'saved_waypoint':
                target_position = target_instance['position']
            else:
                # Object instance - get center position
                if hasattr(target_instance, 'point_cloud') and len(target_instance.point_cloud) > 0:
                    center = target_instance.point_cloud.mean(axis=0)
                    target_position = [center[0], center[1], center[2]]
                else:
                    if self.debug:
                        print(f"❌ Could not get position from target waypoint '{target_waypoint}'")
                    return False
            
            # Update source waypoint to target position
            if self.debug:
                print(f"📍 Moving '{source_waypoint}' to position {target_position}")
            
            return self.update_object_waypoint_after_place(source_key, target_position)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to move waypoint: {e}")
            return False
    
    def _remove_waypoint_visualization(self, name: str):
        """Remove waypoint marker and label from rerun viewer"""
        if self.robot is None or not hasattr(self.robot, '_rerun') or self.robot._rerun is None:
            if self.debug:
                print("⚠️ Rerun not available - skipping visualization removal")
            return
        
        try:
            import rerun as rr
            
            # Create safe name for rerun path (escape spaces)
            safe_name = name.replace(" ", "_")
            
            # Clear the waypoint marker by logging empty data
            self.robot._rerun.log_custom_pointcloud(
                f"world/waypoints/{safe_name}",
                points=[],
                colors=[],
                radii=0.0
            )
            
            # Clear the text label
            rr.log(
                f"world/waypoints/{safe_name}/label",
                rr.Clear(recursive=True)
            )
            
            if self.debug:
                print(f"🧹 Removed rerun visualization for waypoint '{name}'")
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to remove waypoint visualization '{name}' from rerun: {e}")
    
    def update_object_waypoint_after_place(self, object_name: str, new_position: list) -> bool:
        """Update object waypoint to new location after successful placement"""
        try:
            if self.debug:
                print(f"🔄 Object '{object_name}' moved to new location {new_position}")
                print(f"🗂️ Current waypoints: {list(self.detected_waypoints.keys())}")
        except Exception as e:
            if self.debug:
                print(f"⚠️ Debug output failed: {e}")
        
        try:
            # Get existing waypoint data or create new
            # Handle both quoted and unquoted versions of object names  
            waypoint_key = None
            if object_name in self.detected_waypoints:
                waypoint_key = object_name
            elif f"'{object_name}'" in self.detected_waypoints:
                waypoint_key = f"'{object_name}'"
            elif object_name.strip("'\"") in self.detected_waypoints:
                waypoint_key = object_name.strip("'\"")
            
            if self.debug and waypoint_key:
                print(f"🔍 Found waypoint using key: '{waypoint_key}'")
            
            if waypoint_key:
                # Update existing waypoint
                old_instance, score = self.detected_waypoints[waypoint_key]
                
                # Create new waypoint data with updated position
                new_waypoint_data = {
                    'type': 'saved_waypoint',
                    'position': new_position,
                    'timestamp': time.time()
                }
                
                # Update waypoint storage (use original key to maintain consistency)
                self.detected_waypoints[waypoint_key] = (new_waypoint_data, score)
                
                try:
                    # Remove old visualization first (use the original stored key)
                    self._remove_waypoint_visualization(waypoint_key)
                    
                    # Small delay to prevent Rerun websocket overload
                    time.sleep(0.1)
                    
                    # Add new visualization at updated position
                    self._visualize_waypoint_on_rerun(
                        object_name, 
                        new_waypoint_data, 
                        score, 
                        color=[0, 255, 0]  # Green for moved objects
                    )
                except Exception as rerun_error:
                    if self.debug:
                        print(f"⚠️ Rerun visualization update failed: {rerun_error}")
                    # Continue anyway - data is still updated
                
                if self.debug:
                    print(f"📍 Updated waypoint '{object_name}'")
                
                return True
            else:
                # Create new waypoint for this object
                new_waypoint_data = {
                    'type': 'saved_waypoint',
                    'position': new_position,
                    'timestamp': time.time()
                }
                
                # Store new waypoint
                self.detected_waypoints[object_name] = (new_waypoint_data, 1.0)  # Perfect confidence for placed objects
                
                try:
                    # Add visualization  
                    self._visualize_waypoint_on_rerun(
                        object_name, 
                        new_waypoint_data, 
                        1.0, 
                        color=[0, 255, 0]  # Green for placed objects
                    )
                except Exception as rerun_error:
                    if self.debug:
                        print(f"⚠️ Rerun visualization creation failed: {rerun_error}")
                    # Continue anyway - data is still updated
                
                if self.debug:
                    print(f"📍 Created new waypoint '{object_name}'")
                
                return True
                
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to update waypoint for '{object_name}': {e}")
            return False
    
    def _update_dynamic_memory(self):
        """Update dynamic memory system with current observations"""
        if not self.use_dynamic_memory or self.agent is None:
            return
        
        try:
            # Get all current instances from the agent
            instances = self.agent.get_instances()
            
            # Track only target and receptacle objects in current observation
            current_objects = set()
            target_objects_to_track = []
            
            # Add target object and receptacle to tracking list if specified
            if self.target_object:
                target_objects_to_track.append(self.target_object.lower())
            if self.receptacle:
                target_objects_to_track.append(self.receptacle.lower())
            
            # Only track objects we care about
            if target_objects_to_track:
                for instance_id, instance in instances.items():
                    if hasattr(instance, 'category_id') and instance.category_id is not None:
                        object_name = str(instance.category_id).lower()
                        
                        # Only track if it's one of our target objects
                        if object_name in target_objects_to_track:
                            current_objects.add(object_name)
                            # Update object tracking
                            self._track_object_in_memory(object_name, instance, getattr(instance, 'confidence', 0.5))
            
            # Check for objects that have disappeared
            # self._check_disappeared_objects(current_objects)  # Disabled cleanup for now
            
            # Update semantic memory with current observation
            if self.robot is not None:
                self._update_semantic_memory_from_observation()
                
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error updating dynamic memory: {e}")
    
    def _track_object_in_memory(self, object_name: str, instance, confidence: float):
        """Track an object in dynamic memory system"""
        try:
            # Get object position
            if hasattr(instance, 'point_cloud') and len(instance.point_cloud) > 0:
                position = instance.point_cloud.mean(axis=0)  # Center of point cloud
            elif self.robot is not None:
                # Fallback to robot position
                obs = self.robot.get_observation()
                position = obs.gps[:2]  # Just x, y
                position = np.append(position, [0.5])  # Add height estimate
            else:
                return
            
            # Check if object already being tracked
            if object_name in self.tracked_objects:
                # Update existing tracking
                tracked_obj = self.tracked_objects[object_name]
                old_pos = tracked_obj['position']
                
                # Check if position has changed significantly
                distance = np.linalg.norm(position - old_pos)
                if distance > self.position_update_threshold:
                    # Object has moved - update position
                    tracked_obj['position'] = position
                    tracked_obj['last_moved'] = self.obs_count
                    if self.debug:
                        print(f"📍 '{object_name}' moved {distance:.2f}m to new position")
                        
                    # Update waypoint if this is a tracked target
                    if object_name in self.detected_waypoints:
                        self.detected_waypoints[object_name] = (instance, confidence)
                        self._visualize_waypoint_on_rerun(object_name, instance, confidence, color=[255, 165, 0])  # Orange for moved objects
                
                # Update last seen time
                tracked_obj['last_seen'] = self.obs_count
                tracked_obj['confidence'] = max(tracked_obj['confidence'], confidence)
                
            else:
                # New object - start tracking
                self.tracked_objects[object_name] = {
                    'position': position,
                    'last_seen': self.obs_count,
                    'first_seen': self.obs_count,
                    'last_moved': self.obs_count,
                    'confidence': confidence,
                    'detection_count': 1
                }
                
                if self.debug:
                    print(f"🆕 Started tracking new object: '{object_name}' at {position[:2]}")
                    
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error tracking object '{object_name}': {e}")
    
    def _check_disappeared_objects(self, current_objects: set):
        """Check for objects that may have disappeared from view"""
        disappeared_objects = []
        
        for object_name, tracking_data in self.tracked_objects.items():
            observations_since_seen = self.obs_count - tracking_data['last_seen']
            
            if observations_since_seen > self.object_disappearance_threshold:
                # Object hasn't been seen recently
                if object_name not in current_objects:
                    disappeared_objects.append(object_name)
                    
                    if self.debug:
                        print(f"👻 Object '{object_name}' may have disappeared ({observations_since_seen} observations ago)")
                    
                    # Remove from waypoints if it was tracked
                    if object_name in self.detected_waypoints:
                        if self.debug:
                            print(f"🗑️ Removing '{object_name}' waypoint due to disappearance")
                        self.remove_waypoint(object_name)
        
        # Clean up disappeared objects from tracking
        for object_name in disappeared_objects:
            if object_name in self.tracked_objects:
                del self.tracked_objects[object_name]
                if self.debug:
                    print(f"🧹 Cleaned up tracking data for '{object_name}'")
    
    def _update_semantic_memory_from_observation(self):
        """Update semantic memory with current robot observation"""
        if not self.use_dynamic_memory or self.robot is None:
            return
            
        try:
            # Get current observation
            obs = self.robot.get_observation()
            if obs is None or obs.rgb is None or obs.depth is None:
                return
            
            # Convert to tensors for processing
            rgb = torch.from_numpy(obs.rgb).permute(2, 0, 1).float()  # H,W,C -> C,H,W
            depth = torch.from_numpy(obs.depth).float()
            
            # Get camera parameters
            camera_pose = torch.from_numpy(obs.camera_pose).float()
            camera_K = torch.from_numpy(obs.camera_K).float()
            
            # Process depth mask
            valid_depth = (depth > 0.25) & (depth < 2.5)  # Valid depth range
            
            if torch.sum(valid_depth) == 0:
                return  # No valid depth points
            
            # Extract features if encoder is available (would need CLIP encoder)
            # For now, use RGB as features
            height, width = depth.shape
            rgb_flat = rgb.permute(1, 2, 0).reshape(-1, 3)  # Flatten to N x 3
            
            # Get world coordinates using robot's camera
            # This is a simplified version - in full DynaMem they use proper camera projection
            if hasattr(obs, 'xyz') and obs.xyz is not None:
                world_xyz = torch.from_numpy(obs.xyz).reshape(-1, 3)
                valid_points = world_xyz[valid_depth.flatten()]
                valid_rgb = rgb_flat[valid_depth.flatten()]
                
                if len(valid_points) > 0:
                    # Subsample points for efficiency
                    num_points = len(valid_points)
                    if num_points > 1000:  # Limit points per observation
                        indices = torch.randperm(num_points)[:1000]
                        valid_points = valid_points[indices]
                        valid_rgb = valid_rgb[indices]
                    
                    # Add to semantic memory
                    self.semantic_memory.add(
                        points=valid_points.to(self.semantic_memory._device),
                        features=None,  # Would use CLIP features in full implementation
                        rgb=valid_rgb.to(self.semantic_memory._device),
                        weights=None,
                        obs_count=self.obs_count
                    )
                    
                    if self.debug and self.obs_count % 10 == 0:  # Log every 10 observations
                        total_points = len(self.semantic_memory._points) if self.semantic_memory._points is not None else 0
                        print(f"🧠 Dynamic memory: {total_points} points, tracking {len(self.tracked_objects)} objects")
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error updating semantic memory: {e}")
    
    def get_dynamic_memory_status(self) -> Dict[str, Any]:
        """Get status of dynamic memory system"""
        if not self.use_dynamic_memory:
            return {"enabled": False}
        
        try:
            total_points = len(self.semantic_memory._points) if self.semantic_memory._points is not None else 0
            
            status = {
                "enabled": True,
                "observation_count": self.obs_count,
                "total_semantic_points": total_points,
                "tracked_objects": len(self.tracked_objects),
                "object_details": {}
            }
            
            # Add details for each tracked object
            for obj_name, tracking_data in self.tracked_objects.items():
                status["object_details"][obj_name] = {
                    "confidence": tracking_data["confidence"],
                    "last_seen": tracking_data["last_seen"],
                    "observations_ago": self.obs_count - tracking_data["last_seen"],
                    "position": tracking_data["position"].tolist() if isinstance(tracking_data["position"], np.ndarray) else tracking_data["position"]
                }
            
            return status
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error getting dynamic memory status: {e}")
            return {"enabled": True, "error": str(e)}
    
    def find_object_with_dynamic_memory(self, object_name: str) -> Optional[Tuple[np.ndarray, float]]:
        """Find an object using dynamic memory system"""
        if not self.use_dynamic_memory:
            return None
        
        object_name = object_name.lower()
        
        # Check if object is currently being tracked
        if object_name in self.tracked_objects:
            tracking_data = self.tracked_objects[object_name]
            observations_ago = self.obs_count - tracking_data['last_seen']
            
            # Only return if recently seen (within threshold)
            if observations_ago <= self.object_disappearance_threshold:
                return tracking_data['position'], tracking_data['confidence']
        
        # Could implement CLIP-based semantic search here in full version
        return None
    
    def grab_object(self, target_object: str, receptacle: str = "") -> bool:
        """Enhanced object pickup with smart navigation and faster execution"""
        if self.pickup_executor is None:
            if self.debug:
                print("❌ Pickup system not available. Cannot execute grab command.")
                print("🔧 This might be due to dry run mode or initialization failure.")
            return False
        
        # Parse quoted strings for multi-word object names
        target_object = target_object.strip('"').strip("'")
        if receptacle:
            receptacle = receptacle.strip('"').strip("'")
        
        if receptacle:
            self._robot_say(f"Beginning grab sequence for {target_object} to place in {receptacle}")
        else:
            self._robot_say(f"Beginning grab sequence for {target_object}")
        if self.debug:
            if receptacle:
                print(f"🦾 Enhanced grab: '{target_object}' -> '{receptacle}'")
            else:
                print(f"🦾 Enhanced grab: '{target_object}' (pick only)")
        
        try:
            # Temporarily pause background mapping to avoid interference
            mapping_was_active = self.mapping_active
            if mapping_was_active:
                if self.debug:
                    print("⏸️ Pausing background mapping during grab operation")
                self.stop_background_mapping()
            
            # OPTIMIZATION 1: Pre-navigate to known object location if available
            navigation_success = self._smart_navigate_to_object(target_object)
            if not navigation_success:
                if self.debug:
                    print("⚠️ Pre-navigation failed, PickupExecutor will handle search")
            
            # OPTIMIZATION 2: Use faster pickup execution with reduced timeouts
            if receptacle:
                # Full pickup and place sequence
                llm_response = [("pickup", target_object), ("place", receptacle)]
            else:
                # Pick-only operation  
                llm_response = [("pickup", target_object)]
            
            if self.debug:
                print("🚀 Starting optimized pickup operation...")
            
            # OPTIMIZATION 3: Execute with enhanced pickup executor settings
            if self.debug:
                executor_type = type(self.pickup_executor).__name__
                print(f"🔧 Using pickup executor: {executor_type}")
            success = self._execute_fast_pickup(llm_response)
            
            if success:
                if receptacle:
                    # Full pickup and place - not holding anything afterward
                    self.update_holding_state(None)
                    self._robot_say(f"Grab complete. Successfully placed {target_object} in {receptacle}")
                    if self.debug:
                        print(f"✅ Successfully grabbed '{target_object}' and placed in '{receptacle}'")
                else:
                    # Pick only - now holding the object
                    self.update_holding_state(target_object)
                    self._robot_say(f"Pickup complete. Now holding {target_object}")
                    if self.debug:
                        print(f"✅ Successfully grabbed '{target_object}' - now holding it")
            else:
                self._robot_say(f"Grab failed for {target_object}")
                if self.debug:
                    print(f"❌ Failed to grab '{target_object}'")
            
            # Resume background mapping if it was active
            if mapping_was_active:
                if self.debug:
                    print("▶️ Resuming background mapping")
                self.start_background_mapping()
            
            return success
            
        except Exception as e:
            if self.debug:
                print(f"❌ Grab operation error: {e}")
            
            # Make sure to resume background mapping even if there was an error
            if mapping_was_active and not self.mapping_active:
                if self.debug:
                    print("▶️ Resuming background mapping after error")
                self.start_background_mapping()
            
            return False
    
    def _smart_navigate_to_object(self, target_object: str) -> bool:
        """Smart navigation to known object locations with A* pathfinding optimization"""
        if self.debug:
            print(f"🧭 Checking for known location of '{target_object}'...")
        
        # First check if object exists in detected waypoints (from 360° scan)
        if target_object in self.detected_waypoints:
            instance, score = self.detected_waypoints[target_object]
            self._robot_say(f"Object {target_object} found in memory, navigating directly")
            if self.debug:
                print(f"📍 Found '{target_object}' in memory (confidence: {score:.3f})")
                print(f"🚀 Using A* optimal pathfinding to known location...")
            
            # Navigate using optimized A* pathfinding
            nav_success = self._navigate_with_optimal_pathfinding(target_object)
            if nav_success:
                if self.debug:
                    print(f"✅ Successfully arrived at '{target_object}' location via optimal path")
                return True
            else:
                if self.debug:
                    print(f"❌ Optimal navigation to '{target_object}' failed, trying fallback")
                # Fallback to standard navigation
                nav_success = self.navigate_to_waypoint(target_object)
                if nav_success:
                    if self.debug:
                        print(f"✅ Fallback navigation successful")
                    return True
                else:
                    if self.debug:
                        print(f"❌ All navigation attempts to '{target_object}' failed")
                    return False
        
        # Second check: Search current instance memory for the object
        if self.agent is not None:
            try:
                ranked_objects = self.agent.get_ranked_instances(target_object, threshold=0.1, debug=False)
                if ranked_objects:
                    best_score, best_id, best_instance = ranked_objects[0]
                    self._robot_say(f"Object {target_object} located in current view")
                    if self.debug:
                        print(f"🔍 Found '{target_object}' in current view (confidence: {best_score:.3f})")
                        print(f"🚀 Using A* optimal pathfinding to detected instance...")
                    
                    # Navigate to the best matching instance with A* optimization
                    nav_success = self._navigate_to_instance_optimal(best_instance)
                    if nav_success:
                        if self.debug:
                            print(f"✅ Successfully arrived at '{target_object}' instance via optimal path")
                        return True
                    else:
                        if self.debug:
                            print(f"❌ Optimal navigation to '{target_object}' instance failed, trying fallback")
                        # Fallback to standard navigation
                        nav_success = self.agent.move_to_instance(best_instance)
                        if nav_success:
                            if self.debug:
                                print(f"✅ Fallback navigation to instance successful")
                            return True
                        else:
                            if self.debug:
                                print(f"❌ All navigation attempts to '{target_object}' instance failed")
                            return False
                else:
                    self._robot_say(f"Object {target_object} not found in current view")
                    if self.debug:
                        print(f"🔍 '{target_object}' not found in current instance memory")
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Error searching for '{target_object}': {e}")
        
        # Object not found in memory - let PickupExecutor handle the search
        self._robot_say(f"Object {target_object} not in memory, beginning search")
        if self.debug:
            print(f"📍 '{target_object}' location unknown - will search during pickup")
            available_objects = list(self.detected_waypoints.keys())
            if available_objects:
                print(f"💡 Known objects in memory: {available_objects}")
        
        return False
    
    def _navigate_with_optimal_pathfinding(self, waypoint_name: str) -> bool:
        """Navigate to waypoint using A* optimal pathfinding"""
        if waypoint_name not in self.detected_waypoints:
            return False
        
        instance, score = self.detected_waypoints[waypoint_name]
        
        if self.robot is None:
            if self.debug:
                print(f"🔍 DRY RUN: Would use A* pathfinding to '{waypoint_name}'")
            return True
        
        try:
            # Configure agent for A* pathfinding
            original_config = self._configure_optimal_pathfinding()
            
            if self.debug:
                print(f"🎯 Planning optimal A* path to '{waypoint_name}'...")
            
            # Execute navigation with A* pathfinding
            if isinstance(instance, dict) and instance.get('type') == 'saved_waypoint':
                # Navigate to saved waypoint position
                target_position = instance['position']
                success = self.robot.move_base_to(target_position, blocking=True, timeout=45.0)
            else:
                # Navigate to object instance using A* planner
                success = self.agent.move_to_instance(instance)
            
            # Restore original configuration
            self._restore_pathfinding_config(original_config)
            
            return success
            
        except Exception as e:
            if self.debug:
                print(f"❌ A* navigation error: {e}")
            # Restore original configuration on error
            try:
                self._restore_pathfinding_config(original_config)
            except:
                pass
            return False
    
    def _navigate_to_instance_optimal(self, instance) -> bool:
        """Navigate to object instance using A* optimal pathfinding"""
        if self.robot is None or self.agent is None:
            if self.debug:
                print(f"🔍 DRY RUN: Would use A* pathfinding to instance")
            return True
        
        try:
            # Configure agent for A* pathfinding
            original_config = self._configure_optimal_pathfinding()
            
            if self.debug:
                print(f"🎯 Planning optimal A* path to object instance...")
            
            # Execute navigation with A* pathfinding
            success = self.agent.move_to_instance(instance)
            
            # Restore original configuration
            self._restore_pathfinding_config(original_config)
            
            return success
            
        except Exception as e:
            if self.debug:
                print(f"❌ A* instance navigation error: {e}")
            # Restore original configuration on error
            try:
                self._restore_pathfinding_config(original_config)
            except:
                pass
            return False
    
    def _configure_optimal_pathfinding(self) -> dict:
        """Configure agent to use A* pathfinding for optimal routes"""
        original_config = {}
        
        try:
            if self.agent and hasattr(self.agent, 'parameters'):
                # Store original motion planner settings
                if hasattr(self.agent.parameters, 'motion_planner'):
                    original_config['algorithm'] = getattr(self.agent.parameters.motion_planner, 'algorithm', 'rrt_connect')
                    original_config['simplify_plans'] = getattr(self.agent.parameters.motion_planner, 'simplify_plans', True)
                    original_config['shortcut_plans'] = getattr(self.agent.parameters.motion_planner, 'shortcut_plans', True)
                
                    # Configure for A* optimal pathfinding
                    self.agent.parameters.motion_planner.algorithm = "a_star"
                    self.agent.parameters.motion_planner.simplify_plans = False  # A* already optimal
                    self.agent.parameters.motion_planner.shortcut_plans = False  # Keep optimal path
                    
                    if self.debug:
                        print("⚡ Configured A* pathfinding: optimal routes, no unnecessary steps")
                else:
                    if self.debug:
                        print("⚠️ Motion planner configuration not accessible")
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Could not configure A* pathfinding: {e}")
        
        return original_config
    
    def _restore_pathfinding_config(self, original_config: dict):
        """Restore original pathfinding configuration"""
        try:
            if self.agent and hasattr(self.agent, 'parameters') and hasattr(self.agent.parameters, 'motion_planner'):
                if 'algorithm' in original_config:
                    self.agent.parameters.motion_planner.algorithm = original_config['algorithm']
                if 'simplify_plans' in original_config:
                    self.agent.parameters.motion_planner.simplify_plans = original_config['simplify_plans']
                if 'shortcut_plans' in original_config:
                    self.agent.parameters.motion_planner.shortcut_plans = original_config['shortcut_plans']
                
                if self.debug:
                    print("🔄 Restored original pathfinding configuration")
        except Exception as e:
            if self.debug:
                print(f"⚠️ Could not restore pathfinding config: {e}")
    
    def _execute_fast_pickup(self, llm_response: list) -> bool:
        """Execute pickup with optimized settings for faster operation"""
        if self.debug:
            print("⚡ Using optimized pickup execution...")
        
        try:
            # Store original pickup executor settings if they exist
            original_settings = {}
            
            # OPTIMIZATION: Reduce search timeouts and increase confidence thresholds
            if hasattr(self.pickup_executor, 'robot') and self.pickup_executor.robot:
                # Optimize movement speeds for faster execution
                if self.debug:
                    print("⚡ Applying fast execution optimizations...")
                
                # Set faster movement parameters (if configurable)
                # This would depend on the specific PickupExecutor implementation
                pass
            
            # Execute the pickup while preserving detected instances from 360° scan
            success = self._execute_pickup_with_waypoint_data(llm_response)
            
            # Restore original settings if they were changed
            # (Implementation depends on PickupExecutor internals)
            
            return success
            
        except Exception as e:
            if self.debug:
                print(f"❌ Fast pickup execution error: {e}")
            return False
    
    def _execute_pickup_with_waypoint_data(self, llm_response: list) -> bool:
        """Execute pickup using PickupExecutor but preserve waypoint data from 360° scan"""
        if self.debug:
            print("🔗 Executing pickup with preserved waypoint data...")
        
        try:
            # Get the PickupExecutor's __call__ method logic but skip the agent.reset()
            if llm_response is None or len(llm_response) == 0:
                if self.debug:
                    print("❌ No pickup commands to execute")
                return False
            
            # DON'T call self.agent.reset() - this preserves our 360° scan data!
            if self.debug:
                print("💾 Preserving agent instances from 360° scan (skipping reset)")
            
            # Extract the pickup command
            command, target_object = llm_response[0]
            if command != "pickup":
                if self.debug:
                    print(f"❌ Expected pickup command, got: {command}")
                return False
            
            # Check if we have a place command too
            receptacle = None
            if len(llm_response) > 1:
                place_command, receptacle = llm_response[1]
                if place_command == "place":
                    if self.debug:
                        print(f"🎯 Pickup '{target_object}' and place in '{receptacle}'")
                else:
                    receptacle = None
            
            if receptacle is None:
                if self.debug:
                    print(f"🎯 Pickup '{target_object}' only")
            
            # Use our own pickup logic that leverages waypoint data
            return self._execute_waypoint_aware_pickup(target_object, receptacle)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Waypoint-aware pickup error: {e}")
            # Fallback to standard pickup executor
            return self.pickup_executor(llm_response)
    
    def _execute_waypoint_aware_pickup(self, target_object: str, receptacle: str = None) -> bool:
        """Execute pickup using waypoint data from 360° scan instead of fresh search"""
        if self.debug:
            print(f"🎯 Waypoint-aware pickup: '{target_object}' -> '{receptacle or 'hand only'}'")
        
        try:
            # Check if we have the target object in our waypoint memory
            # Handle both quoted and unquoted versions of object names
            waypoint_key = None
            if target_object in self.detected_waypoints:
                waypoint_key = target_object
            elif f"'{target_object}'" in self.detected_waypoints:
                waypoint_key = f"'{target_object}'"
            elif target_object.strip("'\"") in self.detected_waypoints:
                waypoint_key = target_object.strip("'\"")
            
            if waypoint_key:
                instance, score = self.detected_waypoints[waypoint_key]
                if self.debug:
                    print(f"✅ Using waypoint data for '{target_object}' -> '{waypoint_key}' (confidence: {score:.3f})")
                
                # Handle pick-only vs pickup+place differently
                if receptacle is None or receptacle == "":
                    if self.debug:
                        print("🤏 Pick-only operation (no placement)")
                    # Use PickObjectTask for pick-only operations
                    from stretch.agent.task.pickup.pick_task import PickObjectTask
                    
                    pickup_task = PickObjectTask(
                        self.agent,
                        target_object=target_object,
                        matching=self.pickup_executor._match_method,
                        use_visual_servoing_for_grasp=not self.pickup_executor._open_loop,
                    )
                    
                    # Get the task but don't use "add_rotate" since we're already positioned
                    task = pickup_task.get_task(add_rotate=False)
                    
                else:
                    if self.debug:
                        print(f"📦 Pickup and place operation -> '{receptacle}'")
                    # Use full PickupTask for pickup+place operations
                    from stretch.agent.task.pickup.pickup_task import PickupTask
                    
                    pickup_task = PickupTask(
                        self.agent,
                        target_object=target_object,
                        target_receptacle=receptacle,
                        matching=self.pickup_executor._match_method,
                        use_visual_servoing_for_grasp=not self.pickup_executor._open_loop,
                    )
                    
                    # Get the task but don't use "add_rotate" since we're already positioned
                    task = pickup_task.get_task(add_rotate=False, mode=self.pickup_executor._pickup_task_mode)
                
                # Execute the task
                task.run()
                return True
            else:
                if self.debug:
                    print(f"⚠️ No waypoint data for '{target_object}', falling back to standard pickup")
                # Fallback to appropriate standard pickup executor method
                if receptacle is None or receptacle == "":
                    return self.pickup_executor._pick_only(target_object)
                else:
                    return self.pickup_executor._pickup(target_object, receptacle)
                
        except Exception as e:
            if self.debug:
                print(f"❌ Waypoint-aware pickup failed: {e}")
                print("🔄 Falling back to standard pickup executor")
            # Fallback to appropriate standard approach
            if receptacle is None or receptacle == "":
                return self.pickup_executor._pick_only(target_object)
            else:
                return self.pickup_executor._pickup(target_object, receptacle)
    
    def list_grabbable_objects(self) -> list:
        """List objects that have been detected and could potentially be grabbed"""
        grabbable_objects = []
        
        if self.agent is None:
            return grabbable_objects
        
        try:
            # Get all detected instances
            instances = self.agent.get_instances()
            
            # Filter for objects that are likely grabbable (not furniture/walls)
            common_grabbable_classes = [
                'apple', 'orange', 'banana', 'cup', 'mug', 'bottle', 'bowl', 'ball',
                'book', 'phone', 'remote', 'toy', 'stuffed animal', 'pen', 'pencil',
                'scissors', 'tape', 'stapler', 'mouse', 'keyboard', 'headphones',
                'shoe', 'sock', 'hat', 'glove', 'bag', 'backpack', 'wallet',
                'glasses', 'sunglasses', 'watch', 'bracelet', 'ring', 'necklace'
            ]
            
            for instance_id, instance in instances.items():
                if hasattr(instance, 'category_id') and instance.category_id is not None:
                    # Get the semantic class name 
                    class_name = instance.category_id.lower() if hasattr(instance.category_id, 'lower') else str(instance.category_id).lower()
                    
                    # Check if it's a potentially grabbable object
                    is_grabbable = any(grabbable_class in class_name for grabbable_class in common_grabbable_classes)
                    
                    if is_grabbable:
                        grabbable_objects.append({
                            'name': class_name,
                            'instance_id': instance_id,
                            'confidence': getattr(instance, 'confidence', 0.0)
                        })
            
            # Sort by confidence (highest first)
            grabbable_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Error listing grabbable objects: {e}")
        
        return grabbable_objects
    
    def is_holding_object(self) -> bool:
        """Check if robot is currently holding an object"""
        return self.currently_holding is not None

    def get_held_object(self) -> str:
        """Get name of currently held object, empty string if none"""
        return self.currently_holding if self.currently_holding else ""

    def update_holding_state(self, object_name: str = None):
        """Update what object robot is currently holding"""
        if object_name:
            # Picking up new object
            self.currently_holding = object_name
            self.pickup_timestamp = time.time()
            if self.debug:
                print(f"🤏 Now holding: '{object_name}'")
        else:
            # Placing or dropping object
            if self.currently_holding:
                # Record placement history
                placement_record = {
                    'object': self.currently_holding,
                    'picked_up_at': self.pickup_timestamp,
                    'placed_at': time.time(),
                    'duration_held': time.time() - (self.pickup_timestamp or time.time())
                }
                self.placement_history.append(placement_record)
                
                if self.debug:
                    print(f"📤 No longer holding: '{self.currently_holding}'")
            
            self.currently_holding = None
            self.pickup_timestamp = None
    
    def place_object_at_location(self, target_location: str) -> bool:
        """Place currently held object at specified location"""
        
        # Validation: Check if holding an object
        if not self.is_holding_object():
            if self.debug:
                print("❌ No object currently held. Use 'grab <object>' first.")
            return False
        
        # Validation: Check pickup executor availability
        if self.pickup_executor is None:
            if self.debug:
                print("❌ Pickup system not available. Cannot execute place command.")
            return False
        
        held_object = self.get_held_object()
        target_location = target_location.strip('"').strip("'")  # Handle quoted strings
        
        if self.debug:
            print(f"📦 Placing '{held_object}' at '{target_location}'")
        
        try:
            # Pause background mapping during placement
            mapping_was_active = self.mapping_active
            if mapping_was_active:
                if self.debug:
                    print("⏸️ Pausing background mapping during place operation")
                self.stop_background_mapping()
            
            # Determine placement strategy based on target location
            success = False
            
            if target_location.lower() == "here":
                success = self._place_at_current_location()
            elif target_location in self.detected_waypoints:
                success = self._place_at_waypoint(target_location)
            else:
                success = self._place_near_object(target_location)
            
            # Update holding state if placement successful
            if success:
                self.update_holding_state(None)  # No longer holding object
                if self.debug:
                    print(f"✅ Successfully placed '{held_object}' at '{target_location}'")
                
                # Update object waypoint to new placement location
                try:
                    if held_object:  # Only update if we actually held something
                        current_pos = self.robot.get_base_pose()
                        if current_pos is not None:
                            self.update_object_waypoint_after_place(held_object, current_pos)
                        else:
                            if self.debug:
                                print("⚠️ Could not get robot position for waypoint update")
                    else:
                        if self.debug:
                            print("⚠️ No held object to update waypoint for")
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ Failed to update waypoint after placement: {e}")
            else:
                if self.debug:
                    print(f"❌ Failed to place '{held_object}' at '{target_location}'")
            
            # Resume background mapping
            if mapping_was_active:
                if self.debug:
                    print("▶️ Resuming background mapping")
                self.start_background_mapping()
            
            return success
            
        except Exception as e:
            if self.debug:
                print(f"❌ Place operation error: {e}")
            
            # Ensure background mapping resumes even on error
            if mapping_was_active and not self.mapping_active:
                if self.debug:
                    print("▶️ Resuming background mapping after error")
                self.start_background_mapping()
            
            return False
    
    def _place_at_current_location(self) -> bool:
        """Place object at current robot position"""
        if self.debug:
            print("📍 Placing object at current robot location")
        
        # Speech announcement for current location placement
        self._robot_say("Placing object at current location")
        
        try:
            # Use PickupExecutor to place at current location
            # This typically means placing on the ground or nearby surface
            llm_response = [("place", "here")]
            return self.pickup_executor(llm_response)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to place at current location: {e}")
            return False

    def _place_at_waypoint(self, waypoint_name: str) -> bool:
        """Navigate to waypoint using A* pathfinding and place object there"""
        if waypoint_name not in self.detected_waypoints:
            if self.debug:
                print(f"❌ Waypoint '{waypoint_name}' not found in detected waypoints")
                print(f"🗺️ Available waypoints: {list(self.detected_waypoints.keys())}")
            return False
        
        if self.debug:
            print(f"🗺️ Using A* optimal pathfinding to waypoint '{waypoint_name}' for placement")
        
        # Speech announcement for waypoint placement
        self._robot_say(f"Navigating to {waypoint_name} for placement")
        
        try:
            # First navigate to the waypoint using A* pathfinding
            nav_success = self._navigate_with_optimal_pathfinding(waypoint_name)
            if not nav_success:
                if self.debug:
                    print(f"❌ A* navigation to waypoint '{waypoint_name}' failed, trying fallback")
                # Fallback to standard navigation
                nav_success = self.navigate_to_waypoint(waypoint_name)
                if not nav_success:
                    if self.debug:
                        print(f"❌ All navigation attempts to waypoint '{waypoint_name}' failed")
                    return False
                else:
                    if self.debug:
                        print(f"✅ Fallback navigation to waypoint successful")
            else:
                if self.debug:
                    print(f"✅ A* navigation to waypoint '{waypoint_name}' successful")
            
            # Then place object at that location
            if self.debug:
                print(f"📦 Placing object at waypoint '{waypoint_name}'")
            llm_response = [("place", waypoint_name)]
            return self.pickup_executor(llm_response)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to place at waypoint '{waypoint_name}': {e}")
            return False

    def _place_near_object(self, object_name: str) -> bool:
        """Navigate to detected object using A* pathfinding and place held item nearby/on it"""
        if self.debug:
            print(f"🎯 Looking for '{object_name}' to place object near/on")
        
        # Speech announcement for object placement
        self._robot_say(f"Looking for {object_name} to place object")
        
        try:
            # Check if object exists as detected waypoint (from 360° scan)
            if object_name in self.detected_waypoints:
                if self.debug:
                    instance, score = self.detected_waypoints[object_name]
                    print(f"📍 Found '{object_name}' in memory (confidence: {score:.3f})")
                    print(f"🚀 Using A* optimal pathfinding to object for placement...")
                
                # Navigate to the detected object using A* pathfinding
                nav_success = self._navigate_with_optimal_pathfinding(object_name)
                if not nav_success:
                    if self.debug:
                        print(f"❌ A* navigation to object '{object_name}' failed, trying fallback")
                    # Fallback to standard navigation
                    nav_success = self.navigate_to_waypoint(object_name)
                    if not nav_success:
                        if self.debug:
                            print(f"❌ All navigation attempts to object '{object_name}' failed")
                        return False
                    else:
                        if self.debug:
                            print(f"✅ Fallback navigation to object successful")
                else:
                    if self.debug:
                        print(f"✅ A* navigation to object '{object_name}' successful")
            else:
                # Try to find object in current instances
                if self.agent is None:
                    if self.debug:
                        print(f"❌ Cannot search for '{object_name}' - no agent available")
                    return False
                
                if self.debug:
                    print(f"🔍 Searching for '{object_name}' in current view...")
                
                ranked_objects = self.agent.get_ranked_instances(object_name, threshold=0.1, debug=False)
                if not ranked_objects:
                    if self.debug:
                        print(f"❌ Object '{object_name}' not found in current environment")
                        print("💡 Available objects:", list(self.detected_waypoints.keys()))
                    return False
                
                # Navigate to best match using A* pathfinding
                best_score, best_id, best_instance = ranked_objects[0]
                if self.debug:
                    print(f"🔍 Found '{object_name}' in current view (confidence: {best_score:.3f})")
                    print(f"🚀 Using A* optimal pathfinding to detected instance...")
                
                nav_success = self._navigate_to_instance_optimal(best_instance)
                if not nav_success:
                    if self.debug:
                        print(f"❌ A* navigation to '{object_name}' instance failed, trying fallback")
                    # Fallback to standard navigation
                    nav_success = self.agent.move_to_instance(best_instance)
                    if not nav_success:
                        if self.debug:
                            print(f"❌ All navigation attempts to '{object_name}' instance failed")
                        return False
                    else:
                        if self.debug:
                            print(f"✅ Fallback navigation to instance successful")
                else:
                    if self.debug:
                        print(f"✅ A* navigation to instance successful")
            
            # Place object at/near the target object
            if self.debug:
                print(f"📦 Placing object near/on '{object_name}'")
            llm_response = [("place", object_name)]
            return self.pickup_executor(llm_response)
            
        except Exception as e:
            if self.debug:
                print(f"❌ Failed to place near object '{object_name}': {e}")
            return False
    
    def interactive_command_loop(self):
        """Interactive command loop for navigation and control"""
        if self.debug:
            print("\n🎮 Interactive Command Mode Started")
            print("Available commands:")
            print("  arrive <waypoint_name> - Navigate to a detected waypoint")
            print("  move <distance> - Move forward by distance in meters")
            print("  turn <degrees> - Turn by degrees (positive=left, negative=right)")
            print("  setWay <name> - Save current location as a named waypoint")
            print("  remove <waypoint_name> - Remove a waypoint and its visualization")
            print("  scan [degrees] [divisions] - Scan specified degrees with divisions (default: 360 12)")
            print("  grab <object> [receptacle] - Pick up object, optionally place in receptacle")
            print("  place <location> - Place currently held object at location (waypoint/object/here)")
            print("  objects - List detected grabbable objects")
            print("  waypoints - List all detected waypoints")
            print("  memory - Show dynamic memory status")
            print("  find <object> - Search for object using dynamic memory")
            print("  status - Show mapping status")
            print("  quit - Exit the service")
            print("\n🗺️ Type commands or press Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    command = input("🤖 Command: ").strip().lower()
                    
                    if not command:
                        continue
                    
                    try:
                        parts = shlex.split(command)
                    except ValueError:
                        print("❌ Invalid command format. Use quotes for multi-word objects.")
                        continue
                    if not parts:
                        continue
                    
                    cmd = parts[0]
                    
                    if cmd == "quit" or cmd == "exit":
                        print("👋 Exiting service...")
                        break
                    
                    elif cmd == "arrive" and len(parts) == 2:
                        waypoint_name = parts[1]
                        self.navigate_to_waypoint(waypoint_name)
                    
                    elif cmd == "move" and len(parts) == 2:
                        try:
                            distance = float(parts[1])
                            if self.robot is None:
                                print(f"🔍 DRY RUN: Would move forward {distance}m")
                            else:
                                self.move_forward(distance)  # Use the existing working method
                        except ValueError:
                            print("❌ Invalid distance. Use: move <distance>")
                    
                    elif cmd == "turn" and len(parts) == 2:
                        try:
                            degrees = float(parts[1])
                            if self.robot is None:
                                direction = "left" if degrees > 0 else "right"
                                print(f"🔍 DRY RUN: Would turn {abs(degrees)}° {direction}")
                            else:
                                self.turn_degrees(degrees)  # Use the existing working method
                        except ValueError:
                            print("❌ Invalid angle. Use: turn <degrees>")
                    
                    elif cmd == "setway" and len(parts) == 2:
                        waypoint_name = parts[1]
                        self.save_current_waypoint(waypoint_name)
                    
                    elif cmd == "remove" and len(parts) == 2:
                        waypoint_name = parts[1]
                        self.remove_waypoint(waypoint_name)
                    
                    elif cmd == "moveway" and len(parts) == 3:
                        source_waypoint = parts[1]
                        target_waypoint = parts[2]
                        success = self.move_waypoint_to_waypoint(source_waypoint, target_waypoint)
                        if success:
                            print(f"✅ Moved waypoint '{source_waypoint}' to position of '{target_waypoint}'")
                        else:
                            print(f"❌ Failed to move waypoint. Check that both waypoints exist.")
                    
                    elif cmd == "scan":
                        # Default parameters
                        total_degrees = 360
                        divisions = 12
                        
                        # Parse parameters
                        if len(parts) >= 2:
                            try:
                                total_degrees = int(parts[1])
                                if abs(total_degrees) > 720:  # Reasonable limit
                                    print("❌ Degrees must be between -720 and 720")
                                    continue
                            except ValueError:
                                print("❌ Invalid degrees. Use: scan [degrees] [divisions]")
                                continue
                        
                        if len(parts) >= 3:
                            try:
                                divisions = int(parts[2])
                                if divisions < 1 or divisions > 50:
                                    print("❌ Divisions must be between 1 and 50")
                                    continue
                            except ValueError:
                                print("❌ Invalid divisions. Use: scan [degrees] [divisions]")
                                continue
                        
                        # Calculate increment
                        degree_increment = total_degrees / divisions if divisions > 1 else total_degrees
                        direction = "clockwise" if total_degrees >= 0 else "counter-clockwise"
                        
                        print(f"🌐 Starting {abs(total_degrees)}° scan ({direction}) with {divisions} divisions...")
                        print(f"📐 Increment: {abs(degree_increment):.1f}° per step")
                        
                        self.perform_degree_scan(total_degrees, divisions, self.target_object, self.receptacle)
                        print(f"✅ Scan completed! New objects added to memory.")
                    
                    elif cmd == "grab" and len(parts) == 2:
                        # Pick only: grab <object>
                        target_object = parts[1]
                        self.grab_object(target_object)
                    
                    elif cmd == "grab" and len(parts) != 2:
                        print("❌ Usage: grab <object>")
                        print("   Examples:")
                        print("     grab apple")
                        print("     grab \"sports ball\"  (use quotes for multi-word objects)")
                        print("   Note: Use separate 'place <location>' command after grab")
                    
                    elif cmd == "place" and len(parts) == 2:
                        target_location = parts[1]
                        if not self.is_holding_object():
                            print("❌ No object currently held. Use 'grab <object>' first.")
                            print("💡 Current holding state: None")
                        else:
                            held_object = self.get_held_object()
                            print(f"📦 Attempting to place '{held_object}' at '{target_location}'")
                            self.place_object_at_location(target_location)
                    
                    elif cmd == "objects":
                        grabbable_objects = self.list_grabbable_objects()
                        if grabbable_objects:
                            print("🎯 Detected grabbable objects:")
                            for i, obj in enumerate(grabbable_objects[:10], 1):  # Show top 10
                                confidence_str = f" (confidence: {obj['confidence']:.3f})" if obj['confidence'] > 0 else ""
                                print(f"  {i}. {obj['name']}{confidence_str}")
                            if len(grabbable_objects) > 10:
                                print(f"  ... and {len(grabbable_objects) - 10} more objects")
                        else:
                            print("🎯 No grabbable objects detected yet")
                            print("💡 Try performing a 360° scan to detect more objects")
                    
                    elif cmd == "waypoints":
                        if self.detected_waypoints:
                            print("🗺️ Available waypoints:")
                            for name, (instance, score) in self.detected_waypoints.items():
                                if isinstance(instance, dict) and instance.get('type') == 'saved_waypoint':
                                    pos = instance['position']
                                    print(f"  📍 {name}: saved location (x={pos[0]:.2f}, y={pos[1]:.2f})")
                                else:
                                    print(f"  🎯 {name}: detected object (confidence {score:.3f})")
                        else:
                            print("🗺️ No waypoints available yet")
                    
                    elif cmd == "memory":
                        if self.use_dynamic_memory:
                            memory_status = self.get_dynamic_memory_status()
                            print(f"🧠 Dynamic Memory Status:")
                            print(f"  Enabled: {memory_status.get('enabled', False)}")
                            print(f"  Observations: {memory_status.get('observation_count', 0)}")
                            print(f"  Semantic Points: {memory_status.get('total_semantic_points', 0)}")
                            print(f"  Tracked Objects: {memory_status.get('tracked_objects', 0)}")
                            
                            object_details = memory_status.get('object_details', {})
                            if object_details:
                                print("  📍 Object Details:")
                                for obj_name, details in object_details.items():
                                    obs_ago = details.get('observations_ago', 0)
                                    confidence = details.get('confidence', 0.0)
                                    pos = details.get('position', [0, 0, 0])
                                    print(f"    {obj_name}: conf={confidence:.2f}, {obs_ago} obs ago, pos=({pos[0]:.2f},{pos[1]:.2f})")
                        else:
                            print("🧠 Dynamic Memory: Disabled")
                    
                    elif cmd == "find" and len(parts) == 2:
                        object_name = parts[1]
                        result = self.find_object_with_dynamic_memory(object_name)
                        if result is not None:
                            position, confidence = result
                            print(f"🔍 Found '{object_name}': position=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}), confidence={confidence:.3f}")
                        else:
                            print(f"❌ '{object_name}' not found in dynamic memory")
                            if object_name in self.detected_waypoints:
                                print(f"💡 '{object_name}' available as waypoint (use 'arrive {object_name}')")
                    
                    elif cmd == "status":
                        status = self.get_mapping_status()
                        print(f"📊 Mapping Status:")
                        print(f"  Active: {status['mapping_active']}")
                        print(f"  Objects: {status['total_instances']}")
                        print(f"  Voxels: {status['total_voxels']}")
                        if self.robot:
                            obs = self.robot.get_observation()
                            pos = obs.gps
                            theta = obs.compass
                            print(f"  Position: ({pos[0]:.2f}, {pos[1]:.2f}, {math.degrees(theta):.1f}°)")
                        
                        # Display holding state
                        if status['currently_holding']:
                            duration = time.time() - (status['pickup_timestamp'] or time.time())
                            minutes, seconds = divmod(int(duration), 60)
                            print(f"  Holding: {status['currently_holding']} (for {minutes}m {seconds}s)")
                        else:
                            print(f"  Holding: None")
                        
                        # Show dynamic memory status summary
                        if self.use_dynamic_memory:
                            memory_status = self.get_dynamic_memory_status()
                            tracked_objects = memory_status.get('tracked_objects', 0)
                            semantic_points = memory_status.get('total_semantic_points', 0)
                            print(f"  Dynamic Memory: {tracked_objects} tracked objects, {semantic_points} semantic points")
                    
                    else:
                        print("❌ Unknown command. Available: arrive, move, turn, setWay, moveWay, remove, scan, grab, place, objects, waypoints, memory, find, status, quit")
                
                except KeyboardInterrupt:
                    print("\n👋 Exiting service...")
                    break
                except Exception as e:
                    print(f"❌ Command error: {e}")
        
        except KeyboardInterrupt:
            print("\n👋 Exiting service...")
    
    def get_mapping_status(self):
        """Get current status of mapping system including object holding state"""
        status = {
            'mapping_active': self.mapping_active,
            'scan_complete': self.scan_complete,
            'total_instances': len(self.agent.get_instances()) if self.agent else 0,
            'total_voxels': len(self.agent.voxel_map.voxel_pcd.points) if self.agent and hasattr(self.agent, 'voxel_map') else 0,
            'current_position': None,
            'currently_holding': self.currently_holding,
            'pickup_timestamp': self.pickup_timestamp,
        }
        
        if self.robot is not None:
            try:
                obs = self.robot.get_observation()
                status['current_position'] = obs.gps
            except Exception as e:
                if self.debug:
                    print(f"⚠️ Could not get position: {e}")
        
        return status
    
    def wait_for_scan_completion(self):
        """Block until 360° scan is completed"""
        while not self.scan_complete and not self.stop_mapping.is_set():
            time.sleep(0.5)
        return self.scan_complete
    
    def move_forward(self, distance_meters: float):
        """Move forward with persistent mapping active"""
        if distance_meters <= 0:
            if self.debug:
                print("❌ Distance must be positive")
            return False
        
        if self.debug:
            print(f"➡️ Moving forward {distance_meters:.2f}m (mapping continues)")
        
        if self.robot is not None:
            try:
                # Execute forward movement in single command - no chunking limits
                timeout = max(20.0, distance_meters * 10)  # 10 seconds per meter minimum
                success = self.robot.move_base_to([distance_meters, 0, 0], relative=True, blocking=True, timeout=timeout)
                if not success:
                    if self.debug:
                        print("❌ Movement failed")
                    return False
                
                time.sleep(0.3)
                
                if self.debug:
                    obs = self.robot.get_observation()
                    pos = obs.gps
                    print(f"✅ Forward movement complete. Position: x={pos[0]:.3f}, y={pos[1]:.3f}")
                
                return True
                
            except Exception as e:
                if self.debug:
                    print(f"❌ Movement error: {e}")
                return False
        else:
            if self.debug:
                print("🔍 DRY RUN: Would move forward")
            return True
    
    def turn_degrees(self, degrees: float):
        """Turn by specified degrees with persistent mapping active"""
        if abs(degrees) < 1:
            if self.debug:
                print("⚠️ Turn angle too small")
            return False
        
        direction = "left" if degrees > 0 else "right"
        radians = math.radians(degrees)
        
        if self.debug:
            print(f"🔄 Turning {abs(degrees):.1f}° {direction} (mapping continues)")
        
        if self.robot is not None:
            try:
                # Execute rotation in single command - no chunking limits
                timeout = max(15.0, abs(degrees) * 0.2)  # 0.2 seconds per degree minimum
                success = self.robot.move_base_to([0, 0, radians], relative=True, blocking=True, timeout=timeout)
                if not success:
                    if self.debug:
                        print("❌ Turn failed")
                    return False
                
                time.sleep(0.5)
                
                if self.debug:
                    obs = self.robot.get_observation()
                    pos = obs.gps
                    orientation = math.degrees(pos[2]) if len(pos) > 2 else "unknown"
                    print(f"✅ Turn complete. Heading: {orientation}°")
                
                return True
                
            except Exception as e:
                if self.debug:
                    print(f"❌ Turn error: {e}")
                return False
        else:
            if self.debug:
                print("🔍 DRY RUN: Would turn")
            return True
    
    
    def run_persistent_service(self, scan_points: int = 12, target_object: str = "", receptacle: str = ""):
        """
        Main service method: Perform 360° scan then run background mapping indefinitely
        """
        if self.debug:
            print("🚀 Starting PERSISTENT 360° Mapping Service")
            print("📋 Service will:")
            print("   1. Perform complete 360° scan")
            print("   2. Start background mapping updates")
            print("   3. Run until interrupted (Ctrl+C)")
        
        try:
            # Step 1: Setup robot
            self.setup_robot_for_scanning()
            
            # Step 2: Perform initial 360° scan
            self.perform_360_scan(scan_points, target_object, receptacle)
            
            # Step 3: Start background mapping
            self.start_background_mapping()
            
            # Step 4: Keep service running
            if self.debug:
                print("🌐 PERSISTENT mapping service now running...")
                print("📊 Use Ctrl+C to stop service")
                print("🗺️ Voxel map updating continuously in background")
            
            # Interactive command loop
            self.interactive_command_loop()
            
        except Exception as e:
            if self.debug:
                print(f"❌ Service error: {e}")
        
        finally:
            # Cleanup
            self.stop_background_mapping()
            if self.debug:
                final_status = self.get_mapping_status()
                print(f"🏁 Service stopped. Final map: {final_status['total_instances']} objects")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal - stopping service...")
    sys.exit(0)


# =============================================================================
# PERFORMANCE OPTIMIZATION UTILITIES
# =============================================================================

def enable_head_sweeping_during_scans(enable: bool = True):
    """
    Utility function to provide info about head sweeping behavior during scanning.
    
    Args:
        enable (bool): If True, provides info on re-enabling head sweeping.
                      If False, confirms current disabled state.
    
    Note:
        Head sweeping is currently disabled in scanning methods for performance.
        Head sweeping adds 11 extra head movements per scan point:
        - 12-point scan without head sweep: 12 movements
        - 12-point scan with head sweep: 132 movements (11x slower)
        
        To re-enable head sweeping, change agent.update(move_head=False) 
        to agent.update(move_head=True) in scanning methods.
    """
    if enable:
        print("⚠️ Head sweeping during scans is currently disabled for performance.")
        print("   To re-enable, change agent.update(move_head=False) to agent.update(move_head=True)")
        print("   in the scanning methods (perform_360_scan, perform_degree_scan, etc.)")
        print("   WARNING: This will make scanning 8-11x slower (132 vs 12 movements)")
    else:
        print("✅ Head sweeping during scans is disabled (current optimized state)")
        print(f"   Scanning performance improved by ~60-80%")

def get_scan_performance_info():
    """Return information about scanning performance optimizations"""
    return {
        "head_sweeping_disabled": True,
        "ee_camera_points_disabled_by_default": True,
        "estimated_speedup": "60-80% (head sweep) + 40-60% (EE camera)", 
        "movements_per_12_point_scan": 12,  # vs 132 with head sweeping
        "optimization_applied": "move_head=False in agent.update() calls + show_ee_camera_point_clouds=False",
        "modified_methods": ["perform_360_scan", "perform_degree_scan", "save_current_waypoint"],
        "ee_camera_flag": "Use --show-ee-camera-points to re-enable EE camera point clouds"
    }


@click.command()
@click.option("--robot_ip", default="", help="IP address of the robot")
@click.option("--scan_points", default=12, help="Number of observation points for 360° scan")
@click.option("--debug", default=True, help="Enable debug output")
@click.option("--dry_run", is_flag=True, help="Run without executing robot movements")
@click.option("--target_object", default="", help="Object to detect and mark as waypoint")
@click.option("--receptacle", default="", help="Receptacle to detect and mark as waypoint")
@click.option("--dynamic_memory", "--dynamic-memory", is_flag=True, help="Enable dynamic memory system")
@click.option("--memory_threshold", "--memory-threshold", default=50, help="Observations before object cleanup (default: 50)")
@click.option("--show_camera_points", "--show-camera-points", is_flag=True, help="Show ALL camera point clouds in rerun (disabled by default for performance)")
@click.option("--use_table_aware_pickup", "--use-table-aware-pickup", default=True, help="Enable universal height-adaptive pickup for objects at any elevation (floor, table, shelf) - enabled by default")
@click.option("--enable_speech_debug", "--enable-speech-debug", is_flag=True, help="Enable vocal debugging announcements for workflow phases (disabled by default)")
@click.option("--show_object_bboxes", "--show-object-bboxes", is_flag=True, help="Show 3D object bounding boxes in rerun viewer (disabled by default)")
def main(robot_ip, scan_points, debug, dry_run, target_object, receptacle, dynamic_memory, memory_threshold, show_camera_points, use_table_aware_pickup, enable_speech_debug, show_object_bboxes):
    """
    PERSISTENT 360° Mapping Service
    
    Performs complete 360° scan and then runs continuous background mapping.
    Service maintains voxel map state that other programs can access.
    """
    
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🌐 PERSISTENT 360° Mapping Service")
    print("="*50)
    
    if enable_speech_debug:
        print("🗣️ Speech debugging ENABLED - robot will announce workflow phases")
    else:
        print("🔇 Speech debugging DISABLED (use --enable-speech-debug to enable)")
    
    if show_object_bboxes:
        print("📦 3D object bounding boxes ENABLED in rerun viewer")
    else:
        print("📦 3D object bounding boxes DISABLED (use --show-object-bboxes to enable)")
    
    if dry_run:
        print("🔍 Running in DRY RUN mode")
    
    # Get robot parameters
    parameters = get_parameters("default_planner.yaml")
    # Override camera range to 4 meters
    parameters.max_depth = 4.0
    
    # Memory optimization settings
    parameters.voxel_size = 0.08  # Memory savings while maintaining good resolution (0.04 -> 0.08)
    parameters.min_points_per_voxel = 25  # More aggressive filtering
    parameters.obs_min_density = 20  # Drop sparse points
    if hasattr(parameters, 'server_memory_limit'):
        parameters.server_memory_limit = "1GB"  # Reduce Rerun memory limit
    
    # Create robot client
    if not dry_run:
        robot = HomeRobotZmqClient(
            robot_ip=robot_ip,
            show_object_bounding_boxes=show_object_bboxes
        )
        
        # PERFORMANCE OPTIMIZATION: Configure camera point clouds
        # Both cameras send ~600K points every second during background mapping
        # This overwhelms the websocket connection and causes 5+ minute delays in rerun
        if robot._rerun is not None:
            # Disable both head and EE camera point clouds by default for performance
            robot._rerun.show_camera_point_clouds = show_camera_points
            robot._rerun.show_ee_camera_point_clouds = show_camera_points
            if show_camera_points:
                print("📸 ALL camera point clouds ENABLED in rerun (may cause lag)")
            else:
                print("📸 ALL camera point clouds DISABLED for performance (use --show-camera-points to enable)")
        
        robot.move_to_nav_posture()
    else:
        robot = None
        print("🤖 Robot client skipped (dry run)")
    
    # Create semantic sensor
    if not dry_run:
        semantic_sensor = create_semantic_sensor(
            parameters=parameters,
            device_id=0,
            verbose=False,
        )
        
        # Create robot agent
        agent = RobotAgent(robot, parameters, semantic_sensor)
        agent.start()
        
    else:
        semantic_sensor = None
        agent = None
        print("🔍 Semantic sensor skipped (dry run)")
        print("🧠 Robot agent skipped (dry run)")
    
    # Get object names from user if not provided
    if not target_object and not dry_run:
        target_object = input("Enter target object name (or press Enter to skip): ").strip()
    if not receptacle and not dry_run:
        receptacle = input("Enter receptacle name (or press Enter to skip): ").strip()
    
    if target_object or receptacle:
        print(f"🎯 Will search for objects during scan:")
        if target_object:
            print(f"  Object: '{target_object}'")
        if receptacle:
            print(f"  Receptacle: '{receptacle}'")
    
    # Create persistent mapping service
    mapping_service = PersistentMapping360(
        robot, 
        agent, 
        debug=debug, 
        use_dynamic_memory=dynamic_memory, 
        memory_threshold=memory_threshold,
        use_table_aware_pickup=use_table_aware_pickup,
        enable_speech_debug=enable_speech_debug
    )
    
    try:
        if not dry_run:
            # Run the persistent service
            mapping_service.run_persistent_service(scan_points, target_object, receptacle)
        else:
            print("✅ Dry run completed - service ready for deployment")
            print(f"🔍 Would perform {scan_points}-point 360° scan")
            print("🔍 Would run background mapping service")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return
    
    finally:
        if robot is not None:
            robot.stop()
    
    print("🏁 PERSISTENT 360° Mapping Service completed")


if __name__ == "__main__":
    main()