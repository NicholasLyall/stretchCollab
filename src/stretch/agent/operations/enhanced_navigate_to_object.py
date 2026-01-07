# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Enhanced navigation to object with visual servoing integration
# This replaces the slow navigation+pregrasp sequence with visual servoing approach

import math
import time
import numpy as np

from stretch.agent.operations.navigate import NavigateToObjectOperation
from stretch.motion.kinematics import HelloStretchIdx
import stretch.motion.constants as constants
from stretch.utils.geometry import point_global_to_base

try:
    from stretch.agent.operations.visual_servo_grasp import VisualServoGraspOperation, YOLOServoPerception, RegulatePollTimeout
    VISUAL_SERVO_AVAILABLE = True
except ImportError:
    VISUAL_SERVO_AVAILABLE = False


class EnhancedNavigateToObjectOperation(NavigateToObjectOperation):
    """
    Enhanced navigation that integrates visual servoing for the final approach.
    
    This operation:
    1. Uses standard navigation to get close to the object 
    2. Switches to visual servoing for precise final approach
    3. Eliminates the jerky transition between navigation and manipulation modes
    4. Provides much better accuracy in the final positioning
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TEMPORARILY DISABLED: Visual servoing is causing objects to be lost
        self.use_visual_servoing_approach = False  # Was: VISUAL_SERVO_AVAILABLE
        self.final_approach_distance = 1.0  # Start visual servoing when within 1m
        self.visual_servo_precision = 0.15  # 15cm precision for approach (vs 3cm for grasp)
        self.invert_signs = False  # Can be set to True if coordinate system still inverted
        
    def run(self):
        """Enhanced run with visual servoing final approach"""
        
        if not self.use_visual_servoing_approach:
            # Fallback to standard navigation
            super().run()
            return
            
        self.intro("executing enhanced motion plan with visual servoing approach.")
        self.robot.move_to_nav_posture()

        # Check if already within reach
        if self.agent.within_reach_of(self.get_target()):
            self.warn("Already within reach of object!")
            self._orient_toward_object()
            return

        # Execute standard trajectory to get close
        assert self.plan is not None, "Plan should exist from can_start()"
        
        # Get the final goal position
        final_xyt = self.plan.trajectory[-1].state
        object_xyz = self.get_target().get_center()
        
        # Calculate distance to object
        object_distance = np.linalg.norm(object_xyz[:2] - self.robot.get_base_pose()[:2])
        
        if object_distance > self.final_approach_distance:
            # Execute most of the trajectory using standard navigation
            self.info(f"Using standard navigation to get within {self.final_approach_distance}m of object")
            
            # Execute trajectory but stop before the final approach
            truncated_plan = self._truncate_plan_for_visual_servoing()
            if truncated_plan is not None:
                self.robot.execute_trajectory(truncated_plan, final_timeout=30.0)
            else:
                # If can't truncate, execute full plan but be ready for visual servoing
                self.robot.execute_trajectory(self.plan, final_timeout=30.0)
        
        # Now use visual servoing for final approach and positioning
        self.info("Switching to visual servoing for precise final approach")
        success = self._visual_servo_final_approach()
        
        if not success:
            self.warn("Visual servoing approach failed, using standard positioning")
            # Fallback to standard final positioning
            self.robot.move_base_to(final_xyt, blocking=True, timeout=30.0)
            self._orient_toward_object()
    
    def _truncate_plan_for_visual_servoing(self):
        """Truncate the navigation plan to stop before final approach distance"""
        try:
            object_xyz = self.get_target().get_center()
            
            # Find the waypoint that's closest to final_approach_distance from object
            best_waypoint_idx = 0
            best_distance_diff = float('inf')
            
            for i, waypoint in enumerate(self.plan.trajectory):
                waypoint_pos = waypoint.state[:2]
                distance_to_object = np.linalg.norm(object_xyz[:2] - waypoint_pos)
                distance_diff = abs(distance_to_object - self.final_approach_distance)
                
                if distance_diff < best_distance_diff:
                    best_distance_diff = distance_diff
                    best_waypoint_idx = i
            
            # Only truncate if we found a reasonable stopping point
            if best_waypoint_idx < len(self.plan.trajectory) - 2:
                # Create truncated plan
                from stretch.motion import PlanResult
                truncated_plan = PlanResult(
                    success=True,
                    trajectory=self.plan.trajectory[:best_waypoint_idx + 1]
                )
                return truncated_plan
            
        except Exception as e:
            print(f"[ENHANCED NAV] Failed to truncate plan: {e}")
        
        return None
    
    def _visual_servo_final_approach(self):
        """Use visual servoing for final approach to object"""
        try:
            # Initialize visual servoing components
            perception = YOLOServoPerception(self.agent)
            perception.set_target(self.agent.target_object)
            loop_regulator = RegulatePollTimeout(10)  # 10Hz for approach (vs 15Hz for grasp)
            
            # Switch to manipulation mode for visual servoing
            self.robot.switch_to_manipulation_mode()
            time.sleep(0.2)  # Allow mode switch to complete
            
            # Position arm for visual servoing
            self._prepare_arm_for_visual_servoing()
            
            max_approach_time = 30.0  # 30 seconds max for final approach
            start_time = time.time()
            iteration_count = 0
            
            while time.time() - start_time < max_approach_time:
                loop_regulator.start_loop()
                iteration_count += 1
                
                # Get current observation
                servo_obs = self.robot.get_servo_observation()
                if servo_obs.ee_rgb is None:
                    continue
                
                # Detect target object
                detection = perception.detect_object(servo_obs.ee_rgb, servo_obs.ee_depth)
                
                if detection is None or detection['confidence'] < 0.3:
                    # Lost object or low confidence
                    if iteration_count > 20:  # Give it a few tries
                        print("[ENHANCED NAV] Lost object during visual servoing approach")
                        break
                    continue
                
                # Get object center and distance
                object_center = detection['center']
                object_depth = detection['depth']
                
                # Check if we're close enough to proceed to grasping
                if object_depth < self.visual_servo_precision:
                    print(f"[ENHANCED NAV] Reached target distance: {object_depth:.3f}m < {self.visual_servo_precision}m")
                    break
                
                # Compute approach motion with corrected coordinate system transformation
                image_center_x = servo_obs.ee_rgb.shape[1] // 2
                image_center_y = servo_obs.ee_rgb.shape[0] // 2
                
                # Standard camera coordinate mapping (matches original visual_servoing_demo.py)
                # dx = horizontal error (for base rotation), dy = vertical error (for wrist pitch)
                dx = object_center[0] - image_center_x  # Horizontal pixel error
                dy = object_center[1] - image_center_y  # Vertical pixel error
                
                # Simple proportional control for approach
                base_motion_needed = abs(dx) > 30 or abs(dy) > 30  # pixels
                approach_motion_needed = object_depth > self.visual_servo_precision + 0.05
                
                if base_motion_needed or approach_motion_needed:
                    success = self._execute_approach_motion(dx, dy, object_depth)
                    if not success:
                        print("[ENHANCED NAV] Motion execution failed")
                        break
                
                # Progress feedback
                if iteration_count % 10 == 0:
                    print(f"[ENHANCED NAV] Approaching: depth={object_depth:.3f}m, error=({dx:.0f},{dy:.0f})px")
                
                # Regulate loop timing
                loop_regulator.wait_to_finish_loop()
            
            print(f"[ENHANCED NAV] Visual servoing approach completed in {time.time() - start_time:.1f}s")
            return True
            
        except Exception as e:
            print(f"[ENHANCED NAV] Visual servoing approach failed: {e}")
            return False
    
    def _prepare_arm_for_visual_servoing(self):
        """Position arm optimally for visual servoing approach"""
        try:
            # Get current joint state
            joint_state = self.robot.get_joint_positions()
            
            # Position arm for good camera view of object
            object_xyz = self.get_target().get_center()
            xyt = self.robot.get_base_pose()
            relative_object_xyz = point_global_to_base(object_xyz, xyt)
            
            # Compute optimal wrist pitch for viewing object
            ee_pos, ee_rot = self.robot.get_robot_model().manip_fk(joint_state)
            dy = abs(ee_pos[1] - relative_object_xyz[1])
            dz = abs(ee_pos[2] - relative_object_xyz[2])
            optimal_pitch = -np.pi/2 + np.arctan2(dy, dz)
            
            # Set arm configuration for approach
            joint_state[HelloStretchIdx.WRIST_PITCH] = np.clip(optimal_pitch, -1.57, 0.5)
            joint_state[HelloStretchIdx.ARM] = np.clip(joint_state[HelloStretchIdx.ARM], 0.1, 0.3)  # Moderate extension
            joint_state[HelloStretchIdx.LIFT] = np.clip(joint_state[HelloStretchIdx.LIFT], 0.3, 0.8)  # Good height
            
            # Move to position
            self.robot.arm_to(joint_state, head=constants.look_at_ee, blocking=True)
            time.sleep(0.5)  # Allow settling
            
        except Exception as e:
            print(f"[ENHANCED NAV] Arm positioning failed: {e}")
    
    def _execute_approach_motion(self, dx, dy, depth):
        """Execute motion commands for visual servoing approach with correct coordinate mapping"""
        try:
            # Conservative approach gains (slower than grasping)
            base_step = 0.05  # Match original base_x_step
            approach_gain = 0.3
            
            # Get current base pose
            current_xyt = self.robot.get_base_pose()
            
            # Compute motion increments with corrected sign convention
            # Based on analysis: original visual servoing uses negative sign for base rotation
            base_rotation_increment = 0
            if abs(dx) > 30:  # Significant horizontal error (align_x_threshold)
                # Test corrected sign convention (may need to be inverted if still wrong)
                if dx > 30:
                    # Object is to the right of center, rotate base counterclockwise (positive)
                    base_rotation_increment = base_step * 0.5  
                elif dx < -30:
                    # Object is to the left of center, rotate base clockwise (negative)
                    base_rotation_increment = -base_step * 0.5
                
                # Option to invert signs if coordinate system is still wrong
                if self.invert_signs:
                    base_rotation_increment = -base_rotation_increment
                    print(f"[ENHANCED NAV] Using inverted sign convention")
                    
                print(f"[ENHANCED NAV] Visual error dx={dx:.0f}, commanding base rotation: {base_rotation_increment:.3f}")
                    
            approach_increment = 0
            if depth > self.visual_servo_precision + 0.05:  # Too far
                approach_increment = min(0.1, approach_gain * (depth - self.visual_servo_precision))
            
            # Clamp movements for safety (smaller limits for approach vs grasping)
            base_rotation_increment = np.clip(base_rotation_increment, -0.05, 0.05)
            approach_increment = np.clip(approach_increment, 0, 0.1)
            
            # Execute motion if needed
            motion_executed = False
            
            if abs(base_rotation_increment) > 0.01:
                new_xyt = current_xyt.copy()
                new_xyt[2] += base_rotation_increment
                self.robot.move_base_to(new_xyt, blocking=True, timeout=10.0)
                motion_executed = True
                
            if approach_increment > 0.01:
                new_xyt = current_xyt.copy()
                # Move forward in the direction robot is facing
                new_xyt[0] += approach_increment * np.cos(current_xyt[2])
                new_xyt[1] += approach_increment * np.sin(current_xyt[2])
                self.robot.move_base_to(new_xyt, blocking=True, timeout=10.0)
                motion_executed = True
                
            return True
            
        except Exception as e:
            print(f"[ENHANCED NAV] Motion execution failed: {e}")
            return False
    
    def _orient_toward_object(self):
        """Orient the robot toward the object (standard behavior)"""
        try:
            xyz = self.get_target().get_center()
            start_xyz = self.robot.get_base_pose()[:2]
            theta = math.atan2(
                xyz[1] - self.robot.get_base_pose()[1], 
                xyz[0] - self.robot.get_base_pose()[0]
            )
            self.robot.move_base_to(
                np.array([start_xyz[0], start_xyz[1], theta + np.pi / 2]),
                blocking=True,
                timeout=30.0,
            )
        except Exception as e:
            print(f"[ENHANCED NAV] Orientation failed: {e}")
    
    def was_successful(self):
        """Check if the enhanced navigation was successful"""
        if self.use_visual_servoing_approach:
            # Check if we're reasonably close to the target with good orientation
            try:
                object_xyz = self.get_target().get_center()
                robot_xyt = self.robot.get_base_pose()
                
                # Distance check
                distance = np.linalg.norm(object_xyz[:2] - robot_xyt[:2])
                distance_ok = distance < self.visual_servo_precision + 0.1
                
                # Mode check
                mode_ok = self.robot.in_manipulation_mode()
                
                return distance_ok and mode_ok
                
            except Exception:
                return False
        else:
            return super().was_successful()