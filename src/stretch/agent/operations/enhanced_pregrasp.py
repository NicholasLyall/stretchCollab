# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Enhanced pregrasp operation that works smoothly with visual servoing

import numpy as np
import time

from stretch.agent.operations.pregrasp import PreGraspObjectOperation
from stretch.motion import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base
import stretch.motion.constants as constants


class EnhancedPreGraspObjectOperation(PreGraspObjectOperation):
    """
    Enhanced pregrasp operation that:
    1. Works smoothly with the enhanced navigation
    2. Doesn't cause sudden jerky motions
    3. Prepares optimally for visual servoing grasp
    4. Reduces transition issues between operations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # More liberal distance threshold since we use visual servoing
        self.grasp_distance_threshold = 1.2  # Was 0.8, now more permissive
        
    def can_start(self):
        """Enhanced can_start with better distance checking"""
        self.plan = None
        if self.agent.current_object is None:
            return False
        elif self.agent.is_instance_unreachable(self.agent.current_object):
            return False

        start = self.robot.get_base_pose()
        if not self.navigation_space.is_valid(start):
            self.error(
                f"{self.name}: Robot is in an invalid configuration. It is probably too close to geometry, or localization has failed."
            )
            return False

        # Enhanced distance checking
        object_xyz = self.agent.current_object.point_cloud.mean(axis=0)
        dist = np.linalg.norm(object_xyz[:2] - start[:2])
        
        if dist > self.grasp_distance_threshold:
            self.error(f"Object is too far away for pregrasp: {dist:.3f}m > {self.grasp_distance_threshold}m")
            return False
            
        self.cheer(f"{self.name}: Object is at good distance for enhanced pregrasp: {dist:.3f}m")
        return True

    def run(self):
        """Enhanced run method with smoother transitions"""
        
        self.intro("Enhanced pregrasp positioning with smooth transitions.")
        
        # Check if already in manipulation mode from enhanced navigation
        if not self.robot.in_manipulation_mode():
            self.info("Switching to manipulation mode")
            self.robot.switch_to_manipulation_mode()
            time.sleep(0.3)  # Allow mode switch to complete smoothly
        else:
            self.info("Already in manipulation mode - no jarring mode switch needed")

        # Get current state
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        # Get object position
        object_xyz = self.agent.current_object.point_cloud.mean(axis=0)
        xyt = self.robot.get_base_pose()
        relative_object_xyz = point_global_to_base(object_xyz, xyt)

        # Enhanced arm positioning that works well with visual servoing
        optimal_joint_state = self._compute_optimal_pregrasp_pose(joint_state, relative_object_xyz)
        
        # Smooth transition to pregrasp pose
        success = self._smooth_transition_to_pose(joint_state, optimal_joint_state)
        
        if not success:
            self.warn("Smooth transition failed, using direct positioning")
            self.robot.arm_to(optimal_joint_state, blocking=True)

    def _compute_optimal_pregrasp_pose(self, current_joint_state, relative_object_xyz):
        """Compute optimal arm pose for pregrasp that works well with visual servoing"""
        
        optimal_state = current_joint_state.copy()
        
        # Get current end effector position
        ee_pos, ee_rot = self.robot.get_robot_model().manip_fk(current_joint_state)
        
        # Compute optimal wrist pitch for object viewing
        if self.use_pitch_from_vertical:
            dy = abs(ee_pos[1] - relative_object_xyz[1])
            dz = abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
            
            # Enhanced pitch calculation that avoids extreme angles
            target_pitch = -np.pi / 2 + pitch_from_vertical
            target_pitch = np.clip(target_pitch, -1.4, 0.3)  # Avoid extreme positions
        else:
            target_pitch = -np.pi / 2 - 0.1  # Default safe angle
            
        # Set optimal joint positions
        optimal_state[HelloStretchIdx.WRIST_PITCH] = target_pitch
        
        # Optimize arm extension for object distance
        object_distance = np.linalg.norm(relative_object_xyz)
        if object_distance < 0.3:
            # Very close - retract arm slightly
            optimal_state[HelloStretchIdx.ARM] = max(0.05, current_joint_state[HelloStretchIdx.ARM] - 0.1)
        elif object_distance > 0.6:
            # Far - extend arm moderately
            optimal_state[HelloStretchIdx.ARM] = min(0.4, current_joint_state[HelloStretchIdx.ARM] + 0.1)
        # Otherwise keep current arm position
        
        # Optimize lift for object height
        if relative_object_xyz[2] < 0.2:
            # Low object - lower the arm
            optimal_state[HelloStretchIdx.LIFT] = max(0.2, current_joint_state[HelloStretchIdx.LIFT] - 0.1)
        elif relative_object_xyz[2] > 0.8:
            # High object - raise the arm  
            optimal_state[HelloStretchIdx.LIFT] = min(0.9, current_joint_state[HelloStretchIdx.LIFT] + 0.1)
            
        # Ensure safe joint limits
        optimal_state[HelloStretchIdx.ARM] = np.clip(optimal_state[HelloStretchIdx.ARM], 0.02, 0.52)
        optimal_state[HelloStretchIdx.LIFT] = np.clip(optimal_state[HelloStretchIdx.LIFT], 0.15, 1.1)
        optimal_state[HelloStretchIdx.WRIST_YAW] = 0.0  # Keep yaw neutral
        optimal_state[HelloStretchIdx.WRIST_ROLL] = 0.0  # Keep roll neutral
        
        return optimal_state
        
    def _smooth_transition_to_pose(self, start_state, target_state):
        """Smoothly transition from current pose to target pose to avoid jerking"""
        
        try:
            # Calculate the difference between current and target
            state_diff = target_state - start_state
            max_diff = np.max(np.abs(state_diff))
            
            if max_diff < 0.1:
                # Small change - move directly
                self.robot.arm_to(target_state, head=constants.look_at_ee, blocking=True)
                return True
                
            # Large change - use intermediate steps
            num_steps = max(2, int(max_diff / 0.15))  # Steps to keep changes small
            
            for step in range(num_steps + 1):
                # Linear interpolation between start and target
                alpha = step / num_steps
                intermediate_state = start_state + alpha * state_diff
                
                # Ensure joint limits
                intermediate_state[HelloStretchIdx.ARM] = np.clip(intermediate_state[HelloStretchIdx.ARM], 0.02, 0.52)
                intermediate_state[HelloStretchIdx.LIFT] = np.clip(intermediate_state[HelloStretchIdx.LIFT], 0.15, 1.1)
                intermediate_state[HelloStretchIdx.WRIST_PITCH] = np.clip(intermediate_state[HelloStretchIdx.WRIST_PITCH], -1.57, 0.5)
                
                # Move to intermediate position
                self.robot.arm_to(intermediate_state, head=constants.look_at_ee, blocking=True)
                
                # Small pause between steps for smooth motion
                if step < num_steps:
                    time.sleep(0.1)
                    
            return True
            
        except Exception as e:
            print(f"[ENHANCED PREGRASP] Smooth transition failed: {e}")
            return False
            
    def was_successful(self):
        """Enhanced success check"""
        # Check that we're in manipulation mode and arm is positioned reasonably
        if not self.robot.in_manipulation_mode():
            return False
            
        try:
            # Check that arm is in a reasonable position for grasping
            joint_state = self.robot.get_joint_positions()
            
            # Basic sanity checks
            arm_ok = 0.02 <= joint_state[HelloStretchIdx.ARM] <= 0.52
            lift_ok = 0.15 <= joint_state[HelloStretchIdx.LIFT] <= 1.1
            pitch_ok = -1.57 <= joint_state[HelloStretchIdx.WRIST_PITCH] <= 0.5
            
            return arm_ok and lift_ok and pitch_ok
            
        except Exception:
            return False