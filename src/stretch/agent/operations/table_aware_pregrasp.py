# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import numpy as np

from stretch.agent.operations.pregrasp import PreGraspObjectOperation
from stretch.motion import HelloStretchIdx
from stretch.utils.geometry import point_global_to_base
from stretch.utils.logger import Logger

logger = Logger(__name__)


class TableAwarePreGraspObjectOperation(PreGraspObjectOperation):
    """
    Height-adaptive pregrasp positioning that safely approaches objects at any elevation.
    
    Key Features:
    - Always starts from high arm position to avoid surface collisions
    - Detects surface height underneath object and maintains safety margin
    - Calculates optimal wrist pitch for any object height
    - Ensures gripper never goes below detected surface + safety margin
    - Works universally for floor, table, shelf, and counter objects
    """

    def __init__(
        self,
        *args,
        safety_margin: float = 0.05,
        executor=None,
        high_start_clearance: float = 0.3,  # Start 30cm above object for safety
        **kwargs
    ):
        """Initialize height-adaptive pregrasp operation.
        
        Args:
            safety_margin: Height margin to maintain above detected surfaces
            executor: Reference to TableAwarePickupExecutor for height constraint access
            high_start_clearance: How far above object to start arm positioning
        """
        super().__init__(*args, **kwargs)
        
        self.safety_margin = safety_margin
        self.executor = executor
        self.high_start_clearance = high_start_clearance
        
        # Override parent's distance threshold to get closer before visual servoing
        self.grasp_distance_threshold = 0.6  # Get closer (60cm) before starting visual servoing
        
        logger.info(f"[TABLE-AWARE PREGRASP] Height-adaptive pregrasp initialized")
        logger.info(f"[TABLE-AWARE PREGRASP] Safety margin: {safety_margin}m, Start clearance: {high_start_clearance}m")

    def can_start(self):
        """Enhanced can_start check with height constraint validation."""
        
        # Speech announcement for pre-grasp phase
        if hasattr(self.agent, 'enable_speech_debug') and self.agent.enable_speech_debug:
            self.agent.robot_say("Positioning arm for grasp")
        
        # Run parent validation first
        if not super().can_start():
            return False
            
        # Additional height constraint validation
        if self.agent.current_object is None:
            logger.warning("[TABLE-AWARE PREGRASP] No current object set")
            return False
            
        try:
            # Get height constraints for the object
            if self.executor is not None:
                height_constraints = self.executor.get_height_constraints_for_object(self.agent.current_object)
                
                # Check if we can safely approach this object
                min_gripper_height = height_constraints['min_gripper_height']
                object_height = height_constraints['object_height']
                recommended_start_height = height_constraints['recommended_start_height']
                
                logger.info(f"[HEIGHT CONSTRAINTS] Surface: {height_constraints['surface_height']:.3f}m")
                logger.info(f"[HEIGHT CONSTRAINTS] Min gripper: {min_gripper_height:.3f}m") 
                logger.info(f"[HEIGHT CONSTRAINTS] Object: {object_height:.3f}m")
                logger.info(f"[HEIGHT CONSTRAINTS] Start height: {recommended_start_height:.3f}m")
                
                # Validate constraints make sense
                if min_gripper_height > object_height + 0.05:  # Need 5cm clearance
                    logger.error(f"[HEIGHT CONSTRAINTS] Invalid: min gripper {min_gripper_height:.3f}m > object {object_height:.3f}m + clearance")
                    return False
                    
                # Check robot arm reach limits
                if recommended_start_height > 1.8:  # Stretch arm limit approximation
                    logger.warning(f"[HEIGHT CONSTRAINTS] Start height {recommended_start_height:.3f}m may exceed arm reach")
                    
            self.cheer(f"[TABLE-AWARE PREGRASP] Height constraints validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"[HEIGHT CONSTRAINTS] Error validating height constraints: {e}")
            return False

    def calculate_safe_arm_position(self, object_xyz, base_pose, height_constraints):
        """
        Calculate safe arm positioning that respects height constraints.
        
        Args:
            object_xyz: Object position in global coordinates
            base_pose: Robot base pose
            height_constraints: Height constraint dictionary from executor
            
        Returns:
            dict: Safe joint positions and approach parameters
        """
        try:
            # Convert object position to robot base frame
            relative_object_xyz = point_global_to_base(object_xyz, base_pose)
            
            # Get current joint state
            obs = self.robot.get_observation()
            joint_state = obs.joint.copy()
            
            # Calculate safe starting position (high above object)
            target_object_height = height_constraints['object_height']
            recommended_start_height = height_constraints['recommended_start_height']
            min_gripper_height = height_constraints['min_gripper_height']
            
            # Start with arm in high position - ALWAYS RESET HIGH FIRST
            # Force high position regardless of current lift (like place operation)
            print("🚀🚀🚀 TABLE-AWARE PREGRASP: Using 'reset HIGH first' approach (like place operation) 🚀🚀🚀")
            current_lift = joint_state[HelloStretchIdx.LIFT]
            safe_start_lift = max(
                target_object_height + self.high_start_clearance,  # High above object (0.3m default)
                min_gripper_height + 0.2  # Well above surface constraint
            )
            
            # Clamp to robot limits (Stretch can lift to ~1.1m)
            safe_start_lift = min(safe_start_lift, 1.05)  # Leave margin from maximum
            safe_start_lift = max(safe_start_lift, 0.1)   # Above minimum
            
            joint_state[HelloStretchIdx.LIFT] = safe_start_lift
            
            # Calculate wrist pitch for looking at object from this height
            # More conservative pitch calculation for height-adaptive approach
            dx = relative_object_xyz[0]  # Forward distance to object
            dy = relative_object_xyz[1]  # Left-right offset
            dz_approach = safe_start_lift - target_object_height  # Height difference
            
            # Calculate pitch to look down at object
            if self.use_pitch_from_vertical and abs(dz_approach) > 0.01:
                # Look down at the object from our starting height
                pitch_angle = np.arctan2(abs(dz_approach), abs(dx)) if abs(dx) > 0.01 else np.pi/4
                # Limit pitch to reasonable range
                pitch_angle = np.clip(pitch_angle, 0.0, np.pi/2 - 0.1)  # Not quite straight down
                joint_state[HelloStretchIdx.WRIST_PITCH] = -pitch_angle  # Negative for looking down
            else:
                # Default pitch for horizontal viewing
                joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi / 2
                
            # Ensure arm is extended appropriately for object distance
            arm_extension = joint_state[HelloStretchIdx.ARM]
            object_distance = np.linalg.norm(relative_object_xyz[:2])  # Distance in X-Y plane
            
            # Adjust arm extension based on object distance and height
            if object_distance > 0.6:  # Object is far
                target_arm_extension = min(0.4, arm_extension + 0.1)  # Extend more
            elif object_distance < 0.3:  # Object is close
                target_arm_extension = max(0.0, arm_extension - 0.1)  # Retract
            else:
                target_arm_extension = arm_extension  # Keep current
                
            joint_state[HelloStretchIdx.ARM] = target_arm_extension
            
            position_info = {
                'joint_state': joint_state,
                'safe_lift_height': safe_start_lift,
                'min_allowed_height': min_gripper_height,
                'object_height': target_object_height,
                'approach_pitch': joint_state[HelloStretchIdx.WRIST_PITCH],
                'arm_extension': target_arm_extension
            }
            
            logger.info(f"[SAFE POSITIONING] Lift: {safe_start_lift:.3f}m (min: {min_gripper_height:.3f}m)")
            logger.info(f"[SAFE POSITIONING] Pitch: {np.degrees(position_info['approach_pitch']):.1f}°")
            logger.info(f"[SAFE POSITIONING] Arm extension: {target_arm_extension:.3f}m")
            
            return position_info
            
        except Exception as e:
            logger.error(f"[SAFE POSITIONING] Error calculating safe arm position: {e}")
            raise

    def run(self):
        """Run height-adaptive pregrasp positioning."""
        
        logger.info("[TABLE-AWARE PREGRASP] Starting height-adaptive pregrasp positioning")
        
        # Switch to manipulation mode
        self.robot.move_to_manip_posture()
        
        # Get object and robot state
        object_xyz = self.agent.current_object.point_cloud.mean(axis=0)
        base_pose = self.robot.get_base_pose()
        
        # Get height constraints if executor available
        if self.executor is not None:
            height_constraints = self.executor.get_height_constraints_for_object(self.agent.current_object)
            
            # Calculate safe arm positioning
            position_info = self.calculate_safe_arm_position(object_xyz, base_pose, height_constraints)
            
            # Execute safe positioning with two-step collision avoidance (like place operation)
            logger.info("[TABLE-AWARE PREGRASP] Moving to safe high starting position")
            self.robot.switch_to_manipulation_mode()
            
            # Get current joint state
            current_joint_state = self.robot.get_joint_positions()
            target_joint_state = position_info['joint_state']
            target_lift = target_joint_state[HelloStretchIdx.LIFT]
            current_lift = current_joint_state[HelloStretchIdx.LIFT]
            
            # Two-step movement to avoid collision: UP first, THEN extend
            if target_lift > current_lift:
                logger.info("🚫🚫🚫 HIGH RESET DETECTED - USING TWO-STEP MOVEMENT TO AVOID TABLE COLLISION! 🚫🚫🚫")
                
                # Step 1: Go UP first (keep current arm extension to avoid collision)
                safe_up_posture = current_joint_state.copy()
                safe_up_posture[HelloStretchIdx.LIFT] = target_lift
                logger.info(f"[STEP 1] Lifting to {target_lift:.3f}m height first")
                self.robot.arm_to(safe_up_posture, blocking=True)
                
                # Step 2: THEN extend arm and adjust pitch (now safe above table)
                elevated_posture = safe_up_posture.copy()
                elevated_posture[HelloStretchIdx.ARM] = target_joint_state[HelloStretchIdx.ARM]
                elevated_posture[HelloStretchIdx.WRIST_PITCH] = target_joint_state[HelloStretchIdx.WRIST_PITCH]
                logger.info(f"[STEP 2] Extending arm to {target_joint_state[HelloStretchIdx.ARM]:.3f}m")
                self.robot.arm_to(elevated_posture, blocking=True)
            else:
                # Single movement if we're already high enough
                logger.info("[SINGLE STEP] Moving to position (already at safe height)")
                self.robot.arm_to(target_joint_state, blocking=True)
            
            # Store height constraints for grasp operation
            if hasattr(self.agent, 'height_constraints'):
                self.agent.height_constraints.update(height_constraints)
            else:
                self.agent.height_constraints = height_constraints
                
            logger.info(f"[TABLE-AWARE PREGRASP] Positioned at {position_info['safe_lift_height']:.3f}m height")
            logger.info(f"[TABLE-AWARE PREGRASP] Will maintain minimum {position_info['min_allowed_height']:.3f}m during grasp")
            
        else:
            # Fallback to parent behavior if no executor
            logger.warning("[TABLE-AWARE PREGRASP] No executor reference, using standard pregrasp")
            super().run()

    def was_successful(self) -> bool:
        """Check if pregrasp positioning was successful."""
        
        # Check parent success conditions
        if not super().was_successful():
            return False
            
        # Additional height constraint validation
        if self.executor is not None and self.agent.current_object is not None:
            try:
                # Verify we're positioned safely
                obs = self.robot.get_observation()
                current_lift = obs.joint[HelloStretchIdx.LIFT]
                
                height_constraints = self.executor.get_height_constraints_for_object(self.agent.current_object)
                min_required_height = height_constraints['min_gripper_height']
                
                if current_lift < min_required_height - 0.01:  # 1cm tolerance
                    logger.error(f"[PREGRASP VALIDATION] Gripper too low: {current_lift:.3f}m < {min_required_height:.3f}m")
                    return False
                    
                logger.info(f"[PREGRASP VALIDATION] Success - positioned at safe height {current_lift:.3f}m")
                return True
                
            except Exception as e:
                logger.error(f"[PREGRASP VALIDATION] Error validating position: {e}")
                return False
        else:
            return True  # No additional validation without executor