# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
import numpy as np
import torch

from stretch.agent.operations.place_object import PlaceObjectOperation
from stretch.motion import HelloStretchIdx
from stretch.utils.logger import Logger
from stretch.utils.geometry import point_global_to_base

logger = Logger(__name__)


class TableAwarePlaceObjectOperation(PlaceObjectOperation):
    """
    Height-adaptive placement operation that approaches from appropriate height based on target surface.
    
    Key Features:
    - Starts from elevated position for high surfaces (tables, shelves)
    - Uses standard low approach for ground-level surfaces
    - Mirrors the pickup behavior: high->low for elevated, low->high for ground
    - Surface collision avoidance with safety margins
    - Graceful fallback if height detection fails
    """

    def __init__(
        self,
        *args,
        safety_margin: float = 0.05,
        executor=None,
        **kwargs
    ):
        """Initialize height-adaptive placement operation.
        
        Args:
            safety_margin: Height margin to maintain above detected surfaces
            executor: Reference to TableAwarePickupExecutor for height constraint access
        """
        super().__init__(*args, **kwargs)
        
        self.safety_margin = safety_margin
        self.executor = executor
        
        logger.info(f"[TABLE-AWARE PLACE] Universal height-adaptive placement initialized")
        logger.info(f"[TABLE-AWARE PLACE] Safety margin: {self.safety_margin}m")
        logger.info(f"[TABLE-AWARE PLACE] Works for ALL surface heights (floor, table, shelf)")

    def detect_receptacle_height(self) -> float:
        """
        Detect the height of the target receptacle surface.
        
        Returns:
            float: Height of receptacle surface in meters
        """
        try:
            target = self.get_target()
            if target is None:
                logger.warning("[TABLE-AWARE PLACE] No target receptacle found")
                return 0.0

            # Use executor's height detection if available (preferred - matches pickup)
            if self.executor is not None and hasattr(self.executor, 'detect_surface_height_under_object'):
                surface_height = self.executor.detect_surface_height_under_object(target)
                logger.info(f"[TABLE-AWARE PLACE] Using executor surface detection: {surface_height:.3f}m")
                return surface_height
            
            # Fallback: use point cloud max height
            if hasattr(target, 'point_cloud') and target.point_cloud is not None and len(target.point_cloud) > 0:
                if isinstance(target.point_cloud, torch.Tensor):
                    max_height = target.point_cloud[:, 2].max().item()
                else:
                    max_height = float(np.max(target.point_cloud[:, 2]))
                logger.info(f"[TABLE-AWARE PLACE] Using point cloud max height: {max_height:.3f}m")
                return max_height
            
            # Final fallback
            logger.warning("[TABLE-AWARE PLACE] No height detection method available, assuming floor level")
            return 0.0
            
        except Exception as e:
            logger.error(f"[TABLE-AWARE PLACE] Error detecting receptacle height: {e}")
            return 0.0

    def get_adaptive_starting_posture(self, target_height: float) -> np.ndarray:
        """
        Calculate adaptive starting arm posture for ANY surface height.
        Always starts elevated and approaches downward - just like pickup!
        
        Args:
            target_height: Height of target placement surface
            
        Returns:
            np.ndarray: Joint positions for starting posture
        """
        try:
            current_joint_state = self.robot.get_joint_positions()
            
            # UNIVERSAL approach: Always start elevated and approach from above
            # This mirrors the pickup behavior exactly - no thresholds!
            
            # Calculate elevated starting position (target + clearance, like pickup)
            elevated_lift = min(target_height + 0.25, 1.0)  # Start 25cm above target, max 1.0m
            elevated_lift = max(elevated_lift, 0.3)  # Minimum reasonable height
            
            # NEVER GO DOWN: If current position is higher, stay there or go higher
            current_lift = current_joint_state[HelloStretchIdx.LIFT]
            if elevated_lift < current_lift:
                elevated_lift = max(current_lift, current_lift + 0.1)  # Stay current or go higher
                logger.info(f"[TABLE-AWARE PLACE] Current lift {current_lift:.3f}m > target {elevated_lift:.3f}m, keeping high position")
                print(f"🔝🔝🔝 STAYING HIGH: current={current_lift:.3f}m, target={elevated_lift:.3f}m 🔝🔝🔝")
            
            # Create elevated starting posture (same as pickup approach)
            elevated_posture = current_joint_state.copy()
            elevated_posture[HelloStretchIdx.LIFT] = elevated_lift
            elevated_posture[HelloStretchIdx.ARM] = 0.15  # Slightly extended arm  
            elevated_posture[HelloStretchIdx.WRIST_PITCH] = -1.1  # Angled down for top-down approach
            
            logger.info(f"[TABLE-AWARE PLACE] UNIVERSAL elevated approach for {target_height:.3f}m surface")
            logger.info(f"[TABLE-AWARE PLACE] Starting from {elevated_lift:.3f}m height (like pickup)")
            logger.info(f"[TABLE-AWARE PLACE] Will approach downward to place object")
            
            return elevated_posture
                
        except Exception as e:
            logger.error(f"[TABLE-AWARE PLACE] Error calculating adaptive posture: {e}")
            # Safe fallback: still use elevated approach
            fallback_posture = current_joint_state.copy()
            fallback_posture[HelloStretchIdx.LIFT] = 0.5  # Safe middle height
            fallback_posture[HelloStretchIdx.WRIST_PITCH] = -1.1
            return fallback_posture

    def run(self) -> None:
        """
        Execute height-adaptive placement with appropriate starting posture.
        Follows the same pattern as TableAwareGraspObjectOperation.
        """
        logger.info("[TABLE-AWARE PLACE] Starting height-adaptive placement run() method")
        print("🚀🚀🚀 HEIGHT-ADAPTIVE PLACEMENT RUNNING - THIS SHOULD BE VISIBLE! 🚀🚀🚀")
        print("🚀🚀🚀 MY HEIGHT-ADAPTIVE RUN() METHOD IS EXECUTING! 🚀🚀🚀")
        print("🚀🚀🚀 NOT THE PARENT'S LOW-RESET RUN() METHOD! 🚀🚀🚀")
        
        # Speech announcement for place phase
        if hasattr(self.agent, 'enable_speech_debug') and self.agent.enable_speech_debug:
            self.agent.robot_say("Placing object")
        
        # LOUD DEBUG SPEECH - announce it right before placement
        self.agent.robot_say("USING HEIGHT ENHANCED PLACE!")
        
        self.intro("Height-adaptive placement of object on receptacle.")
        self._successful = False

        # Get initial (carry) joint posture
        obs = self.robot.get_observation()
        joint_state = obs.joint
        model = self.robot.get_robot_model()

        # Detect target receptacle height first
        receptacle_height = self.detect_receptacle_height()
        logger.info(f"[TABLE-AWARE PLACE] Target receptacle height: {receptacle_height:.3f}m")
        print(f"🎯🎯🎯 RECEPTACLE HEIGHT DETECTED: {receptacle_height:.3f}m 🎯🎯🎯")

        # Switch to manipulation mode first
        self.robot.switch_to_manipulation_mode()

        # Get adaptive starting posture based on receptacle height
        adaptive_posture = self.get_adaptive_starting_posture(receptacle_height)
        
        current_lift = joint_state[HelloStretchIdx.LIFT]
        target_lift = adaptive_posture[HelloStretchIdx.LIFT]
        
        print(f"🔥🔥🔥 ABOUT TO MOVE TO HIGH POSITION: lift={target_lift:.3f}m 🔥🔥🔥")
        
        # Move to height-adaptive starting position with collision avoidance
        logger.info("[TABLE-AWARE PLACE] Moving to height-adaptive starting position")
        
        if target_lift > current_lift:  # Only for going UP (high reset)
            print("🚫🚫🚫 HIGH RESET DETECTED - USING TWO-STEP MOVEMENT TO AVOID TABLE COLLISION! 🚫🚫🚫")
            
            # Step 1: Go UP first (keep current arm extension to avoid collision)
            print("⬆️⬆️⬆️ STEP 1: LIFTING ARM UP FIRST (NO EXTENSION)! ⬆️⬆️⬆️")
            safe_up_posture = joint_state.copy()
            safe_up_posture[HelloStretchIdx.LIFT] = target_lift
            # Keep current arm position - don't extend yet
            self.robot.arm_to(safe_up_posture, blocking=True)
            print("✅ STEP 1 COMPLETE: ARM IS NOW HIGH AND SAFE!")
            
            # Step 2: THEN extend arm forward (now safe above table)
            print("➡️➡️➡️ STEP 2: NOW EXTENDING ARM FORWARD (SAFE ABOVE TABLE)! ➡️➡️➡️")
            elevated_posture = safe_up_posture.copy()
            elevated_posture[HelloStretchIdx.ARM] = adaptive_posture[HelloStretchIdx.ARM]
            elevated_posture[HelloStretchIdx.WRIST_PITCH] = adaptive_posture[HelloStretchIdx.WRIST_PITCH]
            self.robot.arm_to(elevated_posture, blocking=True)
            print("✅✅✅ STEP 2 COMPLETE: ARM FULLY POSITIONED WITHOUT COLLISION! ✅✅✅")
            
        else:
            # For low positions or lateral movements, single movement is fine
            print("🆙🆙🆙 SINGLE MOVEMENT (NO COLLISION RISK)! 🆙🆙🆙")
            self.robot.arm_to(adaptive_posture, blocking=True)
            print("✅✅✅ SINGLE MOVEMENT COMPLETED! ✅✅✅")

        # Get object xyz coords
        xyt = self.robot.get_base_pose()
        placement_xyz = self.sample_placement_position(xyt)
        logger.info(f"[TABLE-AWARE PLACE] Place object at {placement_xyz}")

        # Get the center of the object point cloud so that we can place there
        relative_object_xyz = point_global_to_base(placement_xyz, xyt)

        # Get current joint state after adaptive positioning
        joint_state = self.robot.get_joint_positions()

        # Compute the angles necessary
        if self.use_pitch_from_vertical:
            ee_pos, ee_rot = model.manip_fk(joint_state)
            dy = np.abs(ee_pos[1] - relative_object_xyz[1])
            dz = np.abs(ee_pos[2] - relative_object_xyz[2])
            pitch_from_vertical = np.arctan2(dy, dz)
        else:
            pitch_from_vertical = 0.0

        # UNIVERSAL approach: Always use downward angle (like pickup)
        joint_state[HelloStretchIdx.WRIST_PITCH] = -np.pi/2 + pitch_from_vertical
        logger.info(f"[TABLE-AWARE PLACE] Using universal top-down approach for {receptacle_height:.3f}m surface")

        self.robot.arm_to(joint_state)
        ee_pos, ee_rot = model.manip_fk(joint_state)

        # Get max xyz of target
        max_xyz = self.get_target().point_cloud.max(axis=0)[0]

        # Placement is at xy = object_xyz[:2], z = max_xyz[2] + margin
        place_xyz = np.array(
            [relative_object_xyz[0], relative_object_xyz[1], max_xyz[2] + self.place_height_margin]
        )

        if self.show_place_in_voxel_grid:
            self.agent.get_voxel_map().show(
                orig=place_xyz, xyt=xyt, footprint=self.robot_model.get_footprint()
            )

        target_joint_positions, success = self._get_place_joint_state(
            pos=place_xyz, quat=ee_rot, joint_state=joint_state
        )
        
        self.attempt(f"Trying to place object using UNIVERSAL height-adaptive approach at {place_xyz}.")
        
        if self.talk:
            self.agent.robot_say(f"Placing object from above onto the {receptacle_height:.1f}m surface.")
                
        if not success:
            self.error("Could not place object with height-adaptive approach!")
            return

        # Move to the target joint state
        self.robot.switch_to_manipulation_mode()
        self.robot.arm_to(target_joint_positions, blocking=True)
        time.sleep(0.5)

        # Open the gripper
        self.robot.open_gripper(blocking=True)
        time.sleep(0.5)

        # Move directly up
        target_joint_positions_lifted = target_joint_positions.copy()
        target_joint_positions_lifted[HelloStretchIdx.LIFT] += self.lift_distance
        self.robot.arm_to(target_joint_positions_lifted, blocking=True)

        # Return arm to navigation posture
        self.robot.move_to_nav_posture()
        self._successful = True

        self.agent.robot_say("Universal height-adaptive placement completed.")
        self.cheer("Successfully placed object using UNIVERSAL height-adaptive approach.")

    def was_successful(self):
        return self._successful