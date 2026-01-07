# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
from typing import List, Optional
import numpy as np

from stretch.agent.operations.search_for_object import SearchForObjectOnFloorOperation
from stretch.utils.logger import Logger

logger = Logger(__name__)


class TableAwareSearchForObjectOperation(SearchForObjectOnFloorOperation):
    """
    Height-adaptive search operation that finds objects at any elevation.
    
    Unlike SearchForObjectOnFloorOperation, this searches for objects on any surface:
    - Floor objects (height < 0.3m)
    - Table objects (height 0.3m - 1.5m) 
    - Shelf objects (height > 1.5m)
    
    Features:
    - Universal height detection and filtering
    - Surface-aware object validation
    - Height constraint preparation for subsequent operations
    """

    def __init__(
        self,
        *args,
        safety_margin: float = 0.05,
        executor=None,
        min_object_height: float = 0.02,  # 2cm minimum (avoid floor noise)
        max_object_height: float = 2.5,   # 2.5m maximum (reasonable ceiling)
        **kwargs
    ):
        """Initialize height-adaptive search operation.
        
        Args:
            safety_margin: Height margin to maintain above surfaces
            executor: Reference to TableAwarePickupExecutor for height constraint access
            min_object_height: Minimum object height to consider (meters)
            max_object_height: Maximum object height to consider (meters)
        """
        # Initialize parent with on_floor_only=False to search all heights
        super().__init__(*args, on_floor_only=False, **kwargs)
        
        self.safety_margin = safety_margin
        self.executor = executor
        self.min_object_height = min_object_height
        self.max_object_height = max_object_height
        
        logger.info(f"[TABLE-AWARE SEARCH] Height-adaptive search initialized")
        logger.info(f"[TABLE-AWARE SEARCH] Height range: {min_object_height}m - {max_object_height}m")
        logger.info(f"[TABLE-AWARE SEARCH] Safety margin: {safety_margin}m")

    def is_object_at_valid_height(self, instance) -> bool:
        """
        Check if object is at a valid height for pickup.
        
        Args:
            instance: Object instance to check
            
        Returns:
            bool: True if object height is within valid range
        """
        try:
            if not hasattr(instance, 'point_cloud') or len(instance.point_cloud) == 0:
                logger.warning(f"[HEIGHT CHECK] Instance {instance.global_id} has no point cloud")
                return False
                
            # Get object height statistics
            object_points = instance.point_cloud
            object_height = object_points.mean(axis=0)[2].item()
            object_min_height = object_points[:, 2].min().item()
            object_max_height = object_points[:, 2].max().item()
            
            # Check if object is within valid height range
            if object_height < self.min_object_height:
                logger.debug(f"[HEIGHT CHECK] Object too low: {object_height:.3f}m < {self.min_object_height}m")
                return False
                
            if object_height > self.max_object_height:
                logger.debug(f"[HEIGHT CHECK] Object too high: {object_height:.3f}m > {self.max_object_height}m")
                return False
                
            # Additional check: ensure object has reasonable thickness (not just noise)
            object_thickness = object_max_height - object_min_height
            if object_thickness < 0.01:  # Less than 1cm thick
                logger.debug(f"[HEIGHT CHECK] Object too thin: {object_thickness:.3f}m")
                return False
                
            logger.debug(f"[HEIGHT CHECK] Valid object at {object_height:.3f}m (thickness: {object_thickness:.3f}m)")
            return True
            
        except Exception as e:
            logger.error(f"[HEIGHT CHECK] Error checking object height: {e}")
            return False

    def can_pick_object_safely(self, instance) -> bool:
        """
        Check if object can be picked up safely considering height constraints.
        
        Args:
            instance: Object instance to check
            
        Returns:
            bool: True if object can be safely approached and grasped
        """
        try:
            # First check basic height validity
            if not self.is_object_at_valid_height(instance):
                return False
                
            # If we have executor reference, get detailed height constraints
            if self.executor is not None:
                height_constraints = self.executor.get_height_constraints_for_object(instance)
                
                # Validate that we can maintain minimum gripper height
                min_gripper_height = height_constraints['min_gripper_height']
                object_height = height_constraints['object_height']
                
                # Ensure we can approach the object from above
                if min_gripper_height > object_height + 0.1:  # Need 10cm clearance above object
                    logger.warning(f"[SAFETY CHECK] Cannot approach object: min gripper {min_gripper_height:.3f}m > object {object_height:.3f}m + clearance")
                    return False
                    
                # Check if we can reach the object with arm extended
                # This is a simplified check - full IK validation happens in grasp operation
                if object_height > 1.8:  # Stretch robot arm reach limit approximation
                    logger.warning(f"[SAFETY CHECK] Object too high for arm reach: {object_height:.3f}m > 1.8m")
                    return False
                    
                logger.debug(f"[SAFETY CHECK] Object safe to pick: surface={height_constraints['surface_height']:.3f}m, object={object_height:.3f}m")
                
            return True
            
        except Exception as e:
            logger.error(f"[SAFETY CHECK] Error checking pickup safety: {e}")
            return False

    def update_instances_found(self, instances: List) -> None:
        """
        Override parent method to apply height-based filtering.
        
        Args:
            instances: List of detected instances to filter
        """
        # Apply parent filtering first (name matching, etc.)
        super().update_instances_found(instances)
        
        # Then apply height-based filtering
        if not hasattr(self, 'instances_found'):
            self.instances_found = []
            
        original_count = len(self.instances_found)
        
        # Filter instances by height and safety constraints
        height_valid_instances = []
        for instance in self.instances_found:
            if self.can_pick_object_safely(instance):
                height_valid_instances.append(instance)
            else:
                logger.debug(f"[HEIGHT FILTER] Filtered out instance {instance.global_id} (height/safety constraints)")
                
        self.instances_found = height_valid_instances
        filtered_count = original_count - len(self.instances_found)
        
        if filtered_count > 0:
            logger.info(f"[HEIGHT FILTER] Filtered {filtered_count} objects due to height/safety constraints")
            logger.info(f"[HEIGHT FILTER] {len(self.instances_found)} height-valid objects remain")

    def get_found_instances_summary(self) -> str:
        """
        Get a summary of found instances with height information.
        
        Returns:
            str: Summary string for logging
        """
        if not hasattr(self, 'instances_found') or not self.instances_found:
            return "No height-valid instances found"
            
        summaries = []
        for instance in self.instances_found:
            try:
                if hasattr(instance, 'point_cloud') and len(instance.point_cloud) > 0:
                    object_height = instance.point_cloud.mean(axis=0)[2].item()
                    if self.executor:
                        constraints = self.executor.get_height_constraints_for_object(instance)
                        object_type = constraints['object_type']
                        summaries.append(f"ID{instance.global_id}@{object_height:.2f}m({object_type})")
                    else:
                        summaries.append(f"ID{instance.global_id}@{object_height:.2f}m")
                else:
                    summaries.append(f"ID{instance.global_id}@unknown_height")
            except Exception as e:
                summaries.append(f"ID{instance.global_id}@error")
                
        return f"{len(self.instances_found)} objects: {', '.join(summaries)}"

    def was_successful(self) -> bool:
        """
        Check if search was successful (found at least one height-valid object).
        
        Returns:
            bool: True if at least one valid object was found
        """
        success = super().was_successful()
        
        if success and hasattr(self, 'instances_found'):
            height_summary = self.get_found_instances_summary()
            logger.info(f"[SEARCH SUCCESS] {height_summary}")
            
        return success

    def run(self):
        """Run the height-adaptive search operation."""
        # Speech announcement for search phase
        if hasattr(self.agent, 'enable_speech_debug') and self.agent.enable_speech_debug:
            self.agent.robot_say(f"Beginning search for {self.object_class}")
        logger.info(f"[TABLE-AWARE SEARCH] Starting height-adaptive search for '{self.object_class}'")
        
        # Run the parent search operation
        super().run()
        
        # Log final results with height information
        if hasattr(self, 'instances_found') and self.instances_found:
            summary = self.get_found_instances_summary()
            if hasattr(self.agent, 'enable_speech_debug') and self.agent.enable_speech_debug:
                self.agent.robot_say(f"Object {self.object_class} located")
            logger.info(f"[TABLE-AWARE SEARCH] Found {summary}")
            
            # Set current object for agent (choose first valid instance)
            if len(self.instances_found) > 0:
                self.agent.current_object = self.instances_found[0]
                if self.executor:
                    constraints = self.executor.get_height_constraints_for_object(self.instances_found[0])
                    logger.info(f"[TABLE-AWARE SEARCH] Selected object type: {constraints['object_type']}")
        else:
            if hasattr(self.agent, 'enable_speech_debug') and self.agent.enable_speech_debug:
                self.agent.robot_say(f"Search failed. Object {self.object_class} not found")
            logger.info(f"[TABLE-AWARE SEARCH] No instances of '{self.object_class}' found")


# Alias for backward compatibility with existing operation imports
TableAwareSearchForObjectOnFloorOperation = TableAwareSearchForObjectOperation