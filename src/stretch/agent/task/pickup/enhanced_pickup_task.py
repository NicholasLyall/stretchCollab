# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Enhanced pickup task with visual servoing integrated in navigation/approach phases

from typing import Optional

import stretch.utils.logger as logger
from stretch.agent.operations import (
    GoToNavOperation,
    GraspObjectOperation,
    NavigateToObjectOperation,
    PlaceObjectOperation,
    RotateInPlaceOperation,
    SearchForObjectOnFloorOperation,
    SearchForReceptacleOperation,
)

# Import enhanced operations
try:
    from stretch.agent.operations.enhanced_navigate_to_object import EnhancedNavigateToObjectOperation
    ENHANCED_NAVIGATE_AVAILABLE = True
except ImportError:
    from stretch.agent.operations import NavigateToObjectOperation
    EnhancedNavigateToObjectOperation = NavigateToObjectOperation
    ENHANCED_NAVIGATE_AVAILABLE = False

try:
    from stretch.agent.operations.enhanced_pregrasp import EnhancedPreGraspObjectOperation
    ENHANCED_PREGRASP_AVAILABLE = True
except ImportError:
    from stretch.agent.operations import PreGraspObjectOperation
    EnhancedPreGraspObjectOperation = PreGraspObjectOperation
    ENHANCED_PREGRASP_AVAILABLE = False

from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Task


class EnhancedPickupTask:
    """
    Enhanced pickup task that integrates visual servoing in the navigation and approach phases
    instead of only at the final grasp. This provides:
    
    1. Much better accuracy during object approach
    2. Smoother transitions between operations  
    3. No jerky motion at the final grasp
    4. Overall faster and more reliable operation
    """

    def __init__(
        self,
        agent: RobotAgent,
        target_object: Optional[str] = None,
        target_receptacle: Optional[str] = None,
        use_enhanced_visual_servoing: bool = True,
        matching: str = "feature",
    ) -> None:
        self.agent = agent
        
        # Task information
        self.agent.target_object = target_object
        self.agent.target_receptacle = target_receptacle
        self.target_object = target_object
        self.target_receptacle = target_receptacle
        self.use_enhanced_visual_servoing = (
            use_enhanced_visual_servoing and 
            ENHANCED_NAVIGATE_AVAILABLE and 
            ENHANCED_PREGRASP_AVAILABLE
        )

        assert matching in ["feature", "class"], f"Invalid instance matching method: {matching}"
        self.matching = matching

        # Robot interfaces
        self.robot = self.agent.robot
        self.voxel_map = self.agent.get_voxel_map()
        self.navigation_space = self.agent.space
        self.semantic_sensor = self.agent.semantic_sensor
        self.parameters = self.agent.parameters
        self.instance_memory = self.agent.get_voxel_map().instances
        
        assert self.instance_memory is not None, "Instance memory must be configured!"

        # Reset state
        self.current_object = None
        self.current_receptacle = None
        self.agent.reset_object_plans()
        
        # Log configuration
        if self.use_enhanced_visual_servoing:
            print("[ENHANCED PICKUP] Using enhanced visual servoing in navigation/approach phases")
        else:
            print("[ENHANCED PICKUP] Using standard operations")

    def get_task(self, add_rotate: bool = False, mode: str = "one_shot") -> Task:
        """Create enhanced task plan with visual servoing integration"""
        
        if mode == "one_shot":
            return self.get_one_shot_task(add_rotate=add_rotate, matching=self.matching)
        else:
            raise ValueError(f"Enhanced pickup task only supports 'one_shot' mode, got: {mode}")

    def get_one_shot_task(self, add_rotate: bool = False, matching: str = "feature") -> Task:
        """Create enhanced one-shot task with visual servoing in navigation phases"""

        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation(
            "go to navigation mode", self.agent, retry_on_failure=True
        )

        if add_rotate:
            # Spin in place to find objects
            rotate_in_place = RotateInPlaceOperation(
                name="rotate_in_place", agent=self.agent, parent=go_to_navigation_mode
            )

        # Look for the target receptacle
        search_for_receptacle = SearchForReceptacleOperation(
            name=f"search_for_{self.target_receptacle}",
            agent=self.agent,
            parent=rotate_in_place if add_rotate else go_to_navigation_mode,
            retry_on_failure=True,
            match_method="feature",
        )

        # Search for target object
        search_for_object = SearchForObjectOnFloorOperation(
            name=f"search_for_{self.target_object}_on_floor",
            agent=self.agent,
            retry_on_failure=True,
            match_method="feature",
        )

        # Set target objects
        search_for_object.set_target_object_class(self.target_object)
        search_for_receptacle.set_target_object_class(self.target_receptacle)

        # Enhanced navigation to object with visual servoing final approach
        if self.use_enhanced_visual_servoing:
            go_to_object = EnhancedNavigateToObjectOperation(
                name="enhanced_go_to_object",
                agent=self.agent,
                parent=search_for_object,
                on_cannot_start=search_for_object,
                to_receptacle=False,
                for_manipulation=True,
            )
            print("[ENHANCED PICKUP] Using enhanced navigation with visual servoing")
        else:
            go_to_object = NavigateToObjectOperation(
                name="go_to_object",
                agent=self.agent,
                parent=search_for_object,
                on_cannot_start=search_for_object,
                to_receptacle=False,
            )
            print("[ENHANCED PICKUP] Using standard navigation")

        # Navigate to receptacle (standard - no need to enhance this)
        go_to_receptacle = NavigateToObjectOperation(
            name="go_to_receptacle",
            agent=self.agent,
            on_cannot_start=search_for_receptacle,
            to_receptacle=True,
        )

        # Enhanced pregrasp operation with smooth transitions
        if self.use_enhanced_visual_servoing:
            pregrasp_object = EnhancedPreGraspObjectOperation(
                name="enhanced_prepare_to_grasp",
                agent=self.agent,
                on_failure=None,
                on_cannot_start=go_to_object,
                retry_on_failure=True,
            )
            print("[ENHANCED PICKUP] Using enhanced pregrasp with smooth transitions")
        else:
            pregrasp_object = PreGraspObjectOperation(
                name="prepare_to_grasp",
                agent=self.agent,
                on_failure=None,
                on_cannot_start=go_to_object,
                retry_on_failure=True,
            )
            print("[ENHANCED PICKUP] Using standard pregrasp")

        # Final grasp operation (standard to avoid jerky transitions)
        grasp_object = GraspObjectOperation(
            name=f"grasp_the_{self.target_object}",
            agent=self.agent,
            parent=pregrasp_object,
            on_failure=pregrasp_object,
            on_cannot_start=go_to_object,
            retry_on_failure=False,
        )
        grasp_object.set_target_object_class(self.agent.target_object)
        grasp_object.servo_to_grasp = False  # Disable visual servoing here
        grasp_object.use_visual_servoing = False  # Ensure it's disabled
        grasp_object.match_method = matching

        # Place object operation
        place_object_on_receptacle = PlaceObjectOperation(
            "place_object_on_receptacle",
            self.agent,
            on_cannot_start=go_to_receptacle,
            require_object=True,
        )

        # Build the task
        task = Task()
        task.add_operation(go_to_navigation_mode)
        if add_rotate:
            task.add_operation(rotate_in_place)
        task.add_operation(search_for_receptacle)
        task.add_operation(search_for_object)
        task.add_operation(go_to_object)
        task.add_operation(pregrasp_object)
        task.add_operation(grasp_object)
        task.add_operation(go_to_receptacle)
        task.add_operation(place_object_on_receptacle)

        return task