# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from typing import Optional

from stretch.agent.operations import (
    GoToNavOperation,
    NavigateToObjectOperation,
    OpenLoopGraspObjectOperation,
    PlaceObjectOperation,
    RotateInPlaceOperation,
    SearchForReceptacleOperation,
)
from stretch.agent.operations.table_aware_pregrasp import TableAwarePreGraspObjectOperation
from stretch.agent.operations.table_aware_grasp_object import TableAwareGraspObjectOperation
from stretch.agent.operations.table_aware_search import TableAwareSearchForObjectOperation
from stretch.agent.operations.table_aware_place_object import TableAwarePlaceObjectOperation
from stretch.agent.robot_agent import RobotAgent
from stretch.core.task import Operation, Task
from stretch.utils.logger import Logger

logger = Logger(__name__)


class TableAwarePickupTask:
    """
    Height-adaptive robot task for pickup and place operations at any elevation.
    
    Features:
    - Universal height adaptation (works for floor, table, shelf objects)
    - Surface detection and collision avoidance
    - Safe pickup from any height and placement to any receptacle
    - Always starts from high position and approaches from above
    - Maintains safety margins above detected surfaces
    """

    def __init__(
        self,
        agent: RobotAgent,
        target_object: Optional[str] = None,
        target_receptacle: Optional[str] = None,
        use_visual_servoing_for_grasp: bool = True,
        matching: str = "feature",
        safety_margin: float = 0.05,
        executor=None,
    ) -> None:
        """Initialize height-adaptive pickup task.
        
        Args:
            agent: Robot agent
            target_object: Name of object to pick up
            target_receptacle: Name of receptacle to place object in
            use_visual_servoing_for_grasp: Whether to use visual servoing for grasping
            matching: Object matching method ("feature" or "class")
            safety_margin: Height margin to maintain above surfaces
            executor: Reference to TableAwarePickupExecutor for height constraint access
        """
        self.agent = agent

        # Task information
        self.agent.target_object = target_object
        self.agent.target_receptacle = target_receptacle
        self.target_object = target_object
        self.target_receptacle = target_receptacle
        self.use_visual_servoing_for_grasp = use_visual_servoing_for_grasp

        assert matching in ["feature", "class"], f"Invalid instance matching method: {matching}"
        self.matching = matching

        # Sync these things
        self.robot = self.agent.robot
        self.voxel_map = self.agent.get_voxel_map()
        self.navigation_space = self.agent.space
        self.semantic_sensor = self.agent.semantic_sensor
        self.parameters = self.agent.parameters
        self.instance_memory = self.agent.get_voxel_map().instances
        assert (
            self.instance_memory is not None
        ), "Make sure instance memory was created! This is configured in parameters file."

        # Height-adaptive configuration
        self.safety_margin = safety_margin
        self.executor = executor

        self.current_object = None
        self.current_receptacle = None
        self.agent.reset_object_plans()
        
        logger.info(f"[TABLE-AWARE PICKUP] Initialized height-adaptive pickup task")
        logger.info(f"[TABLE-AWARE PICKUP] Object: '{target_object}' -> Receptacle: '{target_receptacle}'")
        logger.info(f"[TABLE-AWARE PICKUP] Safety margin: {safety_margin}m, Visual servoing: {use_visual_servoing_for_grasp}")

    def get_task(self, add_rotate: bool = False, mode: str = "one_shot") -> Task:
        """Create a height-adaptive task plan with universal surface awareness.

        Args:
            add_rotate (bool, optional): Whether to add a rotate operation to explore the robot's area. Defaults to False.
            mode (str, optional): Type of task to create. Currently only "one_shot" supported. Defaults to "one_shot".

        Returns:
            Task: Executable task plan for height-adaptive pickup and place operations.
        """

        if mode == "one_shot":
            return self.get_one_shot_task(add_rotate=add_rotate, matching=self.matching)
        else:
            raise NotImplementedError(f"Task mode '{mode}' not implemented for table-aware pickup.")

    def get_one_shot_task(self, add_rotate: bool = False, matching: str = "feature") -> Task:
        """Create a height-adaptive one-shot pickup+place task that works at any elevation."""

        # Put the robot into navigation mode
        go_to_navigation_mode = GoToNavOperation(
            "go to navigation mode", self.agent, retry_on_failure=True
        )

        if add_rotate:
            # Spin in place to find objects.
            rotate_in_place = RotateInPlaceOperation(
                "rotate_in_place", self.agent, parent=go_to_navigation_mode
            )

        # Height-adaptive search for target object
        search_for_object = TableAwareSearchForObjectOperation(
            name=f"search_for_{self.target_object}_at_any_height",
            agent=self.agent,
            retry_on_failure=True,
            match_method=matching,
            require_receptacle=True,  # We need to place it somewhere
            safety_margin=self.safety_margin,
            executor=self.executor,
        )
        if self.agent.target_object is not None:
            search_for_object.set_target_object_class(self.agent.target_object)

        # Search for receptacle (standard operation, already height-aware)
        search_for_receptacle = SearchForReceptacleOperation(
            name=f"search_for_{self.target_receptacle}",
            agent=self.agent,
            retry_on_failure=True,
            match_method=matching,
        )
        if self.agent.target_receptacle is not None:
            search_for_receptacle.set_target_receptacle_class(self.agent.target_receptacle)

        # Navigate to target object (handles any height)
        go_to_object = NavigateToObjectOperation(
            name="go_to_object",
            agent=self.agent,
            parent=search_for_object,
            on_cannot_start=search_for_object,
            to_receptacle=False,
        )

        # Height-adaptive pregrasp positioning for object
        pregrasp_object = TableAwarePreGraspObjectOperation(
            name="prepare_to_grasp_with_height_awareness",
            agent=self.agent,
            on_failure=None,
            on_cannot_start=go_to_object,
            retry_on_failure=True,
            safety_margin=self.safety_margin,
            executor=self.executor,
        )

        # Height-adaptive grasping with surface collision avoidance
        grasp_object: Operation = None
        if self.use_visual_servoing_for_grasp:
            grasp_object = TableAwareGraspObjectOperation(
                f"grasp_the_{self.target_object}_safely",
                self.agent,
                parent=pregrasp_object,
                on_failure=pregrasp_object,
                on_cannot_start=go_to_object,
                retry_on_failure=False,
                safety_margin=self.safety_margin,
                executor=self.executor,
            )
            grasp_object.set_target_object_class(self.agent.target_object)
            grasp_object.servo_to_grasp = True
            grasp_object.match_method = matching
        else:
            # Open-loop grasping with height awareness
            grasp_object = OpenLoopGraspObjectOperation(
                f"grasp_the_{self.target_object}_open_loop",
                self.agent,
                parent=pregrasp_object,
                on_failure=pregrasp_object,
                on_cannot_start=go_to_object,
                retry_on_failure=False,
            )
            grasp_object.set_target_object_class(self.agent.target_object)
            grasp_object.match_method = matching

        # Navigate to receptacle for placement
        go_to_receptacle = NavigateToObjectOperation(
            name="go_to_receptacle",
            agent=self.agent,
            parent=grasp_object,
            on_cannot_start=search_for_receptacle,
            to_receptacle=True,
        )

        # Place the object in the receptacle using height-adaptive operation
        logger.info("[TABLE-AWARE TASK] Creating TableAwarePlaceObjectOperation for height-adaptive placement")
        print("📦📦📦 CREATING HEIGHT-ADAPTIVE PLACE OPERATION - THIS SHOULD BE VISIBLE! 📦📦📦")
        place_object = TableAwarePlaceObjectOperation(
            name="table_aware_place_object_in_receptacle",
            agent=self.agent,
            parent=go_to_receptacle,
            on_failure=go_to_receptacle,
            on_cannot_start=search_for_receptacle,
            safety_margin=self.safety_margin,
            executor=self.executor,  # Pass executor for height detection
        )
        logger.info("[TABLE-AWARE TASK] Height-adaptive placement operation created successfully")
        place_object.set_target_object_class(self.agent.target_object)
        place_object.set_target_receptacle_class(self.agent.target_receptacle)
        place_object.match_method = matching

        # Build the task
        task = Task()
        task.add_operation(go_to_navigation_mode)
        if add_rotate:
            task.add_operation(rotate_in_place)
        task.add_operation(search_for_object)
        task.add_operation(search_for_receptacle)
        task.add_operation(go_to_object)
        task.add_operation(pregrasp_object)
        task.add_operation(grasp_object)
        task.add_operation(go_to_receptacle)
        task.add_operation(place_object)

        # Success connections
        if add_rotate:
            task.connect_on_success(go_to_navigation_mode.name, rotate_in_place.name)
            task.connect_on_success(rotate_in_place.name, search_for_object.name)
        else:
            task.connect_on_success(go_to_navigation_mode.name, search_for_object.name)

        task.connect_on_success(search_for_object.name, search_for_receptacle.name)
        task.connect_on_success(search_for_receptacle.name, go_to_object.name)
        task.connect_on_success(go_to_object.name, pregrasp_object.name)
        task.connect_on_success(pregrasp_object.name, grasp_object.name)
        task.connect_on_success(grasp_object.name, go_to_receptacle.name)
        task.connect_on_success(go_to_receptacle.name, place_object.name)

        # Failure connections - retry search operations when things fail
        task.connect_on_failure(pregrasp_object.name, search_for_object.name)
        task.connect_on_failure(grasp_object.name, search_for_object.name)
        task.connect_on_failure(place_object.name, search_for_receptacle.name)

        # Navigation failures
        if add_rotate:
            task.connect_on_failure(go_to_object.name, rotate_in_place.name)
            task.connect_on_failure(go_to_receptacle.name, rotate_in_place.name)
        else:
            task.connect_on_failure(go_to_object.name, search_for_object.name)
            task.connect_on_failure(go_to_receptacle.name, search_for_receptacle.name)

        # Terminate on successful placement
        task.terminate_on_success(place_object.name)

        return task


if __name__ == "__main__":
    from stretch.agent.robot_agent import RobotAgent
    from stretch.agent.zmq_client import HomeRobotZmqClient

    robot = HomeRobotZmqClient()

    # Create a robot agent with instance memory
    agent = RobotAgent(robot, create_semantic_sensor=True)

    task = TableAwarePickupTask(agent, target_object="apple", target_receptacle="bowl").get_task(add_rotate=False)
    task.run()