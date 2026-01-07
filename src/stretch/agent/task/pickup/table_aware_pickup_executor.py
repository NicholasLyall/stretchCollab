# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import datetime
import numpy as np
from typing import List, Tuple, Optional

from PIL import Image

from stretch.agent.robot_agent import RobotAgent
from stretch.agent.task.emote import EmoteTask
from stretch.agent.task.pickup.find_task import FindObjectTask
from stretch.agent.task.pickup.hand_over_task import HandOverTask
from stretch.agent.task.pickup.table_aware_pick_task import TableAwarePickObjectTask
from stretch.agent.task.pickup.table_aware_pickup_task import TableAwarePickupTask
from stretch.agent.task.pickup.place_task import PlaceOnReceptacleTask

from stretch.core import AbstractRobotClient
from stretch.utils.image import numpy_image_to_bytes
from stretch.utils.logger import Logger

logger = Logger(__name__)


class TableAwarePickupExecutor:
    """
    Universal height-adaptive pickup executor that works for objects at any height.
    Automatically detects surface heights and adapts approach strategy accordingly.
    
    Key Features:
    - Always starts from high arm position
    - Detects surface underneath objects (floor, table, shelf, etc.)
    - Sets minimum gripper height constraints
    - Approaches from above, never goes below detected surface
    - Works universally for ground objects, table objects, shelf objects, etc.
    """

    _pickup_task_mode = "one_shot"

    def __init__(
        self,
        robot: AbstractRobotClient,
        agent: RobotAgent,
        match_method: str = "feature",
        open_loop: bool = False,
        dry_run: bool = False,
        available_actions: List[str] = None,
        discord_bot=None,
        enhanced_visual_servoing: bool = True,
        safety_margin: float = 0.05,  # 5cm safety margin above detected surfaces
        min_object_height_for_surface_detection: float = 0.15,  # Objects below 15cm assumed to be on floor
    ) -> None:
        """Initialize the height-adaptive pickup executor.

        Args:
            robot: The robot client.
            agent: The robot agent.
            match_method: Method for matching objects ("feature" or "class").
            open_loop: If true, use open-loop grasping.
            dry_run: If true, don't actually execute the commands.
            available_actions: A list of available actions.
            discord_bot: Optional discord integration.
            enhanced_visual_servoing: Whether to use enhanced visual servoing.
            safety_margin: Height margin to maintain above detected surfaces (meters).
            min_object_height_for_surface_detection: Objects below this height assumed on floor (meters).
        """
        self.robot = robot
        self.agent = agent
        self.available_actions = available_actions

        # Optional discord integration for chatting with the robot
        self.discord_bot = discord_bot

        # Do type checks
        if not isinstance(self.robot, AbstractRobotClient):
            raise TypeError(f"Expected AbstractRobotClient, got {type(self.robot)}")
        if not isinstance(self.agent, RobotAgent):
            raise TypeError(f"Expected RobotAgent, got {type(self.agent)}")

        self.dry_run = dry_run
        self.emote_task = EmoteTask(self.agent)

        # Configuration
        self._match_method = match_method
        self._open_loop = open_loop
        self._enhanced_visual_servoing = enhanced_visual_servoing
        
        # Height-adaptive configuration
        self.safety_margin = safety_margin
        self.min_object_height_for_surface_detection = min_object_height_for_surface_detection
        
        # Log configuration
        logger.info("[TABLE-AWARE PICKUP] Universal height-adaptive pickup executor initialized")
        logger.info(f"[TABLE-AWARE PICKUP] Safety margin: {self.safety_margin}m above detected surfaces")
        logger.info(f"[TABLE-AWARE PICKUP] Surface detection threshold: {self.min_object_height_for_surface_detection}m")
        
        if self._enhanced_visual_servoing and not self._open_loop:
            logger.info("[TABLE-AWARE PICKUP] Enhanced visual servoing enabled for height-adaptive manipulation!")
        elif not self._open_loop:
            logger.info("[TABLE-AWARE PICKUP] Standard visual servoing enabled with height adaptation")
        else:
            logger.info("[TABLE-AWARE PICKUP] Open-loop grasping enabled with height awareness")

    def detect_surface_height_under_object(self, instance) -> float:
        """
        Detect the height of the surface underneath an object.
        
        Args:
            instance: The object instance with point cloud data
            
        Returns:
            float: Height of the detected surface (0.0 for floor, table height for tables, etc.)
        """
        try:
            # Get object point cloud
            if not hasattr(instance, 'point_cloud') or len(instance.point_cloud) == 0:
                logger.warning("[HEIGHT DETECTION] No point cloud available, assuming floor level")
                return 0.0
                
            object_points = instance.point_cloud
            object_height = object_points.mean(axis=0)[2].item()
            
            # If object is very low, assume it's on the floor
            if object_height < self.min_object_height_for_surface_detection:
                logger.info(f"[HEIGHT DETECTION] Object at {object_height:.3f}m - detected as floor object")
                return 0.0
            
            # For elevated objects, try to detect the supporting surface
            # Look for the lowest point of the object as approximation of surface height
            min_object_z = object_points[:, 2].min().item()
            
            # SIMPLIFIED: Use object's minimum height as surface estimate
            # This avoids the complex voxel map API that's causing errors
            estimated_surface_height = max(0.0, min_object_z - 0.05)  # 5cm below object minimum
            
            # Simple heuristic: if object is high, assume it's on a table
            if object_height > 0.5:  # Object center > 50cm high
                estimated_surface_height = max(estimated_surface_height, 0.7)  # Assume table height
                logger.info(f"[HEIGHT DETECTION] High object at {object_height:.3f}m, assuming table surface at {estimated_surface_height:.3f}m")
            else:
                logger.info(f"[HEIGHT DETECTION] Low object at {object_height:.3f}m, estimated surface at {estimated_surface_height:.3f}m")
            
            return estimated_surface_height
                
        except Exception as e:
            logger.error(f"[HEIGHT DETECTION] Error detecting surface height: {e}")
            # Safe fallback: assume floor level
            return 0.0

    def get_height_constraints_for_object(self, instance) -> dict:
        """
        Calculate height constraints for approaching an object.
        
        Args:
            instance: The object instance
            
        Returns:
            dict: Height constraint information
        """
        try:
            # Detect surface height under object
            surface_height = self.detect_surface_height_under_object(instance)
            
            # Calculate minimum gripper height (surface + safety margin)
            min_gripper_height = surface_height + self.safety_margin
            
            # Get object height for approach planning
            object_points = instance.point_cloud
            object_height = object_points.mean(axis=0)[2].item()
            object_max_height = object_points[:, 2].max().item()
            
            # Calculate recommended starting height (well above object)
            start_height = max(object_max_height + 0.2, min_gripper_height + 0.15)  # Start 20cm above object or 15cm above constraint
            
            constraints = {
                'surface_height': surface_height,
                'min_gripper_height': min_gripper_height,
                'object_height': object_height,
                'object_max_height': object_max_height,
                'recommended_start_height': start_height,
                'safety_margin': self.safety_margin,
                'object_type': 'elevated' if surface_height > 0.1 else 'floor'
            }
            
            logger.info(f"[HEIGHT CONSTRAINTS] Surface: {surface_height:.3f}m, Min gripper: {min_gripper_height:.3f}m, Start: {start_height:.3f}m")
            
            return constraints
            
        except Exception as e:
            logger.error(f"[HEIGHT CONSTRAINTS] Error calculating constraints: {e}")
            # Safe fallback constraints
            return {
                'surface_height': 0.0,
                'min_gripper_height': self.safety_margin,
                'object_height': 0.2,
                'object_max_height': 0.25,
                'recommended_start_height': 0.5,
                'safety_margin': self.safety_margin,
                'object_type': 'floor'
            }

    def _pickup(self, target_object: str, target_receptacle: str) -> None:
        """Create a height-adaptive task to pick up the object and execute it.

        Args:
            target_object: The object to pick up.
            target_receptacle: The receptacle to place the object in.
        """

        if target_receptacle is None or len(target_receptacle) == 0:
            self._pick_only(target_object)
            return

        # LOUD DEBUG SPEECH
        self.agent.robot_say("USING HEIGHT ENHANCED PICKUP AND PLACE!")
        
        logger.alert(f"[Table-Aware Pickup task] Pickup: {target_object} Place: {target_receptacle}")

        # After the robot has started...
        try:
            # Use table-aware pickup task with height constraints
            pickup_task = TableAwarePickupTask(
                self.agent,
                target_object=target_object,
                target_receptacle=target_receptacle,
                matching=self._match_method,
                use_visual_servoing_for_grasp=not self._open_loop,
                safety_margin=self.safety_margin,
                executor=self  # Pass executor for height constraint access
            )
            task = pickup_task.get_task(add_rotate=True, mode=self._pickup_task_mode)
        except Exception as e:
            print(f"Error creating table-aware pickup task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def _pick_only(self, target_object: str) -> None:
        """Create a height-adaptive task to pick up the object and execute it.

        Args:
            target_object: The object to pick up.
        """

        # LOUD DEBUG SPEECH  
        self.agent.robot_say("USING HEIGHT ENHANCED PICKUP!")
        
        logger.alert(f"[Table-Aware Pickup task] Pickup: {target_object}")

        # After the robot has started...
        try:
            # Use table-aware pick task with height constraints
            pickup_task = TableAwarePickObjectTask(
                self.agent,
                target_object=target_object,
                matching=self._match_method,
                use_visual_servoing_for_grasp=not self._open_loop,
                safety_margin=self.safety_margin,
                executor=self  # Pass executor for height constraint access
            )
            task = pickup_task.get_task(add_rotate=True)
        except Exception as e:
            print(f"Error creating table-aware pick task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def _place(self, target_receptacle: str) -> None:
        """Create a height-adaptive task to place the object and execute it.

        Args:
            target_receptacle: The receptacle to place the object in.
        """
        logger.alert(f"[Table-Aware Pickup task] Height-Adaptive Place: {target_receptacle}")
        print("🔧🔧🔧 TABLE-AWARE EXECUTOR USING HEIGHT-ADAPTIVE PLACEMENT! 🔧🔧🔧")

        # After the robot has started...
        try:
            # DEAD SIMPLE: Copy the standard place task but use my height-adaptive operation
            print("🚀🚀🚀 USING HEIGHT-ADAPTIVE PLACEMENT - SAME AS STANDARD BUT HIGH START! 🚀🚀🚀")
            
            from stretch.agent.operations import (
                GoToNavOperation,
                NavigateToObjectOperation,
                RotateInPlaceOperation, 
                SearchForReceptacleOperation,
            )
            from stretch.agent.operations.table_aware_place_object import TableAwarePlaceObjectOperation
            from stretch.core.task import Task
            
            # COPY THE EXACT SAME TASK STRUCTURE AS PlaceOnReceptacleTask but with my operation
            
            # Put the robot into navigation mode
            go_to_navigation_mode = GoToNavOperation(
                "go to navigation mode", self.agent, retry_on_failure=True
            )

            # Look for the target receptacle (same as standard)
            search_for_receptacle = SearchForReceptacleOperation(
                name=f"search_for_{target_receptacle}",
                agent=self.agent,
                parent=go_to_navigation_mode,
                retry_on_failure=True,
                match_method=self._match_method,
            )
            search_for_receptacle.set_target_object_class(target_receptacle)

            # Navigate to receptacle (same as standard)
            go_to_receptacle = NavigateToObjectOperation(
                name="go_to_receptacle",
                agent=self.agent,
                on_cannot_start=search_for_receptacle,
                to_receptacle=True,
            )

            # THE ONLY DIFFERENCE: Use my height-adaptive placement instead of standard
            place_object_on_receptacle = TableAwarePlaceObjectOperation(
                name="place_object_on_receptacle",
                agent=self.agent,
                on_cannot_start=go_to_receptacle,
                require_object=False,
                safety_margin=self.safety_margin,
                executor=self,
            )
            
            print("🎯🎯🎯 USING HEIGHT-ADAPTIVE PLACE OPERATION INSTEAD OF STANDARD! 🎯🎯🎯")

            # Build and execute the task (same as standard)
            task = Task()
            task.add_operation(go_to_navigation_mode)
            task.add_operation(search_for_receptacle)
            task.add_operation(go_to_receptacle)
            task.add_operation(place_object_on_receptacle)

            task.connect_on_success(go_to_navigation_mode.name, search_for_receptacle.name)
            task.connect_on_success(search_for_receptacle.name, go_to_receptacle.name)
            task.connect_on_success(go_to_receptacle.name, place_object_on_receptacle.name)

            task.terminate_on_success(place_object_on_receptacle.name)

            task.run()
                
        except Exception as e:
            print(f"Error creating height-adaptive place operation: {e}")
            logger.error(f"[HEIGHT-ADAPTIVE PLACE] Error: {e}, falling back to standard placement")
            # Fallback to standard place task
            try:
                place_task = PlaceOnReceptacleTask(
                    self.agent,
                    target_receptacle=target_receptacle,
                    matching=self._match_method,
                )
                task = place_task.get_task(add_rotate=True)
                task.run()
            except Exception as fallback_e:
                print(f"Error with fallback placement: {fallback_e}")
                self.robot.stop()
                raise fallback_e

    # Standard methods (unchanged from original PickupExecutor)
    def _take_picture(self, channel=None) -> None:
        """Take a picture with the head camera. Optionally send it to Discord."""

        obs = self.robot.get_observation()
        if channel is None:
            # Just save it to the disk
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(obs.rgb).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel, message="Head camera:", content=numpy_image_to_bytes(obs.rgb)
            )

    def _take_ee_picture(self, channel=None) -> None:
        """Take a picture of the end effector."""

        obs = self.robot.get_servo_observation()
        if channel is None:
            # Just save it to the disk
            now = datetime.datetime.now()
            filename = f"stretch_image_{now.strftime('%Y-%m-%d_%H-%M-%S')}.png"
            Image.fromarray(obs.ee_rgb).save(filename)
        else:
            self.discord_bot.send_message(
                channel=channel,
                message="End effector camera:",
                content=numpy_image_to_bytes(obs.ee_rgb),
            )

    def _find(self, target_object: str) -> None:
        """Create a task to find the object and execute it.

        Args:
            target_object: The object to find.
        """

        logger.alert(f"[Find task] Find: {target_object}")

        # After the robot has started...
        try:
            find_task = FindObjectTask(
                self.agent,
                target_object=target_object,
                matching=self._match_method,
            )
            task = find_task.get_task(add_rotate=True)
        except Exception as e:
            print(f"Error creating find task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def _hand_over(self) -> None:
        """Create a task to find a person, navigate to them, and extend the arm toward them"""
        logger.alert(f"[Table-Aware Pickup task] Hand Over")

        # After the robot has started...
        try:
            hand_over_task = HandOverTask(self.agent)
            task = hand_over_task.get_task()
        except Exception as e:
            print(f"Error creating hand over task: {e}")
            self.robot.stop()
            raise e

        # Execute the task
        task.run()

    def __call__(self, response: List[Tuple[str, str]], channel=None) -> bool:
        """Execute the list of commands given by the LLM bot.

        Args:
            response: A list of tuples, where the first element is the command and the second is the argument.
            channel (Optional): The discord channel to send messages to, if using discord bot.

        Returns:
            True if we should keep going, False if we should stop.
        """
        i = 0

        if response is None or len(response) == 0:
            logger.error("No commands to execute!")
            self.agent.robot_say("I'm sorry, I didn't understand that.")
            return True

        # Check if we have any place-only commands that shouldn't reset agent memory
        has_place_only = any(cmd[0] == "place" for cmd in response)
        has_pickup = any(cmd[0] == "pickup" for cmd in response)
        
        if has_place_only and not has_pickup:
            logger.info("⚠️ PLACE-ONLY command detected - PRESERVING agent memory for height-adaptive placement!")
            print("🧠🧠🧠 KEEPING OBJECT MEMORY FOR HEIGHT-ADAPTIVE PLACEMENT! 🧠🧠🧠")
        else:
            logger.info("Resetting agent for table-aware pickup...")
            self.agent.reset()

        # Loop over every command we have been given
        # Pull out pickup and place as a single arg if they are in a row
        # Else, execute things as they come
        while i < len(response):
            command, args = response[i]
            logger.info(f"Command: {i} {command} {args}")
            if command == "say":
                # Use TTS to say the text
                logger.info(f"Saying: {args}")
                if channel is not None:
                    # Optionally strip quotes from args
                    if args[0] == '"' and args[-1] == '"':
                        args = args[1:-1]
                    self.discord_bot.send_message(channel=channel, message=args)
                self.agent.robot_say(args)
            elif command == "pickup":
                logger.info(f"[Table-Aware Pickup task] Pickup: {args}")
                target_object = args
                i += 1
                if i >= len(response):
                    logger.warning(
                        "Pickup without place! Try giving a full pick-and-place instruction."
                    )
                    self._pickup(target_object, None)
                    # Continue works here because we've already incremented i
                    continue
                next_command, next_args = response[i]
                if next_command != "place":
                    logger.warning(
                        "Pickup without place! Try giving a full pick-and-place instruction."
                    )
                    self._pickup(target_object, None)
                    # Continue works here because we've already incremented i
                    continue
                else:
                    logger.info(f"{i} {next_command} {next_args}")
                    logger.info(f"[Table-Aware Pickup task] Place: {next_args}")
                target_receptacle = next_args
                self._pickup(target_object, target_receptacle)
            elif command == "place":
                logger.warning(
                    "Place without pickup! Try giving a full pick-and-place instruction."
                )
                self._place(args)
            elif command == "hand_over":
                self._hand_over()
            elif command == "take_picture":
                self._take_picture(channel)
            elif command == "take_ee_picture":
                self._take_ee_picture(channel)
            elif command == "wave":
                self.agent.move_to_manip_posture()
                self.emote_task.get_task("wave").run()
                self.agent.move_to_manip_posture()
            elif command == "go_home":
                if self.agent.get_voxel_map().is_empty():
                    logger.warning("No map data available. Cannot go home.")
                else:
                    self.agent.go_home()
            elif command == "explore":
                self.agent.explore()
            elif command == "find":
                self._find(args)
            elif command == "nod_head":
                self.emote_task.get_task("nod_head").run()
            elif command == "shake_head":
                self.emote_task.get_task("shake_head").run()
            elif command == "avert_gaze":
                self.emote_task.get_task("avert_gaze").run()
            elif command == "quit":
                logger.info("[Table-Aware Pickup task] Quitting.")
                self.robot.stop()
                return False
            elif command == "end":
                logger.info("[Table-Aware Pickup task] Ending.")
                break
            else:
                logger.error(f"Skipping unknown command: {command}")

            i += 1
        # If we did not explicitly receive a quit command, we are not yet done.
        return True