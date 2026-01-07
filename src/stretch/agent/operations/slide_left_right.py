# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import time
from stretch.agent.base import ManagedOperation


class SlideLeftRightOperation(ManagedOperation):
    """Slide the robot left and right to scan a table instead of rotating in place."""

    def __init__(self, *args, table_width=2.0, slide_steps=5, table_height=0.8, 
                 min_height=0.6, max_height=1.2, **kwargs):
        """Initialize the slide operation.
        
        Args:
            table_width (float): Total width to scan across table (meters)
            slide_steps (int): Number of scan positions
            table_height (float): Height to maintain while scanning (meters)
            min_height (float): Minimum allowed height to prevent hitting table (meters)
            max_height (float): Maximum allowed height for safety (meters)
        """
        super().__init__(*args, **kwargs)
        self.table_width = table_width
        self.slide_steps = slide_steps
        self.table_height = max(min_height, min(table_height, max_height))  # Clamp height
        self.min_height = min_height
        self.max_height = max_height
        self._successful = False
        
        # Log safety constraints
        if table_height != self.table_height:
            print(f"[TABLE_SCAN] WARNING: Requested height {table_height}m clamped to {self.table_height}m")
            print(f"[TABLE_SCAN] Height constraints: min={min_height}m, max={max_height}m")

    def can_start(self) -> bool:
        self.attempt(f"Table scan: {self.slide_steps} positions across {self.table_width}m at height {self.table_height}m")
        return True

    def run(self) -> None:
        """Slide left to right across table to scan for objects."""
        self.intro(f"Starting table scan: {self.slide_steps} positions across {self.table_width}m")
        self._successful = False
        
        try:
            # Set up robot for table scanning
            print(f"[TABLE_SCAN] Setting up robot at table height {self.table_height}m")
            self._setup_for_table_scan()
            
            # Calculate slide distance per step
            step_distance = self.table_width / (self.slide_steps - 1) if self.slide_steps > 1 else 0
            print(f"[TABLE_SCAN] Step distance: {step_distance:.2f}m")
            
            # Move to leftmost position
            print(f"[TABLE_SCAN] Moving to leftmost position...")
            leftmost_distance = self.table_width / 2
            self._slide_left(leftmost_distance)
            
            # Scan from left to right
            for step in range(self.slide_steps):
                print(f"[TABLE_SCAN] === SCAN POSITION {step + 1}/{self.slide_steps} ===")
                
                # Take observation at current position
                obs = self.robot.get_observation()
                print(f"[TABLE_SCAN] Position {step + 1}: GPS = {obs.gps}")
                
                # Let the agent process this observation
                self.agent.update(
                    obs=obs,
                    update_map=True,
                    compute_new_frontier=True,
                )
                
                # Move to next position (except for last step)
                if step < self.slide_steps - 1:
                    print(f"[TABLE_SCAN] Moving to next scan position...")
                    self._slide_right(step_distance)
                    time.sleep(0.5)  # Brief pause for stability
            
            print(f"[TABLE_SCAN] Scan complete! Processed {self.slide_steps} positions")
            self._successful = True
            
        except Exception as e:
            self.error(f"Table scan failed: {e}")
            self._successful = False

    def _setup_for_table_scan(self):
        """Set up robot for table scanning at specified height."""
        print(f"[TABLE_SCAN] Setting up robot for table scan...")
        
        # Ensure in navigation mode for base movements
        self.robot.move_to_nav_posture()
        time.sleep(0.5)
        
        # Switch to manipulation mode to set height
        print(f"[TABLE_SCAN] Setting table height to {self.table_height}m...")
        self.robot.move_to_manip_posture()
        self.robot.switch_to_manipulation_mode()
        
        # Set lift height for table work
        # [x, lift_height, arm_extension, wrist_yaw, wrist_pitch, wrist_roll]
        self.robot.arm_to([0, self.table_height, 0, 0, 0, 0], blocking=True)
        print(f"[TABLE_SCAN] Robot positioned at table height {self.table_height}m")
        
        # Switch back to navigation mode for base movements
        self.robot.move_to_nav_posture()
        time.sleep(0.5)

    def _slide_left(self, distance):
        """Slide robot left using relative movement."""
        print(f"[TABLE_SCAN] Sliding LEFT {distance:.2f}m")
        try:
            # Use relative movement: [x, y, theta] = [0, -distance, 0] for left slide
            self.robot.move_base_to([0, -distance, 0], relative=True, blocking=True, timeout=15.0)
            time.sleep(0.2)  # Brief pause for stability
        except Exception as e:
            print(f"[TABLE_SCAN] WARNING: Left slide failed: {e}")

    def _slide_right(self, distance):
        """Slide robot right using relative movement."""
        print(f"[TABLE_SCAN] Sliding RIGHT {distance:.2f}m")
        try:
            # Use relative movement: [x, y, theta] = [0, +distance, 0] for right slide
            self.robot.move_base_to([0, distance, 0], relative=True, blocking=True, timeout=15.0)
            time.sleep(0.2)  # Brief pause for stability
        except Exception as e:
            print(f"[TABLE_SCAN] WARNING: Right slide failed: {e}")

    def was_successful(self) -> bool:
        return self._successful