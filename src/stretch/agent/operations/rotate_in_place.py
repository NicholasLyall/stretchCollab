# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from stretch.agent.base import ManagedOperation


class RotateInPlaceOperation(ManagedOperation):
    """Rotate the robot in place. Number of steps is determined by parameters file."""

    def can_start(self) -> bool:
        self.attempt(f"Rotating for {self.parameters['agent']['in_place_rotation_steps']} steps.")
        return True

    def run(self) -> None:
        steps = self.parameters["agent"]["in_place_rotation_steps"]
        self.intro(f"rotating for {steps} steps.")
        self._successful = False
        try:
            print(f"[ROTATE] Starting rotation with {steps} steps and increased timeout")
            self.agent.rotate_in_place(
                steps=steps,
                visualize=False,
                verbose=True,  # Enable verbose for debugging
            )
            self._successful = True
            print(f"[ROTATE] Successfully completed {steps} rotation steps")
        except Exception as e:
            self.error(f"Rotation failed: {e}")
            self._successful = False

    def was_successful(self) -> bool:
        return self._successful
