# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from .emote import (
    ApproachOperation,
    AvertGazeOperation,
    NodHeadOperation,
    ShakeHeadOperation,
    TestOperation,
    WaveOperation,
    WithdrawOperation,
)

# from .grasp_closed_loop import ClosedLoopGraspObjectOperation
from .explore import ExploreOperation
from .extend_arm import ExtendArm
from .go_home import GoHomeOperation
from .go_to import GoToOperation
from .grasp_object import GraspObjectOperation
from .grasp_open_loop import OpenLoopGraspObjectOperation
from .navigate import NavigateToObjectOperation
from .open_gripper import OpenGripper
from .place_object import PlaceObjectOperation

# Table-aware operations for height adaptation
try:
    from .table_aware_place_object import TableAwarePlaceObjectOperation
except ImportError:
    # Table-aware place operation not available - will fall back to standard operation
    pass
from .pregrasp import PreGraspObjectOperation

# Enhanced operations for visual servoing integration
try:
    from .enhanced_navigate_to_object import EnhancedNavigateToObjectOperation
    from .enhanced_pregrasp import EnhancedPreGraspObjectOperation
except ImportError:
    # Enhanced operations not available - will fall back to standard operations
    pass
from .rotate_in_place import RotateInPlaceOperation
from .search_for_object import SearchForObjectOnFloorOperation, SearchForReceptacleOperation
# Add SearchForObjectOperation alias for compatibility
SearchForObjectOperation = SearchForObjectOnFloorOperation
from .speak import SpeakOperation
from .switch_mode import GoToNavOperation
from .update import UpdateOperation
from .utility_operations import SetCurrentObjectOperation, SetCurrentReceptacleOperation
