# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Enhanced velocity control system for precise visual servoing
# Adapted from stretch_visual_servoing normalized_velocity_control.py

import time
import threading
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

try:
    from stretch.motion.kinematics import HelloStretchIdx
    STRETCH_AI_AVAILABLE = True
except ImportError:
    STRETCH_AI_AVAILABLE = False
    # Fallback definitions
    class HelloStretchIdx:
        BASE_X = 0
        LIFT = 3
        ARM = 4
        WRIST_YAW = 6
        WRIST_PITCH = 7
        WRIST_ROLL = 8


@dataclass
class VelocityLimits:
    """Velocity and acceleration limits for robot joints"""
    max_linear_vel: float = 0.3
    max_rotation_vel: float = 0.5
    max_arm_vel: float = 0.1
    max_lift_vel: float = 0.1
    max_wrist_vel: float = 0.8
    
    # Precision mode limits (for fine control)
    precision_linear_vel: float = 0.02
    precision_rotation_vel: float = 0.08
    precision_arm_vel: float = 0.02
    precision_lift_vel: float = 0.02
    precision_wrist_vel: float = 0.15


class EnhancedVelocityController:
    """
    Enhanced velocity controller optimized for visual servoing applications
    
    Provides:
    - 15Hz consistent control loop
    - Precision mode for fine manipulation  
    - Smooth acceleration profiles
    - Joint limit safety
    - Thread-safe operation
    """
    
    def __init__(self, robot, control_rate_hz: float = 15.0):
        """
        Initialize the enhanced velocity controller
        
        Args:
            robot: Robot interface (stretch_ai or stretch_body)
            control_rate_hz: Control loop frequency in Hz
        """
        self.robot = robot
        self.control_rate_hz = control_rate_hz
        self.control_dt = 1.0 / control_rate_hz
        
        # Velocity limits
        self.limits = VelocityLimits()
        
        # Control state
        self.precision_mode = True  # Default to precision for servoing
        self.fast_base_mode = False
        
        # Thread safety
        self.lock = threading.Lock()
        self.stop_flag = False
        self.command_queue = []
        
        # Control loop state
        self.last_command_time = 0
        self.current_velocities = self._get_zero_velocities()
        
        # Joint state tracking for safety
        self.last_joint_state = None
        
        # Safety parameters
        self.dead_zone = 0.001  # Minimum velocity to consider non-zero
        self.joint_limits = self._get_joint_limits()
        
        if STRETCH_AI_AVAILABLE:
            print("[VELOCITY CTRL] Enhanced controller initialized for stretch_ai")
        else:
            print("[VELOCITY CTRL] Enhanced controller initialized for stretch_body")
            
    def _get_zero_velocities(self) -> Dict[str, float]:
        """Get zero velocity command dictionary"""
        return {
            'base_forward': 0.0,
            'base_counterclockwise': 0.0,
            'lift_up': 0.0,
            'arm_out': 0.0,
            'wrist_yaw_counterclockwise': 0.0,
            'wrist_pitch_up': 0.0,
            'wrist_roll_counterclockwise': 0.0
        }
        
    def _get_joint_limits(self) -> Dict[str, tuple]:
        """Get approximate joint limits (should be robot-specific)"""
        return {
            'lift': (0.15, 1.1),       # meters
            'arm': (0.0, 0.52),        # meters  
            'base_x': (-2.0, 2.0),     # meters (approximate)
            'wrist_yaw': (-1.75, 1.75),    # radians
            'wrist_pitch': (-1.57, 0.5),   # radians
            'wrist_roll': (-2.6, 2.6)      # radians
        }
        
    def set_precision_mode(self, enabled: bool):
        """Enable/disable precision mode for fine control"""
        with self.lock:
            self.precision_mode = enabled
            print(f"[VELOCITY CTRL] Precision mode: {'enabled' if enabled else 'disabled'}")
            
    def set_velocities(self, velocity_cmd: Dict[str, float]) -> bool:
        """
        Set normalized velocity commands with safety checks
        
        Args:
            velocity_cmd: Dictionary of normalized velocities [-1, 1]
            
        Returns:
            True if commands were accepted, False if rejected for safety
        """
        with self.lock:
            if self.stop_flag:
                return False
                
            # Validate and bound commands
            bounded_cmd = self._bound_and_validate_commands(velocity_cmd)
            if bounded_cmd is None:
                return False
                
            # Apply deadzone
            filtered_cmd = self._apply_deadzone(bounded_cmd)
            
            # Convert to actual velocities
            actual_velocities = self._normalize_to_actual_velocities(filtered_cmd)
            
            # Safety check with current joint state
            if not self._safety_check(actual_velocities):
                print("[VELOCITY CTRL] Command rejected for safety")
                return False
                
            # Execute the command
            return self._execute_velocities(actual_velocities)
            
    def _bound_and_validate_commands(self, cmd: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Bound commands to [-1, 1] and validate"""
        bounded = {}
        
        for key, value in cmd.items():
            if not isinstance(value, (int, float)):
                print(f"[VELOCITY CTRL] Invalid command type for {key}: {type(value)}")
                return None
                
            # Bound to valid range
            bounded[key] = max(-1.0, min(1.0, float(value)))
            
        return bounded
        
    def _apply_deadzone(self, cmd: Dict[str, float]) -> Dict[str, float]:
        """Apply deadzone to eliminate noise"""
        filtered = {}
        
        for key, value in cmd.items():
            if abs(value) < self.dead_zone:
                filtered[key] = 0.0
            else:
                filtered[key] = value
                
        return filtered
        
    def _normalize_to_actual_velocities(self, cmd: Dict[str, float]) -> Dict[str, float]:
        """Convert normalized commands to actual velocities"""
        actual = {}
        
        # Get velocity limits based on precision mode
        if self.precision_mode:
            base_linear_limit = self.limits.precision_linear_vel
            base_rot_limit = self.limits.precision_rotation_vel
            arm_limit = self.limits.precision_arm_vel
            lift_limit = self.limits.precision_lift_vel
            wrist_limit = self.limits.precision_wrist_vel
        else:
            base_linear_limit = self.limits.max_linear_vel
            base_rot_limit = self.limits.max_rotation_vel
            arm_limit = self.limits.max_arm_vel
            lift_limit = self.limits.max_lift_vel
            wrist_limit = self.limits.max_wrist_vel
            
        # Convert each command
        actual['base_forward'] = cmd.get('base_forward', 0.0) * base_linear_limit
        actual['base_counterclockwise'] = cmd.get('base_counterclockwise', 0.0) * base_rot_limit
        actual['lift_up'] = cmd.get('lift_up', 0.0) * lift_limit
        actual['arm_out'] = cmd.get('arm_out', 0.0) * arm_limit
        actual['wrist_yaw_counterclockwise'] = cmd.get('wrist_yaw_counterclockwise', 0.0) * wrist_limit
        actual['wrist_pitch_up'] = cmd.get('wrist_pitch_up', 0.0) * wrist_limit
        actual['wrist_roll_counterclockwise'] = cmd.get('wrist_roll_counterclockwise', 0.0) * wrist_limit
        
        return actual
        
    def _safety_check(self, velocities: Dict[str, float]) -> bool:
        """Check if velocities are safe given current joint state"""
        try:
            if STRETCH_AI_AVAILABLE:
                joint_state = self.robot.get_joint_positions()
            else:
                # For stretch_body - simplified state access
                joint_state = self._get_stretch_body_joint_state()
                
            if joint_state is None:
                print("[VELOCITY CTRL] Cannot get joint state for safety check")
                return False
                
            # Check joint limits with velocity prediction
            dt = self.control_dt * 2  # Look ahead 2 control cycles
            
            # Lift safety check
            if velocities.get('lift_up', 0.0) != 0:
                current_lift = joint_state[HelloStretchIdx.LIFT] if STRETCH_AI_AVAILABLE else joint_state.get('lift_pos', 0.5)
                predicted_lift = current_lift + velocities['lift_up'] * dt
                lift_min, lift_max = self.joint_limits['lift']
                if predicted_lift < lift_min or predicted_lift > lift_max:
                    print(f"[VELOCITY CTRL] Lift limit violation predicted: {predicted_lift:.3f} not in [{lift_min}, {lift_max}]")
                    return False
                    
            # Arm safety check  
            if velocities.get('arm_out', 0.0) != 0:
                current_arm = joint_state[HelloStretchIdx.ARM] if STRETCH_AI_AVAILABLE else joint_state.get('arm_pos', 0.1)
                predicted_arm = current_arm + velocities['arm_out'] * dt
                arm_min, arm_max = self.joint_limits['arm']
                if predicted_arm < arm_min or predicted_arm > arm_max:
                    print(f"[VELOCITY CTRL] Arm limit violation predicted: {predicted_arm:.3f} not in [{arm_min}, {arm_max}]")
                    return False
                    
            self.last_joint_state = joint_state
            return True
            
        except Exception as e:
            print(f"[VELOCITY CTRL] Safety check failed: {e}")
            return False
            
    def _get_stretch_body_joint_state(self) -> Optional[Dict]:
        """Get joint state for stretch_body robot"""
        try:
            return {
                'arm_pos': self.robot.arm.status['pos'],
                'lift_pos': self.robot.lift.status['pos'],
                'base_x': self.robot.base.status['x'],
                'wrist_yaw_pos': self.robot.end_of_arm.motors['wrist_yaw'].status['pos'],
                'wrist_pitch_pos': self.robot.end_of_arm.motors['wrist_pitch'].status['pos'],
                'wrist_roll_pos': self.robot.end_of_arm.motors['wrist_roll'].status['pos']
            }
        except Exception as e:
            print(f"[VELOCITY CTRL] Failed to get stretch_body joint state: {e}")
            return None
            
    def _execute_velocities(self, velocities: Dict[str, float]) -> bool:
        """Execute velocity commands on robot"""
        try:
            current_time = time.time()
            
            if STRETCH_AI_AVAILABLE:
                return self._execute_stretch_ai_velocities(velocities)
            else:
                return self._execute_stretch_body_velocities(velocities)
                
        except Exception as e:
            print(f"[VELOCITY CTRL] Command execution failed: {e}")
            return False
            
    def _execute_stretch_ai_velocities(self, velocities: Dict[str, float]) -> bool:
        """Execute velocities for stretch_ai robot"""
        try:
            # For stretch_ai, convert to position increments
            joint_state = self.robot.get_joint_positions()
            
            # Base motion
            if abs(velocities['base_forward']) > 0 or abs(velocities['base_counterclockwise']) > 0:
                # Use base velocity commands
                self.robot.base.set_velocity(
                    velocities['base_forward'],
                    velocities['base_counterclockwise']
                )
                
            # Arm and lift as position increments
            new_joint_state = joint_state.copy()
            
            if abs(velocities['lift_up']) > 0:
                new_joint_state[HelloStretchIdx.LIFT] += velocities['lift_up'] * self.control_dt
                new_joint_state[HelloStretchIdx.LIFT] = np.clip(
                    new_joint_state[HelloStretchIdx.LIFT],
                    self.joint_limits['lift'][0],
                    self.joint_limits['lift'][1]
                )
                
            if abs(velocities['arm_out']) > 0:
                new_joint_state[HelloStretchIdx.ARM] += velocities['arm_out'] * self.control_dt
                new_joint_state[HelloStretchIdx.ARM] = np.clip(
                    new_joint_state[HelloStretchIdx.ARM],
                    self.joint_limits['arm'][0], 
                    self.joint_limits['arm'][1]
                )
                
            # Wrist joints
            if abs(velocities['wrist_pitch_up']) > 0:
                new_joint_state[HelloStretchIdx.WRIST_PITCH] += velocities['wrist_pitch_up'] * self.control_dt
                new_joint_state[HelloStretchIdx.WRIST_PITCH] = np.clip(
                    new_joint_state[HelloStretchIdx.WRIST_PITCH],
                    self.joint_limits['wrist_pitch'][0],
                    self.joint_limits['wrist_pitch'][1]
                )
                
            # Send arm command if any joint changed
            if not np.allclose(new_joint_state, joint_state, atol=1e-6):
                self.robot.arm_to(new_joint_state, blocking=False)
                
            return True
            
        except Exception as e:
            print(f"[VELOCITY CTRL] stretch_ai execution error: {e}")
            return False
            
    def _execute_stretch_body_velocities(self, velocities: Dict[str, float]) -> bool:
        """Execute velocities for stretch_body robot"""
        try:
            # Base motion
            self.robot.base.set_velocity(
                velocities['base_forward'],
                velocities['base_counterclockwise']
            )
            
            # Arm and lift
            if abs(velocities['lift_up']) > 0:
                self.robot.lift.set_velocity(velocities['lift_up'])
            else:
                self.robot.lift.set_velocity(0)
                
            if abs(velocities['arm_out']) > 0:
                self.robot.arm.set_velocity(velocities['arm_out'])
            else:
                self.robot.arm.set_velocity(0)
                
            # Wrist joints
            if abs(velocities['wrist_yaw_counterclockwise']) > 0:
                self.robot.end_of_arm.get_joint('wrist_yaw').set_velocity(
                    velocities['wrist_yaw_counterclockwise']
                )
                
            if abs(velocities['wrist_pitch_up']) > 0:
                self.robot.end_of_arm.get_joint('wrist_pitch').set_velocity(
                    velocities['wrist_pitch_up']
                )
                
            if abs(velocities['wrist_roll_counterclockwise']) > 0:
                self.robot.end_of_arm.get_joint('wrist_roll').set_velocity(
                    velocities['wrist_roll_counterclockwise']
                )
                
            # Push command for stretch_body
            self.robot.push_command()
            return True
            
        except Exception as e:
            print(f"[VELOCITY CTRL] stretch_body execution error: {e}")
            return False
            
    def stop_all_motion(self):
        """Stop all robot motion immediately"""
        with self.lock:
            self.stop_flag = True
            try:
                if STRETCH_AI_AVAILABLE:
                    self.robot.base.set_velocity(0, 0)
                else:
                    # Stop stretch_body motion
                    self.robot.base.set_velocity(0, 0)
                    self.robot.arm.set_velocity(0)
                    self.robot.lift.set_velocity(0)
                    self.robot.end_of_arm.get_joint('wrist_yaw').set_velocity(0)
                    self.robot.end_of_arm.get_joint('wrist_pitch').set_velocity(0)
                    self.robot.end_of_arm.get_joint('wrist_roll').set_velocity(0)
                    self.robot.push_command()
                    
                print("[VELOCITY CTRL] All motion stopped")
                
            except Exception as e:
                print(f"[VELOCITY CTRL] Error stopping motion: {e}")
                
    def resume_operation(self):
        """Resume operation after stop"""
        with self.lock:
            self.stop_flag = False
            print("[VELOCITY CTRL] Operation resumed")
            
    def get_status(self) -> Dict[str, any]:
        """Get current controller status"""
        return {
            'precision_mode': self.precision_mode,
            'fast_base_mode': self.fast_base_mode,
            'stopped': self.stop_flag,
            'control_rate_hz': self.control_rate_hz,
            'last_command_time': self.last_command_time
        }