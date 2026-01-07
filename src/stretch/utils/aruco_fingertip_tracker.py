# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# Advanced ArUco fingertip tracking for precise visual servoing
# Adapted from stretch_visual_servoing for stretch_ai integration

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class FingertipPose:
    """Represents a fingertip pose with position and orientation"""
    position: np.ndarray  # 3D position in camera frame
    x_axis: np.ndarray    # Fingertip x-axis direction
    y_axis: np.ndarray    # Fingertip y-axis direction  
    z_axis: np.ndarray    # Fingertip z-axis direction
    confidence: float     # Detection confidence [0, 1]
    image_center: Tuple[int, int]  # Center pixel location


class AdvancedArUcoFingertipTracker:
    """
    Advanced ArUco-based fingertip tracking for precise manipulation
    
    Uses ArUco markers on gripper fingertips to provide 6DOF pose tracking
    with sub-centimeter accuracy for visual servoing applications.
    """
    
    def __init__(self, 
                 marker_dict_type=cv2.aruco.DICT_6X6_250,
                 left_marker_id=200,
                 right_marker_id=201):
        """
        Initialize the fingertip tracker
        
        Args:
            marker_dict_type: ArUco dictionary type
            left_marker_id: ID of left fingertip marker
            right_marker_id: ID of right fingertip marker
        """
        # ArUco detection setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(marker_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Optimize detection parameters for gripper markers
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 23
        self.aruco_params.adaptiveThreshWinSizeStep = 10
        self.aruco_params.adaptiveThreshConstant = 7
        self.aruco_params.minMarkerPerimeterRate = 0.03
        self.aruco_params.maxMarkerPerimeterRate = 4.0
        self.aruco_params.polygonalApproxAccuracyRate = 0.03
        self.aruco_params.minCornerDistanceRate = 0.05
        self.aruco_params.minDistanceToBorder = 3
        self.aruco_params.minMarkerDistanceRate = 0.05
        
        # Marker IDs
        self.left_marker_id = left_marker_id
        self.right_marker_id = right_marker_id
        
        # Calibration parameters (should be calibrated for specific robot)
        self.marker_size_m = 0.015  # 15mm markers
        
        # Transform from marker center to fingertip
        # These should be calibrated - using approximate values
        self.marker_to_fingertip_offset = {
            'left': np.array([0.0, 0.0, 0.02]),   # 2cm forward from marker
            'right': np.array([0.0, 0.0, 0.02])
        }
        
        # Detection history for smoothing
        self.detection_history = {
            'left': [],
            'right': []
        }
        self.history_length = 5
        
        # Tracking state
        self.last_detection_time = 0
        self.tracking_lost_threshold = 1.0  # seconds
        
    def detect_fingertips(self, 
                         color_image: np.ndarray,
                         depth_image: np.ndarray,
                         camera_info: Optional[Dict] = None) -> Optional[Dict[str, FingertipPose]]:
        """
        Detect fingertip poses using ArUco markers
        
        Args:
            color_image: RGB image from gripper camera
            depth_image: Corresponding depth image
            camera_info: Camera intrinsic parameters (K matrix, distortion)
            
        Returns:
            Dictionary mapping 'left'/'right' to FingertipPose objects, or None if no detection
        """
        if color_image is None or color_image.size == 0:
            return None
            
        # Convert to grayscale for ArUco detection
        if len(color_image.shape) == 3:
            gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = color_image
            
        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        
        if ids is None or len(ids) == 0:
            return None
            
        fingertips = {}
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [self.left_marker_id, self.right_marker_id]:
                side = 'left' if marker_id == self.left_marker_id else 'right'
                
                # Get marker corners and center
                marker_corners = corners[i][0]
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                
                # Validate marker detection quality
                if not self._validate_marker_detection(marker_corners, color_image.shape):
                    continue
                
                # Estimate pose if camera calibration available
                if camera_info is not None:
                    pose = self._estimate_marker_pose_with_calibration(
                        marker_corners, camera_info, depth_image, center_x, center_y
                    )
                else:
                    pose = self._estimate_marker_pose_from_depth(
                        marker_corners, depth_image, center_x, center_y
                    )
                    
                if pose is not None:
                    # Transform from marker to fingertip
                    fingertip_pose = self._transform_marker_to_fingertip(pose, side)
                    
                    # Add to detection history for smoothing
                    self._add_to_history(side, fingertip_pose, current_time)
                    
                    # Get smoothed pose
                    smoothed_pose = self._get_smoothed_pose(side)
                    if smoothed_pose is not None:
                        fingertips[side] = smoothed_pose
                        
        self.last_detection_time = current_time
        return fingertips if fingertips else None
        
    def _validate_marker_detection(self, corners: np.ndarray, image_shape: Tuple) -> bool:
        """Validate marker detection quality"""
        # Check if marker is too close to image border
        border_margin = 10
        h, w = image_shape[:2]
        
        for corner in corners:
            x, y = corner
            if x < border_margin or x > w - border_margin:
                return False
            if y < border_margin or y > h - border_margin:
                return False
                
        # Check marker size (not too small or too large)
        marker_area = cv2.contourArea(corners)
        min_area, max_area = 100, 10000  # pixel area thresholds
        if marker_area < min_area or marker_area > max_area:
            return False
            
        return True
        
    def _estimate_marker_pose_with_calibration(self,
                                             marker_corners: np.ndarray,
                                             camera_info: Dict,
                                             depth_image: np.ndarray,
                                             center_x: int,
                                             center_y: int) -> Optional[Dict]:
        """Estimate marker pose using camera calibration"""
        try:
            camera_matrix = np.array(camera_info.get('K', np.eye(3))).reshape(3, 3)
            dist_coeffs = np.array(camera_info.get('D', [0, 0, 0, 0, 0]))
            
            # Estimate pose using solvePnP
            object_points = np.array([
                [-self.marker_size_m/2, -self.marker_size_m/2, 0],
                [self.marker_size_m/2, -self.marker_size_m/2, 0],
                [self.marker_size_m/2, self.marker_size_m/2, 0],
                [-self.marker_size_m/2, self.marker_size_m/2, 0]
            ], dtype=np.float32)
            
            success, rvec, tvec = cv2.solvePnP(
                object_points, marker_corners.astype(np.float32),
                camera_matrix, dist_coeffs
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                return {
                    'position': tvec.flatten(),
                    'rotation_matrix': rotation_matrix,
                    'image_center': (center_x, center_y),
                    'confidence': 0.9  # High confidence with calibration
                }
        except Exception as e:
            print(f"[ArUco] Pose estimation failed: {e}")
            
        return None
        
    def _estimate_marker_pose_from_depth(self,
                                       marker_corners: np.ndarray,
                                       depth_image: np.ndarray,
                                       center_x: int,
                                       center_y: int) -> Optional[Dict]:
        """Estimate marker pose using depth image (fallback method)"""
        # Get depth at marker center
        if (center_y >= depth_image.shape[0] or center_x >= depth_image.shape[1] or
            center_y < 0 or center_x < 0):
            return None
            
        center_depth = depth_image[center_y, center_x]
        
        # Validate depth
        if center_depth <= 0.001 or center_depth > 2.0:  # Invalid or too far
            # Try to get depth from surrounding region
            region_size = 5
            y_start = max(0, center_y - region_size)
            y_end = min(depth_image.shape[0], center_y + region_size)
            x_start = max(0, center_x - region_size)
            x_end = min(depth_image.shape[1], center_x + region_size)
            
            depth_region = depth_image[y_start:y_end, x_start:x_end]
            valid_depths = depth_region[depth_region > 0.001]
            
            if len(valid_depths) == 0:
                return None
                
            center_depth = np.median(valid_depths)
            
        # Estimate 3D position (simplified camera model)
        # This assumes camera center at image center - should use actual calibration
        image_center_x = depth_image.shape[1] / 2
        image_center_y = depth_image.shape[0] / 2
        
        # Rough focal length estimate (should use calibration)
        focal_length = 400.0  # pixels
        
        # Convert to 3D coordinates
        x_3d = (center_x - image_center_x) * center_depth / focal_length
        y_3d = (center_y - image_center_y) * center_depth / focal_length
        z_3d = center_depth
        
        # Estimate orientation from marker corners (simplified)
        rotation_matrix = self._estimate_orientation_from_corners(marker_corners)
        
        confidence = 0.6  # Lower confidence without full calibration
        
        return {
            'position': np.array([x_3d, y_3d, z_3d]),
            'rotation_matrix': rotation_matrix,
            'image_center': (center_x, center_y),
            'confidence': confidence
        }
        
    def _estimate_orientation_from_corners(self, corners: np.ndarray) -> np.ndarray:
        """Estimate orientation from marker corner geometry"""
        # Simple orientation estimation based on marker shape
        # This is a fallback when full pose estimation isn't available
        
        # Get marker edges
        edge1 = corners[1] - corners[0]
        edge2 = corners[3] - corners[0]
        
        # Normalize edges
        edge1_norm = edge1 / np.linalg.norm(edge1)
        edge2_norm = edge2 / np.linalg.norm(edge2)
        
        # Create rotation matrix (simplified - assumes marker is roughly frontal)
        x_axis = np.array([edge1_norm[0], edge1_norm[1], 0])
        y_axis = np.array([edge2_norm[0], edge2_norm[1], 0])
        z_axis = np.array([0, 0, 1])
        
        # Normalize and orthogonalize
        x_axis = x_axis / np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        return rotation_matrix
        
    def _transform_marker_to_fingertip(self, marker_pose: Dict, side: str) -> FingertipPose:
        """Transform marker pose to fingertip pose"""
        # Get marker position and orientation
        marker_position = marker_pose['position']
        marker_rotation = marker_pose['rotation_matrix']
        
        # Apply offset from marker to fingertip
        offset = self.marker_to_fingertip_offset[side]
        fingertip_position = marker_position + marker_rotation @ offset
        
        # Extract orientation axes
        x_axis = marker_rotation[:, 0]
        y_axis = marker_rotation[:, 1]
        z_axis = marker_rotation[:, 2]
        
        return FingertipPose(
            position=fingertip_position,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            confidence=marker_pose['confidence'],
            image_center=marker_pose['image_center']
        )
        
    def _add_to_history(self, side: str, pose: FingertipPose, timestamp: float):
        """Add detection to history for smoothing"""
        self.detection_history[side].append({
            'pose': pose,
            'timestamp': timestamp
        })
        
        # Keep only recent history
        if len(self.detection_history[side]) > self.history_length:
            self.detection_history[side].pop(0)
            
    def _get_smoothed_pose(self, side: str) -> Optional[FingertipPose]:
        """Get temporally smoothed pose"""
        history = self.detection_history[side]
        if not history:
            return None
            
        # Simple averaging for smoothing (could use more sophisticated filtering)
        positions = [h['pose'].position for h in history]
        avg_position = np.mean(positions, axis=0)
        
        # Use most recent orientation (orientation is harder to average)
        latest_pose = history[-1]['pose']
        
        return FingertipPose(
            position=avg_position,
            x_axis=latest_pose.x_axis,
            y_axis=latest_pose.y_axis,
            z_axis=latest_pose.z_axis,
            confidence=min(1.0, np.mean([h['pose'].confidence for h in history])),
            image_center=latest_pose.image_center
        )
        
    def get_grasp_center(self, fingertips: Dict[str, FingertipPose]) -> Optional[Tuple[np.ndarray, float]]:
        """
        Compute optimal grasp center from fingertip poses
        
        Args:
            fingertips: Dictionary with 'left' and 'right' FingertipPose objects
            
        Returns:
            Tuple of (grasp_center_3d, confidence) or None
        """
        if 'left' not in fingertips or 'right' not in fingertips:
            return None
            
        left_fingertip = fingertips['left']
        right_fingertip = fingertips['right']
        
        # Compute center between fingertips
        grasp_center = (left_fingertip.position + right_fingertip.position) / 2.0
        
        # Confidence is minimum of both detections
        confidence = min(left_fingertip.confidence, right_fingertip.confidence)
        
        # Adjust confidence based on fingertip separation (should be reasonable)
        separation = np.linalg.norm(left_fingertip.position - right_fingertip.position)
        expected_separation = 0.1  # 10cm expected gripper width
        separation_error = abs(separation - expected_separation) / expected_separation
        separation_confidence = max(0.1, 1.0 - separation_error)
        
        final_confidence = confidence * separation_confidence
        
        return grasp_center, final_confidence
        
    def visualize_detections(self, 
                           image: np.ndarray,
                           fingertips: Dict[str, FingertipPose],
                           draw_axes: bool = True,
                           axis_length: float = 0.02) -> np.ndarray:
        """
        Visualize fingertip detections on image
        
        Args:
            image: Input image to draw on
            fingertips: Detected fingertip poses
            draw_axes: Whether to draw coordinate axes
            axis_length: Length of axes in meters
            
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        
        colors = {'left': (255, 0, 0), 'right': (0, 255, 0)}  # BGR
        
        for side, fingertip in fingertips.items():
            color = colors.get(side, (255, 255, 255))
            center = fingertip.image_center
            
            # Draw fingertip center
            cv2.circle(vis_image, center, 6, color, -1)
            cv2.circle(vis_image, center, 8, (0, 0, 0), 2)
            
            # Draw confidence
            text = f"{side}: {fingertip.confidence:.2f}"
            cv2.putText(vis_image, text, (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
            # Draw coordinate axes if requested
            if draw_axes:
                # This would require camera projection - simplified here
                pass
                
        return vis_image
        
    def is_tracking_valid(self) -> bool:
        """Check if tracking is currently valid"""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        return (current_time - self.last_detection_time) < self.tracking_lost_threshold