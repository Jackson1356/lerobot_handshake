"""
Handshake detection module for human-robot interaction.

This module provides robust handshake gesture detection using MediaPipe pose estimation
with enhanced criteria to prevent false positives (e.g., when touching face).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import mediapipe as mp
    import cv2
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class ImprovedHandshakeDetector:
    """
    Enhanced handshake detection with face-touch prevention.
    
    Uses 8 criteria for robust detection:
    1. Hand Extended (laterally from body)
    2. Correct Height (around shoulder level)  
    3. Arm Extended (elbow angle > 90°)
    4. Forward Position (hand toward camera)
    5. Not Touching Face (distance from nose)
    6. Outward Direction (forearm not pointing toward face)
    7. Lateral Extension (hand away from body centerline)
    8. Not At Face Level (hand not at exact face height)
    """
    
    def __init__(self, detection_confidence: float = 0.7, confidence_threshold: float = 0.8):
        """
        Initialize handshake detector.
        
        Args:
            detection_confidence: MediaPipe detection confidence threshold
            confidence_threshold: Minimum confidence required for handshake detection
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe opencv-python")
        
        self.detection_confidence = detection_confidence
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
            model_complexity=1
        )
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=detection_confidence,
            max_num_hands=2
        )
        
        # Detection history for stability
        self.gesture_history = []
        self.history_length = 5
        
        # Colors for visualization
        self.COLOR_READY = (0, 255, 0)      # Green
        self.COLOR_NOT_READY = (0, 0, 255)  # Red
        self.COLOR_PARTIAL = (0, 255, 255)  # Yellow
        self.COLOR_TEXT = (255, 255, 255)   # White
        
    def detect_handshake_gesture(self, frame: np.ndarray, visualize: bool = True) -> Dict[str, Any]:
        """
        Detect handshake gesture with enhanced criteria.
        
        Args:
            frame: Input image as numpy array (BGR format)
            visualize: Whether to draw pose annotations on frame
            
        Returns:
            Dictionary containing:
            - 'ready': bool, whether handshake gesture detected
            - 'confidence': float, detection confidence (0-1)
            - 'criteria_met': list, which criteria are satisfied
            - 'hand_position': tuple, (x, y) of extended hand in pixels
            - 'annotated_frame': np.ndarray, frame with annotations (if visualize=True)
            - 'arm_analysis': dict, detailed analysis of arm pose
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = frame.copy() if visualize else None
        
        # Process pose and hands
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # Initialize result
        result = {
            'ready': False,
            'confidence': 0.0,
            'criteria_met': [],
            'hand_position': None,
            'annotated_frame': annotated_frame,
            'arm_analysis': None
        }
        
        # Analyze pose if detected
        if pose_results.pose_landmarks:
            ready, confidence, hand_pos, criteria_met, arm_analysis = self._analyze_handshake_pose(
                pose_results.pose_landmarks
            )
            result.update({
                'ready': ready,
                'confidence': confidence,
                'hand_position': hand_pos,
                'criteria_met': criteria_met,
                'arm_analysis': arm_analysis
            })
            
            if visualize:
                # Draw pose with color based on readiness
                pose_color = self.COLOR_READY if ready else (
                    self.COLOR_PARTIAL if confidence > 0.5 else self.COLOR_NOT_READY
                )
                self._draw_pose_with_color(annotated_frame, pose_results.pose_landmarks, pose_color)
        
        # Draw hand landmarks
        if visualize and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Add gesture history for stability
        self.gesture_history.append(result['ready'])
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
        
        # Stable detection requires consistency over multiple frames
        if len(self.gesture_history) >= 3:
            stable_ready = sum(self.gesture_history[-3:]) >= 2
            result['ready'] = stable_ready
        
        # Add visualization overlay
        if visualize:
            self._add_detection_overlay(annotated_frame, result)
            result['annotated_frame'] = annotated_frame
        
        return result
    
    def _draw_pose_with_color(self, image, landmarks, color):
        """Draw pose landmarks with custom color."""
        landmark_spec = self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        connection_spec = self.mp_drawing.DrawingSpec(color=color, thickness=2)
        
        self.mp_drawing.draw_landmarks(
            image, landmarks, self.mp_pose.POSE_CONNECTIONS, landmark_spec, connection_spec
        )
    
    def _analyze_handshake_pose(self, landmarks) -> Tuple[bool, float, Optional[Tuple[int, int]], list, dict]:
        """Analyze pose with enhanced face proximity checks."""
        # Get landmarks
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Face landmarks for proximity checking
        nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        # Check both arms with face proximity
        right_ready, right_conf, right_pos, right_criteria, right_analysis = self._check_arm_handshake(
            right_shoulder, right_elbow, right_wrist, "right", nose
        )
        left_ready, left_conf, left_pos, left_criteria, left_analysis = self._check_arm_handshake(
            left_shoulder, left_elbow, left_wrist, "left", nose
        )
        
        # Use arm with higher confidence
        if right_conf > left_conf:
            return right_ready, right_conf, right_pos, right_criteria, right_analysis
        else:
            return left_ready, left_conf, left_pos, left_criteria, left_analysis
    
    def _check_arm_handshake(self, shoulder, elbow, wrist, side: str, nose) -> Tuple[bool, float, Tuple[int, int], list, dict]:
        """Enhanced arm check with face proximity and direction analysis."""
        criteria_met = []
        criteria_details = {}
        
        # Convert to numpy
        shoulder_pos = np.array([shoulder.x, shoulder.y, shoulder.z])
        elbow_pos = np.array([elbow.x, elbow.y, elbow.z])
        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])
        nose_pos = np.array([nose.x, nose.y, nose.z])
        
        # Criterion 1: Hand extended sideways
        if side == "right":
            hand_extended = wrist.x > shoulder.x + 0.1
        else:
            hand_extended = wrist.x < shoulder.x - 0.1
        
        if hand_extended:
            criteria_met.append("Hand Extended")
        
        # Criterion 2: Appropriate height
        height_diff = abs(wrist.y - shoulder.y)
        appropriate_height = height_diff < 0.3
        if appropriate_height:
            criteria_met.append("Correct Height")
        
        # Criterion 3: Arm extended (elbow angle)
        upper_arm = shoulder_pos - elbow_pos
        forearm = wrist_pos - elbow_pos
        
        cos_angle = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        elbow_angle_degrees = np.degrees(np.arccos(cos_angle))
        
        arm_extended = elbow_angle_degrees > 90
        if arm_extended:
            criteria_met.append("Arm Extended")
        
        # Criterion 4: Forward position
        forward_position = wrist.z < shoulder.z + 0.1
        if forward_position:
            criteria_met.append("Forward Position")
        
        # Criterion 5: NOT touching face (strict threshold)
        face_distance = np.linalg.norm(wrist_pos - nose_pos)
        not_touching_face = face_distance > 0.25
        if not_touching_face:
            criteria_met.append("Not Touching Face")
        
        # Criterion 6: Forearm pointing outward (strict)
        forearm_direction = wrist_pos - elbow_pos
        elbow_to_nose = nose_pos - elbow_pos
        
        cos_face_angle = np.dot(forearm_direction, elbow_to_nose) / (
            np.linalg.norm(forearm_direction) * np.linalg.norm(elbow_to_nose)
        )
        cos_face_angle = np.clip(cos_face_angle, -1.0, 1.0)
        face_angle_degrees = np.degrees(np.arccos(cos_face_angle))
        
        outward_direction = face_angle_degrees > 90
        if outward_direction:
            criteria_met.append("Outward Direction")
        
        # Criterion 7: Lateral extension from body centerline
        center_x = (shoulder.x + nose.x) / 2
        if side == "right":
            lateral_extension = wrist.x > center_x + 0.15
        else:
            lateral_extension = wrist.x < center_x - 0.15
        
        if lateral_extension:
            criteria_met.append("Lateral Extension")
        
        # Criterion 8: Hand not at face level
        hand_face_height_diff = abs(wrist.y - nose.y)
        not_at_face_level = hand_face_height_diff > 0.1
        if not_at_face_level:
            criteria_met.append("Not At Face Level")
        
        # Store details for debugging
        criteria_details = {
            'face_distance': face_distance,
            'face_angle': face_angle_degrees,
            'elbow_angle': elbow_angle_degrees,
            'lateral_distance': abs(wrist.x - center_x),
            'face_height_diff': hand_face_height_diff
        }
        
        # Enhanced confidence calculation with 8 criteria
        confidence = len(criteria_met) / 8.0
        is_ready = confidence >= self.confidence_threshold and len(criteria_met) >= 6
        
        hand_position = (int(wrist.x * 640), int(wrist.y * 480))
        
        arm_analysis = {
            'side': side,
            'criteria_details': criteria_details,
            'total_criteria_met': len(criteria_met),
            'total_criteria': 8
        }
        
        return is_ready, confidence, hand_position, criteria_met, arm_analysis
    
    def _add_detection_overlay(self, frame, result):
        """Add detection overlay with criteria visualization."""
        height, width = frame.shape[:2]
        
        # Main status
        status_text = "🤝 HANDSHAKE READY!" if result['ready'] else "⏳ Waiting for handshake..."
        status_color = self.COLOR_READY if result['ready'] else self.COLOR_NOT_READY
        
        cv2.putText(frame, status_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        # Confidence
        conf_text = f"Confidence: {result['confidence']:.2f}"
        cv2.putText(frame, conf_text, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)
        
        # Criteria checklist
        y_offset = 120
        cv2.putText(frame, "Detection Criteria:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        
        criteria_list = [
            "Hand Extended", "Correct Height", "Arm Extended", 
            "Forward Position", "Not Touching Face", "Outward Direction",
            "Lateral Extension", "Not At Face Level"
        ]
        
        for i, criterion in enumerate(criteria_list):
            y_pos = y_offset + 30 + (i * 20)
            
            if criterion in result['criteria_met']:
                status = "✓"
                color = self.COLOR_READY
            else:
                status = "✗"
                color = self.COLOR_NOT_READY
            
            text = f"{status} {criterion}"
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def reset_history(self):
        """Reset gesture detection history."""
        self.gesture_history = []
    
    def is_handshake_ready(self, frame: np.ndarray) -> bool:
        """
        Simple boolean check for handshake readiness.
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            True if handshake gesture is detected, False otherwise
        """
        result = self.detect_handshake_gesture(frame, visualize=False)
        return result['ready']
    
    def get_handshake_confidence(self, frame: np.ndarray) -> float:
        """
        Get handshake detection confidence.
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            Confidence score between 0 and 1
        """
        result = self.detect_handshake_gesture(frame, visualize=False)
        return result['confidence'] 