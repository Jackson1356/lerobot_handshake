#!/usr/bin/env python

"""
Real-time handshake detection test script - IMPROVED VERSION.
Fixes false positives when touching face by adding:
- Face proximity detection
- Forearm direction analysis
"""

import argparse
import time
from typing import Dict, Any
import numpy as np

# Try to import MediaPipe and OpenCV
try:
    import mediapipe as mp
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
    print(f"✅ MediaPipe available")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("❌ Install with: pip install mediapipe opencv-python")
    exit(1)


class ImprovedHandshakeDetector:
    """Enhanced handshake detection with face-touch prevention."""
    
    def __init__(self, detection_confidence: float = 0.7):
        self.detection_confidence = detection_confidence
        
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
        
        # Detection history
        self.gesture_history = []
        self.history_length = 5
        
        # Colors
        self.COLOR_READY = (0, 255, 0)      # Green
        self.COLOR_NOT_READY = (0, 0, 255)  # Red
        self.COLOR_PARTIAL = (0, 255, 255)  # Yellow
        self.COLOR_TEXT = (255, 255, 255)   # White
        
    def detect_handshake_gesture(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect handshake with enhanced face-touch prevention."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = frame.copy()
        
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        result = {
            'ready': False,
            'confidence': 0.0,
            'criteria_met': [],
            'hand_position': None,
            'annotated_frame': annotated_frame,
            'arm_analysis': None
        }
        
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
            
            pose_color = self.COLOR_READY if ready else (
                self.COLOR_PARTIAL if confidence > 0.5 else self.COLOR_NOT_READY
            )
            
            self._draw_pose_with_color(annotated_frame, pose_results.pose_landmarks, pose_color)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        # Gesture history for stability
        self.gesture_history.append(result['ready'])
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
        
        if len(self.gesture_history) >= 3:
            stable_ready = sum(self.gesture_history[-3:]) >= 2
            result['ready'] = stable_ready
        
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
    
    def _analyze_handshake_pose(self, landmarks) -> tuple:
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
    
    def _check_arm_handshake(self, shoulder, elbow, wrist, side: str, nose) -> tuple:
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
        
        # IMPROVED Criterion 5: NOT touching face (more strict threshold)
        face_distance = np.linalg.norm(wrist_pos - nose_pos)
        not_touching_face = face_distance > 0.25  # Increased from 0.15 to 0.25
        if not_touching_face:
            criteria_met.append("Not Touching Face")
        
        # IMPROVED Criterion 6: Forearm pointing outward (more strict)
        forearm_direction = wrist_pos - elbow_pos
        elbow_to_nose = nose_pos - elbow_pos
        
        cos_face_angle = np.dot(forearm_direction, elbow_to_nose) / (
            np.linalg.norm(forearm_direction) * np.linalg.norm(elbow_to_nose)
        )
        cos_face_angle = np.clip(cos_face_angle, -1.0, 1.0)
        face_angle_degrees = np.degrees(np.arccos(cos_face_angle))
        
        outward_direction = face_angle_degrees > 90  # Increased from 60 to 90 degrees
        if outward_direction:
            criteria_met.append("Outward Direction")
        
        # NEW Criterion 7: Wrist should be laterally away from center line
        # For genuine handshake, wrist should be significantly to the side
        center_x = (shoulder.x + nose.x) / 2  # Body center line
        if side == "right":
            lateral_extension = wrist.x > center_x + 0.15
        else:
            lateral_extension = wrist.x < center_x - 0.15
        
        if lateral_extension:
            criteria_met.append("Lateral Extension")
        
        # NEW Criterion 8: Hand height relative to face (should not be at face level)
        hand_face_height_diff = abs(wrist.y - nose.y)
        not_at_face_level = hand_face_height_diff > 0.1  # Hand should not be at exact face level
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
        is_ready = confidence >= 0.8 and len(criteria_met) >= 6  # Need 6/8 criteria (80% confidence)
        
        hand_position = (int(wrist.x * 640), int(wrist.y * 480))
        
        arm_analysis = {
            'side': side,
            'criteria_details': criteria_details,
            'total_criteria_met': len(criteria_met),
            'total_criteria': 8
        }
        
        return is_ready, confidence, hand_position, criteria_met, arm_analysis
    
    def _add_detection_overlay(self, frame, result):
        """Add detection overlay with new criteria."""
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
            y_pos = y_offset + 30 + (i * 20)  # Reduced spacing to fit 8 criteria
            
            if criterion in result['criteria_met']:
                status = "✓"
                color = self.COLOR_READY
            else:
                status = "✗"
                color = self.COLOR_NOT_READY
            
            text = f"{status} {criterion}"
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Instructions
        instructions = ["Press 'q' to quit", "Press 's' to save", "Press 'r' to reset"]
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 250, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)


def main():
    parser = argparse.ArgumentParser(description="Improved handshake detection")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--detection_confidence", type=float, default=0.7)
    parser.add_argument("--save_screenshots", action="store_true")
    
    args = parser.parse_args()
    
    print("🤝 Improved Handshake Detection (Face-Touch Prevention)")
    print("=" * 60)
    
    detector = ImprovedHandshakeDetector(args.detection_confidence)
    
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    window_name = "🤝 Improved Handshake Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("✅ Starting improved detection...")
    print("Now prevents false positives when touching face!")
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame = cv2.flip(frame, 1)
            
            detection_result = detector.detect_handshake_gesture(frame)
            
            if detection_result['ready']:
                detection_count += 1
            
            cv2.imshow(window_name, detection_result['annotated_frame'])
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and args.save_screenshots:
                filename = f"improved_handshake_{int(time.time())}.jpg"
                cv2.imwrite(filename, detection_result['annotated_frame'])
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                detector.gesture_history = []
                print("🔄 Reset")
            
            if detection_result['ready'] and detection_count == 1:
                print(f"🎯 Handshake detected! Confidence: {detection_result['confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Stopped")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Done!")


if __name__ == "__main__":
    main() 