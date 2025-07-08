#!/usr/bin/env python

"""
Test handshake detection with live camera feed.

Example usage:

```bash
# Basic usage with default camera
python -m lerobot.test_handshake

# Specify camera and confidence threshold
python -m lerobot.test_handshake \
    --camera_id=0 \
    --confidence_threshold=0.8

# Custom camera resolution
python -m lerobot.test_handshake \
    --camera_id=1 \
    --confidence_threshold=0.7 \
    --width=1280 \
    --height=720
```

Controls:
- Press 'q' to quit
- Press any other key to see current detection status
"""

import cv2
import time
import logging
from dataclasses import dataclass
from pprint import pformat

import draccus

from lerobot.common.handshake_detection import ImprovedHandshakeDetector
from lerobot.common.utils.utils import init_logging


@dataclass
class TestHandshakeConfig:
    # Camera configuration
    camera_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    
    # Detection configuration
    confidence_threshold: float = 0.8
    detection_confidence: float = 0.7


def test_handshake_detection(cfg: TestHandshakeConfig):
    """Test handshake detection with live camera feed."""
    
    print("🤖 Handshake Detection Test")
    print("=" * 40)
    print(f"📺 Opening camera {cfg.camera_id}...")
    
    # Initialize camera
    cap = cv2.VideoCapture(cfg.camera_id)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {cfg.camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Camera: {actual_width}x{actual_height}")
    
    # Initialize handshake detector
    try:
        detector = ImprovedHandshakeDetector(
            detection_confidence=cfg.detection_confidence,
            confidence_threshold=cfg.confidence_threshold
        )
        print(f"✅ Handshake detector ready! (threshold: {cfg.confidence_threshold})")
        print("🎯 Extend your hand toward the camera!")
        print("\nControls: Press 'q' to quit, any other key for status")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install: pip install mediapipe opencv-python")
        cap.release()
        return
    
    start_time = time.time()
    frame_count = 0
    handshake_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            
            # Detect handshake
            result = detector.detect_handshake_gesture(frame, visualize=True)
            
            # Count detections
            if result['ready']:
                handshake_count += 1
            
            # Add simple status overlay
            display_frame = result.get('annotated_frame', frame)
            
            # Add frame counter
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, display_frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detections: {handshake_count}", 
                       (10, display_frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Handshake Detection Test (Press q to quit)', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key != 255:  # Any other key pressed
                conf = result['confidence']
                criteria = len(result['criteria_met'])
                ready_status = "🤝 READY" if result['ready'] else "⏳ NOT READY"
                print(f"{ready_status} | Confidence: {conf:.2f} | Criteria: {criteria}/8")
                if result['criteria_met']:
                    print(f"  Met: {', '.join(result['criteria_met'])}")
    
    except KeyboardInterrupt:
        print("\n⚡ Interrupted by user")
    
    finally:
        # Cleanup and show results
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Runtime: {elapsed:.1f}s")
        print(f"   Average FPS: {fps_actual:.1f}")
        print(f"   Handshakes detected: {handshake_count}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed!")


@draccus.wrap()
def test_handshake(cfg: TestHandshakeConfig):
    init_logging()
    logging.info(pformat(cfg.__dict__))
    test_handshake_detection(cfg)


if __name__ == "__main__":
    test_handshake() 