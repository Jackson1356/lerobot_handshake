#!/usr/bin/env python

"""Simple camera test to diagnose OpenCV window issues."""

import cv2
import time

def test_camera():
    print("🔧 Simple Camera Test")
    print("=" * 30)
    
    # Test OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # Try different camera backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_ANY, "Any available")
    ]
    
    cap = None
    for backend, name in backends:
        print(f"Trying {name}...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"✅ Success with {name}")
            break
        else:
            print(f"❌ Failed with {name}")
            cap.release()
    
    if not cap or not cap.isOpened():
        print("❌ No camera backend worked")
        return
    
    # Test frame reading
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame")
        cap.release()
        return
    
    print(f"✅ Frame size: {frame.shape}")
    
    # Create window
    window_name = "Simple Camera Test - Press 'q' to quit"
    print(f"Creating window: {window_name}")
    
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Force window to foreground (Windows specific)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
    except:
        pass
    
    print("📺 Window should appear now...")
    print("If you don't see it, check:")
    print("1. Alt+Tab to see if window is hidden")
    print("2. Check taskbar for Python/OpenCV window")
    print("3. Try moving mouse to trigger window focus")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Add frame counter and instructions
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "If you see this, OpenCV works!", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Print status every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frame {frame_count}, FPS: {fps:.1f}")
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' pressed - quitting")
            break
        elif key != 255:  # Any other key
            print(f"Key pressed: {key} (char: '{chr(key) if 32 <= key <= 126 else '?'}')")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed!")

if __name__ == "__main__":
    test_camera() 