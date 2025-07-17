#!/usr/bin/env python3
"""
Test script to verify that delta timestamps are causing the handshake data issue.
"""

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def test_delta_timestamps():
    """Test how delta timestamps affect handshake data."""
    print("🧪 TESTING DELTA TIMESTAMPS EFFECT")
    print("=" * 60)
    
    # Test 1: Direct loading (no delta timestamps)
    print("1. Direct loading (no delta timestamps):")
    try:
        dataset_direct = LeRobotDataset("HS_01/handshake_dataset")
        sample_direct = dataset_direct[0]
        if 'observation.handshake' in sample_direct:
            print(f"   ✅ observation.handshake: {sample_direct['observation.handshake'].tolist()}")
        else:
            print("   ❌ observation.handshake missing")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: With action delta timestamps (like ACT policy)
    print("\n2. With action delta timestamps (like ACT policy):")
    try:
        # ACT policy uses action_delta_indices = [0, 1, 2, ..., 99]
        delta_timestamps = {
            "action": [i / 20.0 for i in range(100)]  # 20 FPS, 100 frames = 5 seconds
        }
        dataset_delta = LeRobotDataset("HS_01/handshake_dataset", delta_timestamps=delta_timestamps)
        sample_delta = dataset_delta[0]
        if 'observation.handshake' in sample_delta:
            print(f"   ✅ observation.handshake: {sample_delta['observation.handshake'].tolist()}")
        else:
            print("   ❌ observation.handshake missing")
        
        # Check action shape
        if 'action' in sample_delta:
            print(f"   Action shape: {sample_delta['action'].shape}")
        else:
            print("   ❌ Action missing")
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Check if handshake data is available in future frames
    print("\n3. Check handshake data availability in future frames:")
    try:
        dataset = LeRobotDataset("HS_01/handshake_dataset")
        
        # Check multiple frames to see if handshake data is consistent
        for i in range(5):
            sample = dataset[i]
            if 'observation.handshake' in sample:
                handshake = sample['observation.handshake']
                print(f"   Frame {i}: {handshake.tolist()}")
            else:
                print(f"   Frame {i}: Missing handshake data")
                
    except Exception as e:
        print(f"   Error: {e}")


def test_act_like_loading():
    """Test loading exactly like ACT policy would."""
    print("\n" + "=" * 60)
    print("TESTING ACT-LIKE LOADING")
    print("=" * 60)
    
    try:
        # Simulate ACT policy delta timestamps
        delta_timestamps = {
            "action": [i / 20.0 for i in range(100)]  # chunk_size=100, fps=20
        }
        
        dataset = LeRobotDataset("HS_01/handshake_dataset", delta_timestamps=delta_timestamps)
        
        # Test a few samples
        for i in range(3):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"   Keys: {list(sample.keys())}")
            
            if 'observation.handshake' in sample:
                handshake = sample['observation.handshake']
                print(f"   observation.handshake: shape={handshake.shape}, values={handshake.tolist()}")
            else:
                print("   ❌ observation.handshake missing")
                
            if 'action' in sample:
                action = sample['action']
                print(f"   action: shape={action.shape}")
            else:
                print("   ❌ action missing")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_delta_timestamps()
    test_act_like_loading() 