#!/usr/bin/env python3
"""
Test script to see what make_dataset(cfg) returns vs direct loading.
"""

import torch
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


def test_make_dataset():
    """Test what make_dataset(cfg) returns."""
    print("🧪 TESTING MAKE_DATASET")
    print("=" * 60)
    
    # Create a minimal config for testing
    config_dict = {
        "dataset": {
            "repo_id": "HS_01/handshake_dataset"
        },
        "policy": {
            "type": "act",
            "device": "cpu"
        },
        "batch_size": 8,
        "steps": 100,
        "log_freq": 10,
        "save_freq": 100,
        "eval_freq": 0,
        "save_checkpoint": False,
        "resume": False,
        "seed": None,
        "num_workers": 0,
        "output_dir": "./test_output",
        "wandb": {
            "enable": False
        }
    }
    
    # Create config object
    cfg = TrainPipelineConfig.from_dict(config_dict)
    
    print("Loading dataset with make_dataset(cfg)...")
    try:
        dataset = make_dataset(cfg)
        print(f"✅ Successfully loaded dataset with make_dataset")
        print(f"   Dataset type: {type(dataset)}")
        print(f"   Number of frames: {dataset.num_frames}")
        print(f"   Number of episodes: {dataset.num_episodes}")
        
        # Check features
        print(f"\nDataset features:")
        for key, feature in dataset.features.items():
            print(f"   {key}: {feature}")
        
        # Test a sample
        sample = dataset[0]
        print(f"\nSample keys: {list(sample.keys())}")
        
        if 'observation.handshake' in sample:
            handshake_data = sample['observation.handshake']
            print(f"✅ observation.handshake found: shape={handshake_data.shape}")
            print(f"   Values: {handshake_data.tolist()}")
        else:
            print("❌ observation.handshake NOT found in make_dataset sample")
            
    except Exception as e:
        print(f"❌ Error loading with make_dataset: {e}")
        import traceback
        traceback.print_exc()


def test_direct_vs_make_dataset():
    """Compare direct loading vs make_dataset."""
    print("\n" + "=" * 60)
    print("COMPARING DIRECT vs MAKE_DATASET")
    print("=" * 60)
    
    # Direct loading
    print("Direct loading:")
    try:
        direct_dataset = LeRobotDataset("HS_01/handshake_dataset")
        direct_sample = direct_dataset[0]
        print(f"   Direct sample keys: {list(direct_sample.keys())}")
        if 'observation.handshake' in direct_sample:
            print(f"   Direct handshake: {direct_sample['observation.handshake'].tolist()}")
    except Exception as e:
        print(f"   Direct loading error: {e}")
    
    # Make dataset loading
    print("\nMake dataset loading:")
    try:
        config_dict = {
            "dataset": {"repo_id": "HS_01/handshake_dataset"},
            "policy": {"type": "act", "device": "cpu"},
            "batch_size": 8,
            "steps": 100,
            "log_freq": 10,
            "save_freq": 100,
            "eval_freq": 0,
            "save_checkpoint": False,
            "resume": False,
            "seed": None,
            "num_workers": 0,
            "output_dir": "./test_output",
            "wandb": {"enable": False}
        }
        cfg = TrainPipelineConfig.from_dict(config_dict)
        make_dataset_obj = make_dataset(cfg)
        make_sample = make_dataset_obj[0]
        print(f"   Make dataset sample keys: {list(make_sample.keys())}")
        if 'observation.handshake' in make_sample:
            print(f"   Make dataset handshake: {make_sample['observation.handshake'].tolist()}")
        else:
            print("   ❌ Make dataset missing observation.handshake")
    except Exception as e:
        print(f"   Make dataset error: {e}")


if __name__ == "__main__":
    test_make_dataset()
    test_direct_vs_make_dataset() 