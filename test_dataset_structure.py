#!/usr/bin/env python3
"""
Test script to check the dataset structure and see what keys are available.
"""

import torch
from lerobot.common.datasets.factory import make_dataset
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs import parser

def test_dataset_structure():
    """Test the dataset structure to understand what keys are available."""
    
    # Create a simple config for testing
    config_dict = {
        "dataset": {
            "repo_id": "your-username/handshake_dataset",  # Replace with actual dataset
            "root": None,
            "episodes": None,
            "revision": None,
            "image_transforms": {"enable": False},
            "use_imagenet_stats": False,
            "video_backend": None,
        },
        "policy": {
            "type": "act",
            "device": "cpu",
            "use_amp": False,
            "n_obs_steps": 1,
            "chunk_size": 100,
            "push_to_hub": False,
        },
        "optimizer": {
            "type": "adamw",
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 100,
        },
        "batch_size": 2,
        "num_workers": 0,
        "steps": 100,
        "log_freq": 10,
        "save_freq": 100,
        "save_checkpoint": True,
        "resume": False,
        "checkpoint_path": None,
        "output_dir": "./output",
        "seed": 42,
        "eval_freq": 0,
        "eval": {
            "batch_size": 1,
            "use_async_envs": False,
            "n_episodes": 1,
        },
        "env": None,
        "wandb": {
            "enable": False,
            "project": None,
        },
    }
    
    # Create config object
    cfg = TrainPipelineConfig.from_dict(config_dict)
    
    try:
        print("Creating dataset...")
        dataset = make_dataset(cfg)
        
        print(f"Dataset features: {list(dataset.features.keys())}")
        
        # Try to get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            
            # Check for handshake-related keys
            handshake_keys = [k for k in sample.keys() if 'handshake' in k.lower()]
            print(f"Handshake-related keys: {handshake_keys}")
            
            if "observation.handshake" in sample:
                handshake_data = sample["observation.handshake"]
                print(f"Handshake data shape: {handshake_data.shape}")
                print(f"Handshake data type: {type(handshake_data)}")
                print(f"Handshake data sample: {handshake_data}")
            else:
                print("No 'observation.handshake' found in sample")
                
        else:
            print("Dataset is empty")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This is expected if the dataset doesn't exist or has issues")

if __name__ == "__main__":
    test_dataset_structure() 