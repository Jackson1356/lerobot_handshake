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
    
    # Create a simple config for testing using the parser
    parser.set_defaults(
        dataset_repo_id="your-username/handshake_dataset",  # Replace with actual dataset
        dataset_root=None,
        dataset_episodes=None,
        dataset_revision=None,
        dataset_image_transforms_enable=False,
        dataset_use_imagenet_stats=False,
        dataset_video_backend=None,
        policy_type="act",
        policy_device="cpu",
        policy_use_amp=False,
        policy_n_obs_steps=1,
        policy_chunk_size=100,
        policy_push_to_hub=False,
        optimizer_type="adamw",
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-4,
        optimizer_grad_clip_norm=1.0,
        scheduler_type="cosine",
        scheduler_warmup_steps=100,
        batch_size=2,
        num_workers=0,
        steps=100,
        log_freq=10,
        save_freq=100,
        save_checkpoint=True,
        resume=False,
        checkpoint_path=None,
        output_dir="./output",
        seed=42,
        eval_freq=0,
        eval_batch_size=1,
        eval_use_async_envs=False,
        eval_n_episodes=1,
        env=None,
        wandb_enable=False,
        wandb_project=None,
    )
    
    # Create config object using the parser
    cfg = parser.parse_args([])
    cfg = TrainPipelineConfig.from_args(cfg)
    
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