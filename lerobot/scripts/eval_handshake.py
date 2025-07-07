#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluate a trained policy on real handshake interactions with humans using SO-101 robot.

This script evaluates policies by having them perform handshakes with real people,
using handshake detection to determine when to start and measure success.

Usage examples:

Evaluate a trained handshake model:
```
python lerobot/scripts/eval_handshake.py \
    --policy.path=outputs/train/handshake_act/checkpoints/010000/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{main: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --eval.n_episodes=10 \
    --eval.episode_time_s=30 \
    --eval.reset_time_s=10 \
    --output_dir=outputs/eval/handshake_results
```

Evaluate with video recording:
```
python lerobot/scripts/eval_handshake.py \
    --policy.path=lerobot/your_trained_handshake_model \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{main: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --eval.n_episodes=5 \
    --eval.save_videos=true \
    --eval.handshake_timeout_s=45
```
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import numpy as np
import torch
from termcolor import colored
from tqdm import trange

from lerobot.common.cameras import CameraConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.common.handshake_detection import ImprovedHandshakeDetector
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.common.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class HandshakeEvalConfig:
    """Configuration for handshake evaluation parameters."""
    # Number of handshake episodes to evaluate
    n_episodes: int = 10
    # Maximum time to wait for person to extend hand (seconds)
    handshake_timeout_s: float = 10.0
    # Time for each handshake interaction (seconds)
    episode_time_s: float = 30.0
    # Reset time between episodes (seconds)
    reset_time_s: float = 15.0
    # Handshake detection confidence threshold (0-1)
    handshake_confidence_threshold: float = 0.8
    # Minimum handshake confidence required for success evaluation
    success_confidence_threshold: float = 0.9
    # Whether to save evaluation videos
    save_videos: bool = False
    # Whether to display camera feed during evaluation
    display_data: bool = True
    # Whether to use voice announcements
    play_sounds: bool = True


@dataclass
class HandshakeEvalPipelineConfig:
    """Complete configuration for handshake evaluation pipeline."""
    robot: RobotConfig
    policy: PreTrainedConfig
    eval: HandshakeEvalConfig = HandshakeEvalConfig()
    output_dir: str = "outputs/eval/handshake"
    device: str = "cpu"
    
    def __post_init__(self):
        # Parse policy path if provided via CLI
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def wait_for_handshake_detection(
    robot: Robot, 
    handshake_detector: ImprovedHandshakeDetector,
    main_camera_name: str,
    timeout_s: float,
    confidence_threshold: float,
    display_data: bool = False
) -> tuple[bool, float]:
    """
    Wait for handshake detection and return whether detected and final confidence.
    
    Returns:
        (detected: bool, confidence: float)
    """
    log_say("Waiting for person to extend their hand for handshake...", True)
    
    start_time = time.perf_counter()
    max_confidence = 0.0
    
    while time.perf_counter() - start_time < timeout_s:
        observation = robot.get_observation()
        
        if main_camera_name not in observation:
            continue
            
        frame = observation[main_camera_name]
        detection_result = handshake_detector.detect_handshake_gesture(frame, visualize=display_data)
        
        max_confidence = max(max_confidence, detection_result['confidence'])
        
        if detection_result['ready'] and detection_result['confidence'] >= confidence_threshold:
            log_say("Handshake detected! Starting policy evaluation...", True)
            return True, detection_result['confidence']
            
        if display_data and 'annotated_frame' in detection_result:
            cv2.imshow('Handshake Detection', detection_result['annotated_frame'])
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
                
        time.sleep(0.1)  # Small delay to prevent overwhelming the system
    
    log_say("Timeout waiting for handshake detection.", True)
    return False, max_confidence


def evaluate_handshake_episode(
    robot: Robot,
    policy: PreTrainedPolicy,
    handshake_detector: ImprovedHandshakeDetector,
    main_camera_name: str,
    episode_time_s: float,
    success_confidence_threshold: float,
    display_data: bool = False,
    save_video: bool = False,
    episode_ix: int = 0,
    output_dir: Path = None
) -> dict[str, Any]:
    """
    Evaluate a single handshake episode.
    
    Returns:
        Dictionary with episode metrics and data
    """
    episode_start_time = time.perf_counter()
    frames_for_video = []
    handshake_confidences = []
    policy_actions = []
    robot_states = []
    timestamps = []
    
    policy.reset()
    
    log_say(f"Starting handshake episode {episode_ix + 1}...", True)
    
    step = 0
    max_steps = int(episode_time_s * 30)  # Assume ~30 FPS
    
    while step < max_steps:
        step_start_time = time.perf_counter()
        
        # Get robot observation
        observation = robot.get_observation()
        
        if main_camera_name not in observation:
            continue
            
        frame = observation[main_camera_name]
        
        # Run handshake detection
        detection_result = handshake_detector.detect_handshake_gesture(frame, visualize=display_data)
        handshake_confidences.append(detection_result['confidence'])
        
        # Prepare observation for policy
        policy_observation = {}
        for key, value in observation.items():
            if key.startswith('observation.'):
                policy_observation[key] = torch.from_numpy(value).unsqueeze(0).float()
            elif isinstance(value, np.ndarray):
                # Camera observations
                policy_observation[f'observation.images.{key}'] = torch.from_numpy(value).unsqueeze(0)
        
        # Add handshake detection features
        policy_observation['observation.handshake_ready'] = torch.tensor([[detection_result['ready']]], dtype=torch.bool)
        policy_observation['observation.handshake_confidence'] = torch.tensor([[detection_result['confidence']]], dtype=torch.float32)
        
        if detection_result['hand_position']:
            policy_observation['observation.hand_position_x'] = torch.tensor([[detection_result['hand_position'][0]]], dtype=torch.float32)
            policy_observation['observation.hand_position_y'] = torch.tensor([[detection_result['hand_position'][1]]], dtype=torch.float32)
        else:
            policy_observation['observation.hand_position_x'] = torch.tensor([[0.0]], dtype=torch.float32)
            policy_observation['observation.hand_position_y'] = torch.tensor([[0.0]], dtype=torch.float32)
        
        # Move to device
        device = next(policy.parameters()).device
        for key in policy_observation:
            policy_observation[key] = policy_observation[key].to(device)
        
        # Get policy action
        with torch.inference_mode():
            action = policy.select_action(policy_observation)
        
        # Convert action to robot format
        action_np = action.squeeze(0).cpu().numpy()
        policy_actions.append(action_np.copy())
        
        # Convert action to robot action dict
        robot_action = {}
        action_names = robot.action_names  # Get joint names from robot
        for i, name in enumerate(action_names):
            if i < len(action_np):
                robot_action[name] = action_np[i]
        
        # Apply action to robot
        robot.send_action(robot_action)
        
        # Store data
        robot_states.append(observation.copy())
        timestamps.append(time.perf_counter() - episode_start_time)
        
        # Save frame for video if needed
        if save_video:
            if display_data and 'annotated_frame' in detection_result:
                frames_for_video.append(detection_result['annotated_frame'].copy())
            else:
                frames_for_video.append(frame.copy())
        
        # Display frame if requested
        if display_data and 'annotated_frame' in detection_result:
            cv2.imshow('Handshake Evaluation', detection_result['annotated_frame'])
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        step += 1
        
        # Control loop timing (aim for ~30 FPS)
        elapsed = time.perf_counter() - step_start_time
        if elapsed < 1/30:
            time.sleep(1/30 - elapsed)
    
    episode_end_time = time.perf_counter()
    episode_duration = episode_end_time - episode_start_time
    
    # Analyze episode results
    avg_confidence = np.mean(handshake_confidences) if handshake_confidences else 0.0
    max_confidence = np.max(handshake_confidences) if handshake_confidences else 0.0
    handshake_detected_frames = sum(1 for conf in handshake_confidences if conf > success_confidence_threshold)
    handshake_detection_rate = handshake_detected_frames / len(handshake_confidences) if handshake_confidences else 0.0
    
    # Determine success - handshake completed successfully
    success = (
        avg_confidence > success_confidence_threshold * 0.7 and  # Average confidence reasonably high
        max_confidence > success_confidence_threshold and        # Peak confidence high enough  
        handshake_detection_rate > 0.3                          # Sustained detection over time
    )
    
    # Save video if requested
    video_path = None
    if save_video and frames_for_video and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / f"eval_episode_{episode_ix:03d}.mp4"
        
        # Convert frames to video
        if frames_for_video:
            height, width = frames_for_video[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (width, height))
            
            for frame in frames_for_video:
                video_writer.write(frame)
            video_writer.release()
    
    episode_metrics = {
        "episode_ix": episode_ix,
        "success": success,
        "duration_s": episode_duration,
        "avg_handshake_confidence": avg_confidence,
        "max_handshake_confidence": max_confidence,
        "handshake_detection_rate": handshake_detection_rate,
        "total_steps": step,
        "avg_confidence": avg_confidence,
        "video_path": str(video_path) if video_path else None,
    }
    
    log_say(f"Episode {episode_ix + 1} completed: {'SUCCESS' if success else 'FAILURE'} "
            f"(avg_conf: {avg_confidence:.3f}, max_conf: {max_confidence:.3f})", True)
    
    return episode_metrics


def eval_handshake_policy(
    robot: Robot,
    policy: PreTrainedPolicy, 
    cfg: HandshakeEvalConfig,
    output_dir: Path
) -> dict[str, Any]:
    """
    Evaluate a policy on handshake tasks.
    
    Returns:
        Dictionary with evaluation results and metrics
    """
    log_say(f"Starting handshake evaluation with {cfg.n_episodes} episodes", True)
    
    # Initialize handshake detector
    handshake_detector = ImprovedHandshakeDetector(
        detection_confidence=0.7,
        confidence_threshold=cfg.handshake_confidence_threshold
    )
    
    # Get main camera name (assume first camera)
    camera_names = list(robot.cameras.keys())
    if not camera_names:
        raise ValueError("Robot must have at least one camera configured for handshake evaluation")
    main_camera_name = camera_names[0]
    
    log_say(f"Using camera '{main_camera_name}' for handshake detection", True)
    
    all_episodes = []
    start_time = time.perf_counter()
    
    for episode_ix in range(cfg.n_episodes):
        log_say(f"\n=== Episode {episode_ix + 1}/{cfg.n_episodes} ===", True)
        
        # Wait for handshake detection
        detected, detection_confidence = wait_for_handshake_detection(
            robot,
            handshake_detector,
            main_camera_name,
            cfg.handshake_timeout_s,
            cfg.handshake_confidence_threshold,
            cfg.display_data
        )
        
        if not detected:
            log_say(f"Episode {episode_ix + 1}: No handshake detected within timeout", True)
            episode_metrics = {
                "episode_ix": episode_ix,
                "success": False,
                "duration_s": 0.0,
                "avg_handshake_confidence": 0.0,
                "max_handshake_confidence": detection_confidence,
                "handshake_detection_rate": 0.0,
                "total_steps": 0,
                "timeout": True,
                "video_path": None,
            }
        else:
            # Run handshake episode
            episode_metrics = evaluate_handshake_episode(
                robot,
                policy,
                handshake_detector,
                main_camera_name,
                cfg.episode_time_s,
                cfg.success_confidence_threshold,
                cfg.display_data,
                cfg.save_videos,
                episode_ix,
                output_dir / "videos" if cfg.save_videos else None
            )
            episode_metrics["timeout"] = False
        
        all_episodes.append(episode_metrics)
        
        # Reset period between episodes
        if episode_ix < cfg.n_episodes - 1:  # Don't wait after last episode
            log_say(f"Reset period: {cfg.reset_time_s}s before next episode", True)
            time.sleep(cfg.reset_time_s)
    
    total_time = time.perf_counter() - start_time
    
    # Calculate aggregate metrics
    successful_episodes = [ep for ep in all_episodes if ep["success"]]
    detected_episodes = [ep for ep in all_episodes if not ep.get("timeout", False)]
    
    aggregated_metrics = {
        "total_episodes": cfg.n_episodes,
        "successful_episodes": len(successful_episodes),
        "success_rate": len(successful_episodes) / cfg.n_episodes * 100,
        "detection_rate": len(detected_episodes) / cfg.n_episodes * 100,
        "avg_episode_duration": np.mean([ep["duration_s"] for ep in detected_episodes]) if detected_episodes else 0.0,
        "avg_handshake_confidence": np.mean([ep["avg_handshake_confidence"] for ep in all_episodes]),
        "avg_max_confidence": np.mean([ep["max_handshake_confidence"] for ep in all_episodes]),
        "total_eval_time_s": total_time,
        "avg_time_per_episode_s": total_time / cfg.n_episodes,
    }
    
    results = {
        "per_episode": all_episodes,
        "aggregated": aggregated_metrics,
        "config": asdict(cfg),
    }
    
    # Clean up
    if cfg.display_data:
        cv2.destroyAllWindows()
    
    return results


@parser.wrap()
def eval_handshake_main(cfg: HandshakeEvalPipelineConfig):
    """Main function for handshake evaluation."""
    init_logging()
    logging.info("Starting handshake policy evaluation")
    logging.info(pformat(asdict(cfg)))
    
    # Check device
    device = get_safe_torch_device(cfg.device, log=True)
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
    
    # Initialize robot
    logging.info("Connecting to robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect(calibrate=False)
    
    try:
        # Load policy
        logging.info("Loading policy...")
        policy = make_policy(cfg.policy)
        policy.to(device)
        policy.eval()
        
        # Run evaluation
        logging.info("Starting evaluation...")
        results = eval_handshake_policy(robot, policy, cfg.eval, output_dir)
        
        # Print results summary
        print("\n" + "="*60)
        print("HANDSHAKE EVALUATION RESULTS")
        print("="*60)
        agg = results["aggregated"]
        print(f"Episodes: {agg['total_episodes']}")
        print(f"Success Rate: {agg['success_rate']:.1f}%")
        print(f"Detection Rate: {agg['detection_rate']:.1f}%")
        print(f"Avg Handshake Confidence: {agg['avg_handshake_confidence']:.3f}")
        print(f"Avg Episode Duration: {agg['avg_episode_duration']:.1f}s")
        print(f"Total Evaluation Time: {agg['total_eval_time_s']:.1f}s")
        print("="*60)
        
        # Save detailed results
        results_file = output_dir / "eval_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Detailed results saved to {results_file}")
        
    finally:
        # Disconnect robot
        robot.disconnect()
        logging.info("Robot disconnected")
    
    logging.info("Handshake evaluation completed!")


if __name__ == "__main__":
    eval_handshake_main()
