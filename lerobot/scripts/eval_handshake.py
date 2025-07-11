"""Evaluate a handshake policy on real SO-101 robot hardware with handshake detection.

Usage examples:

Evaluate a trained handshake policy with live handshake detection:

```shell
python -m lerobot.scripts.eval_handshake \
    --policy.path=outputs/train/your_username/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower_arm \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": "/dev/video1", "width": 640, "height": 480, "fps": 30}}' \
    --eval.n_episodes=10 \
    --eval.handshake_confidence_threshold=0.8 \
    --eval.handshake_timeout_s=15.0 \
    --eval.episode_time_s=10.0 \
    --output_dir=./eval_results \
    --save_video=true \
    --display_data=true
```

The script integrates the same handshake detection and features used during training:
- Waits for handshake detection before starting each episode
- Processes the same 4 handshake features: [ready, confidence, pos_x, pos_y]
- Records handshake-specific metrics during evaluation
- Saves videos of successful handshake interactions
"""

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import numpy as np
import rerun as rr
import torch
from termcolor import colored

from lerobot.common.handshake_detection import ImprovedHandshakeDetector
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.robots import Robot, make_robot_from_config
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig


@dataclass
class HandshakeEvalConfig:
    """Configuration for handshake-specific evaluation settings."""
    n_episodes: int = 10
    episode_time_s: float = 10.0
    handshake_confidence_threshold: float = 0.8
    handshake_timeout_s: float = 15.0
    handshake_detection_delay: float = 1.0
    success_confidence_threshold: float = 0.7
    max_episodes_rendered: int = 5
    use_amp: bool = False
    batch_size: int = 1  # Always 1 for real robot evaluation


@dataclass 
class HandshakeEvalPipelineConfig(EvalPipelineConfig):
    eval: HandshakeEvalConfig = HandshakeEvalConfig()


def wait_for_handshake_detection(
    robot: Robot,
    handshake_detector: ImprovedHandshakeDetector,
    camera_name: str,
    timeout_s: float,
    confidence_threshold: float,
    detection_delay: float,
    display_data: bool = False,
    episode_ix: int = 0,
) -> tuple[bool, dict]:
    """
    Wait for handshake detection before starting evaluation episode.
    
    Returns:
        Tuple of (handshake_detected, detection_result)
    """
    start_time = time.perf_counter()
    detection_start_time = None
    last_status_update = 0
    
    log_say(f"Episode {episode_ix}: Waiting for person to extend their hand for handshake...", True)
    
    while time.perf_counter() - start_time < timeout_s:
        try:
            observation = robot.get_observation()
            
            if camera_name not in observation:
                continue
                
            frame = observation[camera_name]
            
            # Detect handshake gesture
            detection_result = handshake_detector.detect_handshake_gesture(frame, visualize=display_data)
            
            if detection_result['ready'] and detection_result['confidence'] >= confidence_threshold:
                if detection_start_time is None:
                    detection_start_time = time.perf_counter()
                    log_say(f"Handshake detected! Waiting {detection_delay} seconds before starting episode...", True)
                
                # Wait for the specified delay after detection
                if time.perf_counter() - detection_start_time >= detection_delay:
                    log_say("Starting handshake evaluation episode now!", True)
                    return True, detection_result
            else:
                # Reset detection timer if gesture is lost
                detection_start_time = None
            
            if display_data:
                # Update status every second
                current_time = time.perf_counter()
                if current_time - last_status_update >= 1.0:
                    status_text = f"EVAL WAITING | Episode: {episode_ix} | Time remaining: {timeout_s - (current_time - start_time):.1f}s"
                    rr.log("status", rr.TextLog(status_text, level=rr.TextLogLevel.INFO))
                    last_status_update = current_time
                
                # Display robot joints and camera feed
                annotated_frame = detection_result.get('annotated_frame')
                robot_joints = {}
                for obs, val in observation.items():
                    if isinstance(val, float) and obs.endswith('.pos'):
                        robot_joints[obs] = val
                    elif isinstance(val, np.ndarray):
                        if obs == camera_name and annotated_frame is not None:
                            rr.log("camera_with_pose", rr.Image(annotated_frame), static=True)
                            rr.log("camera_raw", rr.Image(val), static=True)
                        else:
                            rr.log("camera_raw", rr.Image(val), static=True)
                
                if robot_joints:
                    for joint_name, joint_val in robot_joints.items():
                        rr.log(f"robot_joints/{joint_name}", rr.Scalar(joint_val))
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt received in handshake detection")
            break
        except Exception as e:
            logging.error(f"Error in handshake detection loop: {e}")
            time.sleep(0.1)
            continue
    
    log_say(f"Episode {episode_ix}: Handshake detection timeout.", True)
    return False, {}


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
    Run a single handshake evaluation episode.
    
    Returns:
        Dictionary with episode metrics and data
    """
    policy.eval()
    policy.reset()
    
    device = get_device_from_parameters(policy)
    episode_start_time = time.perf_counter()
    
    # Tracking variables
    all_actions = []
    all_rewards = []
    handshake_confidences = []
    hand_positions = []
    robot_states = []
    timestamps = []
    frames_for_video = []
    
    step = 0
    max_steps = int(episode_time_s * 30)  # Assume ~30 FPS
    last_status_update = 0
    
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
        
        # Process hand position
        if detection_result['hand_position']:
            hand_positions.append(detection_result['hand_position'])
        else:
            hand_positions.append([0.0, 0.0])
        
        # Prepare observation for policy (same format as training)
        policy_observation = {}
        for key, value in observation.items():
            if key.startswith('observation.'):
                policy_observation[key] = torch.from_numpy(value).unsqueeze(0).float()
            elif isinstance(value, np.ndarray):
                # Camera observations
                policy_observation[f'observation.images.{key}'] = torch.from_numpy(value).unsqueeze(0)
        
        # Add handshake detection features (same as training pipeline)
        handshake_data = np.array([
            float(detection_result['ready']),
            detection_result['confidence'],
            hand_positions[-1][0],
            hand_positions[-1][1]
        ])
        policy_observation['observation.handshake'] = torch.from_numpy(handshake_data).unsqueeze(0).float()
        
        # Move to device
        for key in policy_observation:
            policy_observation[key] = policy_observation[key].to(device)
        
        # Get policy action
        with torch.inference_mode():
            action = policy.select_action(policy_observation)
        
        # Convert action to robot format
        action_np = action.squeeze(0).cpu().numpy()
        all_actions.append(action_np.copy())
        
        # Convert action to robot action dict
        robot_action = {}
        action_names = robot.action_features  # Get joint names from robot
        for i, name in enumerate(action_names):
            if i < len(action_np):
                robot_action[name] = action_np[i]
        
        # Apply action to robot
        robot.send_action(robot_action)
        
        # Store data
        robot_states.append(observation.copy())
        timestamps.append(time.perf_counter() - episode_start_time)
        
        # Simple reward based on handshake confidence (higher confidence = better handshake)
        reward = detection_result['confidence'] if detection_result['ready'] else 0.0
        all_rewards.append(reward)
        
        # Save frame for video if needed
        if save_video:
            if display_data and 'annotated_frame' in detection_result:
                frames_for_video.append(detection_result['annotated_frame'].copy())
            else:
                frames_for_video.append(frame.copy())
        
        # Display frame if requested
        if display_data:
            # Update status every second
            current_time = time.perf_counter()
            if current_time - last_status_update >= 1.0:
                actual_fps = step / (current_time - episode_start_time) if current_time - episode_start_time > 0 else 0
                status_text = f"EVAL EPISODE {episode_ix} | Step: {step}/{max_steps} | Time: {current_time - episode_start_time:.1f}s | FPS: {actual_fps:.1f}"
                rr.log("status", rr.TextLog(status_text, level=rr.TextLogLevel.INFO))
                last_status_update = current_time
            
            # Display robot data
            annotated_frame = detection_result.get('annotated_frame')
            robot_joints = {}
            for obs, val in observation.items():
                if isinstance(val, float) and obs.endswith('.pos'):
                    robot_joints[obs] = val
                elif isinstance(val, np.ndarray):
                    if obs == main_camera_name and annotated_frame is not None:
                        rr.log("camera_with_pose", rr.Image(annotated_frame), static=True)
                        rr.log("camera_raw", rr.Image(val), static=True)
                    else:
                        rr.log("camera_raw", rr.Image(val), static=True)
            
            if robot_joints:
                for joint_name, joint_val in robot_joints.items():
                    rr.log(f"robot_joints/{joint_name}", rr.Scalar(joint_val))

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
    total_reward = np.sum(all_rewards)
    
    # Determine success based on sustained high confidence
    success = max_confidence >= success_confidence_threshold
    
    # Calculate hand position statistics
    valid_positions = [pos for pos in hand_positions if pos[0] > 0 and pos[1] > 0]
    avg_hand_position = np.mean(valid_positions, axis=0) if valid_positions else [0.0, 0.0]
    hand_position_variance = np.var(valid_positions, axis=0) if len(valid_positions) > 1 else [0.0, 0.0]
    
    # Save video if requested
    video_path = None
    if save_video and frames_for_video and output_dir:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"eval_episode_{episode_ix}.mp4"
        write_video(str(video_path), np.array(frames_for_video), fps=30)
    
    episode_data = {
        "episode_ix": episode_ix,
        "success": success,
        "episode_duration": episode_duration,
        "total_reward": total_reward,
        "avg_handshake_confidence": avg_confidence,
        "max_handshake_confidence": max_confidence,
        "num_steps": step,
        "avg_hand_position_x": avg_hand_position[0],
        "avg_hand_position_y": avg_hand_position[1],
        "hand_position_variance_x": hand_position_variance[0],
        "hand_position_variance_y": hand_position_variance[1],
        "valid_hand_detections": len(valid_positions),
        "video_path": str(video_path) if video_path else None,
    }
    
    return episode_data


def eval_handshake_policy(
    robot: Robot,
    policy: PreTrainedPolicy, 
    handshake_detector: ImprovedHandshakeDetector,
    main_camera_name: str,
    cfg: HandshakeEvalConfig,
    output_dir: Path,
    display_data: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a handshake policy over multiple episodes.
    
    Returns:
        Dictionary with aggregated metrics and per-episode data
    """
    start_time = time.time()
    policy.eval()
    
    episode_results = []
    successful_episodes = 0
    
    log_say(f"Starting handshake evaluation with {cfg.n_episodes} episodes", True)
    
    for episode_ix in range(cfg.n_episodes):
        try:
            # Wait for handshake detection
            handshake_detected, initial_detection = wait_for_handshake_detection(
                robot=robot,
                handshake_detector=handshake_detector,
                camera_name=main_camera_name,
                timeout_s=cfg.handshake_timeout_s,
                confidence_threshold=cfg.handshake_confidence_threshold,
                detection_delay=cfg.handshake_detection_delay,
                display_data=display_data,
                episode_ix=episode_ix,
            )
            
            if not handshake_detected:
                log_say(f"Episode {episode_ix}: Skipping due to handshake detection timeout", True)
                continue
            
            # Run evaluation episode
            episode_data = evaluate_handshake_episode(
                robot=robot,
                policy=policy,
                handshake_detector=handshake_detector,
                main_camera_name=main_camera_name,
                episode_time_s=cfg.episode_time_s,
                success_confidence_threshold=cfg.success_confidence_threshold,
                display_data=display_data,
                save_video=episode_ix < cfg.max_episodes_rendered,
                episode_ix=episode_ix,
                output_dir=output_dir,
            )
            
            episode_results.append(episode_data)
            
            if episode_data["success"]:
                successful_episodes += 1
                log_say(f"Episode {episode_ix}: SUCCESS (confidence: {episode_data['max_handshake_confidence']:.3f})", True)
            else:
                log_say(f"Episode {episode_ix}: Failed (confidence: {episode_data['max_handshake_confidence']:.3f})", True)
            
            # Brief pause between episodes
            time.sleep(2.0)
            
        except KeyboardInterrupt:
            log_say("Evaluation interrupted by user", True)
            break
        except Exception as e:
            logging.error(f"Error in episode {episode_ix}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # Calculate aggregated metrics
    if episode_results:
        aggregated_metrics = {
            "success_rate": (successful_episodes / len(episode_results)) * 100,
            "avg_total_reward": np.mean([ep["total_reward"] for ep in episode_results]),
            "avg_episode_duration": np.mean([ep["episode_duration"] for ep in episode_results]),
            "avg_handshake_confidence": np.mean([ep["avg_handshake_confidence"] for ep in episode_results]),
            "max_handshake_confidence": np.max([ep["max_handshake_confidence"] for ep in episode_results]),
            "avg_hand_position_x": np.mean([ep["avg_hand_position_x"] for ep in episode_results]),
            "avg_hand_position_y": np.mean([ep["avg_hand_position_y"] for ep in episode_results]),
            "avg_valid_detections": np.mean([ep["valid_hand_detections"] for ep in episode_results]),
            "total_episodes": len(episode_results),
            "successful_episodes": successful_episodes,
            "total_evaluation_time": total_time,
            "avg_time_per_episode": total_time / len(episode_results) if episode_results else 0,
        }
    else:
        aggregated_metrics = {
            "success_rate": 0.0,
            "total_episodes": 0,
            "successful_episodes": 0,
            "total_evaluation_time": total_time,
        }
    
    return {
        "aggregated": aggregated_metrics,
        "per_episode": episode_results,
    }


@parser.wrap()
def eval_handshake_main(cfg: HandshakeEvalPipelineConfig):
    """Main handshake evaluation function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Initialize Rerun for visualization
    if cfg.display_data:
        _init_rerun(session_name="handshake_evaluation")
    
    # Check device availability
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize robot
    logging.info("Connecting to robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    
    # Initialize handshake detector
    try:
        handshake_detector = ImprovedHandshakeDetector(
            confidence_threshold=cfg.eval.handshake_confidence_threshold
        )
        logging.info(f"Handshake detector initialized with confidence threshold: {cfg.eval.handshake_confidence_threshold}")
    except ImportError as e:
        logging.error(f"Failed to initialize handshake detector: {e}")
        logging.error("Please install required dependencies: pip install mediapipe opencv-python")
        raise
    
    # Determine main camera name
    if not hasattr(robot.config, 'cameras') or not robot.config.cameras:
        raise ValueError("Robot must have at least one camera configured for handshake evaluation")
    
    main_camera_name = list(robot.config.cameras.keys())[0]
    logging.info(f"Using camera '{main_camera_name}' for handshake detection")
    
    # Load policy
    logging.info("Loading policy...")
    policy = make_policy(cfg=cfg.policy)
    policy.to(device)
    policy.eval()
    
    try:
        # Run evaluation
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.eval.use_amp else nullcontext():
            eval_results = eval_handshake_policy(
                robot=robot,
                policy=policy,
                handshake_detector=handshake_detector,
                main_camera_name=main_camera_name,
                cfg=cfg.eval,
                output_dir=output_dir,
                display_data=cfg.display_data,
            )
        
        # Print results
        print("\n" + "="*50)
        print("HANDSHAKE EVALUATION RESULTS")
        print("="*50)
        for key, value in eval_results["aggregated"].items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
        print("="*50)
        
        # Save results
        results_file = output_dir / "eval_results.json"
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logging.info(f"Evaluation results saved to: {results_file}")
        
    finally:
        # Cleanup
        robot.disconnect()
        cv2.destroyAllWindows()
    
    logging.info("Handshake evaluation completed!")


if __name__ == "__main__":
    init_logging()
    eval_handshake_main()
