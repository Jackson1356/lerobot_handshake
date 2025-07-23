"""
Evaluate a trained handshake policy on real robot hardware.

This script loads a trained handshake policy and evaluates it by:
1. Waiting for a person to extend their hand for a handshake
2. Running the trained policy to perform the handshake
3. Collecting metrics on handshake success and performance

Example usage:

```bash
python -m lerobot.scripts.eval_handshake \
    --policy.path=outputs/train/handshake_policy_v1/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower_arm \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": "/dev/video1", "width": 640, "height": 480, "fps": 30}}' \
    --eval.num_episodes=10 \
    --eval.episode_time_s=30 \
    --eval.handshake_timeout_s=15 \
    --display_data=true
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import rerun as rr
import torch

from lerobot.common.cameras import CameraConfig  # noqa: F401
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.datasets.utils import build_dataset_frame
from lerobot.common.handshake_detection import ImprovedHandshakeDetector
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.common.robots import so101_follower  # Ensure so101_follower is registered
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class HandshakeEvalConfig:
    """Configuration for handshake policy evaluation."""
    num_episodes: int = 10
    episode_time_s: float = 30.0
    handshake_timeout_s: float = 10.0
    handshake_confidence_threshold: float = 0.8
    handshake_detection_delay: float = 1.0
    fps: int = 20
    handshake_detection_fps: int = 10


@dataclass
class HandshakeEvalPipelineConfig:
    robot: RobotConfig
    eval: HandshakeEvalConfig = HandshakeEvalConfig()
    policy: PreTrainedConfig | None = None
    output_dir: Path = Path("outputs/eval_handshake")
    display_data: bool = False
    device: str = "cuda"
    seed: int | None = None
    
    def __post_init__(self):
        # Parse policy from CLI if provided
        policy_path = parser.get_path_arg("policy")
        if self.policy is None and policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
    
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def wait_for_handshake_detection(
    robot: Robot, 
    handshake_detector: ImprovedHandshakeDetector,
    camera_name: str,
    timeout_s: float,
    confidence_threshold: float,
    detection_delay: float,
    display_data: bool = False,
    episode_number: int = 0,
) -> bool:
    """
    Wait for handshake detection from the person.
    
    Returns:
        True if handshake detected, False if timeout
    """
    start_time = time.perf_counter()
    detection_start_time = None
    last_status_update = 0
    
    log_say("Waiting for person to extend their hand for handshake...", True)
    
    while time.perf_counter() - start_time < timeout_s:
        try:
            observation = robot.get_observation()
            frame = observation[camera_name]
            
            # Detect handshake gesture
            detection_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
        
            if detection_result['ready'] and detection_result['confidence'] >= confidence_threshold:
                if detection_start_time is None:
                    detection_start_time = time.perf_counter()
                    log_say(f"Handshake detected! Waiting {detection_delay} seconds before starting evaluation...", True)
                
                # Wait for the specified delay after detection
                if time.perf_counter() - detection_start_time >= detection_delay:
                    log_say("Starting handshake evaluation now!", True)
                    return True
            else:
                # Reset detection timer if gesture is lost
                detection_start_time = None
            
            if display_data:
                # Update status every second
                current_time = time.perf_counter()
                if current_time - last_status_update >= 1.0:
                    status_text = f"WAITING | Episode: {episode_number} | Time remaining: {timeout_s - (current_time - start_time):.1f}s"
                    rr.log("status", rr.TextLog(status_text, level=rr.TextLogLevel.INFO))
                    last_status_update = current_time
                
                # Get pose overlay for camera feed using detection result
                annotated_frame = None
                if camera_name in observation and detection_result and 'annotated_frame' in detection_result:
                    annotated_frame = detection_result['annotated_frame']
                
                # Robot joint positions (6 values only) - grouped in single chart
                robot_joints = {}
                for obs, val in observation.items():
                    if isinstance(val, float) and obs.endswith('.pos'):
                        # Collect all robot joint positions for single chart
                        robot_joints[obs] = val
                    elif isinstance(val, np.ndarray):
                        if obs == camera_name and annotated_frame is not None:
                            # Camera with pose detection - consistent path
                            rr.log("camera_with_pose", rr.Image(annotated_frame), static=True)
                            # Raw camera - consistent path
                            rr.log("camera_raw", rr.Image(val), static=True)
                        else:
                            # Raw camera for other cameras
                            rr.log("camera_raw", rr.Image(val), static=True)
                
                # Log all robot joints as single chart
                if robot_joints:
                    # Log each joint individually to create clean grouped chart
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
    
    log_say("Handshake detection timeout. Skipping this episode.", True)
    return False


def evaluate_handshake_episode(
    robot: Robot,
    policy: PreTrainedPolicy,
    handshake_detector: ImprovedHandshakeDetector,
    main_camera_name: str,
    episode_time_s: float,
    fps: int,
    handshake_detection_fps: int,
    display_data: bool = False,
    episode_ix: int = 0,
) -> dict[str, Any]:
    """
    Run a single handshake evaluation episode.
    
    Returns:
        Dictionary with episode metrics
    """
    episode_start_time = time.perf_counter()
    policy.reset()
    
    # Prepare dataset features for policy input (same as record_handshake.py)
    dataset_features = {
        "observation.handshake": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["handshake_ready", "handshake_confidence", "hand_position_x", "hand_position_y"],
        },
    }
    
    handshake_confidences = []
    policy_actions = []
    robot_states = []
    timestamps = []
    
    timestamp = 0
    step = 0
    frame_count = 0
    max_steps = int(episode_time_s * fps)
    last_status_update = 0
    
    # Handshake detection optimization - run at lower frequency
    detection_interval = max(1, fps // handshake_detection_fps)  # Run detection every N frames
    last_handshake_result = None
    
    while step < max_steps:
        step_start_time = time.perf_counter()
        
        # Get robot observation
        observation = robot.get_observation()
        
        if main_camera_name not in observation:
            continue
            
        # Run handshake detection at reduced frequency for better performance
        if main_camera_name in observation:
            # Only run detection every N frames to improve FPS
            if frame_count % detection_interval == 0:
                frame = observation[main_camera_name]
                last_handshake_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
        
            # Use cached result (either fresh or from previous frame)
            if last_handshake_result is not None:
                detection_result = last_handshake_result
                handshake_confidences.append(detection_result['confidence'])
            else:
                # Fallback if no detection result yet
                detection_result = {'ready': False, 'confidence': 0.0, 'hand_position': None}
                handshake_confidences.append(0.0)
        
        # Add handshake features to observation (same as record_handshake.py)
        handshake_ready = float(detection_result['ready'])
        handshake_confidence = detection_result['confidence']
        if detection_result['hand_position'] is not None:
            hand_position_x = float(detection_result['hand_position'][0])
            hand_position_y = float(detection_result['hand_position'][1])
        else:
            hand_position_x = -1.0
            hand_position_y = -1.0
        
        observation["handshake_ready"] = handshake_ready
        observation["handshake_confidence"] = handshake_confidence
        observation["hand_position_x"] = hand_position_x
        observation["hand_position_y"] = hand_position_y
        
        # Build observation frame for policy
        observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
        
        # Get policy action
        with torch.inference_mode():
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        
        # Apply action to robot
        sent_action = robot.send_action(action)
        
        # Store data for analysis
        policy_actions.append(action.copy())
        robot_states.append(observation.copy())
        timestamps.append(time.perf_counter() - episode_start_time)
        
        if display_data:
            # Update status every second 
            current_time = time.perf_counter()
            if current_time - last_status_update >= 1.0:
                # Calculate actual FPS
                actual_fps = frame_count / timestamp if timestamp > 0 else 0
                status_text = f"EVALUATING | Episode: {episode_ix} | Elapsed: {timestamp:.1f}s | Remaining: {episode_time_s - timestamp:.1f}s | Actual FPS: {actual_fps:.1f}"
                rr.log("status", rr.TextLog(status_text, level=rr.TextLogLevel.INFO))
                last_status_update = current_time
            
            # Get pose overlay for camera feed using cached result
            annotated_frame = None
            if main_camera_name in observation and last_handshake_result and 'annotated_frame' in last_handshake_result:
                annotated_frame = last_handshake_result['annotated_frame']
            
            # Robot joint positions (6 values only) - grouped in single chart
            robot_joints = {}
            for obs, val in observation.items():
                if isinstance(val, float) and obs.endswith('.pos'):
                    # Collect all robot joint positions for single chart
                    robot_joints[obs] = val
                elif isinstance(val, np.ndarray):
                    if obs == main_camera_name and annotated_frame is not None:
                        # Camera with pose detection - consistent path
                        rr.log("camera_with_pose", rr.Image(annotated_frame), static=True)
                        # Raw camera - consistent path
                        rr.log("camera_raw", rr.Image(val), static=True)
                    else:
                        # Raw camera for other cameras
                        rr.log("camera_raw", rr.Image(val), static=True)
            
            # Log all robot joints as single chart
            if robot_joints:
                # Log each joint individually to create clean grouped chart
                for joint_name, joint_val in robot_joints.items():
                    rr.log(f"robot_joints/{joint_name}", rr.Scalar(joint_val))
        
        step += 1
        frame_count += 1
        
        # Control loop timing (aim for recording fps)
        elapsed = time.perf_counter() - step_start_time
        if elapsed < 1/fps:
            time.sleep(1/fps - elapsed)
        
        timestamp = time.perf_counter() - episode_start_time
    
    episode_end_time = time.perf_counter()
    episode_duration = episode_end_time - episode_start_time
    
    # Basic episode info
    avg_confidence = np.mean(handshake_confidences) if handshake_confidences else 0.0
    max_confidence = np.max(handshake_confidences) if handshake_confidences else 0.0
    
    episode_metrics = {
        "episode_ix": episode_ix,
        "episode_duration": episode_duration,
        "avg_handshake_confidence": avg_confidence,
        "max_handshake_confidence": max_confidence,
        "num_steps": step,
    }
    
    return episode_metrics


@parser.wrap()
def eval_handshake(cfg: HandshakeEvalPipelineConfig):
    """Main evaluation function for handshake policies."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Initialize Rerun for visualization if requested
    if cfg.display_data:
        _init_rerun(session_name="handshake_evaluation")
    
    # Create output directory
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize robot
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
        raise ValueError("Robot must have at least one camera configured for handshake detection")
    
    main_camera_name = list(robot.config.cameras.keys())[0]
    logging.info(f"Using camera '{main_camera_name}' for handshake detection")
    
    # Load trained policy
    logging.info("Loading trained handshake policy")
    policy = make_policy(cfg=cfg.policy, env_cfg=None)
    policy.eval()

    # Run evaluation episodes
    episode_results = []
    
    for episode_ix in range(cfg.eval.num_episodes):
        logging.info(f"Starting handshake evaluation episode {episode_ix + 1}/{cfg.eval.num_episodes}")
        
        # Wait for handshake detection
        handshake_detected = wait_for_handshake_detection(
            robot=robot,
            handshake_detector=handshake_detector,
            camera_name=main_camera_name,
            timeout_s=cfg.eval.handshake_timeout_s,
            confidence_threshold=cfg.eval.handshake_confidence_threshold,
            detection_delay=cfg.eval.handshake_detection_delay,
            display_data=cfg.display_data,
            episode_number=episode_ix + 1,
        )
        
        if not handshake_detected:
            logging.warning(f"Skipping episode {episode_ix + 1} due to handshake detection timeout")
            continue
        
        # Run evaluation episode
        episode_metrics = evaluate_handshake_episode(
            robot=robot,
            policy=policy,
            handshake_detector=handshake_detector,
            main_camera_name=main_camera_name,
            episode_time_s=cfg.eval.episode_time_s,
            fps=cfg.eval.fps,
            handshake_detection_fps=cfg.eval.handshake_detection_fps,
            display_data=cfg.display_data,
            episode_ix=episode_ix,
        )
        
        episode_results.append(episode_metrics)
        
        logging.info(f"Episode {episode_ix + 1} completed - "
                    f"Max Confidence: {episode_metrics['max_handshake_confidence']:.3f}, "
                    f"Duration: {episode_metrics['episode_duration']:.1f}s")
        
        # Brief pause between episodes
        time.sleep(2.0)
    
    # Calculate final metrics
    if episode_results:
        avg_duration = np.mean([ep["episode_duration"] for ep in episode_results])
        avg_confidence = np.mean([ep["avg_handshake_confidence"] for ep in episode_results])
        avg_max_confidence = np.mean([ep["max_handshake_confidence"] for ep in episode_results])
        
        final_results = {
            "num_episodes_completed": len(episode_results),
            "num_episodes_attempted": cfg.eval.num_episodes,
            "avg_episode_duration": avg_duration,
            "avg_handshake_confidence": avg_confidence,
            "avg_max_handshake_confidence": avg_max_confidence,
            "episode_results": episode_results,
        }
        
        # Save results
        import json
        results_path = cfg.output_dir / "handshake_eval_results.json"
        with open(results_path, "w") as f:
            # Convert numpy types to python types for JSON serialization
            json_results = {}
            for key, value in final_results.items():
                if isinstance(value, np.floating):
                    json_results[key] = float(value)
                elif isinstance(value, np.integer):
                    json_results[key] = int(value)
                elif key == "episode_results":
                    # Handle episode results separately
                    json_results[key] = []
                    for ep in value:
                        ep_clean = {}
                        for ep_key, ep_val in ep.items():
                            if isinstance(ep_val, (list, np.ndarray)):
                                ep_clean[ep_key] = [float(x) if isinstance(x, np.floating) else x for x in ep_val]
                            elif isinstance(ep_val, np.floating):
                                ep_clean[ep_key] = float(ep_val)
                            elif isinstance(ep_val, np.integer):
                                ep_clean[ep_key] = int(ep_val)
                            else:
                                ep_clean[ep_key] = ep_val
                        json_results[key].append(ep_clean)
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("HANDSHAKE EVALUATION SUMMARY")
        print("="*50)
        print(f"Episodes Completed: {len(episode_results)}/{cfg.eval.num_episodes}")
        print(f"Average Episode Duration: {avg_duration:.1f}s")
        print(f"Average Handshake Confidence: {avg_confidence:.3f}")
        print(f"Average Max Handshake Confidence: {avg_max_confidence:.3f}")
        print(f"Results saved to: {results_path}")
        print("="*50)
        
    else:
        logging.error("No episodes were successfully completed")
    
    # Cleanup
    robot.disconnect()
    logging.info("Handshake evaluation completed")


if __name__ == "__main__":
    eval_handshake()
