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
from dataclasses import asdict, dataclass, field
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
from lerobot.common.policies.factory import make_policy, get_policy_class
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig 
from lerobot.common.envs.configs import EnvConfig, PolicyFeature
from lerobot.configs.types import FeatureType
import lerobot.common.robots.so101_follower # noqa: F401


def create_handshake_env_config(robot: Robot, policy_config: PreTrainedConfig | None = None) -> EnvConfig:
    """Create a minimal environment config for handshake evaluation that matches the robot's features and policy requirements."""
    
    # Create features based on robot's observation and action features
    features = {}
    features_map = {}
    
    # Add action features - group all robot actions into a single "action" feature
    action_features = []
    for action_name, action_type in robot.action_features.items():
        if action_type == float:
            action_features.append(action_name)
    
    if action_features:
        features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(len(action_features),))
        features_map["action"] = "action"
    
    # Add observation features
    state_features = []
    for obs_name, obs_type in robot.observation_features.items():
        if obs_type == float:
            # Robot joint positions - collect for state feature
            state_features.append(obs_name)
        elif isinstance(obs_type, tuple) and len(obs_type) == 3:
            # Camera images - map to observation.images.{camera_name}
            features[f"observation.images.{obs_name}"] = PolicyFeature(type=FeatureType.VISUAL, shape=obs_type)
            features_map[f"observation.images.{obs_name}"] = f"observation.images.{obs_name}"
    
    # Add robot state feature (all joint positions grouped together)
    if state_features:
        features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(len(state_features),))
        features_map["observation.state"] = "observation.state"
    
    # Add handshake detection features as environment state
    features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(4,))
    features_map["observation.environment_state"] = "observation.environment_state"
    
    # If we have a policy config, ensure we include any required features that the policy was trained with
    if policy_config and hasattr(policy_config, 'input_features'):
        for key, feature in policy_config.input_features.items():
            if key not in features:
                # Add missing features that the policy expects
                features[key] = feature
                features_map[key] = key
                logging.info(f"Added missing feature from policy config: {key} with shape {feature.shape}")
    
    class HandshakeEnvConfig(EnvConfig):
        task: str = "handshake_evaluation"
        fps: int = 20
        features: dict[str, PolicyFeature] = field(default_factory=lambda: features)
        features_map: dict[str, str] = field(default_factory=lambda: features_map)
        
        @property
        def gym_kwargs(self) -> dict:
            return {}
    
    return HandshakeEnvConfig()


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
    policy: PreTrainedConfig | None = None
    eval: HandshakeEvalConfig = field(default_factory=HandshakeEvalConfig)
    output_dir: Path = Path("outputs/eval_handshake")
    display_data: bool = False
    device: str = "cuda"
    seed: int | None = None
    
    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("A policy path must be provided via --policy.path=local/dir for handshake evaluation")
    
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
        
        # Build observation frame for policy - create the structure the policy expects
        observation_frame = {}
        
        # Add camera images if the policy expects them
        for cam_name, cam_image in observation.items():
            if isinstance(cam_image, np.ndarray) and cam_image.ndim == 3:
                # Check if policy expects this camera
                expected_key = f"observation.images.{cam_name}"
                if expected_key in policy.config.input_features:
                    observation_frame[expected_key] = cam_image
                else:
                    # If policy doesn't expect this camera, try the generic key
                    if "observation.images" in policy.config.input_features:
                        observation_frame["observation.images"] = cam_image
                        break  # Only add one image if using generic key
        
        # Add robot state (all joint positions grouped together)
        if "observation.state" in policy.config.input_features:
            state_values = []
            for obs_name, obs_value in observation.items():
                if isinstance(obs_value, float) and obs_name.endswith('.pos'):
                    state_values.append(obs_value)
            
            if state_values:
                observation_frame["observation.state"] = np.array(state_values, dtype=np.float32)
        
        # Add handshake features as environment state
        if "observation.environment_state" in policy.config.input_features:
            handshake_values = [
                observation["handshake_ready"],
                observation["handshake_confidence"], 
                observation["hand_position_x"],
                observation["hand_position_y"]
            ]
            observation_frame["observation.environment_state"] = np.array(handshake_values, dtype=np.float32)
        
        # Log what we're providing vs what the policy expects
        logging.debug(f"Policy expects: {list(policy.config.input_features.keys())}")
        logging.debug(f"Providing: {list(observation_frame.keys())}")
        
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
    robot = None
    try:
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
    except Exception as e:
        logging.error(f"Failed to initialize or connect to robot: {e}")
        raise
    
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
    
    # Load trained policy (after robot is connected so we can access its features)
    logging.info("Loading trained handshake policy")
    try:
        # For pretrained policies, we need to load them directly without overriding features
        # The make_policy function overrides features even for pretrained policies, which breaks validation
        policy_cls = get_policy_class(cfg.policy.type)
        
        # Load the policy directly using from_pretrained to preserve original features
        policy = policy_cls.from_pretrained(
            pretrained_name_or_path=cfg.policy.pretrained_path,
            config=cfg.policy,
            **{"dataset_stats": None}  # No dataset stats for evaluation
        )
        policy.eval()
        
        logging.info(f"Policy loaded successfully. Input features: {list(policy.config.input_features.keys())}")
        logging.info(f"Policy output features: {list(policy.config.output_features.keys())}")
        
    except Exception as e:
        logging.error(f"Failed to load policy: {e}")
        raise

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
    if robot is not None:
        try:
            robot.disconnect()
        except Exception as e:
            logging.warning(f"Error during robot disconnect: {e}")
    logging.info("Handshake evaluation completed")


if __name__ == "__main__":
    eval_handshake()
