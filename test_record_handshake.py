#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Test version of handshake recording with debugging capabilities.
"""

import logging
import time
import traceback
import cv2
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import rerun as rr

from lerobot.common.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
)
from lerobot.common.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so101_leader,
)
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

# Import handshake detection
try:
    from lerobot.common.handshake_detection import ImprovedHandshakeDetector
except ImportError as e:
    logging.error(f"Failed to import handshake detection: {e}")
    ImprovedHandshakeDetector = None


@dataclass
class HandshakeDatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/handshake_data`).
    repo_id: str
    # A short but accurate description of the handshake task (e.g. "Shake hands with person when they extend their hand.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Handshake detection confidence threshold (0-1). Higher values require more confident detection.
    handshake_confidence_threshold: float = 0.8
    # Wait time in seconds after handshake is detected before starting recording
    handshake_detection_delay: float = 1.0
    # Maximum time to wait for handshake detection before timing out (seconds)
    handshake_timeout_s: float = 30.0
    # Skip handshake detection and start recording immediately (for debugging)
    skip_handshake_detection: bool = False
    # Enable verbose debugging output
    debug_mode: bool = False

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class HandshakeRecordConfig:
    robot: RobotConfig
    dataset: HandshakeDatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def test_robot_connection(robot: Robot, debug_mode: bool = False):
    """Test robot connection and camera access"""
    print("\n=== TESTING ROBOT CONNECTION ===")
    
    try:
        print(f"Robot connected: {robot.is_connected}")
        print(f"Robot type: {robot.name}")
        print(f"Robot ID: {robot.id}")
        
        if hasattr(robot, 'cameras'):
            print(f"Number of cameras: {len(robot.cameras)}")
            for cam_name, cam in robot.cameras.items():
                print(f"  Camera '{cam_name}': connected={cam.is_connected}")
        
        print("\n=== TESTING OBSERVATION ===")
        observation = robot.get_observation()
        print(f"Observation keys: {list(observation.keys())}")
        
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)} = {value}")
        
        return observation
        
    except Exception as e:
        print(f"ERROR testing robot: {e}")
        if debug_mode:
            traceback.print_exc()
        return None


def test_handshake_detection(observation: dict, debug_mode: bool = False):
    """Test handshake detection on current observation"""
    print("\n=== TESTING HANDSHAKE DETECTION ===")
    
    if ImprovedHandshakeDetector is None:
        print("ERROR: Handshake detector not available")
        return None
    
    try:
        detector = ImprovedHandshakeDetector(confidence_threshold=0.5)
        print("Handshake detector initialized successfully")
        
        # Find camera data
        camera_keys = [key for key, val in observation.items() if isinstance(val, np.ndarray) and len(val.shape) == 3]
        print(f"Found camera keys: {camera_keys}")
        
        if not camera_keys:
            print("ERROR: No camera data found in observation")
            return None
        
        # Test detection on first camera
        camera_key = camera_keys[0]
        frame = observation[camera_key]
        print(f"Testing detection on camera '{camera_key}' with frame shape: {frame.shape}")
        
        result = detector.detect_handshake_gesture(frame, visualize=True)
        print(f"Detection result: {result}")
        
        if 'annotated_frame' in result and debug_mode:
            cv2.imshow('Test Handshake Detection', result['annotated_frame'])
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result
        
    except Exception as e:
        print(f"ERROR testing handshake detection: {e}")
        if debug_mode:
            traceback.print_exc()
        return None


def wait_for_handshake_detection_test(
    robot: Robot,
    handshake_detector: ImprovedHandshakeDetector,
    camera_name: str,
    timeout_s: float,
    confidence_threshold: float,
    detection_delay: float,
    debug_mode: bool = False,
    display_data: bool = False,
) -> bool:
    """
    Test version of wait_for_handshake_detection with more debugging.
    """
    print(f"\n=== WAITING FOR HANDSHAKE DETECTION ===")
    print(f"Camera: {camera_name}")
    print(f"Timeout: {timeout_s}s")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Detection delay: {detection_delay}s")
    
    start_time = time.perf_counter()
    detection_start_time = None
    iteration_count = 0
    
    log_say("Waiting for person to extend their hand for handshake...", True)
    
    while time.perf_counter() - start_time < timeout_s:
        iteration_count += 1
        
        try:
            observation = robot.get_observation()
            
            if debug_mode and iteration_count % 30 == 1:  # Log every 30 iterations (~3 seconds)
                print(f"Iteration {iteration_count}: Available keys: {list(observation.keys())}")
                elapsed = time.perf_counter() - start_time
                print(f"Elapsed time: {elapsed:.1f}s / {timeout_s}s")
            
            if camera_name not in observation:
                if iteration_count == 1:  # Only log this once
                    print(f"WARNING: Camera '{camera_name}' not found in observation.")
                    print(f"Available keys: {list(observation.keys())}")
                    # Try to find camera with different names
                    image_keys = [key for key in observation.keys() if isinstance(observation[key], np.ndarray) and len(observation[key].shape) == 3]
                    if image_keys:
                        print(f"Found image keys: {image_keys}, using first one: {image_keys[0]}")
                        camera_name = image_keys[0]
                    else:
                        print("ERROR: No camera data found!")
                        continue
                else:
                    continue
                    
            frame = observation[camera_name]
            
            if debug_mode and iteration_count == 1:
                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            # Detect handshake gesture
            detection_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
            
            if debug_mode and iteration_count % 30 == 1:
                print(f"Detection result: ready={detection_result['ready']}, confidence={detection_result['confidence']:.3f}")
            
            if detection_result['ready'] and detection_result['confidence'] >= confidence_threshold:
                if detection_start_time is None:
                    detection_start_time = time.perf_counter()
                    log_say(f"Handshake detected! Waiting {detection_delay} seconds before starting recording...", True)
                    print(f"Detection confidence: {detection_result['confidence']:.3f}")
                
                # Wait for the specified delay after detection
                if time.perf_counter() - detection_start_time >= detection_delay:
                    log_say("Starting handshake recording now!", True)
                    return True
            else:
                # Reset detection timer if gesture is lost
                if detection_start_time is not None and debug_mode:
                    print("Handshake gesture lost, resetting detection timer")
                detection_start_time = None
            
            # Display annotated frame if requested
            if display_data and 'annotated_frame' in detection_result:
                cv2.imshow('Handshake Detection Test', detection_result['annotated_frame'])
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User requested quit")
                    break
                elif key == ord('s'):
                    print("User requested skip detection")
                    return True
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
        except KeyboardInterrupt:
            print("KeyboardInterrupt received!")
            return False
        except Exception as e:
            print(f"ERROR in handshake detection loop: {e}")
            if debug_mode:
                traceback.print_exc()
            time.sleep(0.1)
    
    log_say("Handshake detection timeout. Skipping this episode.", True)
    return False


@safe_stop_image_writer
def record_handshake_loop_test(
    robot: Robot,
    events: dict,
    fps: int,
    handshake_detector: ImprovedHandshakeDetector | None,
    main_camera_name: str,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    debug_mode: bool = False,
):
    """Test version of record loop with debugging"""
    print(f"\n=== STARTING RECORD LOOP ===")
    print(f"Control time: {control_time_s}s")
    print(f"FPS: {fps}")
    print(f"Main camera: {main_camera_name}")
    
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    frame_count = 0
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()
        frame_count += 1

        if events["exit_early"]:
            print("Exit early requested")
            events["exit_early"] = False
            break

        try:
            observation = robot.get_observation()
            
            if debug_mode and frame_count % 60 == 1:  # Log every 60 frames (~2 seconds)
                print(f"Frame {frame_count}: timestamp={timestamp:.1f}s/{control_time_s}s")

            # Add handshake detection to observation if detector available
            if handshake_detector and main_camera_name in observation:
                frame = observation[main_camera_name]
                handshake_result = handshake_detector.detect_handshake_gesture(frame, visualize=False)
                
                # Add handshake detection data to observation
                observation["handshake_ready"] = int(handshake_result['ready'])  # Convert bool to int
                observation["handshake_confidence"] = handshake_result['confidence']
                if handshake_result['hand_position'] is not None:
                    observation["hand_position_x"] = float(handshake_result['hand_position'][0])
                    observation["hand_position_y"] = float(handshake_result['hand_position'][1])
                else:
                    observation["hand_position_x"] = -1.0  # Invalid position marker
                    observation["hand_position_y"] = -1.0

            if policy is not None or dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

            if policy is not None:
                action_values = predict_action(
                    observation_frame,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )
                action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
            elif policy is None and teleop is not None:
                action = teleop.get_action()
            else:
                if debug_mode and frame_count == 1:
                    print("No policy or teleoperator provided, skipping action generation.")
                continue

            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset.
            sent_action = robot.send_action(action)

            if dataset is not None:
                action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
                frame = {**observation_frame, **action_frame}
                dataset.add_frame(frame, task=single_task)

            if display_data:
                # Display handshake detection results
                if handshake_detector and main_camera_name in observation:
                    frame = observation[main_camera_name]
                    handshake_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
                    if 'annotated_frame' in handshake_result:
                        cv2.imshow('Handshake Detection During Recording', handshake_result['annotated_frame'])
                        cv2.waitKey(1)
                
                for obs, val in observation.items():
                    if isinstance(val, (float, int)):
                        rr.log(f"observation.{obs}", rr.Scalar(val))
                    elif isinstance(val, np.ndarray):
                        rr.log(f"observation.{obs}", rr.Image(val), static=True)
                for act, val in action.items():
                    if isinstance(val, (float, int)):
                        rr.log(f"action.{act}", rr.Scalar(val))

            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            timestamp = time.perf_counter() - start_episode_t
            
        except KeyboardInterrupt:
            print("KeyboardInterrupt in record loop!")
            break
        except Exception as e:
            print(f"ERROR in record loop: {e}")
            if debug_mode:
                traceback.print_exc()
            break

    print(f"Record loop completed: {frame_count} frames, {timestamp:.1f}s")


@parser.wrap()
def test_record_handshake(cfg: HandshakeRecordConfig) -> LeRobotDataset:
    """Test version of handshake recording with extensive debugging"""
    init_logging()
    logging.info("=== STARTING TEST HANDSHAKE RECORDING ===")
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        _init_rerun(session_name="test_handshake_recording")

    print("\n=== CREATING ROBOT ===")
    robot = make_robot_from_config(cfg.robot)
    
    print("\n=== CREATING TELEOPERATOR ===")
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Initialize handshake detector
    handshake_detector = None
    if not cfg.dataset.skip_handshake_detection:
        try:
            if ImprovedHandshakeDetector is None:
                raise ImportError("ImprovedHandshakeDetector not available")
            handshake_detector = ImprovedHandshakeDetector(
                confidence_threshold=cfg.dataset.handshake_confidence_threshold
            )
            print(f"Handshake detector initialized with confidence threshold: {cfg.dataset.handshake_confidence_threshold}")
        except ImportError as e:
            print(f"Failed to initialize handshake detector: {e}")
            print("Please install required dependencies: pip install mediapipe opencv-python")
            if not cfg.dataset.skip_handshake_detection:
                raise
    else:
        print("Skipping handshake detection (debug mode)")

    # Determine main camera name (assume first camera is the main one)
    main_camera_name = None
    if hasattr(robot.config, 'cameras') and robot.config.cameras:
        main_camera_name = list(robot.config.cameras.keys())[0]
        print(f"Using camera '{main_camera_name}' for handshake detection")
    else:
        print("WARNING: No cameras configured")

    # Build dataset features including handshake detection data
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    
    # Add handshake detection features if detector available
    handshake_features = {}
    if handshake_detector:
        handshake_features = {
            "observation.handshake_ready": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "observation.handshake_confidence": {
                "dtype": "float32", 
                "shape": (1,),
                "names": None,
            },
            "observation.hand_position_x": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "observation.hand_position_y": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
        }
    
    dataset_features = {**action_features, **obs_features, **handshake_features}
    print(f"\nDataset features: {list(dataset_features.keys())}")

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    print("\n=== CONNECTING TO ROBOT ===")
    robot.connect()
    if teleop is not None:
        teleop.connect()

    # Test robot connection
    observation = test_robot_connection(robot, cfg.dataset.debug_mode)
    if observation is None:
        print("ERROR: Could not get robot observation")
        return None

    # Test handshake detection if available
    if handshake_detector and observation:
        test_handshake_detection(observation, cfg.dataset.debug_mode)

    listener, events = init_keyboard_listener()
    print("Keyboard listener initialized. Use Space to start/stop, Q to quit, R to re-record")

    recorded_episodes = 0
    try:
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            log_say(f"Preparing to record handshake episode {dataset.num_episodes + 1}", cfg.play_sounds)
            
            # Wait for handshake detection before starting episode (unless skipped)
            handshake_detected = True  # Default to True
            if not cfg.dataset.skip_handshake_detection and handshake_detector and main_camera_name:
                handshake_detected = wait_for_handshake_detection_test(
                    robot=robot,
                    handshake_detector=handshake_detector,
                    camera_name=main_camera_name,
                    timeout_s=cfg.dataset.handshake_timeout_s,
                    confidence_threshold=cfg.dataset.handshake_confidence_threshold,
                    detection_delay=cfg.dataset.handshake_detection_delay,
                    debug_mode=cfg.dataset.debug_mode,
                    display_data=cfg.display_data,
                )
            else:
                log_say("Skipping handshake detection - starting recording immediately", cfg.play_sounds)
            
            if not handshake_detected:
                log_say("Skipping episode due to handshake detection timeout", cfg.play_sounds)
                continue
            
            log_say(f"Recording handshake episode {dataset.num_episodes + 1}", cfg.play_sounds)
            record_handshake_loop_test(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                handshake_detector=handshake_detector,
                main_camera_name=main_camera_name,
                teleop=teleop,
                policy=policy,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
                debug_mode=cfg.dataset.debug_mode,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment for next handshake", cfg.play_sounds)
                # Use regular record loop for reset (without handshake detection)
                from lerobot.record import record_loop
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop=teleop,
                    control_time_s=cfg.dataset.reset_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode", cfg.play_sounds)
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    except KeyboardInterrupt:
        print("\n=== KEYBOARD INTERRUPT RECEIVED ===")
        log_say("Recording interrupted by user", cfg.play_sounds)
    except Exception as e:
        print(f"\n=== ERROR DURING RECORDING ===")
        print(f"Error: {e}")
        if cfg.dataset.debug_mode:
            traceback.print_exc()
    finally:
        print("\n=== CLEANUP ===")
        log_say("Stop recording handshake dataset", cfg.play_sounds, blocking=True)

        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()

        if not is_headless() and listener is not None:
            listener.stop()

        # Close OpenCV windows
        cv2.destroyAllWindows()

    log_say("Exiting", cfg.play_sounds)
    return dataset


if __name__ == "__main__":
    test_record_handshake() 