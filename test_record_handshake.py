#!/usr/bin/env python

# Test version of record_handshake.py with debugging features

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import cv2
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
from lerobot.common.handshake_detection import ImprovedHandshakeDetector
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


def _init_rerun(session_name: str = "data_collection"):
    rr.init(session_name)
    rr.connect("0.0.0.0:9876")
    rr.log("./", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


@dataclass
class TestHandshakeDatasetRecordConfig:
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
    # Enable verbose debugging
    debug_mode: bool = True

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class TestHandshakeRecordConfig:
    robot: RobotConfig
    dataset: TestHandshakeDatasetRecordConfig
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


def wait_for_handshake_detection_debug(
    robot,
    handshake_detector,
    camera_name: str,
    timeout_s: float,
    confidence_threshold: float,
    detection_delay: float,
    display_data: bool = False,
    debug_mode: bool = True,
) -> bool:
    """
    Wait for handshake detection from the person with debugging.
    
    Returns:
        True if handshake detected, False if timeout
    """
    start_time = time.perf_counter()
    detection_start_time = None
    
    log_say("Waiting for person to extend their hand for handshake...", True)
    
    while time.perf_counter() - start_time < timeout_s:
        observation = robot.get_observation()
        
        # Debug: Print available keys in observation
        if debug_mode:
            logging.info(f"DEBUG: Available observation keys: {list(observation.keys())}")
            logging.info(f"DEBUG: Looking for camera: '{camera_name}'")
            
            # Print types and shapes of observation values
            for key, value in observation.items():
                if hasattr(value, 'shape'):
                    logging.info(f"DEBUG: {key}: {type(value)} shape={value.shape}")
                else:
                    logging.info(f"DEBUG: {key}: {type(value)} value={value}")
        
        if camera_name not in observation:
            logging.warning(f"Camera '{camera_name}' not found in observation. Available cameras: {list(observation.keys())}")
            # Try to find camera with 'images' prefix or any image-like data
            image_keys = [key for key in observation.keys() if 'image' in key.lower() or isinstance(observation[key], np.ndarray) and len(observation[key].shape) == 3]
            if image_keys:
                logging.info(f"Found potential image keys: {image_keys}, trying first one...")
                camera_name = image_keys[0]
                logging.info(f"Using camera key: '{camera_name}'")
            else:
                time.sleep(0.5)  # Wait a bit before retrying
                continue
            
        frame = observation[camera_name]
        
        # Debug frame info
        if debug_mode:
            logging.info(f"DEBUG: Frame type: {type(frame)}, shape: {frame.shape if hasattr(frame, 'shape') else 'no shape'}")
        
        # Detect handshake gesture
        try:
            detection_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
            
            if debug_mode:
                logging.info(f"DEBUG: Detection result: ready={detection_result['ready']}, confidence={detection_result['confidence']:.3f}")
            
            if detection_result['ready'] and detection_result['confidence'] >= confidence_threshold:
                if detection_start_time is None:
                    detection_start_time = time.perf_counter()
                    log_say(f"Handshake detected! Waiting {detection_delay} seconds before starting recording...", True)
                
                # Wait for the specified delay after detection
                if time.perf_counter() - detection_start_time >= detection_delay:
                    log_say("Starting handshake recording now!", True)
                    return True
            else:
                # Reset detection timer if gesture is lost
                detection_start_time = None
            
            # Display annotated frame if requested
            if display_data and 'annotated_frame' in detection_result:
                cv2.imshow('Handshake Detection', detection_result['annotated_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logging.error(f"Error in handshake detection: {e}")
            if debug_mode:
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
        
        time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    log_say("Handshake detection timeout. Skipping this episode.", True)
    return False


@safe_stop_image_writer
def record_test_handshake_episode(
    robot,
    events: dict,
    fps: int,
    handshake_detector,
    main_camera_name: str,
    dataset=None,
    teleop=None,
    policy=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    debug_mode: bool = True,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()
        
        if debug_mode:
            logging.info(f"DEBUG: Episode observation keys: {list(observation.keys())}")

        # Add handshake detection to observation
        if main_camera_name in observation:
            frame = observation[main_camera_name]
            try:
                handshake_result = handshake_detector.detect_handshake_gesture(frame, visualize=False)
                
                # Add handshake detection data to observation
                observation["handshake_ready"] = handshake_result['ready']
                observation["handshake_confidence"] = handshake_result['confidence']
                if handshake_result['hand_position'] is not None:
                    observation["hand_position_x"] = float(handshake_result['hand_position'][0])
                    observation["hand_position_y"] = float(handshake_result['hand_position'][1])
                else:
                    observation["hand_position_x"] = -1.0  # Invalid position marker
                    observation["hand_position_y"] = -1.0
                    
                if debug_mode:
                    logging.info(f"DEBUG: Added handshake data - ready: {observation['handshake_ready']}, confidence: {observation['handshake_confidence']:.3f}")
                    
            except Exception as e:
                logging.error(f"Error adding handshake detection to observation: {e}")
                # Add default values
                observation["handshake_ready"] = False
                observation["handshake_confidence"] = 0.0
                observation["hand_position_x"] = -1.0
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
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
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
            if main_camera_name in observation:
                frame = observation[main_camera_name]
                handshake_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
                if 'annotated_frame' in handshake_result:
                    cv2.imshow('Handshake Detection During Recording', handshake_result['annotated_frame'])
                    cv2.waitKey(1)
            
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation.{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act}", rr.Scalar(val))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def test_record_handshake(cfg: TestHandshakeRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info("=== TEST RECORD HANDSHAKE ===")
    logging.info(pformat(asdict(cfg)))
    
    if cfg.display_data:
        _init_rerun(session_name="test_handshake_recording")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Initialize handshake detector
    try:
        handshake_detector = ImprovedHandshakeDetector(
            confidence_threshold=cfg.dataset.handshake_confidence_threshold
        )
        logging.info(f"Handshake detector initialized with confidence threshold: {cfg.dataset.handshake_confidence_threshold}")
    except ImportError as e:
        logging.error(f"Failed to initialize handshake detector: {e}")
        logging.error("Please install required dependencies: pip install mediapipe opencv-python")
        raise

    # Determine main camera name (assume first camera is the main one)
    if not hasattr(robot.config, 'cameras') or not robot.config.cameras:
        raise ValueError("Robot must have at least one camera configured for handshake detection")
    
    main_camera_name = list(robot.config.cameras.keys())[0]
    logging.info(f"Using camera '{main_camera_name}' for handshake detection")

    # Build dataset features including handshake detection data
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    
    # Add handshake detection features
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
    
    if cfg.dataset.debug_mode:
        logging.info("=== DEBUG: Dataset Features ===")
        for key, value in dataset_features.items():
            logging.info(f"  {key}: {value}")

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

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    recorded_episodes = 0
    while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
        log_say(f"TEST: Preparing to record handshake episode {dataset.num_episodes + 1}", cfg.play_sounds)
        
        # Wait for handshake detection before starting episode (unless skipped)
        if cfg.dataset.skip_handshake_detection:
            log_say("TEST: Skipping handshake detection - starting recording immediately", cfg.play_sounds)
            handshake_detected = True
        else:
            handshake_detected = wait_for_handshake_detection_debug(
                robot=robot,
                handshake_detector=handshake_detector,
                camera_name=main_camera_name,
                timeout_s=cfg.dataset.handshake_timeout_s,
                confidence_threshold=cfg.dataset.handshake_confidence_threshold,
                detection_delay=cfg.dataset.handshake_detection_delay,
                display_data=cfg.display_data,
                debug_mode=cfg.dataset.debug_mode,
            )
        
        if not handshake_detected:
            log_say("TEST: Skipping episode due to handshake detection timeout", cfg.play_sounds)
            continue
        
        log_say(f"TEST: Recording handshake episode {dataset.num_episodes + 1}", cfg.play_sounds)
        record_test_handshake_episode(
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
            log_say("TEST: Reset the environment for next handshake", cfg.play_sounds)
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
            log_say("TEST: Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    log_say("TEST: Stop recording handshake dataset", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    # Close OpenCV windows
    cv2.destroyAllWindows()

    log_say("TEST: Exiting", cfg.play_sounds)
    return dataset


if __name__ == "__main__":
    test_record_handshake() 