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


def find_working_camera_index(max_index: int = 3) -> int:
    """
    Find a working camera index by testing multiple indices.
    
    Args:
        max_index: Maximum camera index to test
        
    Returns:
        Working camera index, or raises ValueError if none found
    """
    logging.info("🔍 Searching for working camera...")
    
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    logging.info(f"✅ Found working camera at index {i} ({width}x{height})")
                    cap.release()
                    return i
            cap.release()
        except Exception as e:
            logging.debug(f"Camera index {i} failed: {e}")
    
    raise ValueError(f"❌ No working camera found in indices 0-{max_index-1}")


def update_camera_config_with_working_index(robot_config: RobotConfig) -> RobotConfig:
    """
    Update robot camera configuration with a working camera index.
    
    Args:
        robot_config: Robot configuration with camera settings
        
    Returns:
        Updated robot configuration with working camera index
    """
    if not hasattr(robot_config, 'cameras') or not robot_config.cameras:
        return robot_config
    
    # Find working camera
    try:
        working_index = find_working_camera_index()
    except ValueError as e:
        logging.error(str(e))
        raise
    
    # Update all OpenCV camera configs with working index
    updated_cameras = {}
    for cam_name, cam_config in robot_config.cameras.items():
        if hasattr(cam_config, 'type') and cam_config.type == "opencv":
            # Create new config with working index
            new_config = OpenCVCameraConfig(
                index_or_path=working_index,
                width=cam_config.width,
                height=cam_config.height,
                fps=cam_config.fps,
            )
            updated_cameras[cam_name] = new_config
            logging.info(f"📷 Updated camera '{cam_name}' to use index {working_index}")
        else:
            updated_cameras[cam_name] = cam_config
    
    # Update robot config
    robot_config.cameras = updated_cameras
    return robot_config


@dataclass
class HandshakeDatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 30
    reset_time_s: int | float = 10
    num_episodes: int = 50
    video: bool = True

    handshake_confidence_threshold: float = 0.8
    handshake_detection_delay: float = 1.0
    handshake_timeout_s: float = 10.0

    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class HandshakeRecordConfig:
    robot: RobotConfig
    dataset: HandshakeDatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
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


def wait_for_handshake_detection(
    robot: Robot,
    handshake_detector: ImprovedHandshakeDetector,
    camera_name: str,
    timeout_s: float,
    confidence_threshold: float,
    detection_delay: float,
    display_data: bool = False,
) -> bool:
    """
    Wait for handshake detection from the person.
    
    Returns:
        True if handshake detected, False if timeout
    """
    start_time = time.perf_counter()
    detection_start_time = None
    
    log_say("Waiting for person to extend their hand for handshake...", True)
    
    while time.perf_counter() - start_time < timeout_s:
        try:
            observation = robot.get_observation()
            
            # Guard against None observation
            if observation is None:
                logging.warning("Robot observation is None during handshake detection")
                time.sleep(0.1)
                continue
                
            frame = observation[camera_name]
            
            # Detect handshake gesture
            detection_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
            
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
            
            # Always display data to Rerun during waiting phase with better timeline structure
            current_time = time.perf_counter() - start_time
            
            # Status indicators (separate timeline)
            rr.set_time_sequence("waiting_timeline", int(current_time * 10))
            rr.log("Status/recording_phase", rr.Scalar(0))  # 0 = waiting
            rr.log("Status/time_remaining", rr.Scalar(timeout_s - current_time))
            
            # Handshake detection status (separate timeline)
            rr.set_time_sequence("handshake_timeline", int(current_time * 10))
            rr.log("Handshake/confidence", rr.Scalar(detection_result['confidence']))
            rr.log("Handshake/ready", rr.Scalar(detection_result['ready']))
            
            # Robot joint states only (6 values in separate group)
            rr.set_time_sequence("robot_timeline", int(current_time * 10))
            for obs, val in observation.items():
                if isinstance(val, float) and obs.endswith('.pos'):
                    joint_name = obs.replace('.pos', '')
                    rr.log(f"RobotStates/{joint_name}", rr.Scalar(val))
                elif isinstance(val, np.ndarray) and obs == camera_name:
                    # Show pose detection camera feed (separate timeline)
                    rr.set_time_sequence("camera_timeline", int(current_time * 10))
                    annotated_frame = detection_result.get('annotated_frame')
                    if annotated_frame is not None:
                        rr.log(f"Camera/{obs}_with_pose", rr.Image(annotated_frame), static=True)
            
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


@safe_stop_image_writer
def record_handshake_loop(
    robot: Robot,
    events: dict,
    fps: int,
    handshake_detector: ImprovedHandshakeDetector,
    main_camera_name: str,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    episode_number: int = 0,
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

        # Guard against None observation to prevent TypeError
        if observation is None:
            logging.warning("Robot observation is None, skipping this cycle")
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
            continue

        # Add handshake detection to observation
        if main_camera_name in observation:
            frame = observation[main_camera_name]
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

        # Guard against None action to prevent TypeError
        if action is None:
            logging.warning("Action is None, skipping this cycle")
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)
            continue

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        sent_action = robot.send_action(action)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)

        if display_data:
            # Set timeline for better Rerun organization
            rr.set_time_seconds("timestamp", timestamp)
            
            # Status indicators (completely separate timeline)
            rr.set_time_sequence("status_timeline", int(timestamp * fps))
            rr.log("Status/recording_phase", rr.Scalar(1))  # 1 = recording
            rr.log("Status/episode_number", rr.Scalar(episode_number))
            rr.log("Status/episode_progress", rr.Scalar(min(1.0, timestamp / control_time_s)))
            rr.log("Status/time_remaining", rr.Scalar(max(0, control_time_s - timestamp)))
            
            # Handshake detection status (separate timeline)
            rr.set_time_sequence("handshake_timeline", int(timestamp * fps))
            if main_camera_name in observation:
                rr.log("Handshake/confidence", rr.Scalar(observation.get("handshake_confidence", 0.0)))
                rr.log("Handshake/ready", rr.Scalar(observation.get("handshake_ready", 0)))
            
            # Robot joint states (separate timeline) - only 6 position values
            rr.set_time_sequence("robot_timeline", int(timestamp * fps))
            for obs, val in observation.items():
                if isinstance(val, float) and obs.endswith('.pos'):
                    # Create individual entities for each joint
                    joint_name = obs.replace('.pos', '')
                    rr.log(f"RobotStates/{joint_name}", rr.Scalar(val))
                elif isinstance(val, np.ndarray) and obs == main_camera_name:
                    # Show pose detection camera feed (separate timeline)
                    rr.set_time_sequence("camera_timeline", int(timestamp * fps))
                    frame = observation[main_camera_name]
                    handshake_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
                    if 'annotated_frame' in handshake_result:
                        rr.log(f"Camera/{obs}_with_pose", rr.Image(handshake_result['annotated_frame']), static=True)
            
            # Robot actions (separate timeline) - 6 values
            rr.set_time_sequence("action_timeline", int(timestamp * fps))
            for act, val in action.items():
                if isinstance(val, float):
                    # Create individual entities for each action
                    rr.log(f"RobotActions/{act}", rr.Scalar(val))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record_handshake(cfg: HandshakeRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    
    # Always initialize Rerun for live monitoring (robot states + pose detection)
    _init_rerun(session_name="handshake_recording")

    # Robust camera detection: automatically find working camera index
    try:
        cfg.robot = update_camera_config_with_working_index(cfg.robot)
    except ValueError as e:
        logging.error(f"Camera detection failed: {e}")
        logging.error("Please check your camera connections and try again.")
        raise

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

    # Initialize status indicators in Rerun
    rr.set_time_sequence("status_timeline", 0)
    rr.log("Status/recording_phase", rr.Scalar(0))  # 0 = waiting

    listener, events = init_keyboard_listener()

    recorded_episodes = 0
    while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
        log_say(f"Preparing to record handshake episode {dataset.num_episodes + 1}", cfg.play_sounds)
        
        # Wait for handshake detection before starting episode
        handshake_detected = wait_for_handshake_detection(
            robot=robot,
            handshake_detector=handshake_detector,
            camera_name=main_camera_name,
            timeout_s=cfg.dataset.handshake_timeout_s,
            confidence_threshold=cfg.dataset.handshake_confidence_threshold,
            detection_delay=cfg.dataset.handshake_detection_delay,
            display_data=cfg.display_data,
        )
        
        if not handshake_detected:
            log_say("Skipping episode due to handshake detection timeout", cfg.play_sounds)
            continue
        
        log_say(f"Recording handshake episode {dataset.num_episodes + 1}", cfg.play_sounds)
        record_handshake_loop(
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
            episode_number=dataset.num_episodes + 1,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment for next handshake", cfg.play_sounds)
            
            # Add reset phase indicator to Rerun
            rr.set_time_sequence("status_timeline", int(time.perf_counter()))
            rr.log("Status/recording_phase", rr.Scalar(2))  # 2 = resetting
            
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

    log_say("Stop recording handshake dataset", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    # Close any OpenCV resources (cleanup)
    cv2.destroyAllWindows()

    log_say("Exiting", cfg.play_sounds)
    return dataset


if __name__ == "__main__":
    record_handshake()












