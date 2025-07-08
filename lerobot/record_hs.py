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
            
            # Display annotated frame if requested
            if display_data and 'annotated_frame' in detection_result:
                cv2.imshow('Handshake Detection', detection_result['annotated_frame'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
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

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        sent_action = robot.send_action(action)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)

        if display_data:
            # Integrate pose detection visualization into Rerun viewer
            # (replaces separate OpenCV window with annotated camera feed)
            annotated_frame = None
            if main_camera_name in observation:
                frame = observation[main_camera_name]
                handshake_result = handshake_detector.detect_handshake_gesture(frame, visualize=True)
                if 'annotated_frame' in handshake_result:
                    annotated_frame = handshake_result['annotated_frame']
            
            # Log observations to Rerun
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    # Replace main camera feed with annotated pose detection frame
                    if obs == main_camera_name and annotated_frame is not None:
                        rr.log(f"observation.{obs}_with_pose", rr.Image(annotated_frame), static=True)
                        # Uncomment below to also show raw camera feed for comparison
                        # rr.log(f"observation.{obs}_raw", rr.Image(val), static=True)
                    else:
                        # Log other cameras normally
                        rr.log(f"observation.{obs}", rr.Image(val), static=True)
            
            # Log actions to Rerun
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act}", rr.Scalar(val))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record_handshake(cfg: HandshakeRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="handshake_recording")

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

    log_say("Stop recording handshake dataset", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    # Close OpenCV windows (used during handshake detection wait phase)
    cv2.destroyAllWindows()

    log_say("Exiting", cfg.play_sounds)
    return dataset


if __name__ == "__main__":
    record_handshake()












