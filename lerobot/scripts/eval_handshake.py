"""Evaluate a trained handshake policy on real SO‑101 hardware.

Example
-------
python -m lerobot.scripts.eval_handshake \
    --policy=outputs/train/handshake_policy_v1/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower_arm \
    --robot.cameras='{"front":{"type":"opencv","index_or_path":"/dev/video1","width":640,"height":480,"fps":30}}' \
    --eval.num_episodes=10 \
    --eval.episode_time_s=30 \
    --eval.handshake_timeout_s=15 \
    --display_data=true
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import rerun as rr
import torch

# ---------- LeRobot imports -------------------------------------------------
from lerobot.common.cameras import CameraConfig  # noqa: F401  (side‑effects: register cams)
from lerobot.common.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.common.cameras.realsense.configuration_realsense import (  # noqa: F401
    RealSenseCameraConfig,
)
from lerobot.common.datasets.utils import build_dataset_frame
from lerobot.common.handshake_detection import ImprovedHandshakeDetector
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig

# register the follower variant so `--robot.type=so101_follower` parses
import lerobot.common.robots.so101_follower  # noqa: F401


# ---------------------------------------------------------------------------


@dataclass
class HandshakeEvalConfig:
    """Per‑episode evaluation settings."""
    num_episodes: int = 10
    episode_time_s: float = 30.0
    handshake_timeout_s: float = 10.0
    handshake_confidence_threshold: float = 0.8
    handshake_detection_delay: float = 1.0
    fps: int = 20
    handshake_detection_fps: int = 10


@dataclass
class HandshakeEvalPipelineConfig:
    """Top‑level config parsed from CLI / YAML."""
    robot: RobotConfig
    policy: PreTrainedConfig
    eval: HandshakeEvalConfig
    output_dir: Path = Path("outputs/eval_handshake")
    display_data: bool = False
    device: str = "cuda"
    seed: int | None = None

    # ------------- CLI helpers ---------------------------------------------
    def __post_init__(self) -> None:
        # Allow overriding the entire policy with --policy=<checkpoint_dir>
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=overrides
            )
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        # Treat the *top‑level* policy field as a “path field”
        return ["policy"]


# ---------------------------------------------------------------------------


def wait_for_handshake_detection(
    robot: Robot,
    detector: ImprovedHandshakeDetector,
    camera_name: str,
    timeout_s: float,
    conf_thresh: float,
    delay_s: float,
    display: bool = False,
    episode_idx: int = 0,
) -> bool:
    """Block until a handshake gesture is held for `delay_s` seconds or timeout."""
    start = time.perf_counter()
    detected_at: float | None = None
    last_log = 0.0

    log_say("Waiting for person to extend their hand...", True)

    while time.perf_counter() - start < timeout_s:
        obs = robot.get_observation()
        frame = obs[camera_name]
        res = detector.detect_handshake_gesture(frame, visualize=True)

        ready = res["ready"] and res["confidence"] >= conf_thresh
        if ready:
            detected_at = detected_at or time.perf_counter()
            if time.perf_counter() - detected_at >= delay_s:
                log_say("Handshake detected – starting policy.", True)
                return True
        else:
            detected_at = None

        # minimal visualisation / status
        if display and time.perf_counter() - last_log >= 1.0:
            remaining = timeout_s - (time.perf_counter() - start)
            rr.log(
                "status",
                rr.TextLog(
                    f"WAITING | Episode {episode_idx+1} | "
                    f"Time left: {remaining:.1f}s",
                    level=rr.TextLogLevel.INFO,
                ),
            )
            last_log = time.perf_counter()

        time.sleep(0.05)

    log_say("Handshake timeout – skipping episode.", True)
    return False


# (evaluate_handshake_episode unchanged – omitted for brevity)
# ---------------------------------------------------------------------------
# KEEP the rest of your original evaluate_handshake_episode and helper code
# ---------------------------------------------------------------------------


@parser.wrap()
def eval_handshake(cfg: HandshakeEvalPipelineConfig) -> None:
    """Entry‑point called by draccus."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        _init_rerun(session_name="handshake_evaluation")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ----- robot ----------------------------------------------------------------
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # ----- handshake detector ----------------------------------------------------
    detector = ImprovedHandshakeDetector(
        confidence_threshold=cfg.eval.handshake_confidence_threshold
    )

    main_cam = next(iter(robot.config.cameras.keys()))
    logging.info(f"Using camera '{main_cam}' for gesture detection")

    # ----- policy ----------------------------------------------------------------
    policy = make_policy(cfg=cfg.policy, env_cfg=None)
    policy.eval()

    # ----- evaluation loop -------------------------------------------------------
    results: list[dict[str, Any]] = []

    for ep in range(cfg.eval.num_episodes):
        logging.info(f"Episode {ep+1}/{cfg.eval.num_episodes}")

        if not wait_for_handshake_detection(
            robot,
            detector,
            main_cam,
            cfg.eval.handshake_timeout_s,
            cfg.eval.handshake_confidence_threshold,
            cfg.eval.handshake_detection_delay,
            cfg.display_data,
            ep,
        ):
            continue

        metrics = evaluate_handshake_episode(
            robot,
            policy,
            detector,
            main_cam,
            cfg.eval.episode_time_s,
            cfg.eval.fps,
            cfg.eval.handshake_detection_fps,
            cfg.display_data,
            ep,
        )
        results.append(metrics)
        logging.info(
            f"Episode {ep+1} done – max conf: {metrics['max_handshake_confidence']:.3f}"
        )
        time.sleep(2.0)

    # ----- summary ----------------------------------------------------------------
    if results:
        avg = lambda k: float(np.mean([r[k] for r in results]))
        summary = {
            "episodes_completed": len(results),
            "episodes_attempted": cfg.eval.num_episodes,
            "avg_duration_s": avg("episode_duration"),
            "avg_confidence": avg("avg_handshake_confidence"),
            "avg_max_confidence": avg("max_handshake_confidence"),
            "episode_results": results,
        }
        out = cfg.output_dir / "handshake_eval_results.json"
        out.write_text(json.dumps(summary, indent=2))
        log_say(f"Evaluation complete – summary written to {out}", True)
    else:
        log_say("No successful episodes.", True)

    robot.disconnect()
    logging.info("Evaluation finished.")


if __name__ == "__main__":
    eval_handshake()
