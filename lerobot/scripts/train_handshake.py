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
Training script specialized for handshake interaction tasks.

This script trains policies to perform handshake gestures with humans using SO-101 robot arms.
It expects datasets recorded with handshake detection data and is optimized for imitation learning
of human-robot handshake interactions.

Example usage:

```bash
python lerobot/scripts/train_handshake.py \
    --dataset.repo_id=your_username/handshake_dataset \
    --policy.type=act \
    --output_dir=outputs/train/handshake_act \
    --job_name=handshake_training \
    --policy.device=cuda \
    --wandb.enable=true
```

The script will:
1. Load a handshake dataset (recorded with record_handshake.py)
2. Validate that handshake detection features are present
3. Train a policy to imitate the recorded handshake actions
4. Log handshake-specific metrics during training
5. Save trained models for deployment on SO-101 robots
"""

import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def validate_handshake_dataset(dataset):
    """
    Validate that the dataset contains required handshake detection features.
    
    Args:
        dataset: LeRobotDataset instance
        
    Raises:
        ValueError: If required handshake features are missing
    """
    required_features = [
        "observation.handshake_ready",
        "observation.handshake_confidence", 
        "observation.hand_position_x",
        "observation.hand_position_y"
    ]
    
    missing_features = []
    for feature in required_features:
        if feature not in dataset.features:
            missing_features.append(feature)
    
    if missing_features:
        raise ValueError(
            f"Dataset is missing required handshake detection features: {missing_features}. "
            f"Please record data using record_handshake.py to include handshake detection."
        )
    
    logging.info("✓ Dataset contains all required handshake detection features")


def compute_handshake_metrics(batch: dict) -> dict:
    """
    Compute handshake-specific metrics from a training batch.
    
    Args:
        batch: Training batch dictionary
        
    Returns:
        Dictionary of handshake metrics
    """
    metrics = {}
    
    # Extract handshake detection data if present
    if "observation.handshake_ready" in batch:
        handshake_ready = batch["observation.handshake_ready"]
        if isinstance(handshake_ready, torch.Tensor):
            # Percentage of frames where handshake was detected
            handshake_detection_rate = handshake_ready.float().mean().item() * 100
            metrics["handshake_detection_rate"] = handshake_detection_rate
    
    if "observation.handshake_confidence" in batch:
        handshake_conf = batch["observation.handshake_confidence"]
        if isinstance(handshake_conf, torch.Tensor):
            # Average confidence of handshake detection
            avg_confidence = handshake_conf.mean().item()
            metrics["avg_handshake_confidence"] = avg_confidence
    
    return metrics


def update_handshake_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """
    Update policy with handshake-specific metrics tracking.
    
    This function extends the standard policy update with handshake-specific
    metrics computation and logging.
    """
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        
        # Add handshake-specific metrics to output
        handshake_metrics = compute_handshake_metrics(batch)
        if output_dict is None:
            output_dict = {}
        output_dict.update(handshake_metrics)
    
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train_handshake(cfg: TrainPipelineConfig):
    """
    Train a policy for handshake interaction tasks.
    
    This function specializes the standard training pipeline for handshake tasks,
    including validation of handshake features and tracking of handshake-specific metrics.
    """
    cfg.validate()
    logging.info("=" * 60)
    logging.info("STARTING HANDSHAKE POLICY TRAINING")
    logging.info("=" * 60)
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating handshake dataset")
    dataset = make_dataset(cfg)
    
    # Validate that dataset contains handshake detection features
    validate_handshake_dataset(dataset)

    # For handshake tasks, we typically don't use simulation environments
    # as evaluation is done on real robots during deployment
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.warning(
            "Environment evaluation is enabled but not recommended for handshake tasks. "
            "Handshake policies are typically evaluated on real robots."
        )
        logging.info("Creating evaluation environment")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating handshake policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    # Log training information
    logging.info(colored("Handshake Training Configuration:", "green", attrs=["bold"]))
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"Task: Handshake Interaction Learning")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"Training steps: {cfg.steps} ({format_big_number(cfg.steps)})")
    logging.info(f"Dataset frames: {dataset.num_frames} ({format_big_number(dataset.num_frames)})")
    logging.info(f"Dataset episodes: {dataset.num_episodes}")
    logging.info(f"Learnable parameters: {num_learnable_params} ({format_big_number(num_learnable_params)})")
    logging.info(f"Total parameters: {num_total_params} ({format_big_number(num_total_params)})")

    # Create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    # Define training metrics including handshake-specific ones
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        # Handshake-specific metrics
        "handshake_detection_rate": AverageMeter("hs_rate", ":.1f"),
        "avg_handshake_confidence": AverageMeter("hs_conf", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("🤝 Starting offline training on handshake dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_handshake_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )
        
        # Update handshake-specific metrics
        if output_dict:
            if "handshake_detection_rate" in output_dict:
                train_tracker.handshake_detection_rate = output_dict["handshake_detection_rate"]
            if "avg_handshake_confidence" in output_dict:
                train_tracker.avg_handshake_confidence = output_dict["avg_handshake_confidence"]

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(f"🤝 Handshake Training - {train_tracker}")
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"💾 Checkpoint handshake policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"📊 Eval handshake policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    policy,
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env:
        eval_env.close()
    
    logging.info("🎉 Handshake policy training completed successfully!")
    logging.info(f"📁 Trained model saved in: {cfg.output_dir}")
    logging.info("🚀 Ready for deployment on SO-101 robot!")

    if cfg.policy.push_to_hub:
        logging.info("⬆️  Pushing handshake policy to HuggingFace Hub...")
        policy.push_model_to_hub(cfg)


if __name__ == "__main__":
    init_logging()
    train_handshake()
