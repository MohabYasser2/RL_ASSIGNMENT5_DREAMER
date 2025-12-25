"""Run a trained Dreamer checkpoint in evaluation mode.

Usage examples (PowerShell):
  python scripts\run_evaluation.py --config SpaceInvadersNoFrameskip-v4.yml --checkpoint "checkpoints\SpaceInvadersNoFrameskip-v4_YourRunName_*" --episodes 5 --save_video

This script finds the latest matching checkpoint (if a glob/pattern is provided),
loads the model onto the detected device (GPU if available), and runs
`Dreamer.environmentInteraction` with `evaluation=True`.

This file is created by the assistant on user request and is not executed here.
"""
import argparse
import glob
import os
import sys
import time

import gymnasium as gym
import torch
import numpy as np
import plotly.graph_objects as pgo

from envs import CleanGymWrapper, GymPixelsProcessingWrapper, getEnvProperties
from utils import loadConfig
from dreamer import Dreamer


def find_latest_checkpoint(pattern):
    # Accept either a concrete path or a glob pattern
    if os.path.exists(pattern) and os.path.isfile(pattern):
        return pattern
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    # Choose most recently modified file for 'latest'
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="SpaceInvadersNoFrameskip-v4.yml")
    parser.add_argument("--checkpoint", default=None, help="Exact .pth checkpoint file or glob/pattern")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_out", default=None, help="Output filename (without extension). If omitted, uses config videos folder")
    parser.add_argument("--no_render", action="store_true", help="Run evaluation without render_mode (headless, faster)")
    parser.add_argument("--save_plot", action="store_true", help="Save rewards plot (HTML) after evaluation")
    parser.add_argument("--plot_out", default=None, help="Path to save the rewards plot (HTML). Defaults to config plots folder")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb initialization and syncing for faster evaluation")
    args = parser.parse_args()

    cfg = loadConfig(args.config)

    # Optional Weights & Biases (wandb) integration: initialize a run for evaluation
    try:
        import wandb
        _WANDB_AVAILABLE = True
    except Exception:
        wandb = None
        _WANDB_AVAILABLE = False

    # Respect CLI flag or WANDB_MODE env variable to disable wandb during evaluation
    if args.disable_wandb or os.environ.get("WANDB_MODE", "") == "disabled":
        _WANDB_AVAILABLE = False

    if _WANDB_AVAILABLE:
        try:
            wandb_project = cfg.get("wandbProject", "natural-dreamer")
        except Exception:
            wandb_project = "natural-dreamer"
        try:
            # reinit=True allows multiple runs in the same process during interactive use
            if wandb.run is None:
                wandb.init(project=wandb_project, name=f"{cfg.environmentName}_{cfg.runName}_eval", config=cfg, reinit=True)
        except Exception as _e:
            print(f"Warning: wandb.init() failed: {_e}")

    # device selection (same policy as main.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable cuDNN autotuner for fixed-size inputs (can improve conv performance)
    try:
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    runName = f"{cfg.environmentName}_{cfg.runName}"

    # Build environment like in main.py: resized (64x64) and wrapped
    # Use `render_mode='rgb_array'` only when saving video; otherwise create
    # a headless env (faster/no rendering) when `--no_render` is provided.
    if args.save_video and not args.no_render:
        env_inst = gym.make(cfg.environmentName, render_mode="rgb_array")
    else:
        env_inst = gym.make(cfg.environmentName)
    env = CleanGymWrapper(GymPixelsProcessingWrapper(gym.wrappers.ResizeObservation(env_inst, (64, 64))))

    observationShape, isDiscrete, actionSize, actionLow, actionHigh = getEnvProperties(env)

    # Create Dreamer and load checkpoint
    dreamer = Dreamer(observationShape, isDiscrete, actionSize, actionLow, actionHigh, device, cfg.dreamer)

    ckpt = args.checkpoint
    if ckpt is None:
        # default pattern based on main.py
        base_pattern = os.path.join(cfg.folderNames.checkpointsFolder, f"{runName}_*")
        ckpt = find_latest_checkpoint(base_pattern + ".pth") or find_latest_checkpoint(base_pattern)
    else:
        if any(ch in ckpt for ch in ['*', '?', '[']):
            ckpt = find_latest_checkpoint(ckpt)

    if not ckpt:
        print("No checkpoint found. Tried:")
        if args.checkpoint:
            print(f"  pattern: {args.checkpoint}")
        else:
            print(f"  default: {os.path.join(cfg.folderNames.checkpointsFolder, runName + '_*')}.pth")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt}")
    dreamer.loadCheckpoint(ckpt)

    # Decide video filename base
    if args.save_video and not args.no_render:
        if args.video_out:
            video_base = args.video_out
        else:
            video_base = os.path.join(cfg.folderNames.videosFolder, f"eval_{runName}")
    else:
        video_base = None

    # Run `args.episodes` episodes, collecting per-episode rewards.
    rewards = []
    episode_times = []
    total_start = time.time()
    for i in range(args.episodes):
        ep_start = time.time()
        save_video_this = bool(args.save_video and (i == 0) and (not args.no_render))
        filename = video_base if save_video_this else None
        ep_score = dreamer.environmentInteraction(env, 1, seed=cfg.seed, evaluation=True, saveVideo=save_video_this, filename=filename)
        ep_time = time.time() - ep_start
        episode_times.append(ep_time)
        rewards.append(ep_score if ep_score is not None else 0.0)
        print(f"Episode {i+1}/{args.episodes} score: {rewards[-1]:.2f}  time: {ep_time:.2f}s")
    total_time = time.time() - total_start

    rewards = np.array(rewards, dtype=float)
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    print(f"Evaluation over {len(rewards)} episodes — mean: {mean_reward:.2f}, std: {std_reward:.2f}")
    print(f"Total time: {total_time:.2f}s, avg per-episode: {np.mean(episode_times):.2f}s")

    # Optionally save an interactive HTML plot of per-episode rewards and mean±std
    if args.save_plot:
        if args.plot_out:
            plot_path = args.plot_out
        else:
            plot_path = os.path.join(cfg.folderNames.plotsFolder, f"eval_rewards_{runName}.html")

        x = list(range(1, len(rewards) + 1))
        fig = pgo.Figure()
        # shaded mean±std area polygon
        low = mean_reward - std_reward
        high = mean_reward + std_reward
        xs = [*x, *x[::-1]]
        ys = [*(low for _ in x), *(high for _ in x[::-1])]
        fig.add_trace(pgo.Scatter(x=xs, y=ys, fill='toself', fillcolor='rgba(0,100,80,0.15)', line=dict(color='rgba(0,0,0,0)'), showlegend=True, name='mean±std'))
        fig.add_trace(pgo.Scatter(x=x, y=rewards, mode='lines+markers', name='episode_reward'))
        fig.add_hline(y=mean_reward, line_dash='dash', annotation_text=f"mean={mean_reward:.2f}", annotation_position='top left')
        fig.update_layout(title=f"Evaluation Rewards ({runName}) — mean {mean_reward:.2f}, std {std_reward:.2f}", xaxis_title='Episode', yaxis_title='Reward')
        fig.write_html(plot_path)
        print(f"Saved rewards plot to: {plot_path}")


if __name__ == "__main__":
    main()
