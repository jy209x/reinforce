"""Visually compare a trained DQN with a randomly initialized DQN.

The first episode is played side by side in a Matplotlib window.  The lower
panel compares returns over several episodes using the same reset seeds for
both agents.

Examples:
    python test_dqn_cartpole.py --model-path dqn_cartpole.pt --episodes 5
    python test_dqn_cartpole.py --episodes 5 --save-gif cartpole_compare.gif
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from dqn_cartpole import Config, DQNAgent, make_env, reset_env, seed_everything, step_env


@dataclass
class EpisodeResult:
    return_: float
    frames: list[np.ndarray]


def load_trained_agent(model_path: Path, device: torch.device) -> DQNAgent:
    """Create an agent with the same architecture and load its online network."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Train first with dqn_cartpole.py."
        )

    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("The checkpoint must contain a 'model' state_dict.")

    state_size = int(checkpoint.get("state_size", 4))
    action_size = int(checkpoint.get("action_size", 2))
    agent = DQNAgent(state_size, action_size, device, Config())
    agent.online.load_state_dict(checkpoint["model"])
    agent.target.load_state_dict(agent.online.state_dict())
    agent.online.eval()
    agent.target.eval()
    return agent


def create_untrained_agent(
    state_size: int, action_size: int, device: torch.device
) -> DQNAgent:
    """Create the same DQN architecture but keep its random initialization."""
    agent = DQNAgent(state_size, action_size, device, Config())
    agent.online.eval()
    agent.target.eval()
    return agent


def run_episode(
    env,
    agent: DQNAgent,
    seed: int,
    collect_frames: bool = False,
) -> EpisodeResult:
    """Run one greedy episode and optionally collect RGB frames."""
    state = reset_env(env, seed=seed)
    frames: list[np.ndarray] = []
    if collect_frames:
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame).copy())

    total_return = 0.0
    while True:
        action = agent.act(state, explore=False)
        state, reward, terminated, truncated, _ = step_env(env, action)
        total_return += reward
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame).copy())
        if terminated or truncated:
            break

    return EpisodeResult(total_return, frames)


def compare_agents(
    trained: DQNAgent,
    untrained: DQNAgent,
    episodes: int,
    seed: int,
):
    """Evaluate both agents with matching seeds and collect first-episode frames."""
    trained_env = make_env(render_mode="rgb_array")
    untrained_env = make_env(render_mode="rgb_array")
    trained_returns: list[float] = []
    untrained_returns: list[float] = []
    trained_frames: list[np.ndarray] = []
    untrained_frames: list[np.ndarray] = []

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            trained_result = run_episode(
                trained_env, trained, episode_seed, collect_frames=episode == 0
            )
            untrained_result = run_episode(
                untrained_env, untrained, episode_seed, collect_frames=episode == 0
            )
            trained_returns.append(trained_result.return_)
            untrained_returns.append(untrained_result.return_)
            if episode == 0:
                trained_frames = trained_result.frames
                untrained_frames = untrained_result.frames
            print(
                f"Episode {episode + 1:2d} | "
                f"trained: {trained_result.return_:6.1f} | "
                f"untrained: {untrained_result.return_:6.1f}"
            )
    finally:
        trained_env.close()
        untrained_env.close()

    return trained_returns, untrained_returns, trained_frames, untrained_frames


def show_comparison(
    trained_returns: list[float],
    untrained_returns: list[float],
    trained_frames: list[np.ndarray],
    untrained_frames: list[np.ndarray],
    save_gif: Path | None = None,
    show_window: bool = True,
) -> None:
    """Animate the first episode and show the return comparison below it."""
    if not trained_frames or not untrained_frames:
        raise RuntimeError("No RGB frames were returned by the environments.")

    frame_count = max(len(trained_frames), len(untrained_frames))
    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=(3, 1))
    trained_ax = fig.add_subplot(grid[0, 0])
    untrained_ax = fig.add_subplot(grid[0, 1])
    score_ax = fig.add_subplot(grid[1, :])

    trained_image = trained_ax.imshow(trained_frames[0])
    untrained_image = untrained_ax.imshow(untrained_frames[0])
    for axis, title in (
        (trained_ax, "Trained DQN"),
        (untrained_ax, "Untrained DQN"),
    ):
        axis.set_title(title)
        axis.axis("off")

    x = np.arange(len(trained_returns))
    width = 0.38
    score_ax.bar(x - width / 2, trained_returns, width, label="Trained")
    score_ax.bar(x + width / 2, untrained_returns, width, label="Untrained")
    score_ax.axhline(195, color="tab:green", linestyle="--", linewidth=1,
                     label="Solved threshold (195)")
    score_ax.set_xlabel("Evaluation episode")
    score_ax.set_ylabel("Return")
    score_ax.set_title(
        f"Mean return: trained={np.mean(trained_returns):.1f}, "
        f"untrained={np.mean(untrained_returns):.1f}"
    )
    score_ax.set_xticks(x, [str(i + 1) for i in x])
    score_ax.legend(loc="upper left", ncols=3)
    score_ax.grid(axis="y", alpha=0.25)

    def update(frame_index: int):
        trained_index = min(frame_index, len(trained_frames) - 1)
        untrained_index = min(frame_index, len(untrained_frames) - 1)
        trained_image.set_data(trained_frames[trained_index])
        untrained_image.set_data(untrained_frames[untrained_index])
        trained_ax.set_title(f"Trained DQN | step {trained_index}")
        untrained_ax.set_title(f"Untrained DQN | step {untrained_index}")
        return trained_image, untrained_image

    animation = FuncAnimation(
        fig, update, frames=frame_count, interval=50, blit=False, repeat=False
    )
    # Keep a reference alive until show/save finishes, otherwise Matplotlib may
    # garbage-collect the animation before the window starts playing.
    _ = animation
    fig.tight_layout()
    if save_gif is not None:
        animation.save(save_gif, writer=PillowWriter(fps=20))
        print(f"Saved comparison animation to {save_gif}")
    if show_window:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a trained and an untrained DQN on CartPole"
    )
    parser.add_argument("--model-path", type=Path, default=Path("dqn_cartpole.pt"))
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of matched evaluation episodes.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-gif", type=Path, default=None,
                        help="Optional path for saving the side-by-side GIF.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open a Matplotlib window; useful with --save-gif.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    seed_everything(args.seed)

    trained = load_trained_agent(args.model_path, device)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_size = int(checkpoint.get("state_size", 4))
    action_size = int(checkpoint.get("action_size", 2))
    untrained = create_untrained_agent(state_size, action_size, device)

    print(f"Comparing agents on {device} using {args.episodes} episodes")
    trained_returns, untrained_returns, trained_frames, untrained_frames = compare_agents(
        trained, untrained, args.episodes, args.seed
    )
    show_comparison(
        trained_returns, untrained_returns, trained_frames, untrained_frames,
        save_gif=args.save_gif,
        show_window=not args.no_show,
    )


if __name__ == "__main__":
    main()
