"""Evaluate a saved Acrobot PPO policy without training or changing its weights.

Examples:
    python test_ppo_acrobot.py --episodes 20
    python test_ppo_acrobot.py --episodes 3 --render
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ppo_acrobot import ActorCritic, Config, make_env, reset_env, select_action, step_env


def load_policy(model_path: Path, device: torch.device) -> ActorCritic:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    policy = ActorCritic(
        observation_size=int(checkpoint["observation_size"]),
        action_size=int(checkpoint["action_size"]),
        hidden_size=int(checkpoint.get("config", {}).get("hidden_size", Config().hidden_size)),
    ).to(device)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.eval()
    if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise ValueError("The saved model contains non-finite parameters.")
    return policy


def evaluate_saved_policy(
    policy: ActorCritic,
    device: torch.device,
    episodes: int,
    seed: int,
    render: bool,
) -> None:
    env = make_env(render_mode="human" if render else None)
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    timeouts = 0
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for episode in range(episodes):
                episode_seed = seed + episode
                state = reset_env(env, seed=episode_seed)
                if render:
                    env.render()
                total = 0.0
                steps = 0
                while True:
                    action, _, _ = select_action(policy, state, device, deterministic=True)
                    state, reward, terminated, truncated, _ = step_env(env, action)
                    total += reward
                    steps += 1
                    if terminated or truncated:
                        break
                # In Acrobot-v1, a true termination means the target height
                # was reached. A time limit without termination is a timeout.
                successes += int(terminated)
                timeouts += int(truncated and not terminated)
                returns.append(total)
                lengths.append(steps)
                status = "SUCCESS" if terminated else "TIMEOUT"
                print(
                    f"Episode {episode + 1:2d} | seed {episode_seed} | "
                    f"return {total:7.1f} | steps {steps:3d} | {status}",
                    flush=True,
                )
    finally:
        env.close()
    print(f"Mean return: {np.mean(returns):.2f} (std {np.std(returns):.2f})")
    print(f"Return range: {min(returns):.1f} to {max(returns):.1f}")
    print(f"Mean episode length: {np.mean(lengths):.2f} steps")
    print(f"Success rate: {successes}/{episodes} ({100 * successes / episodes:.1f}%)")
    print(f"Timeouts: {timeouts}/{episodes}")
    print(f"Elapsed: {time.perf_counter() - started:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO Acrobot model")
    parser.add_argument(
        "--model-path", type=Path,
        default=Path(__file__).resolve().with_name("ppo_acrobot.pt"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not args.model_path.is_file():
        parser.error(f"Model file not found: {args.model_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    # This small network generally needs only one CPU thread during inference.
    torch.set_num_threads(1)
    policy = load_policy(args.model_path, device)
    print(f"Loaded {args.model_path}; device={device}; deterministic evaluation", flush=True)
    evaluate_saved_policy(policy, device, args.episodes, args.seed, args.render)


if __name__ == "__main__":
    main()
