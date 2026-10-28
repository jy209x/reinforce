"""PPO-Clip training example on the more challenging Acrobot environment.

Acrobot has a two-link underactuated pendulum, six continuous observations,
three discrete torques, and a sparse step reward.  It is a more demanding
simulation than CartPole while still being available in the base Gym install.

Install one supported environment package if needed:

    pip install torch gymnasium[classic-control]

Examples:
    python ppo_acrobot.py --total-steps 200000
    python ppo_acrobot.py --total-steps 200000 --eval-episodes 5 --render
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam

try:  # Prefer Gymnasium, but remain compatible with gym 0.26.
    import gymnasium as gym
except ImportError:  # pragma: no cover - depends on the user's environment.
    import gym


@dataclass(frozen=True)
class Config:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 128


class ActorCritic(nn.Module):
    """Shared MLP with a categorical actor head and scalar critic head."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor):
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def distribution_value(self, observations: torch.Tensor):
        logits, values = self(observations)
        return Categorical(logits=logits), values


class RolloutBuffer:
    """Stores on-policy transitions and computes GAE advantages."""

    def __init__(self, capacity: int, observation_size: int):
        self.states = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.episode_end = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        next_value: float,
        log_prob: float,
        terminated: bool,
        episode_end: bool,
    ) -> None:
        if self.position >= len(self.rewards):
            raise RuntimeError("RolloutBuffer is full")
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.next_values[i] = next_value
        self.log_probs[i] = log_prob
        self.terminated[i] = float(terminated)
        self.episode_end[i] = float(episode_end)
        self.position += 1

    def tensors(self, config: Config, device: torch.device):
        size = self.position
        advantages = np.zeros(size, dtype=np.float32)
        gae = 0.0
        for i in range(size - 1, -1, -1):
            # A true terminal state has no bootstrap value.  A time-limit
            # truncation may bootstrap, but GAE must not cross into the next
            # episode, hence the separate episode_end mask.
            delta = (
                
                self.rewards[i]
                + config.gamma * (1.0 - self.terminated[i]) * self.next_values[i]
                - self.values[i]
            )
            gae = delta + config.gamma * config.gae_lambda * (
                1.0 - self.episode_end[i]
            ) * gae
            advantages[i] = gae
        returns = advantages + self.values[:size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return (
            torch.as_tensor(self.states[:size], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[:size], dtype=torch.int64, device=device),
            torch.as_tensor(self.log_probs[:size], dtype=torch.float32, device=device),
            torch.as_tensor(returns, dtype=torch.float32, device=device),
            torch.as_tensor(advantages, dtype=torch.float32, device=device),
        )

    def clear(self) -> None:
        self.position = 0


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:  # Older gym versions do not accept reset(seed=...).
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    observation = result[0] if isinstance(result, tuple) else result
    return np.asarray(observation, dtype=np.float32)


def step_env(env: gym.Env, action: int):
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return (np.asarray(observation, dtype=np.float32), float(reward),
                bool(terminated), bool(truncated), info)
    observation, reward, done, info = result
    return (np.asarray(observation, dtype=np.float32), float(reward),
            bool(done), False, info)


def make_env(render_mode: str | None = None):
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    try:
        return gym.make("Acrobot-v1", **kwargs)
    except TypeError:  # Older gym releases may not accept render_mode.
        return gym.make("Acrobot-v1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_action(
    policy: ActorCritic, state: np.ndarray, device: torch.device, deterministic: bool
):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        distribution, value = policy.distribution_value(state_tensor)
        action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(value.item())


def collect_rollout(
    env,
    policy: ActorCritic,
    buffer: RolloutBuffer,
    state: np.ndarray,
    device: torch.device,
):
    episode_returns: list[float] = []
    episode_return = 0.0
    for _ in range(buffer.states.shape[0]):
        action, log_prob, value = select_action(policy, state, device, deterministic=False)
        next_state, reward, terminated, truncated, _ = step_env(env, action)
        next_state_tensor = torch.as_tensor(
            next_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            _, next_value_tensor = policy.distribution_value(next_state_tensor)
        next_value = float(next_value_tensor.item())
        episode_end = terminated or truncated
        buffer.add(state, action, reward, value, next_value, log_prob,
                   terminated, episode_end)
        state = next_state
        episode_return += reward
        if episode_end:
            episode_returns.append(episode_return)
            episode_return = 0.0
            state = reset_env(env)
    return state, episode_returns


def ppo_update(
    policy: ActorCritic,
    optimizer: Adam,
    buffer: RolloutBuffer,
    config: Config,
    device: torch.device,
):
    states, actions, old_log_probs, returns, advantages = buffer.tensors(config, device)
    sample_count = states.shape[0]
    minibatch_size = min(config.minibatch_size, sample_count)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

    for _ in range(config.update_epochs):
        indices = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, minibatch_size):
            batch_indices = indices[start:start + minibatch_size]
            distribution, values = policy.distribution_value(states[batch_indices])
            new_log_probs = distribution.log_prob(actions[batch_indices])
            entropy = distribution.entropy().mean()
            ratio = (new_log_probs - old_log_probs[batch_indices]).exp()
            unclipped = ratio * advantages[batch_indices]
            clipped = torch.clamp(
                ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
            ) * advantages[batch_indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns[batch_indices])
            loss = (
                policy_loss
                + config.value_loss_coef * value_loss
                - config.entropy_coef * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)   # Clip gradients to avoid exploding gradients
            optimizer.step()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.item())
            metrics["updates"] += 1

    for key in ("policy_loss", "value_loss", "entropy"):
        metrics[key] /= max(1, metrics["updates"])
    return metrics


def evaluate(
    policy: ActorCritic,
    episodes: int,
    device: torch.device,
    render: bool = False,
    seed: int | None = None,
) -> float:
    env = make_env(render_mode="human" if render else None)
    returns = []
    try:
        for episode in range(episodes):
            state = reset_env(env, None if seed is None else seed + episode)
            if render:
                env.render()
            total = 0.0
            while True:
                action, _, _ = select_action(policy, state, device, deterministic=True)
                state, reward, terminated, truncated, _ = step_env(env, action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            print(f"Evaluation episode {episode + 1}: return {total:.1f}")
    finally:
        env.close()
    mean_return = float(np.mean(returns))
    print(f"Evaluation mean return: {mean_return:.1f}")
    return mean_return


def train(args: argparse.Namespace) -> None:
    config = Config(
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
    )
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    seed_everything(args.seed)
    env = make_env()
    env.action_space.seed(args.seed)
    observation_size = int(np.prod(env.observation_space.shape))
    action_size = int(env.action_space.n)
    policy = ActorCritic(observation_size, action_size, config.hidden_size).to(device)
    optimizer = Adam(policy.parameters(), lr=config.learning_rate)
    buffer = RolloutBuffer(config.rollout_steps, observation_size)
    state = reset_env(env, args.seed)
    total_steps = 0
    recent_returns: list[float] = []

    print(
        f"Training PPO on Acrobot-v1 with {device}; "
        f"observation_size={observation_size}, action_size={action_size}"
    )
    try:
        while total_steps < args.total_steps:
            buffer.clear()
            state, episode_returns = collect_rollout(env, policy, buffer, state, device)
            total_steps += buffer.position
            recent_returns.extend(episode_returns)
            metrics = ppo_update(policy, optimizer, buffer, config, device)
            if total_steps == buffer.position or total_steps % args.log_interval < buffer.position:
                mean_return = np.mean(recent_returns[-20:]) if recent_returns else float("nan")
                print(
                    f"Steps {total_steps:7d} | episodes {len(recent_returns):4d} | "
                    f"mean(20) {mean_return:7.2f} | "
                    f"policy_loss {metrics['policy_loss']:.3f} | "
                    f"entropy {metrics['entropy']:.3f}"
                )
            if args.eval_episodes and total_steps % args.eval_interval < buffer.position:
                evaluate(policy, args.eval_episodes, device, render=args.render,
                         seed=args.seed + 100_000)
    finally:
        env.close()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": policy.state_dict(),
        "observation_size": observation_size,
        "action_size": action_size,
        "config": asdict(config),
    }, args.model_path)
    print(f"Saved PPO model to {args.model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO-Clip on Gym Acrobot-v1")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-path", type=Path, default=Path("ppo_acrobot.pt"))
    parser.add_argument("--log-interval", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--render", action="store_true",
                        help="Render periodic evaluation episodes in a window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_steps < 1:
        raise ValueError("total-steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if args.eval_episodes < 0:
        raise ValueError("eval-episodes cannot be negative")
    train(args)


if __name__ == "__main__":
    main()
