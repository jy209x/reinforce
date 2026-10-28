"""Train a DQN agent on CartPole.

The script supports both the maintained ``gymnasium`` package and the
compatible ``gym`` 0.26 API.  Install one of them before running:

    pip install torch gymnasium[classic-control]

Examples:
    python dqn_cartpole.py --episodes 500
    python dqn_cartpole.py --episodes 500 --render --eval-episodes 5
    python dqn_cartpole.py --episodes 500 --render-training
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

try:  # Prefer Gymnasium, but keep compatibility with gym 0.26.
    import gymnasium as gym
except ImportError:  # pragma: no cover - depends on the user's environment.
    import gym


@dataclass(frozen=True)
class Config:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 50_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_update_frequency: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    hidden_size: int = 128
    gradient_clip: float = 10.0


class ReplayBuffer:
    """A fixed-size replay buffer storing one-step transitions."""

    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self._buffer.append(
            (np.asarray(state, dtype=np.float32), action, reward,
             np.asarray(next_state, dtype=np.float32), terminated)
        )

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, terminated = zip(*batch)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.int64, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(terminated, dtype=torch.float32, device=device),
        )


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, device: torch.device,
                 config: Config) -> None:
        self.action_size = action_size
        self.device = device
        self.config = config
        self.online = QNetwork(state_size, action_size, config.hidden_size).to(device)
        self.target = QNetwork(state_size, action_size, config.hidden_size).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.total_steps = 0

    def epsilon(self) -> float:
        progress = min(1.0, self.total_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon():
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32,
                                           device=self.device).unsqueeze(0)
            return int(self.online(state_tensor).argmax(dim=1).item())

    def update(self) -> float | None:
        if len(self.replay) < max(self.config.batch_size, self.config.learning_starts):
            return None
        if self.total_steps % self.config.train_frequency != 0:
            return None

        states, actions, rewards, next_states, terminated = self.replay.sample(
            self.config.batch_size, self.device
        )
        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            target_q = rewards + self.config.gamma * (1.0 - terminated) * next_q

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        self.optimizer.step()

        if self.total_steps % self.config.target_update_frequency == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    """Normalize reset output across gym and gymnasium versions."""
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:  # Older gym versions do not accept reset(seed=...).
        if seed is not None and hasattr(env, "seed"):
            env.seed(seed)
        result = env.reset()
    return np.asarray(result[0] if isinstance(result, tuple) else result,
                      dtype=np.float32)


def step_env(env: gym.Env, action: int):
    """Normalize step output and retain terminated separately from truncation."""
    result = env.step(action)
    if len(result) == 5:
        next_state, reward, terminated, truncated, info = result
        return (np.asarray(next_state, dtype=np.float32), float(reward),
                bool(terminated), bool(truncated), info)
    next_state, reward, done, info = result
    return (np.asarray(next_state, dtype=np.float32), float(reward),
            bool(done), False, info)


def make_env(render_mode: str | None = None):
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    try:
        return gym.make("CartPole-v1", **kwargs)
    except TypeError:  # Older gym releases may not accept render_mode.
        return gym.make("CartPole-v1")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(env, agent: DQNAgent, episodes: int, seed: int):
    scores: list[float] = []
    for episode in range(1, episodes + 1):
        state = reset_env(env, seed=seed + episode - 1)
        episode_return = 0.0
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = step_env(env, action)
            # A time-limit truncation is not an MDP terminal state, so it can
            # still bootstrap from the target value during training.
            agent.replay.add(state, action, reward, next_state, terminated)
            agent.total_steps += 1
            agent.update()
            state = next_state
            episode_return += reward
            if terminated or truncated:
                break

        scores.append(episode_return)
        recent = float(np.mean(scores[-100:]))
        if episode == 1 or episode % 10 == 0:
            print(f"Episode {episode:4d} | return {episode_return:6.1f} | "
                  f"mean(100) {recent:6.1f} | epsilon {agent.epsilon():.3f}")
        if len(scores) >= 100 and recent >= 195.0:
            print(f"Solved at episode {episode} (mean return over 100: {recent:.1f}).")
            break
    return scores


def evaluate(env, agent: DQNAgent, episodes: int, render: bool = False) -> None:
    returns = []
    for episode in range(episodes):
        state = reset_env(env)
        # Some Gym versions only create the human window after render() is
        # called explicitly.  Calling it after reset also shows the initial
        # state before the first action.
        if render:
            env.render()
        total = 0.0
        while True:
            action = agent.act(state, explore=False)
            state, reward, terminated, truncated, _ = step_env(env, action)
            total += reward
            if terminated or truncated:
                break
        returns.append(total)
        print(f"Evaluation episode {episode + 1}: return {total:.1f}")
    print(f"Evaluation mean return: {np.mean(returns):.1f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN for Gym CartPole-v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--model-path", type=Path, default=Path("dqn_cartpole.pt"))
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="Evaluate after training; use --render for a visible window.")
    parser.add_argument("--render", action="store_true",
                        help="Render evaluation episodes (requires --eval-episodes > 0).")
    parser.add_argument("--render-training", action="store_true",
                        help="Render every training episode; this is much slower.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.eval_episodes < 0:
        raise ValueError("episodes must be positive and eval-episodes cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    seed_everything(args.seed)

    train_env = make_env(render_mode="human" if args.render_training else None)
    train_env.action_space.seed(args.seed)
    state_size = int(np.prod(train_env.observation_space.shape))
    action_size = int(train_env.action_space.n)
    agent = DQNAgent(state_size, action_size, device, Config())
    print(f"Training on {device} with state_size={state_size}, action_size={action_size}")
    train(train_env, agent, args.episodes, args.seed)
    train_env.close()

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": agent.online.state_dict(), "state_size": state_size,
                "action_size": action_size}, args.model_path)
    print(f"Saved model to {args.model_path}")

    if args.eval_episodes:
        eval_env = make_env(render_mode="human" if args.render else None)
        if args.render:
            print("Rendering evaluation episodes in a Gym window; close it after evaluation.")
        evaluate(eval_env, agent, args.eval_episodes, render=args.render)
        eval_env.close()


if __name__ == "__main__":
    main()
