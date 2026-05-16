"""Optional PyTorch DQN agent for 2048.

The n-tuple learner is stronger per line of code for 2048, but this agent is
useful when the target platform has a GPU and the user wants tensor training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque

from .game2048 import Action, Board, GameResult, legal_actions, max_tile, new_game, step


@dataclass
class DQNConfig:
    gamma: float = 0.99
    epsilon: float = 0.2
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9995
    learning_rate: float = 0.0005
    batch_size: int = 256
    replay_size: int = 50_000
    target_sync: int = 500


class DQNAgent:
    def __init__(self, device: str, config: DQNConfig | None = None, episodes_trained: int = 0) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.nn = nn
        self.device = torch.device(device)
        self.config = config or DQNConfig()
        self.episodes_trained = episodes_trained
        self.steps = 0
        self.policy = _Net().to(self.device)
        self.target = _Net().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.learning_rate)
        self.replay: Deque[tuple[Board, Action, float, Board, bool]] = deque(maxlen=self.config.replay_size)
        self.last_loss: float | None = None

    def choose_action(self, board: Board, rng: random.Random, explore: bool = True) -> Action:
        actions = legal_actions(board)
        if not actions:
            raise ValueError("cannot choose an action from a terminal board")
        if explore and rng.random() < self.config.epsilon:
            return rng.choice(actions)

        with self.torch.no_grad():
            q_values = self.policy(_board_tensor(self.torch, self.device, board).unsqueeze(0))[0]
            illegal = set(range(4)) - set(actions)
            for action in illegal:
                q_values[action] = -1e9
            return int(self.torch.argmax(q_values).item())

    def train_episode(self, rng: random.Random) -> GameResult:
        board = new_game(rng)
        total_reward = 0
        moves = 0
        while legal_actions(board):
            action = self.choose_action(board, rng, explore=True)
            next_board, reward, done = step(board, action, rng)
            self.replay.append((board, action, reward / 2048.0, next_board, done))
            self._learn()
            board = next_board
            total_reward += reward
            moves += 1

        self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon * self.config.epsilon_decay)
        return GameResult(total_reward, max_tile(board), moves, board)

    def play_episode(self, rng: random.Random) -> GameResult:
        board = new_game(rng)
        total_reward = 0
        moves = 0
        while legal_actions(board):
            action = self.choose_action(board, rng, explore=False)
            board, reward, _ = step(board, action, rng)
            total_reward += reward
            moves += 1
        return GameResult(total_reward, max_tile(board), moves, board)

    def save(self, path: str) -> None:
        self.torch.save(
            {
                "kind": "dqn",
                "episodes_trained": self.episodes_trained,
                "steps": self.steps,
                "config": self.config.__dict__,
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str) -> "DQNAgent":
        import torch

        payload = torch.load(path, map_location=device)
        agent = cls(device=device, config=DQNConfig(**payload["config"]), episodes_trained=payload["episodes_trained"])
        agent.steps = payload["steps"]
        agent.policy.load_state_dict(payload["policy"])
        agent.target.load_state_dict(payload["target"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        return agent

    def _learn(self) -> None:
        if len(self.replay) < self.config.batch_size:
            return

        torch = self.torch
        batch = random.sample(self.replay, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        state_t = torch.stack([_board_tensor(torch, self.device, board) for board in states])
        next_t = torch.stack([_board_tensor(torch, self.device, board) for board in next_states])
        action_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q = self.policy(state_t).gather(1, action_t).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target(next_t)
            next_mask = _legal_action_mask(torch, self.device, next_states)
            next_q_values = next_q_values.masked_fill(~next_mask, -1e9)
            next_q = next_q_values.max(dim=1).values
            target = reward_t + self.config.gamma * next_q * (1.0 - done_t)

        loss = self.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.last_loss = float(loss.detach().cpu())

        self.steps += 1
        if self.steps % self.config.target_sync == 0:
            self.target.load_state_dict(self.policy.state_dict())


def _board_tensor(torch, device, board: Board):
    return torch.tensor([value / 16.0 for value in board], dtype=torch.float32, device=device)


def _legal_action_mask(torch, device, boards: tuple[Board, ...]):
    rows = []
    for board in boards:
        legal = set(legal_actions(board))
        rows.append([action in legal for action in range(4)])
    return torch.tensor(rows, dtype=torch.bool, device=device)


class _Net:
    def __new__(cls):
        from torch import nn

        return nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
