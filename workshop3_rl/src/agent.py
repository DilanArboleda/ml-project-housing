"""
Deep Q-Network (DQN) agent for the RealEstateInvestmentEnv.

Implements the three core components of DQN:
1. QNetwork: MLP that approximates Q(s, a).
2. ReplayBuffer: Experience replay to break temporal correlations.
3. DQNAgent: Training loop with target network, epsilon-greedy, and optimization.

References:
- Mnih et al. (2015), "Human-level Control through Deep Reinforcement Learning".
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ==============================================================================
# 1. Neural Network Architecture
# ==============================================================================
class QNetwork(nn.Module):
    """Multi-Layer Perceptron to approximate Q-values for discrete actions.

    Architecture: State_dim -> Hidden1 -> Hidden2 -> Action_dim
    Activation: ReLU
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers dynamically
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
            
        # Output layer (no activation, outputs raw Q-values)
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for all actions."""
        return self.net(state)


# ==============================================================================
# 2. Experience Replay Buffer
# ==============================================================================
class ReplayBuffer:
    """Fixed-size buffer to store and sample (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return current size of the buffer."""
        return len(self.buffer)


# ==============================================================================
# 3. DQN Agent
# ==============================================================================
class DQNAgent:
    """Deep Q-Network agent combining QNetwork, ReplayBuffer, and training logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            state_dim: Dimension of the observation space (8 for our env).
            action_dim: Number of discrete actions (4 for our env).
            config: Dictionary with hyperparameters (from default.yaml).
            device: Torch device (CPU or CUDA).
        """
        # Hyperparameters
        agent_cfg = config.get("agent", config)
        self.gamma = float(agent_cfg.get("gamma", 0.99))
        self.batch_size = int(agent_cfg.get("batch_size", 64))
        self.target_update_freq = int(agent_cfg.get("target_update_freq", 1000))
        self.gradient_clip = float(agent_cfg.get("gradient_clip", 10.0))
        self.train_start = int(agent_cfg.get("train_start", 1000))
        self.hidden_sizes = agent_cfg.get("hidden_sizes", [128, 128])
        self.lr = float(agent_cfg.get("learning_rate", 1e-3))

        # Epsilon-greedy parameters
        self.epsilon = float(agent_cfg.get("epsilon_start", 1.0))
        self.epsilon_end = float(agent_cfg.get("epsilon_end", 0.05))
        self.epsilon_decay = float(agent_cfg.get("epsilon_decay", 0.995))

        # Device and dimensions
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks: Q-network and Target-network
        self.q_network = QNetwork(state_dim, action_dim, self.hidden_sizes).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, self.hidden_sizes).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is only for inference

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        buffer_size = int(agent_cfg.get("buffer_size", 100000))
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Tracking
        self.steps_done = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select an action using epsilon-greedy strategy.

        Args:
            state: Observation vector.
            evaluate: If True, disables exploration (pure greedy).

        Returns:
            Action index.
        """
        if evaluate:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_t)
                return int(q_values.argmax(dim=1).item())

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Exploitation
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def step_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self) -> float | None:
        """Sample a batch and perform one gradient descent step.

        Returns:
            The loss value (float) if training occurred, else None.
        """
        # Wait until buffer has enough samples
        if len(self.replay_buffer) < self.train_start:
            return None

        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a)
        current_q = self.q_network(states_t).gather(1, actions_t)

        # Compute target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q = self.target_network(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (self.gamma * max_next_q * (1.0 - dones_t))

        # Compute loss and backpropagate
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        
        self.optimizer.step()

        # Update target network periodically
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save the best Q-network checkpoint."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        """Load a Q-network checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)