"""
Training script for the DQN agent on the RealEstateInvestmentEnv.

This script orchestrates the full RL pipeline:
1. Environment and agent initialization.
2. Training loop with epsilon-greedy exploration.
3. Periodic evaluation on a held-out validation environment.
4. Checkpoint saving (best model only).
5. TensorBoard logging for learning curves.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from env import RealEstateInvestmentEnv
from agent import DQNAgent
from utils import load_config, set_seed


def evaluate(agent: DQNAgent, env: RealEstateInvestmentEnv, episodes: int = 20) -> float:
    """Evaluate the agent on a fixed set of validation episodes.
    
    Args:
        agent: The DQN agent.
        env: The environment.
        episodes: Number of evaluation episodes.
        
    Returns:
        float: Mean return over evaluation episodes.
    """
    total_returns = []
    for ep in range(episodes):
        # Use fixed seeds for validation so performance is comparable across evaluations
        obs, _ = env.reset(seed=10000 + ep)
        done = False
        ep_return = 0.0
        
        while not done:
            # No exploration during evaluation
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            
        total_returns.append(ep_return)
        
    return float(np.mean(total_returns))


def train(config_path: str) -> None:
    """Main training loop."""
    # --------------------------------------------------------------------------
    # 1. Setup
    # --------------------------------------------------------------------------
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    
    train_cfg = cfg.get("training", {})
    n_episodes = int(train_cfg.get("episodes", 1000))
    eval_every = int(train_cfg.get("eval_every", 50))
    eval_episodes = int(train_cfg.get("eval_episodes", 20))
    
    # Directories
    os.makedirs("../checkpoints", exist_ok=True)
    log_dir = "../runs/workshop3"
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Environment and Agent
    env = RealEstateInvestmentEnv(cfg)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=cfg
    )
    
    print(f"Training on device: {agent.device}")
    print(f"Episodes: {n_episodes}")
    print("-" * 60)
    
    # Tracking
    best_val_return = -float("inf")
    episode_returns = []
    episode_losses = []
    
    start_time = time.time()
    
    # --------------------------------------------------------------------------
    # 2. Training Loop
    # --------------------------------------------------------------------------
    pbar = tqdm(range(1, n_episodes + 1), desc="Training", unit="ep")
    for episode in pbar:
        obs, _ = env.reset(seed=episode)
        done = False
        ep_return = 0.0
        ep_loss = []
        
        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Interact with environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_return += reward
            
            # Train step
            loss = agent.update()
            if loss is not None:
                ep_loss.append(loss)
                
        # End of episode
        agent.step_epsilon()
        episode_returns.append(ep_return)
        
        mean_loss = np.mean(ep_loss) if ep_loss else 0.0
        episode_losses.append(mean_loss)
        
        # Logging to TensorBoard
        writer.add_scalar("Train/Return", ep_return, episode)
        writer.add_scalar("Train/Loss", mean_loss, episode)
        writer.add_scalar("Train/Epsilon", agent.epsilon, episode)
        
        # Update progress bar
        pbar.set_postfix({
            "Return": f"{ep_return:.2f}",
            "Epsilon": f"{agent.epsilon:.3f}"
        })
        
        # ----------------------------------------------------------------------
        # 3. Evaluation and Checkpointing
        # ----------------------------------------------------------------------
        if episode % eval_every == 0:
            val_return = evaluate(agent, env, eval_episodes)
            writer.add_scalar("Val/Return", val_return, episode)
            
            tqdm.write(f"\n[Ep {episode:4d}] Val Return: {val_return:+.4f} | Epsilon: {agent.epsilon:.3f}")
            
            # Save best model
            if val_return > best_val_return:
                best_val_return = val_return
                save_path = "../checkpoints/agent_best.pt"
                agent.save(save_path)
                tqdm.write(f"  -> New best model saved (Val: {best_val_return:+.4f})")
                
    # --------------------------------------------------------------------------
    # 4. Wrap up
    # --------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {elapsed/60:.2f} minutes.")
    print(f"Best validation return: {best_val_return:+.4f}")
    print(f"Checkpoint saved at: ../checkpoints/agent_best.pt")
    
    writer.close()


if __name__ == "__main__":
    # Assuming we run from workshop3_rl/src
    train("../configs/default.yaml")