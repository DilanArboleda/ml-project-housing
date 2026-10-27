import sys
sys.path.append('.')

import numpy as np
from env import RealEstateInvestmentEnv
from agent import DQNAgent
from utils import load_config, set_seed

def main():
    set_seed(42)
    cfg = load_config('../configs/default.yaml')
    env = RealEstateInvestmentEnv(cfg)
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=cfg
    )
    print(f"Agent initialized on device: {agent.device}")
    print(f"Q-Network parameters: {sum(p.numel() for p in agent.q_network.parameters())}")

    # Quick interaction test
    obs, _ = env.reset(seed=42)
    
    for i in range(10):
        action = agent.select_action(obs)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # Save experience
        agent.replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        
        if done:
            break
            
    print(f"Buffer size: {len(agent.replay_buffer)}")
    print("Agent sanity check passed.")

if __name__ == '__main__':
    main()