"""
Quick sanity check for the RealEstateInvestmentEnv.

Runs a short random-policy rollout to verify that the environment
implements the Gymnasium API correctly and produces sensible values.
"""

import sys
sys.path.append('.')

from env import RealEstateInvestmentEnv
from utils import load_config, set_seed


def main() -> None:
    # Reproducibility
    set_seed(42)

    # Load configuration
    cfg = load_config('../configs/default.yaml')

    # Instantiate environment
    env = RealEstateInvestmentEnv(cfg)

    # Reset
    obs, info = env.reset(seed=42)
    print('Initial obs:', obs)
    print('Obs shape:', obs.shape)
    print('Action space:', env.action_space)
    print('Observation space:', env.observation_space)
    print()

    # Random-policy rollout (5 steps)
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f'Step {i + 1}: '
            f'action={action}, '
            f'reward={reward:+.4f}, '
            f'capital={info["capital"]:.3f}, '
            f'owns={info["owns"]}, '
            f'terminated={terminated}, '
            f'truncated={truncated}'
        )

    print()
    print('Sanity check passed: environment runs without errors.')


if __name__ == '__main__':
    main()