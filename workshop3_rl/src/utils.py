"""
Utility functions for Workshop 3 — Reinforcement Learning.

Includes normalization helpers, logging utilities, and reproducibility
setup used across the environment, agent, and training scripts.
"""

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across numpy, random, and torch.

    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(value: float, low: float, high: float) -> float:
    """Normalize a value to [0, 1] given its expected range.

    Args:
        value (float): Value to normalize.
        low (float): Minimum of the expected range.
        high (float): Maximum of the expected range.

    Returns:
        float: Normalized value clipped to [0, 1].
    """
    if high - low < 1e-12:
        return 0.0
    normalized = (value - low) / (high - low)
    return float(np.clip(normalized, 0.0, 1.0))


def load_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary.
    """
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)