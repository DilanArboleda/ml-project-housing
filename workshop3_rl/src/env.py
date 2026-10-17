"""
Custom Gymnasium environment for Workshop 3 — Reinforcement Learning.

Domain: Real Estate Investment Decision-Making (Group 1, Housing Price Prediction).

The agent acts as an investor who decides, month by month, whether to
WAIT, BUY, RENOVATE, or SELL a property, maximizing cumulative net profit
over a fixed horizon under stochastic market dynamics.

This environment implements the MDP specified in:
    workshop3_rl/report/mdp_formulation.pdf

Interface: gymnasium.Env (reset, step, action_space, observation_space, render).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RealEstateInvestmentEnv(gym.Env):
    """Real estate investment environment.

    MDP components:
        - State space: R^8 (normalized continuous features)
        - Action space: Discrete(4) — WAIT, BUY, RENOVATE, SELL
        - Transition: price follows a geometric random walk with drift
        - Reward: dense per-step + terminal
        - Episode end: horizon reached, bankruptcy, or success threshold

    Attributes:
        WAIT, BUY, RENOVATE, SELL (int): Action indices.
        metadata (dict): Gymnasium metadata.
    """

    # Action indices (kept explicit for readability in logs and plots)
    WAIT: int = 0
    BUY: int = 1
    RENOVATE: int = 2
    SELL: int = 3

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the environment with a configuration dictionary.

        Args:
            config (dict | None): Environment configuration. If None, uses
                the default parameters defined below.
        """
        super().__init__()

        # ------------------------------------------------------------------
        # Configuration (with defaults matching configs/default.yaml)
        # ------------------------------------------------------------------
        cfg = config or {}
        env_cfg = cfg.get("environment", cfg)

        self.horizon = int(env_cfg.get("horizon", 120))

        self.initial_price_low = float(env_cfg.get("initial_price_low", 0.8))
        self.initial_price_high = float(env_cfg.get("initial_price_high", 1.2))
        self.mu = float(env_cfg.get("mu_monthly", 0.0025))
        self.sigma = float(env_cfg.get("sigma_monthly", 0.0144))

        self.initial_condition_low = float(env_cfg.get("initial_condition_low", 0.4))
        self.initial_condition_high = float(env_cfg.get("initial_condition_high", 0.8))
        self.condition_decay = float(env_cfg.get("condition_decay_per_month", 0.005))
        self.renovation_gain = float(env_cfg.get("renovation_condition_gain", 0.30))

        self.initial_capital = float(env_cfg.get("initial_capital", 1.0))

        self.buying_cost_rate = float(env_cfg.get("buying_cost_rate", 0.03))
        self.selling_cost_rate = float(env_cfg.get("selling_cost_rate", 0.05))
        self.renovation_cost = float(env_cfg.get("renovation_cost", 0.08))
        self.maintenance_cost = float(env_cfg.get("maintenance_cost_monthly", 0.002))
        self.forced_sale_cost = float(env_cfg.get("forced_sale_cost", 0.10))

        self.invalid_action_penalty = float(env_cfg.get("invalid_action_penalty", 0.05))
        self.idle_capital_penalty = float(env_cfg.get("idle_capital_penalty", 0.001))
        self.idle_capital_threshold = float(env_cfg.get("idle_capital_threshold", 0.5))
        self.condition_bonus_max = float(env_cfg.get("condition_bonus_max", 0.10))

        self.bankruptcy_threshold = float(env_cfg.get("bankruptcy_threshold", 0.0))
        self.success_threshold = float(env_cfg.get("success_threshold", 2.0))

        # ------------------------------------------------------------------
        # Gymnasium spaces
        # ------------------------------------------------------------------
        # Observation: 8 normalized features in [0, 1]
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # ------------------------------------------------------------------
        # Internal state (initialized in reset())
        # ------------------------------------------------------------------
        self._rng: np.random.Generator = np.random.default_rng()
        self._state: dict[str, float] = {}
        self._t: int = 0
        self._price_history: list[float] = []
        self._purchase_price: float = 0.0
        self._episode_reward: float = 0.0

    # ======================================================================
    # Core Gymnasium API
    # ======================================================================
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to a randomized initial state.

        Args:
            seed (int | None): Optional seed for the RNG.
            options (dict | None): Reserved for future use.

        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        initial_price = float(
            self._rng.uniform(self.initial_price_low, self.initial_price_high)
        )
        initial_condition = float(
            self._rng.uniform(self.initial_condition_low, self.initial_condition_high)
        )

        self._t = 0
        self._price_history = [initial_price]
        self._purchase_price = 0.0
        self._episode_reward = 0.0

        self._state = {
            "price": initial_price,
            "market_avg": initial_price,
            "trend_3m": 0.0,
            "owns": 0.0,
            "condition": initial_condition,
            "capital": self.initial_capital,
            "months_held": 0.0,
            "time_step": 0.0,
        }

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one action in the environment.

        Args:
            action (int): Action index in {0, 1, 2, 3}.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action index: {action}")

        # --------------------------------------------------------------
        # 1. Compute reward for the chosen action (updates capital and
        #    internal state via side effects).
        # --------------------------------------------------------------
        reward = self._compute_reward(int(action))

        # --------------------------------------------------------------
        # 2. Advance time and environment dynamics
        # --------------------------------------------------------------
        self._t += 1

        # Price evolves stochastically (only if not just sold)
        self._update_price()

        # Condition decays while owned
        if self._state["owns"] > 0.5:
            self._state["condition"] = max(
                0.0, self._state["condition"] - self.condition_decay
            )
            self._state["months_held"] += 1.0

        # Update derived state features
        self._state["time_step"] = self._t / self.horizon
        self._state["trend_3m"] = self._compute_trend()
        self._state["market_avg"] = self._state["price"]  # simplified proxy

        # --------------------------------------------------------------
        # 3. Check termination conditions
        # --------------------------------------------------------------
        terminated = False
        truncated = False

        # Bankruptcy
        if self._state["capital"] < self.bankruptcy_threshold:
            terminated = True

        # Success threshold (early stop during evaluation only)
        if self._state["capital"] >= self.success_threshold:
            terminated = True

        # Horizon reached
        if self._t >= self.horizon:
            truncated = True
            # Force-sale penalty if still owning at the end
            if self._state["owns"] > 0.5:
                forced_sale = (
                    self._state["price"]
                    * (1.0 + self.condition_bonus_max * self._state["condition"])
                    * (1.0 - self.forced_sale_cost)
                    - self._purchase_price
                )
                reward += forced_sale
                self._state["owns"] = 0.0

        self._episode_reward += reward

        obs = self._get_observation()
        info = self._get_info()
        info["episode_reward"] = self._episode_reward

        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> None | np.ndarray:
        """Render the environment.

        Args:
            mode (str): 'human' prints a text summary; 'rgb_array' returns
                a matplotlib-rendered image.

        Returns:
            np.ndarray | None: RGB array if mode='rgb_array', else None.
        """
        if mode == "human":
            print(
                f"t={self._t:3d} | price={self._state['price']:.3f} | "
                f"capital={self._state['capital']:.3f} | "
                f"owns={int(self._state['owns'])} | "
                f"condition={self._state['condition']:.2f} | "
                f"cum_reward={self._episode_reward:+.3f}"
            )
            return None

        if mode == "rgb_array":
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self._price_history, label="Property price", color="C0")
            ax.set_xlabel("Month")
            ax.set_ylabel("Normalized price")
            ax.set_title(f"Real Estate Investment — Episode t={self._t}")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())
            plt.close(fig)
            return image

        raise ValueError(f"Unsupported render mode: {mode}")

    # ======================================================================
    # Internal helpers
    # ======================================================================
    def _get_observation(self) -> np.ndarray:
        """Build the 8-dimensional normalized observation vector.

        Returns:
            np.ndarray: Observation vector of shape (8,) in [0, 1].
        """
        s = self._state
        obs = np.array(
            [
                np.clip(s["price"], 0.0, 2.0) / 2.0,
                np.clip(s["market_avg"], 0.0, 2.0) / 2.0,
                np.clip((s["trend_3m"] + 1.0) / 2.0, 0.0, 1.0),
                s["owns"],
                s["condition"],
                np.clip(s["capital"] / 2.0, 0.0, 1.0),
                np.clip(s["months_held"] / self.horizon, 0.0, 1.0),
                s["time_step"],
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary diagnostic info (not part of the observation)."""
        return {
            "t": self._t,
            "price": self._state["price"],
            "capital": self._state["capital"],
            "owns": int(self._state["owns"]),
            "condition": self._state["condition"],
        }

    def _is_valid_action(self, action: int) -> bool:
        """Check whether an action is valid given the current ownership state.

        Args:
            action (int): Action index.

        Returns:
            bool: True if valid, False otherwise.
        """
        owns = self._state["owns"] > 0.5
        if action == self.BUY and owns:
            return False
        if action in (self.RENOVATE, self.SELL) and not owns:
            return False
        return True

    def _compute_reward(self, action: int) -> float:
        """Compute the reward for taking an action, updating capital as needed.

        Args:
            action (int): Action index.

        Returns:
            float: Reward value for this step.
        """
        s = self._state
        reward = 0.0

        # -------- Invalid action penalty --------
        if not self._is_valid_action(action):
            return -self.invalid_action_penalty

        # -------- Action-specific dynamics --------
        if action == self.BUY:
            cost = s["price"] * (1.0 + self.buying_cost_rate)
            if s["capital"] < cost:
                return -self.invalid_action_penalty
            s["capital"] -= cost
            s["owns"] = 1.0
            s["months_held"] = 0.0
            self._purchase_price = s["price"]
            reward = 0.0

        elif action == self.RENOVATE:
            s["capital"] -= self.renovation_cost
            s["condition"] = min(1.0, s["condition"] + self.renovation_gain)
            reward = -self.renovation_cost

        elif action == self.SELL:
            condition_bonus = self.condition_bonus_max * s["condition"]
            sale_price = s["price"] * (1.0 + condition_bonus)
            revenue = sale_price * (1.0 - self.selling_cost_rate)
            profit = revenue - self._purchase_price
            s["capital"] += revenue
            s["owns"] = 0.0
            s["months_held"] = 0.0
            reward = profit

        elif action == self.WAIT:
            if s["owns"] > 0.5:
                s["capital"] -= self.maintenance_cost
                reward = -self.maintenance_cost
            elif s["capital"] > self.idle_capital_threshold:
                s["capital"] -= self.idle_capital_penalty
                reward = -self.idle_capital_penalty

        return reward

    def _update_price(self) -> None:
        """Advance the property price by one step of a geometric random walk."""
        shock = self._rng.normal(0.0, 1.0)
        new_price = self._state["price"] * np.exp(self.mu + self.sigma * shock)
        # Clip to avoid pathological values
        new_price = float(np.clip(new_price, 0.05, 5.0))
        self._state["price"] = new_price
        self._price_history.append(new_price)

    def _compute_trend(self) -> float:
        """Compute relative price trend over the last 3 months.

        Returns:
            float: Relative trend in [-1, 1].
        """
        if len(self._price_history) < 4:
            return 0.0
        past = self._price_history[-4]
        current = self._price_history[-1]
        if past < 1e-8:
            return 0.0
        trend = (current - past) / past
        return float(np.clip(trend, -1.0, 1.0))