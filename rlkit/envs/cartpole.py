"""A dependency-free CartPole environment (continuous observations).

Re-implements the classic cart-pole control task (Barto, Sutton & Anderson,
1983) with the same physics and termination bounds as the well-known Gym
version, using only NumPy. Unlike the grid worlds, observations are
continuous 4-vectors ``[x, x_dot, theta, theta_dot]`` and actions are the two
discrete pushes, so this env is a natural target for the policy-gradient
agents (:class:`~rlkit.algorithms.Reinforce`, :class:`~rlkit.algorithms.A2C`)
running over function approximation instead of a lookup table.
"""
from __future__ import annotations

import numpy as np


class CartPole:
    n_actions = 2
    n_features = 4

    # physics constants (identical to the canonical formulation)
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    length = 0.5  # half the pole's length
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    # termination thresholds
    x_threshold = 2.4
    theta_threshold = 12 * np.pi / 180  # 12 degrees in radians

    def __init__(self, max_steps: int = 500, seed: int | None = None):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.total_mass = self.masscart + self.masspole
        self.polemass_length = self.masspole * self.length
        self._state = np.zeros(4, dtype=np.float64)
        self._steps = 0

    def features(self, obs: np.ndarray) -> np.ndarray:
        """CartPole observations are already the feature vector."""
        return np.asarray(obs, dtype=np.float64)

    def reset(self) -> np.ndarray:
        self._state = self.rng.uniform(-0.05, 0.05, size=4)
        self._steps = 0
        return self._state.copy()

    def step(self, action: int):
        if action not in (0, 1):
            raise ValueError(f"invalid action {action}")

        x, x_dot, theta, theta_dot = self._state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # standard cart-pole dynamics
        temp = (force + self.polemass_length * theta_dot ** 2 * sin_t) / self.total_mass
        theta_acc = (self.gravity * sin_t - cos_t * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_t ** 2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_t / self.total_mass

        # semi-implicit Euler integration
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc
        self._state = np.array([x, x_dot, theta, theta_dot])
        self._steps += 1

        failed = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold
            or theta > self.theta_threshold
        )
        truncated = self._steps >= self.max_steps
        done = failed or truncated
        reward = 1.0  # +1 for every step the pole stays up
        return self._state.copy(), reward, done, {"truncated": truncated and not failed}
