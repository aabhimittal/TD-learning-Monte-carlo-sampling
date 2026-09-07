"""Predictive maintenance: run the machine, service it, or replace it.

A machine degrades stochastically while it produces. Running a worn machine
earns revenue but risks a failure that costs far more than a scheduled
service; servicing costs a little and buys back some life; replacing costs a
lot and resets the asset. The optimal policy is a *threshold* policy -- keep
running below some wear level, service above it -- which makes this a compact
way to check that an agent has learned structure rather than memorised a path.

What makes it a good stress test:

* **rare, expensive events.** Failures are low-probability and high-cost, so
  an agent that only ever sees the mean reward learns the wrong policy. This
  is where Q-learning's maximisation bias shows up and Double Q-learning
  earns its keep.
* **an absorbing-ish trap.** Once broken, the machine earns nothing until it
  is replaced, so the agent has to learn a recovery action it is rarely
  rewarded for taking.
* **no goal state.** Like the inventory task, episodes end on a horizon.

States are wear levels ``0 .. n_wear-1`` plus a terminal-looking ``FAILED``
state at index ``n_wear``. Actions are ``RUN``, ``MAINTAIN``, ``REPLACE``.
"""
from __future__ import annotations

import numpy as np

from ..validation import check_finite, check_positive_int, check_probability

RUN, MAINTAIN, REPLACE = 0, 1, 2


class MachineMaintenance:
    """Degrading-machine maintenance scheduling.

    Parameters
    ----------
    n_wear:
        Number of healthy wear levels (0 = new).
    revenue:
        Earned per period the machine runs without failing.
    maintain_cost, replace_cost, failure_cost, downtime_cost:
        Cost of a service, of a new machine, of a failure event, and of each
        period spent broken.
    base_failure_prob, failure_slope:
        Failure probability while running is
        ``base_failure_prob + failure_slope * wear``, clipped to ``[0, 1]``.
    degrade_prob:
        Probability that a successful run advances the wear level by one.
    """

    n_actions = 3
    action_names = ("run", "maintain", "replace")

    def __init__(
        self,
        n_wear: int = 5,
        revenue: float = 10.0,
        maintain_cost: float = 4.0,
        replace_cost: float = 20.0,
        failure_cost: float = 60.0,
        downtime_cost: float = 5.0,
        base_failure_prob: float = 0.02,
        failure_slope: float = 0.12,
        degrade_prob: float = 0.35,
        horizon: int = 60,
        seed: int | None = None,
    ):
        self.n_wear = check_positive_int("n_wear", n_wear)
        self.revenue = check_finite("revenue", revenue)
        self.maintain_cost = check_finite("maintain_cost", maintain_cost)
        self.replace_cost = check_finite("replace_cost", replace_cost)
        self.failure_cost = check_finite("failure_cost", failure_cost)
        self.downtime_cost = check_finite("downtime_cost", downtime_cost)
        self.base_failure_prob = check_probability("base_failure_prob", base_failure_prob)
        self.failure_slope = check_finite("failure_slope", failure_slope)
        if self.failure_slope < 0:
            raise ValueError(f"failure_slope must be >= 0, got {self.failure_slope}")
        self.degrade_prob = check_probability("degrade_prob", degrade_prob)
        self.horizon = check_positive_int("horizon", horizon)
        self.rng = np.random.default_rng(seed)

        self.failed_state = self.n_wear  # the machine is broken, not the episode
        self.n_states = self.n_wear + 1
        self.n_features = self.n_states
        self._state = 0
        self._steps = 0

    # -- helpers ------------------------------------------------------------
    def failure_probability(self, wear: int) -> float:
        """Probability that a run at this wear level ends in a breakdown."""
        if wear >= self.n_wear:
            return 1.0
        return float(np.clip(self.base_failure_prob + self.failure_slope * wear, 0.0, 1.0))

    def one_hot(self, state: int) -> np.ndarray:
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[int(state)] = 1.0
        return vec

    def features(self, state: int) -> np.ndarray:
        return self.one_hot(state)

    def is_terminal(self, state: int) -> bool:
        """``FAILED`` is a trap, not a terminal: only ``REPLACE`` escapes it."""
        return False

    # -- gym-like API -------------------------------------------------------
    def reset(self) -> int:
        self._state = 0
        self._steps = 0
        return self._state

    def step(self, action: int):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid action {action}")
        action = int(action)
        broken = self._state == self.failed_state
        failed_now = False

        if action == REPLACE:
            reward = -self.replace_cost
            self._state = 0
        elif action == MAINTAIN:
            if broken:
                # A service call cannot resurrect a broken machine; you pay the
                # call-out and the downtime and are still broken.
                reward = -self.maintain_cost - self.downtime_cost
            else:
                reward = -self.maintain_cost
                self._state = max(0, self._state - 1)
        else:  # RUN
            if broken:
                reward = -self.downtime_cost
            elif self.rng.random() < self.failure_probability(self._state):
                reward = -self.failure_cost
                self._state = self.failed_state
                failed_now = True
            else:
                reward = self.revenue
                if self.rng.random() < self.degrade_prob:
                    self._state = min(self._state + 1, self.n_wear - 1)

        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {"truncated": truncated, "failed": failed_now,
                "broken": self._state == self.failed_state}
        return self._state, float(reward), truncated, info

    # -- baselines ----------------------------------------------------------
    def threshold_policy(self, maintain_at: int) -> np.ndarray:
        """Run below ``maintain_at`` wear, service at or above it, replace when broken."""
        policy = np.full(self.n_states, RUN, dtype=np.int64)
        for wear in range(self.n_wear):
            if wear >= maintain_at:
                policy[wear] = MAINTAIN
        policy[self.failed_state] = REPLACE
        return policy
