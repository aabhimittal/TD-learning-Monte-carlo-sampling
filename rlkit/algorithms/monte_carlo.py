"""Monte-Carlo control via first-visit sampling.

Monte-Carlo methods learn purely from complete sampled episodes: no
bootstrapping and no model of the environment. We estimate action values by
averaging the observed returns following the first visit to each
``(state, action)`` pair, and improve the policy greedily with respect to the
current estimates (epsilon-greedy for exploration).
"""
from __future__ import annotations

import numpy as np

from ..persistence import SaveLoadMixin
from ..schedules import as_schedule
from ..validation import check_positive_int, check_probability, check_range


class MonteCarloControl(SaveLoadMixin):
    """First-visit MC control with an epsilon-greedy behaviour policy.

    ``epsilon`` accepts a float or a :class:`~rlkit.schedules.Schedule`, which
    is re-read once per training episode.
    """

    _SAVE_ARRAYS = ("Q", "_returns_sum", "_returns_cnt")
    _SAVE_SCALARS = ("n_states", "n_actions", "gamma", "epsilon")

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        self.n_states = check_positive_int("n_states", n_states)
        self.n_actions = check_positive_int("n_actions", n_actions)
        self.gamma = check_range("gamma", gamma, 0.0, 1.0)
        self._episode = 0
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._returns_sum = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._returns_cnt = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

    @property
    def epsilon(self) -> float:
        return self._epsilon_schedule.value(self._episode)

    @epsilon.setter
    def epsilon(self, value) -> None:
        schedule = as_schedule(value)
        check_probability("epsilon", schedule.value(self._episode))
        self._epsilon_schedule = schedule

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def _rollout(self, env, max_steps: int):
        traj = []
        state = env.reset()
        for _ in range(max_steps):
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            traj.append((state, action, reward))
            state = next_state
            if done:
                break
        return traj

    def update_from_episode(self, traj) -> float:
        """First-visit MC update from one trajectory. Returns the episode return."""
        visited = set()
        G = 0.0
        # Walk the trajectory backwards accumulating discounted returns.
        returns = []
        for state, action, reward in reversed(traj):
            G = reward + self.gamma * G
            returns.append((state, action, G))
        returns.reverse()

        ep_return = sum(r for _, _, r in traj)
        for state, action, G in returns:
            if (state, action) in visited:
                continue
            visited.add((state, action))
            self._returns_sum[state, action] += G
            self._returns_cnt[state, action] += 1
            self.Q[state, action] = (
                self._returns_sum[state, action] / self._returns_cnt[state, action]
            )
        return ep_return

    def train(self, env, episodes: int = 2000, max_steps: int = 200, log_every: int = 0):
        history = []
        for ep in range(1, episodes + 1):
            traj = self._rollout(env, max_steps)
            history.append(self.update_from_episode(traj))
            self._episode += 1  # advance any epsilon schedule
            if log_every and ep % log_every == 0:
                avg = np.mean(history[-log_every:])
                print(f"[MC] episode {ep:5d}  avg_return={avg:8.2f}")
        return history
