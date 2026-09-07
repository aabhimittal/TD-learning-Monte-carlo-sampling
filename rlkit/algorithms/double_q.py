"""Variance- and bias-reduced TD control: Double Q-learning and Expected SARSA.

Plain Q-learning bootstraps off ``max_a Q(s', a)``. Because the max of noisy
estimates is a biased estimate of the max, it systematically over-values
actions whose returns are merely *noisy* -- exactly the failure mode you hit
in a plant, a market or an ad auction, where the same action pays out wildly
different amounts. Two standard fixes:

* :class:`DoubleQLearning` (van Hasselt, 2010) keeps two independent tables
  and uses one to *select* the greedy next action and the other to *evaluate*
  it, decoupling the selection noise from the evaluation noise.
* :class:`ExpectedSarsa` replaces the sampled next action of SARSA with the
  expectation over the behaviour policy, removing the variance contributed by
  the exploratory action choice at no extra cost per step.
"""
from __future__ import annotations

import numpy as np

from .td import _TabularTD


class DoubleQLearning(_TabularTD):
    """Off-policy TD control with two decoupled value tables.

    ``Q`` is exposed as the mean of the two tables, so ``act`` and
    ``greedy_policy`` behave exactly as they do for the single-table agents.
    """

    _SAVE_ARRAYS = ("QA", "QB")

    @property
    def Q(self) -> np.ndarray:
        return 0.5 * (self.QA + self.QB)

    @Q.setter
    def Q(self, value) -> None:
        # The base class initialises ``self.Q``; mirror it into both tables so
        # a caller can also warm-start the agent from a single table.
        value = np.asarray(value, dtype=np.float64)
        self.QA = value.copy()
        self.QB = value.copy()

    def act(self, state: int) -> int:
        # Same epsilon-greedy rule as the base class, but summing only the two
        # rows we need instead of materialising the whole averaged table.
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.QA[state] + self.QB[state]))

    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                terminal = self._is_terminal(done, info)

                # Update one table, chosen by a fair coin, using the other for
                # evaluation: A selects the argmax, B says what it is worth.
                if self.rng.random() < 0.5:
                    updating, evaluating = self.QA, self.QB
                else:
                    updating, evaluating = self.QB, self.QA
                if terminal:
                    bootstrap = 0.0
                else:
                    best = int(np.argmax(updating[next_state]))
                    bootstrap = self.gamma * evaluating[next_state, best]
                updating[state, action] += self.alpha * (
                    reward + bootstrap - updating[state, action]
                )

                state = next_state
                total += reward
                if done:
                    break
            self._end_episode()
            history.append(total)
            if log_every and ep % log_every == 0:
                print(f"[DoubleQ] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history


class ExpectedSarsa(_TabularTD):
    """On-policy TD control using the expected value of the next state.

    The target is ``r + gamma * sum_a pi(a|s') Q(s', a)`` with ``pi`` the
    current epsilon-greedy policy. Set ``off_policy=True`` to take the
    expectation under the *greedy* policy instead, which recovers Q-learning
    -- Expected SARSA is the general form both are special cases of.
    """

    _SAVE_SCALARS = _TabularTD._SAVE_SCALARS + ("off_policy",)

    def __init__(self, *args, off_policy: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.off_policy = bool(off_policy)

    def action_probabilities(self, state: int) -> np.ndarray:
        """Epsilon-greedy (or greedy, if ``off_policy``) action distribution."""
        eps = 0.0 if self.off_policy else self.epsilon
        probs = np.full(self.n_actions, eps / self.n_actions)
        best = np.flatnonzero(self.Q[state] == self.Q[state].max())
        # Ties share the greedy mass, so the probabilities always sum to 1.
        probs[best] += (1.0 - eps) / best.size
        return probs

    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                if self._is_terminal(done, info):
                    expected_next = 0.0
                else:
                    probs = self.action_probabilities(next_state)
                    expected_next = float(probs @ self.Q[next_state])
                target = reward + self.gamma * expected_next
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                state = next_state
                total += reward
                if done:
                    break
            self._end_episode()
            history.append(total)
            if log_every and ep % log_every == 0:
                print(f"[ExpSARSA] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history
