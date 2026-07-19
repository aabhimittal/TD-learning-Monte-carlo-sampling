"""Temporal-difference control: SARSA (on-policy) and Q-learning (off-policy).

Unlike Monte-Carlo, TD methods bootstrap: they update value estimates from
other estimates after every single step, without waiting for the episode to
end. The two agents differ only in their TD target:

* **SARSA** uses the value of the action actually taken next
  (``Q[s', a']``) -- it learns the value of the epsilon-greedy policy it
  follows, so it prefers safer paths.
* **Q-learning** uses the greedy next action (``max_a Q[s', a]``) -- it learns
  the value of the optimal policy regardless of the exploratory behaviour.

``TDPrediction`` implements plain TD(0) state-value estimation for a fixed
policy.
"""
from __future__ import annotations

import numpy as np


class _TabularTD:
    def __init__(self, n_states, n_actions, gamma=0.99, alpha=0.5, epsilon=0.1, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)


class Sarsa(_TabularTD):
    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            state = env.reset()
            action = self.act(state)
            total = 0.0
            for _ in range(max_steps):
                next_state, reward, done, _ = env.step(action)
                next_action = self.act(next_state)
                target = reward + (0.0 if done else self.gamma * self.Q[next_state, next_action])
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                state, action = next_state, next_action
                total += reward
                if done:
                    break
            history.append(total)
            if log_every and ep % log_every == 0:
                print(f"[SARSA] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history


class QLearning(_TabularTD):
    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                best_next = 0.0 if done else np.max(self.Q[next_state])
                target = reward + self.gamma * best_next
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                state = next_state
                total += reward
                if done:
                    break
            history.append(total)
            if log_every and ep % log_every == 0:
                print(f"[Q] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history


class TDPrediction:
    """TD(0) state-value estimation ``V(s) <- V(s) + a[r + g V(s') - V(s)]``."""

    def __init__(self, n_states, gamma=0.99, alpha=0.1):
        self.gamma = gamma
        self.alpha = alpha
        self.V = np.zeros(n_states, dtype=np.float64)

    def train(self, env, policy, episodes=1000, max_steps=200):
        """Estimate V for a deterministic ``policy`` (array of action ids)."""
        for _ in range(episodes):
            state = env.reset()
            for _ in range(max_steps):
                action = int(policy[state])
                next_state, reward, done, _ = env.step(action)
                target = reward + (0.0 if done else self.gamma * self.V[next_state])
                self.V[state] += self.alpha * (target - self.V[state])
                state = next_state
                if done:
                    break
        return self.V
