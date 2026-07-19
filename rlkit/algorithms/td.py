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


class NStepSarsa(_TabularTD):
    """On-policy n-step SARSA control (Sutton & Barto, ch. 7).

    Interpolates between one-step SARSA (``n=1``) and Monte-Carlo control
    (``n`` >= episode length). The n-step return bootstraps off
    ``Q[s_{t+n}, a_{t+n}]`` after accumulating ``n`` discounted rewards,
    trading bias for variance as ``n`` grows.
    """

    def __init__(self, *args, n: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n

    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        g_pows = self.gamma ** np.arange(self.n + 1)
        for ep in range(1, episodes + 1):
            states = [env.reset()]
            actions = [self.act(states[0])]
            rewards = [0.0]  # rewards[t] is the reward received entering step t
            total = 0.0
            T = float("inf")
            t = 0
            while True:
                if t < T:
                    s_next, r, done, _ = env.step(actions[t])
                    rewards.append(r)
                    states.append(s_next)
                    total += r
                    if done:
                        T = t + 1
                    else:
                        actions.append(self.act(s_next))
                    if t >= max_steps:  # safety cap
                        T = t + 1
                tau = t - self.n + 1
                if tau >= 0:
                    end = min(tau + self.n, T)
                    G = sum(g_pows[i - tau - 1] * rewards[i] for i in range(tau + 1, int(end) + 1))
                    if tau + self.n < T:
                        G += g_pows[self.n] * self.Q[states[tau + self.n], actions[tau + self.n]]
                    s_tau, a_tau = states[tau], actions[tau]
                    self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])
                t += 1
                if tau == T - 1:
                    break
            history.append(total)
            if log_every and ep % log_every == 0:
                print(f"[{self.n}-SARSA] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history


class SarsaLambda(_TabularTD):
    """SARSA(lambda) with accumulating eligibility traces.

    Eligibility traces provide the same bias/variance interpolation as n-step
    methods but in an online, incremental form: every visited ``(s, a)`` keeps
    a decaying trace, and the single TD error is broadcast to all of them,
    weighted by the trace. ``lam=0`` recovers one-step SARSA; ``lam=1`` (with
    ``gamma=1``) approximates Monte-Carlo.
    """

    def __init__(self, *args, lam: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam

    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            E = np.zeros_like(self.Q)
            state = env.reset()
            action = self.act(state)
            total = 0.0
            for _ in range(max_steps):
                next_state, reward, done, _ = env.step(action)
                total += reward
                next_action = self.act(next_state)
                target = 0.0 if done else self.gamma * self.Q[next_state, next_action]
                delta = reward + target - self.Q[state, action]
                E[state, action] += 1.0
                self.Q += self.alpha * delta * E
                E *= self.gamma * self.lam
                state, action = next_state, next_action
                if done:
                    break
            history.append(total)
            if log_every and ep % log_every == 0:
                print(f"[SARSA(λ)] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
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
