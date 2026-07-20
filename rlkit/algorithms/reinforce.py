"""REINFORCE: Monte-Carlo policy-gradient control.

REINFORCE directly optimises a parameterised stochastic policy
``pi_theta(a|s)`` by ascending the score-function estimate of the return
gradient::

    grad J(theta) = E[ sum_t grad log pi(a_t|s_t) * G_t ]

The policy is a small tanh MLP (see :mod:`rlkit.nn`). We use the softmax
identity ``d log pi(a|s) / d logits = one_hot(a) - pi`` to feed an exact
upstream gradient into the network's backward pass -- no autograd framework
required. Whitening the returns provides a simple, low-variance baseline;
an optional entropy bonus keeps the policy from collapsing prematurely.
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP, Adam, softmax


def discounted_returns(rewards, gamma: float) -> np.ndarray:
    G = 0.0
    out = np.empty(len(rewards), dtype=np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        out[t] = G
    return out


class Reinforce:
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden=(64,),
        gamma: float = 0.99,
        lr: float = 1e-2,
        normalize: bool = True,
        entropy_coef: float = 0.0,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.normalize = normalize
        self.entropy_coef = entropy_coef
        self.rng = np.random.default_rng(seed)
        self.policy = MLP(n_features, hidden, n_actions, seed=seed)
        self.opt = Adam(self.policy.params(), lr=lr)

    def act(self, feat: np.ndarray) -> int:
        logits = self.policy.forward(feat[None])[0]
        probs = softmax(logits)
        return int(self.rng.choice(self.n_actions, p=probs))

    def _rollout(self, env, max_steps):
        feats, actions, rewards = [], [], []
        obs = env.reset()
        for _ in range(max_steps):
            feat = env.features(obs)
            action = self.act(feat)
            obs, reward, done, _ = env.step(action)
            feats.append(feat)
            actions.append(action)
            rewards.append(reward)
            if done:
                break
        return np.array(feats), np.array(actions), rewards

    def greedy_act(self, feat: np.ndarray) -> int:
        return int(np.argmax(self.policy.forward(feat[None])[0]))

    def evaluate(self, env, episodes: int = 20, max_steps: int = 500) -> float:
        """Mean total reward of the deterministic (argmax) policy."""
        totals = []
        for _ in range(episodes):
            obs = env.reset()
            total = 0.0
            for _ in range(max_steps):
                obs, reward, done, _ = env.step(self.greedy_act(env.features(obs)))
                total += reward
                if done:
                    break
            totals.append(total)
        return float(np.mean(totals))

    def _update(self, feats, actions, rewards) -> float:
        returns = discounted_returns(rewards, self.gamma)
        adv = returns.copy()
        if self.normalize and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logits = self.policy.forward(feats)          # (T, A) -- caches activations
        probs = softmax(logits)
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(actions)), actions] = 1.0

        T = len(actions)
        # gradient of the *negative* objective (Adam does descent).
        grad = (probs - onehot) * adv[:, None] / T
        if self.entropy_coef:
            log_p = np.log(probs + 1e-12)
            H = -(probs * log_p).sum(axis=1, keepdims=True)
            grad += self.entropy_coef * probs * (log_p + H) / T

        self.policy.backward(grad)
        self.opt.step(self.policy.grads())
        return float(np.sum(rewards))

    def train(self, env, episodes=1000, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            feats, actions, rewards = self._rollout(env, max_steps)
            history.append(self._update(feats, actions, rewards))
            if log_every and ep % log_every == 0:
                print(f"[REINFORCE] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history

    def greedy_policy(self, n_states, one_hot) -> np.ndarray:
        feats = np.stack([one_hot(s) for s in range(n_states)])
        logits = self.policy.forward(feats)
        return np.argmax(logits, axis=1)
