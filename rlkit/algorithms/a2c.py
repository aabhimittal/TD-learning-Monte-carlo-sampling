"""A2C: (synchronous) Advantage Actor-Critic.

A2C augments REINFORCE with a learned **critic** ``V_phi(s)`` that acts as a
state-dependent baseline. The actor is updated with the *advantage*
``A_t = G_t - V(s_t)`` instead of the raw return, which dramatically reduces
gradient variance while remaining unbiased. The critic is trained by
regression toward the Monte-Carlo returns.

Both networks are numpy MLPs. The actor gradient reuses the softmax score
function from REINFORCE; the critic gradient is the usual mean-squared-error
signal ``2 (V - G)``.
"""
from __future__ import annotations

import numpy as np

from ..nn import MLP, Adam, softmax
from .reinforce import discounted_returns


class A2C:
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden=(64,),
        gamma: float = 0.99,
        actor_lr: float = 1e-2,
        critic_lr: float = 1e-2,
        entropy_coef: float = 0.01,
        seed: int | None = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.rng = np.random.default_rng(seed)

        self.actor = MLP(n_features, hidden, n_actions, seed=seed)
        self.critic = MLP(n_features, hidden, 1, seed=None if seed is None else seed + 1)
        self.actor_opt = Adam(self.actor.params(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.params(), lr=critic_lr)

    def act(self, feat: np.ndarray) -> int:
        logits = self.actor.forward(feat[None])[0]
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
        return int(np.argmax(self.actor.forward(feat[None])[0]))

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
        T = len(actions)

        # --- critic: regress V(s) -> G ------------------------------------
        values = self.critic.forward(feats)[:, 0]        # caches critic activations
        advantage = returns - values
        # normalise advantage for a well-conditioned actor step
        adv_norm = advantage.copy()
        if T > 1:
            adv_norm = (adv_norm - adv_norm.mean()) / (adv_norm.std() + 1e-8)

        critic_grad = (2.0 * (values - returns) / T)[:, None]
        self.critic.backward(critic_grad)
        self.critic_opt.step(self.critic.grads())

        # --- actor: policy gradient weighted by advantage -----------------
        logits = self.actor.forward(feats)               # caches actor activations
        probs = softmax(logits)
        onehot = np.zeros_like(probs)
        onehot[np.arange(T), actions] = 1.0

        grad = (probs - onehot) * adv_norm[:, None] / T
        if self.entropy_coef:
            log_p = np.log(probs + 1e-12)
            H = -(probs * log_p).sum(axis=1, keepdims=True)
            grad += self.entropy_coef * probs * (log_p + H) / T

        self.actor.backward(grad)
        self.actor_opt.step(self.actor.grads())
        return float(np.sum(rewards))

    def train(self, env, episodes=1000, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            feats, actions, rewards = self._rollout(env, max_steps)
            history.append(self._update(feats, actions, rewards))
            if log_every and ep % log_every == 0:
                print(f"[A2C] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history

    def greedy_policy(self, n_states, one_hot) -> np.ndarray:
        feats = np.stack([one_hot(s) for s in range(n_states)])
        logits = self.actor.forward(feats)
        return np.argmax(logits, axis=1)
