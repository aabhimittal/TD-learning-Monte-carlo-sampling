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

from ..persistence import SaveLoadMixin
from ..schedules import as_schedule
from ..validation import check_positive_int, check_probability, check_range


def terminated_only(done, info) -> bool:
    """Distinguish a true terminal state from a time-limit truncation.

    Environments here follow the Gym convention of reporting a truncation as
    ``done=True`` plus ``info["truncated"]=True``. The episode does end, but
    the *MDP* did not reach an absorbing state, so the value of the final
    state is **not** zero and bootstrapping through it is the correct thing to
    do. Treating a truncation as terminal is one of the most common silent
    bugs in applied RL: with a 24-hour episode cap it teaches the agent that
    the world ends every midnight.
    """
    return bool(done) and not bool((info or {}).get("truncated", False))


class _TabularTD(SaveLoadMixin):
    """Shared machinery for the tabular TD agents.

    ``alpha`` and ``epsilon`` accept either a float or a
    :class:`~rlkit.schedules.Schedule`; schedules are advanced once per
    training episode, so ``LinearSchedule(1.0, 0.05, 500)`` anneals
    exploration over the first 500 episodes.
    """

    _SAVE_ARRAYS = ("Q",)
    _SAVE_SCALARS = ("n_states", "n_actions", "gamma", "alpha", "epsilon",
                     "bootstrap_on_truncation")

    def __init__(self, n_states, n_actions, gamma=0.99, alpha=0.5, epsilon=0.1,
                 seed=None, bootstrap_on_truncation: bool = True):
        self.n_states = check_positive_int("n_states", n_states)
        self.n_actions = check_positive_int("n_actions", n_actions)
        self.gamma = check_range("gamma", gamma, 0.0, 1.0)
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        self._episode = 0
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    # -- schedule-aware hyper-parameters ------------------------------------
    @property
    def alpha(self) -> float:
        return self._alpha_schedule.value(self._episode)

    @alpha.setter
    def alpha(self, value) -> None:
        schedule = as_schedule(value)
        check_range("alpha", schedule.value(self._episode), 0.0, 1.0)
        self._alpha_schedule = schedule

    @property
    def epsilon(self) -> float:
        return self._epsilon_schedule.value(self._episode)

    @epsilon.setter
    def epsilon(self, value) -> None:
        schedule = as_schedule(value)
        check_probability("epsilon", schedule.value(self._episode))
        self._epsilon_schedule = schedule

    def _end_episode(self) -> None:
        """Advance the schedules by one episode."""
        self._episode += 1

    def _is_terminal(self, done, info) -> bool:
        """Should the TD target treat this transition as absorbing?"""
        if self.bootstrap_on_truncation:
            return terminated_only(done, info)
        return bool(done)

    # -- behaviour ----------------------------------------------------------
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
                next_state, reward, done, info = env.step(action)
                next_action = self.act(next_state)
                terminal = self._is_terminal(done, info)
                target = reward + (0.0 if terminal else self.gamma * self.Q[next_state, next_action])
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                state, action = next_state, next_action
                total += reward
                if done:
                    break
            self._end_episode()
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
                next_state, reward, done, info = env.step(action)
                best_next = 0.0 if self._is_terminal(done, info) else np.max(self.Q[next_state])
                target = reward + self.gamma * best_next
                self.Q[state, action] += self.alpha * (target - self.Q[state, action])
                state = next_state
                total += reward
                if done:
                    break
            self._end_episode()
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

    _SAVE_SCALARS = _TabularTD._SAVE_SCALARS + ("n",)

    def __init__(self, *args, n: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = check_positive_int("n", n)

    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        g_pows = self.gamma ** np.arange(self.n + 1)
        for ep in range(1, episodes + 1):
            states = [env.reset()]
            actions = [self.act(states[0])]
            rewards = [0.0]  # rewards[t] is the reward received entering step t
            total = 0.0
            T = float("inf")
            # When the episode ends by truncation rather than termination the
            # tail of the n-step return must still bootstrap off Q[s_T, a_T].
            bootstrap_tail = False
            t = 0
            while True:
                if t < T:
                    s_next, r, done, info = env.step(actions[t])
                    rewards.append(r)
                    states.append(s_next)
                    total += r
                    ended = done or t >= max_steps - 1  # safety cap
                    if ended:
                        T = t + 1
                        bootstrap_tail = not self._is_terminal(done, info)
                        if bootstrap_tail:
                            actions.append(self.act(s_next))
                    else:
                        actions.append(self.act(s_next))
                tau = t - self.n + 1
                if tau >= 0:
                    end = min(tau + self.n, T)
                    G = sum(g_pows[i - tau - 1] * rewards[i] for i in range(tau + 1, int(end) + 1))
                    if tau + self.n < T:
                        G += g_pows[self.n] * self.Q[states[tau + self.n], actions[tau + self.n]]
                    elif bootstrap_tail:
                        horizon = int(T) - tau
                        G += g_pows[horizon] * self.Q[states[int(T)], actions[int(T)]]
                    s_tau, a_tau = states[tau], actions[tau]
                    self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])
                t += 1
                if tau == T - 1:
                    break
            self._end_episode()
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

    _SAVE_SCALARS = _TabularTD._SAVE_SCALARS + ("lam",)

    def __init__(self, *args, lam: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = check_probability("lam", lam)

    def train(self, env, episodes=500, max_steps=200, log_every=0):
        history = []
        for ep in range(1, episodes + 1):
            E = np.zeros_like(self.Q)
            state = env.reset()
            action = self.act(state)
            total = 0.0
            for _ in range(max_steps):
                next_state, reward, done, info = env.step(action)
                total += reward
                next_action = self.act(next_state)
                terminal = self._is_terminal(done, info)
                target = 0.0 if terminal else self.gamma * self.Q[next_state, next_action]
                delta = reward + target - self.Q[state, action]
                E[state, action] += 1.0
                self.Q += self.alpha * delta * E
                E *= self.gamma * self.lam
                state, action = next_state, next_action
                if done:
                    break
            self._end_episode()
            history.append(total)
            if log_every and ep % log_every == 0:
                print(f"[SARSA(λ)] episode {ep:5d}  avg_return={np.mean(history[-log_every:]):8.2f}")
        return history


class TDPrediction(SaveLoadMixin):
    """TD(0) state-value estimation ``V(s) <- V(s) + a[r + g V(s') - V(s)]``."""

    _SAVE_ARRAYS = ("V",)
    _SAVE_SCALARS = ("n_states", "gamma", "alpha", "bootstrap_on_truncation")

    def __init__(self, n_states, gamma=0.99, alpha=0.1,
                 bootstrap_on_truncation: bool = True):
        self.n_states = check_positive_int("n_states", n_states)
        self.gamma = check_range("gamma", gamma, 0.0, 1.0)
        self.alpha = check_range("alpha", alpha, 0.0, 1.0)
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        self.V = np.zeros(self.n_states, dtype=np.float64)

    def train(self, env, policy, episodes=1000, max_steps=200):
        """Estimate V for a deterministic ``policy`` (array of action ids)."""
        for _ in range(episodes):
            state = env.reset()
            for _ in range(max_steps):
                action = int(policy[state])
                next_state, reward, done, info = env.step(action)
                terminal = terminated_only(done, info) if self.bootstrap_on_truncation else done
                target = reward + (0.0 if terminal else self.gamma * self.V[next_state])
                self.V[state] += self.alpha * (target - self.V[state])
                state = next_state
                if done:
                    break
        return self.V
