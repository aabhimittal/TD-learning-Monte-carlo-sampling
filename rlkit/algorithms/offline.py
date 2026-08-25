"""Offline (batch) RL: learning a policy from logged data you cannot extend.

This is the setting most industrial RL projects actually start in. You have a
year of logs from the incumbent controller -- a heuristic, a PID loop, a human
operator -- and you are not allowed to run an untested policy against the real
plant to gather more. Everything must be squeezed out of the fixed dataset,
and the honest answer to "is this dataset even sufficient?" matters as much as
the learning rule.

* :class:`BatchQLearning` -- tabular fitted-Q iteration over a replay buffer.
* :class:`OffPolicyMonteCarlo` -- weighted importance sampling from episodes
  logged under a known behaviour policy.

Both expose :meth:`coverage`, the fraction of state-action pairs the dataset
actually visited. A tabular agent can say nothing about a pair it never saw,
so both also support a ``min_count`` guard that keeps the extracted policy
inside the data's support instead of chasing an unsupported optimum.
"""
from __future__ import annotations

import numpy as np

from ..buffers import Batch
from ..persistence import SaveLoadMixin
from ..validation import check_positive_int, check_range

__all__ = ["BatchQLearning", "OffPolicyMonteCarlo"]


def _as_batch(data) -> Batch:
    """Accept a Batch, a ReplayBuffer, or a ``(s, a, r, s', done)`` tuple."""
    if isinstance(data, Batch):
        batch = data
    elif hasattr(data, "all"):  # a ReplayBuffer
        batch = data.all()
    else:
        batch = Batch(*data)
    states = np.asarray(batch.states, dtype=np.int64).reshape(-1)
    actions = np.asarray(batch.actions, dtype=np.int64).reshape(-1)
    rewards = np.asarray(batch.rewards, dtype=np.float64).reshape(-1)
    next_states = np.asarray(batch.next_states, dtype=np.int64).reshape(-1)
    dones = np.asarray(batch.dones, dtype=bool).reshape(-1)
    sizes = {states.size, actions.size, rewards.size, next_states.size, dones.size}
    if len(sizes) != 1:
        raise ValueError("all transition arrays must have the same length")
    if states.size == 0:
        raise ValueError("cannot learn from an empty dataset")
    if not np.all(np.isfinite(rewards)):
        raise ValueError("dataset contains non-finite rewards")
    return Batch(states, actions, rewards, next_states, dones)


class BatchQLearning(SaveLoadMixin):
    """Tabular fitted-Q iteration: repeated Bellman backups over a fixed batch.

    Each sweep recomputes the target ``r + gamma * (1 - done) * max_a Q(s', a)``
    for every logged transition and sets ``Q(s, a)`` to the mean target over
    the transitions that took ``a`` in ``s``. With a tabular representation
    this is exactly the sample-average Bellman-optimality operator, so it
    converges to the optimal Q of the empirical MDP.

    Parameters
    ----------
    unseen_value:
        Value assigned to state-action pairs absent from the dataset. Keep it
        at ``0`` for a neutral prior, or set it low (pessimistic) so that
        unsupported actions are never preferred.
    """

    _SAVE_ARRAYS = ("Q", "counts")
    _SAVE_SCALARS = ("n_states", "n_actions", "gamma", "unseen_value")

    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.99,
                 unseen_value: float = 0.0):
        self.n_states = check_positive_int("n_states", n_states)
        self.n_actions = check_positive_int("n_actions", n_actions)
        self.gamma = check_range("gamma", gamma, 0.0, 1.0)
        self.unseen_value = float(unseen_value)
        self.Q = np.full((self.n_states, self.n_actions), self.unseen_value,
                         dtype=np.float64)
        self.counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)

    def fit(self, data, iterations: int = 100, tol: float = 1e-6) -> list[float]:
        """Run fitted-Q iteration; returns the max Q-change per sweep.

        Stops early once the sweep moves no value by more than ``tol``, which
        is the practical convergence check for the tabular case.
        """
        batch = _as_batch(data)
        iterations = check_positive_int("iterations", iterations)
        if batch.states.max(initial=0) >= self.n_states or batch.states.min(initial=0) < 0:
            raise ValueError("dataset contains a state id outside [0, n_states)")
        if batch.actions.max(initial=0) >= self.n_actions or batch.actions.min(initial=0) < 0:
            raise ValueError("dataset contains an action id outside [0, n_actions)")

        self.counts = np.zeros_like(self.counts)
        np.add.at(self.counts, (batch.states, batch.actions), 1)
        seen = self.counts > 0
        safe_counts = np.where(seen, self.counts, 1)
        not_done = (~batch.dones).astype(np.float64)

        deltas = []
        for _ in range(iterations):
            targets = batch.rewards + self.gamma * not_done * self.Q[batch.next_states].max(axis=1)
            sums = np.zeros_like(self.Q)
            np.add.at(sums, (batch.states, batch.actions), targets)
            new_Q = np.where(seen, sums / safe_counts, self.unseen_value)
            delta = float(np.max(np.abs(new_Q - self.Q)))
            self.Q = new_Q
            deltas.append(delta)
            if delta < tol:
                break
        return deltas

    def coverage(self) -> float:
        """Fraction of ``(state, action)`` pairs present in the dataset."""
        return float((self.counts > 0).mean())

    def greedy_policy(self, min_count: int = 0) -> np.ndarray:
        """Greedy policy, optionally restricted to well-supported actions.

        With ``min_count > 0`` an action is only eligible in a state if the
        dataset contains at least that many examples of it; states with no
        eligible action fall back to the unrestricted argmax.
        """
        if min_count <= 0:
            return np.argmax(self.Q, axis=1)
        supported = self.counts >= min_count
        masked = np.where(supported, self.Q, -np.inf)
        policy = np.argmax(masked, axis=1)
        unsupported_states = ~supported.any(axis=1)
        policy[unsupported_states] = np.argmax(self.Q[unsupported_states], axis=1)
        return policy


class OffPolicyMonteCarlo(SaveLoadMixin):
    """Off-policy MC control with weighted importance sampling.

    Learns the greedy target policy from episodes generated by a *different*
    behaviour policy whose action probabilities were logged. Weighted (rather
    than ordinary) importance sampling is used because its bounded variance is
    what makes the estimate usable on real logs; the price is a small bias
    that vanishes as data accumulates.

    Episodes are sequences of ``(state, action, reward, behaviour_prob)``.
    Because the target policy is deterministic, the backward pass stops at the
    first action the target policy would not have taken -- everything earlier
    has an importance weight of zero.
    """

    _SAVE_ARRAYS = ("Q", "C")
    _SAVE_SCALARS = ("n_states", "n_actions", "gamma", "max_weight", "init_value")

    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.99,
                 max_weight: float = 1e6, init_value: float = 0.0):
        self.n_states = check_positive_int("n_states", n_states)
        self.n_actions = check_positive_int("n_actions", n_actions)
        self.gamma = check_range("gamma", gamma, 0.0, 1.0)
        # Importance weights compound multiplicatively and can explode when the
        # behaviour policy rarely picks the target action; clip for stability.
        self.max_weight = float(max_weight)
        if self.max_weight <= 0:
            raise ValueError(f"max_weight must be > 0, got {self.max_weight}")
        # The backward pass stops at the first action the (greedy) target policy
        # would not have taken, so an unvisited pair that outranks every visited
        # one stalls learning permanently. On a cost-shaped task -- where all
        # returns are negative and the zero-initialised unknowns therefore look
        # best -- set ``init_value`` below the worst achievable return. That is
        # also the pessimism you want offline: never prefer an action on the
        # strength of having no data about it.
        self.init_value = float(init_value)
        self.Q = np.full((self.n_states, self.n_actions), self.init_value,
                         dtype=np.float64)
        self.C = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    def update_from_episode(self, episode) -> float:
        """Incorporate one logged episode; returns its undiscounted return."""
        G = 0.0
        W = 1.0
        for state, action, reward, behaviour_prob in reversed(list(episode)):
            state, action = int(state), int(action)
            if not 0 <= state < self.n_states:
                raise ValueError(f"state {state} outside [0, {self.n_states})")
            if not 0 <= action < self.n_actions:
                raise ValueError(f"action {action} outside [0, {self.n_actions})")
            behaviour_prob = float(behaviour_prob)
            if not 0.0 < behaviour_prob <= 1.0:
                raise ValueError(
                    "behaviour probabilities must lie in (0, 1]; a logged action "
                    f"with probability {behaviour_prob} could never have been taken"
                )
            G = reward + self.gamma * G
            self.C[state, action] += W
            self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
            if action != int(np.argmax(self.Q[state])):
                break  # target policy diverges here: earlier weights are 0
            W = min(W / behaviour_prob, self.max_weight)
        return float(sum(step[2] for step in episode))

    def train_from_dataset(self, dataset, passes: int = 1) -> list[float]:
        """Learn from a list of logged episodes; returns the episode returns.

        Extra ``passes`` simply replay the same log, which helps because the
        target policy -- and therefore which prefixes carry non-zero weight --
        changes as ``Q`` improves.
        """
        passes = check_positive_int("passes", passes)
        returns = []
        for _ in range(passes):
            for episode in dataset:
                returns.append(self.update_from_episode(episode))
        return returns

    def coverage(self) -> float:
        """Fraction of ``(state, action)`` pairs that received any weight."""
        return float((self.C > 0).mean())

    def greedy_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
