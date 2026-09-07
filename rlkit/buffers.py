"""Experience replay: the bridge between online agents and offline (batch) RL.

In a deployed system you rarely get to interact freely with the plant, the
warehouse or the customer -- you get a *log* of past interactions. A replay
buffer is the data structure that turns such a log into something an agent
can learn from, and it is also what lets an online agent reuse each
transition many times instead of throwing it away.

Two variants are provided:

* :class:`ReplayBuffer` -- a fixed-capacity ring buffer with uniform sampling.
* :class:`PrioritizedReplayBuffer` -- samples surprising transitions (large
  TD error) more often, and returns the importance-sampling weights needed to
  correct the resulting bias (Schaul et al., 2016).

Both validate what they are fed: a ``nan`` reward from a broken sensor
should raise at ingest time, not silently poison every batch afterwards.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from .validation import check_finite, check_positive_int, check_probability

__all__ = ["Transition", "Batch", "ReplayBuffer", "PrioritizedReplayBuffer",
           "collect_transitions"]

Transition = namedtuple("Transition", "state action reward next_state done")

#: A batch of transitions as parallel arrays (what the batch agents consume).
Batch = namedtuple("Batch", "states actions rewards next_states dones")


class ReplayBuffer:
    """Fixed-capacity FIFO buffer of transitions with uniform sampling.

    Parameters
    ----------
    capacity:
        Maximum number of transitions kept. Once full, the oldest transition
        is overwritten by each new one.
    seed:
        Seed for the sampling RNG (sampling is reproducible given a seed).

    Notes
    -----
    ``done`` should mean *terminated* (the MDP reached an absorbing state),
    **not** *truncated* (a time limit fired). Bootstrapping through a
    time-limit truncation is correct; bootstrapping through a true terminal
    is not. :func:`collect_transitions` makes that distinction for you.
    """

    def __init__(self, capacity: int = 10_000, seed: int | None = None):
        self.capacity = check_positive_int("capacity", capacity)
        self.rng = np.random.default_rng(seed)
        self._storage: list[Transition] = []
        self._next = 0

    # -- ingest -------------------------------------------------------------
    def add(self, state, action, reward, next_state, done) -> None:
        """Append one transition, evicting the oldest if the buffer is full."""
        reward = check_finite("reward", reward)
        transition = Transition(state, int(action), reward, next_state, bool(done))
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def extend(self, transitions) -> None:
        """Add an iterable of ``(s, a, r, s', done)`` tuples."""
        for t in transitions:
            self.add(*t)

    def clear(self) -> None:
        self._storage.clear()
        self._next = 0

    # -- read ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._storage)

    @property
    def is_full(self) -> bool:
        return len(self._storage) >= self.capacity

    def _batch(self, transitions) -> Batch:
        return Batch(
            states=np.asarray([t.state for t in transitions]),
            actions=np.asarray([t.action for t in transitions], dtype=np.int64),
            rewards=np.asarray([t.reward for t in transitions], dtype=np.float64),
            next_states=np.asarray([t.next_state for t in transitions]),
            dones=np.asarray([t.done for t in transitions], dtype=bool),
        )

    def all(self) -> Batch:
        """Return every stored transition as one :class:`Batch`."""
        if not self._storage:
            raise ValueError("buffer is empty")
        return self._batch(self._storage)

    def sample(self, batch_size: int, replace: bool = False) -> Batch:
        """Draw ``batch_size`` transitions uniformly at random."""
        batch_size = check_positive_int("batch_size", batch_size)
        if not self._storage:
            raise ValueError("cannot sample from an empty buffer")
        if not replace and batch_size > len(self._storage):
            raise ValueError(
                f"cannot draw {batch_size} distinct transitions from a buffer "
                f"holding {len(self._storage)} (pass replace=True to allow repeats)"
            )
        idx = self.rng.choice(len(self._storage), size=batch_size, replace=replace)
        return self._batch([self._storage[i] for i in idx])


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritised replay with importance-sampling correction.

    New transitions enter with the current maximum priority so that every
    transition is replayed at least once. ``sample`` returns the batch, the
    sampled indices (to feed back into :meth:`update_priorities`) and the
    normalised IS weights that de-bias the non-uniform sampling.

    ``alpha=0`` recovers uniform sampling; ``beta=0`` disables the weight
    correction (all weights equal 1).
    """

    def __init__(self, capacity: int = 10_000, alpha: float = 0.6,
                 epsilon: float = 1e-6, seed: int | None = None):
        super().__init__(capacity, seed)
        self.alpha = check_probability("alpha", alpha)
        self.epsilon = check_finite("epsilon", epsilon)
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        self._priorities = np.zeros(self.capacity, dtype=np.float64)

    def add(self, state, action, reward, next_state, done, priority=None) -> None:
        slot = self._next
        super().add(state, action, reward, next_state, done)
        if priority is None:
            known = self._priorities[:len(self._storage)]
            priority = float(known.max()) if known.size and known.max() > 0 else 1.0
        priority = check_finite("priority", priority)
        if priority < 0:
            raise ValueError(f"priority must be >= 0, got {priority}")
        self._priorities[slot] = priority + self.epsilon

    def clear(self) -> None:
        super().clear()
        self._priorities[:] = 0.0

    def _probabilities(self) -> np.ndarray:
        p = self._priorities[:len(self._storage)] ** self.alpha
        total = p.sum()
        if total <= 0 or not np.isfinite(total):  # all-zero priorities -> uniform
            return np.full(len(self._storage), 1.0 / len(self._storage))
        return p / total

    def sample(self, batch_size: int, beta: float = 0.4, replace: bool = True):
        """Return ``(batch, indices, weights)`` sampled by priority."""
        batch_size = check_positive_int("batch_size", batch_size)
        beta = check_finite("beta", beta)
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        if not self._storage:
            raise ValueError("cannot sample from an empty buffer")
        if not replace and batch_size > len(self._storage):
            raise ValueError(
                f"cannot draw {batch_size} distinct transitions from a buffer "
                f"holding {len(self._storage)}"
            )
        probs = self._probabilities()
        idx = self.rng.choice(len(self._storage), size=batch_size, replace=replace,
                              p=probs)
        weights = (len(self._storage) * probs[idx]) ** (-beta)
        weights /= weights.max()  # normalise so the largest weight is 1
        return self._batch([self._storage[i] for i in idx]), idx, weights

    def update_priorities(self, indices, priorities) -> None:
        """Set new priorities (typically ``abs(td_error)``) for ``indices``."""
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        priorities = np.asarray(priorities, dtype=np.float64).reshape(-1)
        if indices.shape != priorities.shape:
            raise ValueError("indices and priorities must have the same length")
        if indices.size and (indices.min() < 0 or indices.max() >= len(self._storage)):
            raise IndexError("priority index out of range for the current buffer")
        if not np.all(np.isfinite(priorities)):
            raise ValueError("priorities must be finite")
        if np.any(priorities < 0):
            raise ValueError("priorities must be >= 0")
        self._priorities[indices] = priorities + self.epsilon


def collect_transitions(env, policy=None, episodes: int = 50, max_steps: int = 200,
                        buffer: ReplayBuffer | None = None, seed: int | None = None):
    """Roll out ``policy`` in ``env`` and log the transitions into a buffer.

    ``policy`` is a callable ``state -> action``; the default is a uniform
    random behaviour policy, which is the usual way to seed an offline
    dataset when no logged data exists yet. Time-limit truncations are stored
    with ``done=False`` so batch agents bootstrap through them correctly.
    """
    rng = np.random.default_rng(seed)
    if buffer is None:
        buffer = ReplayBuffer(capacity=max(1, episodes * max_steps), seed=seed)
    if policy is None:
        def policy(_state):
            return int(rng.integers(env.n_actions))

    for _ in range(check_positive_int("episodes", episodes)):
        state = env.reset()
        for _ in range(check_positive_int("max_steps", max_steps)):
            action = int(policy(state))
            next_state, reward, done, info = env.step(action)
            truncated = bool((info or {}).get("truncated", False))
            buffer.add(state, action, reward, next_state, done and not truncated)
            state = next_state
            if done:
                break
    return buffer
