"""Policy evaluation -- the step between "it trained" and "ship it".

A learning curve is not evidence that a policy is good: it mixes exploration
noise into every episode and reports a single number with no uncertainty. This
module provides the two evaluations an industrial deployment actually needs:

* :func:`evaluate_policy` -- run the fixed (greedy) policy many times and
  report a mean **with a confidence interval**, a success rate and an episode
  length, so "better" can be distinguished from "luckier".
* :func:`off_policy_evaluation` -- estimate the value of a policy you have
  *not* deployed, using logs from the policy that is currently running. This
  is the only honest way to compare a candidate against the incumbent before
  it touches production, and :func:`effective_sample_size` says how much the
  estimate can be trusted.
"""
from __future__ import annotations

import numpy as np

from .validation import check_positive_int, check_range

__all__ = [
    "evaluate_policy",
    "off_policy_evaluation",
    "effective_sample_size",
    "epsilon_greedy_probabilities",
]


def _as_action_fn(policy):
    """Accept either a callable ``state -> action`` or an array of actions."""
    if callable(policy):
        return policy
    table = np.asarray(policy)
    if table.ndim != 1:
        raise ValueError("a tabular policy must be a 1-D array of action ids")
    return lambda state: int(table[int(state)])


def evaluate_policy(env, policy, episodes: int = 100, max_steps: int = 500,
                    gamma: float = 1.0) -> dict:
    """Run ``policy`` in ``env`` and summarise the return distribution.

    Returns a dict with ``mean_return``, ``std_return``, ``stderr``,
    ``ci95`` (a ``(low, high)`` normal-approximation interval on the mean),
    ``mean_discounted_return``, ``mean_length``, ``success_rate`` (fraction of
    episodes that *terminated* rather than hitting the time limit), plus
    ``min_return``/``max_return`` for the worst and best case -- the tail
    usually decides whether a controller is deployable.

    The environment is treated as read-only apart from ``reset``/``step``, and
    no learning happens, so this is safe to call on a live-configured env.
    """
    episodes = check_positive_int("episodes", episodes)
    max_steps = check_positive_int("max_steps", max_steps)
    gamma = check_range("gamma", gamma, 0.0, 1.0)
    act = _as_action_fn(policy)

    returns = np.empty(episodes, dtype=np.float64)
    discounted = np.empty(episodes, dtype=np.float64)
    lengths = np.empty(episodes, dtype=np.int64)
    successes = 0

    for i in range(episodes):
        state = env.reset()
        total = 0.0
        disc = 0.0
        steps = 0
        terminated = False
        for t in range(max_steps):
            state, reward, done, info = env.step(act(state))
            total += reward
            disc += (gamma ** t) * reward
            steps = t + 1
            if done:
                terminated = not bool((info or {}).get("truncated", False))
                break
        returns[i] = total
        discounted[i] = disc
        lengths[i] = steps
        successes += int(terminated)

    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if episodes > 1 else 0.0
    stderr = std / np.sqrt(episodes) if episodes > 1 else 0.0
    return {
        "episodes": episodes,
        "mean_return": mean,
        "std_return": std,
        "stderr": float(stderr),
        "ci95": (mean - 1.96 * stderr, mean + 1.96 * stderr),
        "mean_discounted_return": float(discounted.mean()),
        "mean_length": float(lengths.mean()),
        "success_rate": successes / episodes,
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }


def epsilon_greedy_probabilities(q_row, epsilon: float) -> np.ndarray:
    """Action distribution of an epsilon-greedy policy over one Q-row.

    Useful when logging a behaviour policy: off-policy evaluation needs the
    probability of the action that was actually taken, and reconstructing it
    after the fact is a classic source of subtly wrong estimates.
    """
    q_row = np.asarray(q_row, dtype=np.float64).reshape(-1)
    epsilon = check_range("epsilon", epsilon, 0.0, 1.0)
    n = q_row.size
    probs = np.full(n, epsilon / n)
    best = np.flatnonzero(q_row == q_row.max())
    probs[best] += (1.0 - epsilon) / best.size
    return probs


def effective_sample_size(weights) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` of IS weights.

    An ESS far below the number of logged episodes means the estimate rests on
    a handful of trajectories, however tight its nominal interval looks.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    denom = float(np.sum(w ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum(w) ** 2 / denom)


def off_policy_evaluation(dataset, target_policy, gamma: float = 1.0,
                          method: str = "wis") -> dict:
    """Estimate the value of ``target_policy`` from logged episodes.

    Parameters
    ----------
    dataset:
        Iterable of episodes, each a sequence of
        ``(state, action, reward, behaviour_prob)``.
    target_policy:
        Either a 1-D array of action ids (deterministic) or a
        ``(n_states, n_actions)`` matrix of action probabilities.
    method:
        ``"wis"`` for weighted importance sampling (biased, low variance --
        the sane default on real logs) or ``"is"`` for the unbiased ordinary
        estimator.

    Returns a dict with ``estimate``, ``ess``, ``n_episodes``,
    ``max_weight`` and ``overlap`` (the fraction of episodes with a non-zero
    importance weight). An ``overlap`` of 0 means the log contains no evidence
    about the target policy at all, and the estimate is reported as ``nan``
    rather than a confident zero.
    """
    if method not in ("wis", "is"):
        raise ValueError(f"method must be 'wis' or 'is', got {method!r}")
    gamma = check_range("gamma", gamma, 0.0, 1.0)

    target = np.asarray(target_policy)
    stochastic_target = target.ndim == 2
    if not stochastic_target and target.ndim != 1:
        raise ValueError("target_policy must be a 1-D action array or a 2-D probability matrix")

    weights, returns = [], []
    for episode in dataset:
        episode = list(episode)
        if not episode:
            raise ValueError("dataset contains an empty episode")
        w = 1.0
        g = 0.0
        for t, (state, action, reward, behaviour_prob) in enumerate(episode):
            state, action = int(state), int(action)
            behaviour_prob = float(behaviour_prob)
            if not 0.0 < behaviour_prob <= 1.0:
                raise ValueError(
                    f"behaviour probability must lie in (0, 1], got {behaviour_prob}"
                )
            g += (gamma ** t) * float(reward)
            if w > 0.0:
                p_target = (float(target[state, action]) if stochastic_target
                            else float(int(target[state]) == action))
                w *= p_target / behaviour_prob
        weights.append(w)
        returns.append(g)

    weights = np.asarray(weights, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    total_weight = float(weights.sum())

    if len(weights) == 0:
        raise ValueError("dataset is empty")
    if total_weight <= 0:
        estimate = float("nan")  # no logged trajectory the target could produce
    elif method == "wis":
        estimate = float(np.sum(weights * returns) / total_weight)
    else:
        estimate = float(np.mean(weights * returns))

    return {
        "estimate": estimate,
        "method": method,
        "ess": effective_sample_size(weights),
        "n_episodes": int(weights.size),
        "max_weight": float(weights.max()),
        "overlap": float(np.mean(weights > 0)),
    }
