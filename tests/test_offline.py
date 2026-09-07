"""Offline RL and policy evaluation: learning and judging without a live env.

These cover the workflow an applied project actually follows -- collect logs
from an incumbent policy, fit a candidate offline, then estimate the
candidate's value from those same logs before anyone deploys it.
"""
import numpy as np
import pytest

from rlkit.algorithms import BatchQLearning, OffPolicyMonteCarlo, QLearning
from rlkit.buffers import Batch, ReplayBuffer, collect_transitions
from rlkit.envs import GridWorld
from rlkit.evaluation import (
    effective_sample_size,
    epsilon_greedy_probabilities,
    evaluate_policy,
    off_policy_evaluation,
)


def make_env(**kwargs):
    opts = dict(rows=4, cols=4, start=(0, 0), goals=((3, 3),),
                step_reward=-1.0, max_steps=60)
    opts.update(kwargs)
    return GridWorld(**opts)


def log_episodes(env, episodes, max_steps, seed, policy=None, epsilon=1.0):
    """Roll out an epsilon-greedy behaviour policy, logging its probabilities.

    ``policy=None`` gives a uniform random logger; passing an incumbent policy
    with a small ``epsilon`` mimics the realistic case -- logs from a decent
    controller that explores a little.
    """
    rng = np.random.default_rng(seed)
    n = env.n_actions
    dataset = []
    for _ in range(episodes):
        state = env.reset()
        episode = []
        for _ in range(max_steps):
            probs = np.full(n, epsilon / n)
            if policy is None:
                probs[:] = 1.0 / n
            else:
                probs[int(policy[state])] += 1.0 - epsilon
            action = int(rng.choice(n, p=probs))
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward, float(probs[action])))
            state = next_state
            if done:
                break
        dataset.append(episode)
    return dataset


# --------------------------------------------------------------------------
# fitted-Q iteration
# --------------------------------------------------------------------------
def test_batch_q_learns_from_logged_random_behaviour():
    """The whole point of offline RL: a good policy from mediocre logs."""
    env = make_env()
    buffer = collect_transitions(env, episodes=400, max_steps=60, seed=0)
    agent = BatchQLearning(env.n_states, env.n_actions, gamma=0.99)
    agent.fit(buffer, iterations=200)
    assert evaluate_policy(env, agent.greedy_policy(), episodes=5,
                           max_steps=30)["mean_return"] >= -7


def test_batch_q_converges_and_stops_early():
    env = make_env()
    buffer = collect_transitions(env, episodes=100, max_steps=40, seed=1)
    agent = BatchQLearning(env.n_states, env.n_actions, gamma=0.9)
    deltas = agent.fit(buffer, iterations=500, tol=1e-8)
    assert len(deltas) < 500          # stopped once the sweep stopped moving
    assert deltas[-1] < 1e-8
    assert deltas[-1] < deltas[0]


def test_batch_q_does_not_bootstrap_through_terminals():
    """A ``done`` transition must have value ``r``, not ``r + gamma*V(s')``."""
    data = Batch(
        states=np.array([0, 1]),
        actions=np.array([0, 0]),
        rewards=np.array([1.0, 2.0]),
        next_states=np.array([1, 1]),
        dones=np.array([True, False]),
    )
    agent = BatchQLearning(2, 1, gamma=0.9)
    agent.fit(data, iterations=500, tol=1e-10)
    assert agent.Q[1, 0] == pytest.approx(20.0, rel=1e-3)  # 2 / (1 - 0.9)
    assert agent.Q[0, 0] == pytest.approx(1.0, rel=1e-6)   # terminal: no bootstrap


def test_batch_q_reports_coverage_and_can_stay_in_support():
    """Tabular offline RL knows nothing about pairs the log never contains."""
    env = make_env()
    buffer = ReplayBuffer(capacity=100, seed=0)
    # Log only action 1 (right), from a handful of states.
    collect_transitions(env, policy=lambda s: 1, episodes=10, max_steps=10,
                        buffer=buffer)
    agent = BatchQLearning(env.n_states, env.n_actions, gamma=0.9,
                           unseen_value=5.0)  # deliberately optimistic
    agent.fit(buffer, iterations=50)

    assert 0.0 < agent.coverage() < 1.0
    optimistic = agent.greedy_policy()
    supported = agent.greedy_policy(min_count=1)
    seen_states = np.flatnonzero(agent.counts.sum(axis=1) > 0)
    # Unconstrained, the optimistic prior wins; constrained, only logged
    # actions are eligible, so every visited state keeps action 1.
    assert set(supported[seen_states].tolist()) == {1}
    assert not np.array_equal(optimistic, supported)


def test_batch_q_accepts_buffers_batches_and_raw_arrays():
    env = make_env()
    buffer = collect_transitions(env, episodes=20, max_steps=20, seed=2)
    batch = buffer.all()
    results = []
    for data in (buffer, batch, tuple(batch)):
        agent = BatchQLearning(env.n_states, env.n_actions, gamma=0.9)
        agent.fit(data, iterations=30)
        results.append(agent.Q.copy())
    assert np.allclose(results[0], results[1]) and np.allclose(results[1], results[2])


@pytest.mark.parametrize("bad", [
    Batch(np.array([]), np.array([]), np.array([]), np.array([]), np.array([])),
    Batch(np.array([0, 1]), np.array([0]), np.array([1.0]), np.array([1]),
          np.array([False])),
    Batch(np.array([0]), np.array([0]), np.array([np.nan]), np.array([1]),
          np.array([False])),
    Batch(np.array([99]), np.array([0]), np.array([1.0]), np.array([1]),
          np.array([False])),
    Batch(np.array([0]), np.array([99]), np.array([1.0]), np.array([1]),
          np.array([False])),
])
def test_batch_q_rejects_malformed_datasets(bad):
    with pytest.raises(ValueError):
        BatchQLearning(4, 2, gamma=0.9).fit(bad)


# --------------------------------------------------------------------------
# off-policy Monte Carlo
# --------------------------------------------------------------------------
INCUMBENT = np.array([1, 2, 2, 1, 1, 2, 1, 1, 0])  # a workable 3x3 heuristic


def test_off_policy_mc_learns_the_optimum_from_incumbent_logs():
    """The realistic case: logs from a decent controller that explores a little."""
    env = make_env(rows=3, cols=3, goals=((2, 2),), max_steps=40)
    dataset = log_episodes(env, episodes=400, max_steps=40, seed=0,
                           policy=INCUMBENT, epsilon=0.4)
    agent = OffPolicyMonteCarlo(env.n_states, env.n_actions, gamma=0.95,
                                init_value=-100.0)
    agent.train_from_dataset(dataset, passes=3)
    assert evaluate_policy(env, agent.greedy_policy(), episodes=5,
                           max_steps=20)["mean_return"] == pytest.approx(-3.0)
    assert agent.coverage() > 0.5


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_off_policy_mc_values_near_the_goal_are_exact_from_random_logs(seed):
    """Importance sampling pays off backwards from the terminal state.

    Weighted IS gives near-goal pairs many effective samples and estimates
    them exactly, while pairs far from the goal survive only in rare matching
    suffixes and stay noisy -- which is why the extracted *policy* can still be
    unreliable on uniform logs even when the *values* it is read from look
    healthy. Coverage alone does not tell you the policy is safe to ship.
    """
    env = make_env(rows=3, cols=3, goals=((2, 2),), max_steps=40)
    dataset = log_episodes(env, episodes=1500, max_steps=40, seed=seed)
    agent = OffPolicyMonteCarlo(env.n_states, env.n_actions, gamma=0.95,
                                init_value=-100.0)
    agent.train_from_dataset(dataset, passes=3)

    assert agent.coverage() > 0.8
    assert agent.Q[5, 2] == pytest.approx(0.0)   # (1,2) down  -> goal
    assert agent.Q[7, 1] == pytest.approx(0.0)   # (2,1) right -> goal
    assert agent.Q[4, 1] == pytest.approx(-1.0)  # (1,1) right -> one step out


def test_off_policy_mc_stalls_without_a_pessimistic_init_on_costs():
    """Why ``init_value`` exists, pinned as a test.

    Every return here is negative, so zero-initialised unvisited pairs outrank
    every pair the log actually contains. The backward pass then stops at the
    first step -- the agent learns almost nothing from a perfectly good log.
    """
    env = make_env(rows=3, cols=3, goals=((2, 2),), max_steps=40)
    dataset = log_episodes(env, episodes=400, max_steps=40, seed=0,
                           policy=INCUMBENT, epsilon=0.4)

    optimistic = OffPolicyMonteCarlo(env.n_states, env.n_actions, gamma=0.95)
    optimistic.train_from_dataset(dataset, passes=3)
    pessimistic = OffPolicyMonteCarlo(env.n_states, env.n_actions, gamma=0.95,
                                      init_value=-100.0)
    pessimistic.train_from_dataset(dataset, passes=3)

    assert optimistic.coverage() < 0.25 < pessimistic.coverage()


def test_off_policy_mc_rejects_impossible_behaviour_probabilities():
    agent = OffPolicyMonteCarlo(4, 2, gamma=0.9)
    for bad_prob in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            agent.update_from_episode([(0, 0, 1.0, bad_prob)])


def test_off_policy_mc_rejects_out_of_range_ids():
    agent = OffPolicyMonteCarlo(4, 2, gamma=0.9)
    with pytest.raises(ValueError):
        agent.update_from_episode([(9, 0, 1.0, 0.5)])
    with pytest.raises(ValueError):
        agent.update_from_episode([(0, 7, 1.0, 0.5)])


def test_off_policy_mc_clips_exploding_importance_weights():
    """Rare behaviour actions make weights compound; the clip keeps them finite."""
    agent = OffPolicyMonteCarlo(2, 1, gamma=1.0, max_weight=10.0)
    episode = [(0, 0, 1.0, 1e-6)] * 20  # weight would blow up to 1e120
    agent.update_from_episode(episode)
    assert np.all(np.isfinite(agent.Q)) and np.all(agent.C <= 10.0 * len(episode))


# --------------------------------------------------------------------------
# on-policy evaluation
# --------------------------------------------------------------------------
def test_evaluate_policy_reports_exact_stats_on_a_deterministic_env():
    env = make_env()
    optimal = np.array([1, 1, 2, 2,
                        1, 1, 1, 2,
                        1, 1, 1, 2,
                        1, 1, 1, 0])  # right/down to the goal at (3,3)
    stats = evaluate_policy(env, optimal, episodes=10, max_steps=30, gamma=0.5)
    assert stats["mean_return"] == pytest.approx(-5.0)
    assert stats["std_return"] == 0.0 and stats["stderr"] == 0.0
    assert stats["ci95"] == (pytest.approx(-5.0), pytest.approx(-5.0))
    assert stats["mean_discounted_return"] == pytest.approx(-1.9375)
    assert stats["mean_length"] == pytest.approx(6.0)
    assert stats["success_rate"] == 1.0
    assert stats["min_return"] == stats["max_return"] == pytest.approx(-5.0)


def test_evaluate_policy_reports_zero_success_when_only_truncating():
    """Time-limited episodes are not successes, however good the return looks."""
    env = make_env(obstacles=((2, 3), (3, 2)), max_steps=12)
    stats = evaluate_policy(env, np.zeros(env.n_states, dtype=int), episodes=5,
                            max_steps=50)
    assert stats["success_rate"] == 0.0
    assert stats["mean_length"] == pytest.approx(12.0)


def test_evaluate_policy_accepts_a_callable():
    env = make_env()
    stats = evaluate_policy(env, lambda s: 1, episodes=3, max_steps=10)
    assert stats["episodes"] == 3 and np.isfinite(stats["mean_return"])


def test_evaluate_policy_confidence_interval_brackets_the_mean():
    env = make_env()
    agent = QLearning(env.n_states, env.n_actions, gamma=0.99, alpha=0.5,
                      epsilon=0.1, seed=0)
    agent.train(env, episodes=300)
    stats = evaluate_policy(env, agent.greedy_policy(), episodes=30, max_steps=40)
    lo, hi = stats["ci95"]
    assert lo <= stats["mean_return"] <= hi


@pytest.mark.parametrize("kwargs", [{"episodes": 0}, {"max_steps": 0}, {"gamma": 1.5}])
def test_evaluate_policy_validates_arguments(kwargs):
    env = make_env()
    with pytest.raises(ValueError):
        evaluate_policy(env, np.zeros(env.n_states, dtype=int), **kwargs)


def test_evaluate_policy_rejects_a_two_dimensional_policy_table():
    env = make_env()
    with pytest.raises(ValueError):
        evaluate_policy(env, np.zeros((env.n_states, env.n_actions)))


# --------------------------------------------------------------------------
# off-policy evaluation
# --------------------------------------------------------------------------
def test_wis_recovers_the_on_policy_value_of_the_target():
    env = GridWorld(rows=2, cols=2, start=(0, 0), goals=((1, 1),),
                    step_reward=-1.0, max_steps=10)
    target = np.array([1, 2, 1, 0])  # right then down: a 2-step, -1 return path
    on_policy = evaluate_policy(env, target, episodes=5, max_steps=10)["mean_return"]

    dataset = log_episodes(env, episodes=800, max_steps=10, seed=3)
    result = off_policy_evaluation(dataset, target, gamma=1.0, method="wis")
    assert result["estimate"] == pytest.approx(on_policy, abs=1e-9)
    assert 0.0 < result["overlap"] < 1.0
    assert 0.0 < result["ess"] <= result["n_episodes"]

    ordinary = off_policy_evaluation(dataset, target, gamma=1.0, method="is")
    assert ordinary["estimate"] == pytest.approx(on_policy, abs=0.3)


def test_ope_reports_nan_when_the_log_covers_nothing():
    """No overlap means no evidence -- better a nan than a confident zero."""
    dataset = [[(0, 0, 1.0, 1.0)], [(0, 0, 2.0, 1.0)]]
    result = off_policy_evaluation(dataset, np.array([1, 0]), gamma=1.0)
    assert np.isnan(result["estimate"])
    assert result["overlap"] == 0.0 and result["ess"] == 0.0


def test_ope_supports_a_stochastic_target_policy():
    dataset = [[(0, 0, 1.0, 0.5)], [(0, 1, 3.0, 0.5)]]
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    result = off_policy_evaluation(dataset, probs, gamma=1.0, method="wis")
    assert result["estimate"] == pytest.approx(2.0)  # equal weights: plain mean


def test_ope_validates_its_inputs():
    with pytest.raises(ValueError):
        off_policy_evaluation([[(0, 0, 1.0, 0.5)]], np.array([0]), method="bogus")
    with pytest.raises(ValueError):
        off_policy_evaluation([[]], np.array([0]))
    with pytest.raises(ValueError):
        off_policy_evaluation([[(0, 0, 1.0, 0.0)]], np.array([0]))
    with pytest.raises(ValueError):
        off_policy_evaluation([[(0, 0, 1.0, 0.5)]], np.zeros((2, 2, 2)))


def test_effective_sample_size_bounds():
    assert effective_sample_size([1.0, 1.0, 1.0, 1.0]) == pytest.approx(4.0)
    assert effective_sample_size([1.0, 0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_sample_size([0.0, 0.0]) == 0.0


def test_epsilon_greedy_probabilities_sum_to_one_and_split_ties():
    probs = epsilon_greedy_probabilities([1.0, 1.0, 0.0, 0.0], epsilon=0.2)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == probs[1] == pytest.approx(0.05 + 0.4)
    greedy = epsilon_greedy_probabilities([0.0, 5.0], epsilon=0.0)
    assert greedy.tolist() == [0.0, 1.0]
    uniform = epsilon_greedy_probabilities([0.0, 5.0], epsilon=1.0)
    assert uniform.tolist() == [0.5, 0.5]
