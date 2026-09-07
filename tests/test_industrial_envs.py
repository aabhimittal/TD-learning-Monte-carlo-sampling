"""Inventory control and predictive maintenance.

Two environments modelled on real operational decisions. Both have stochastic,
mixed-sign rewards, end by time limit rather than by reaching a goal, and have
a known-good analytical baseline -- an ``(s, S)`` order-up-to rule and a wear
threshold rule -- so a learned policy can be held to a real standard instead
of merely "improving".
"""
import numpy as np
import pytest

from rlkit.algorithms import DoubleQLearning, QLearning
from rlkit.envs import InventoryControl, MachineMaintenance
from rlkit.envs.maintenance import MAINTAIN, REPLACE, RUN
from rlkit.evaluation import evaluate_policy
from rlkit.schedules import LinearSchedule


# --------------------------------------------------------------------------
# inventory control
# --------------------------------------------------------------------------
def test_inventory_state_stays_within_capacity_under_any_action_sequence():
    """Ordering more than fits is clipped, not an error and not an overflow."""
    env = InventoryControl(capacity=6, demand_mean=0.5, horizon=50, seed=0)
    rng = np.random.default_rng(0)
    env.reset()
    for _ in range(200):
        state, reward, done, info = env.step(int(rng.integers(env.n_actions)))
        assert 0 <= state <= env.capacity
        assert np.isfinite(reward)
        assert info["ordered"] <= env.capacity
        if done:
            env.reset()


def test_inventory_episode_ends_by_truncation_only():
    env = InventoryControl(capacity=4, horizon=5, seed=0)
    env.reset()
    flags = [env.step(0)[3]["truncated"] for _ in range(5)]
    assert flags == [False, False, False, False, True]
    assert not env.is_terminal(0)


def test_inventory_zero_demand_never_reports_shortage():
    """Degenerate but real: a discontinued line with no demand left."""
    env = InventoryControl(capacity=5, demand_mean=0.0, horizon=10, seed=0)
    env.reset()
    for _ in range(10):
        _, _, _, info = env.step(1)
        assert info["demand"] == 0 and info["unmet"] == 0 and info["sold"] == 0


def test_inventory_reward_decomposition_matches_the_cost_model():
    """Pin the reward arithmetic: shaping errors hide in the cost terms."""
    env = InventoryControl(capacity=10, demand_mean=0.0, sale_price=3.0,
                           unit_cost=1.0, fixed_order_cost=2.0, holding_cost=0.5,
                           shortage_cost=4.0, horizon=5, seed=0)
    env.reset()
    _, reward, _, info = env.step(4)  # order 4, no demand -> hold all 4
    assert info["ordered"] == 4
    assert reward == pytest.approx(-1.0 * 4 - 2.0 - 0.5 * 4)
    _, reward_idle, _, _ = env.step(0)  # no order: no fixed cost, still holding 4
    assert reward_idle == pytest.approx(-0.5 * 4)


def test_inventory_zero_capacity_is_a_single_state_mdp():
    env = InventoryControl(capacity=0, demand_mean=2.0, horizon=3, seed=0)
    assert env.n_states == 1 and env.n_actions == 1
    state, reward, _, info = env.step(0) if env.reset() == 0 else (None,) * 4
    assert state == 0 and info["sold"] == 0 and reward <= 0.0  # all demand is lost


def test_inventory_is_reproducible_and_seed_sensitive():
    def run(seed):
        env = InventoryControl(horizon=20, seed=seed)
        env.reset()
        return [env.step(3)[1] for _ in range(20)]

    assert run(7) == run(7)
    assert run(7) != run(8)


@pytest.mark.parametrize("kwargs", [
    {"capacity": -1},
    {"max_order": 20},          # exceeds capacity
    {"demand_mean": -1.0},
    {"horizon": 0},
    {"demand_mean": float("nan")},
])
def test_inventory_rejects_impossible_configurations(kwargs):
    with pytest.raises((ValueError, TypeError)):
        InventoryControl(capacity=kwargs.pop("capacity", 10), **kwargs)


def test_inventory_rejects_invalid_order_quantities():
    env = InventoryControl(capacity=5, max_order=3)
    env.reset()
    for bad in (-1, 4, 99):
        with pytest.raises(ValueError):
            env.step(bad)


def test_q_learning_beats_naive_inventory_baselines():
    """The learned policy must beat both trivial strategies by a clear margin."""
    env = InventoryControl(capacity=10, demand_mean=3.0, horizon=30, seed=0)
    agent = QLearning(env.n_states, env.n_actions, gamma=0.95, alpha=0.1,
                      epsilon=LinearSchedule(0.5, 0.05, decay_steps=2000), seed=0)
    agent.train(env, episodes=4000, max_steps=30)

    learned = evaluate_policy(env, agent.greedy_policy(), episodes=200, max_steps=30)
    never = evaluate_policy(env, np.zeros(env.n_states, dtype=int), episodes=200,
                            max_steps=30)
    always = evaluate_policy(env, np.full(env.n_states, env.max_order), episodes=200,
                             max_steps=30)
    order_up_to = evaluate_policy(env, env.order_up_to_policy(3, 6), episodes=200,
                                  max_steps=30)

    assert learned["mean_return"] > never["mean_return"] + 100
    assert learned["mean_return"] > always["mean_return"]
    # Within 20% of the tuned (s, S) rule, the analytical baseline for this task.
    assert learned["mean_return"] >= 0.8 * order_up_to["mean_return"]


def test_order_up_to_policy_shape_and_clipping():
    env = InventoryControl(capacity=10, max_order=4)
    policy = env.order_up_to_policy(reorder_point=3, order_up_to=10)
    assert policy.shape == (env.n_states,)
    assert policy.max() <= env.max_order      # clipped by the order limit
    assert policy[0] == 4 and policy[4] == 0  # only reorders at or below s=3
    with pytest.raises(ValueError):
        env.order_up_to_policy(3, 99)


# --------------------------------------------------------------------------
# predictive maintenance
# --------------------------------------------------------------------------
def test_maintenance_failed_machine_only_recovers_by_replacement():
    """The trap: running or servicing a broken machine keeps you broken."""
    env = MachineMaintenance(n_wear=3, base_failure_prob=1.0, horizon=20, seed=0)
    env.reset()
    state, _, _, info = env.step(RUN)  # certain failure
    assert state == env.failed_state and info["failed"]

    assert env.step(RUN)[0] == env.failed_state
    assert env.step(MAINTAIN)[0] == env.failed_state
    assert env.step(REPLACE)[0] == 0   # a new machine, wear reset


def test_maintenance_wear_is_bounded_and_maintenance_reduces_it():
    env = MachineMaintenance(n_wear=4, base_failure_prob=0.0, failure_slope=0.0,
                             degrade_prob=1.0, horizon=100, seed=0)
    env.reset()
    for _ in range(20):
        state, _, _, _ = env.step(RUN)
    assert state == env.n_wear - 1  # degradation saturates, never leaves the grid
    assert env.step(MAINTAIN)[0] == env.n_wear - 2


def test_maintenance_failure_probability_rises_with_wear_and_is_clipped():
    env = MachineMaintenance(n_wear=20, base_failure_prob=0.1, failure_slope=0.2)
    probs = [env.failure_probability(w) for w in range(env.n_wear)]
    assert probs[0] == pytest.approx(0.1)
    assert all(b >= a for a, b in zip(probs, probs[1:]))  # monotone
    assert max(probs) <= 1.0                              # clipped, never > 1
    assert env.failure_probability(env.failed_state) == 1.0


def test_maintenance_costs_are_charged_exactly_once():
    env = MachineMaintenance(revenue=10.0, maintain_cost=4.0, replace_cost=20.0,
                             base_failure_prob=0.0, failure_slope=0.0,
                             degrade_prob=0.0, horizon=10, seed=0)
    env.reset()
    assert env.step(RUN)[1] == pytest.approx(10.0)
    assert env.step(MAINTAIN)[1] == pytest.approx(-4.0)
    assert env.step(REPLACE)[1] == pytest.approx(-20.0)


def test_maintenance_rejects_invalid_actions_and_configs():
    env = MachineMaintenance()
    env.reset()
    for bad in (-1, 3, 100):
        with pytest.raises(ValueError):
            env.step(bad)
    with pytest.raises(ValueError):
        MachineMaintenance(n_wear=0)
    with pytest.raises(ValueError):
        MachineMaintenance(base_failure_prob=1.5)
    with pytest.raises(ValueError):
        MachineMaintenance(failure_slope=-0.1)


def test_learned_maintenance_policy_beats_running_until_failure():
    """Rare, expensive failures: the run-to-failure baseline loses badly."""
    env = MachineMaintenance(horizon=60, seed=0)
    agent = QLearning(env.n_states, env.n_actions, gamma=0.95, alpha=0.1,
                      epsilon=0.2, seed=0)
    agent.train(env, episodes=4000, max_steps=60)

    learned = evaluate_policy(env, agent.greedy_policy(), episodes=200, max_steps=60)
    run_forever = evaluate_policy(env, np.full(env.n_states, RUN), episodes=200,
                                  max_steps=60)
    best_threshold = max(
        evaluate_policy(env, env.threshold_policy(t), episodes=200, max_steps=60)
        ["mean_return"] for t in range(env.n_wear)
    )

    assert learned["mean_return"] > 0 > run_forever["mean_return"]
    assert learned["mean_return"] >= 0.9 * best_threshold
    assert agent.greedy_policy()[env.failed_state] == REPLACE  # learned the recovery


@pytest.mark.parametrize("seed", [1, 2])
def test_double_q_is_less_optimistic_than_q_learning_about_rare_failures(seed):
    """Maximisation bias, measured where it actually hurts.

    Failures are rare and cost 20x a shift's revenue, so ``max_a Q`` is taken
    over estimates with very heavy left tails -- and the max of noisy estimates
    over-values whichever action happened to dodge the failures. We compare
    each agent's *own* prediction for the start state against the discounted
    return it really achieves: Q-learning's optimism is consistently larger.
    """
    bias = {}
    for name, cls in (("q", QLearning), ("double", DoubleQLearning)):
        env = MachineMaintenance(horizon=60, base_failure_prob=0.02,
                                 failure_slope=0.05, failure_cost=200.0, seed=seed)
        agent = cls(env.n_states, env.n_actions, gamma=0.95, alpha=0.1,
                    epsilon=0.2, seed=seed)
        agent.train(env, episodes=4000, max_steps=60)
        stats = evaluate_policy(env, agent.greedy_policy(), episodes=300,
                                max_steps=60, gamma=0.95)
        bias[name] = float(agent.Q[0].max()) - stats["mean_discounted_return"]
        assert stats["mean_return"] > 0.0  # both still learn a profitable policy
    assert bias["double"] < bias["q"]


def test_threshold_policy_structure():
    env = MachineMaintenance(n_wear=5)
    policy = env.threshold_policy(maintain_at=2)
    assert policy[:2].tolist() == [RUN, RUN]
    assert policy[2:5].tolist() == [MAINTAIN] * 3
    assert policy[env.failed_state] == REPLACE
