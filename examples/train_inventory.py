"""Learn an inventory-replenishment policy and check it against the (s, S) rule.

    python examples/train_inventory.py

Trains Q-learning with an annealed exploration schedule on the stochastic
inventory task, then evaluates the learned policy alongside the two trivial
baselines and the tuned analytical rule -- with confidence intervals, because
a single average return proves nothing on a stochastic task.
"""
import numpy as np

from rlkit.algorithms import QLearning
from rlkit.envs import InventoryControl
from rlkit.evaluation import evaluate_policy
from rlkit.schedules import LinearSchedule
from rlkit.utils import ascii_sparkline, moving_average

EPISODES = 4000
HORIZON = 30


def main():
    env = InventoryControl(capacity=10, demand_mean=3.0, horizon=HORIZON, seed=0)
    agent = QLearning(
        env.n_states,
        env.n_actions,
        gamma=0.95,
        alpha=0.1,
        epsilon=LinearSchedule(0.5, 0.05, decay_steps=EPISODES // 2),
        seed=0,
    )
    history = agent.train(env, episodes=EPISODES, max_steps=HORIZON, log_every=1000)
    print("\nlearning curve:", ascii_sparkline(moving_average(history, 100)))

    policies = {
        "learned (Q-learning)": agent.greedy_policy(),
        "never order": np.zeros(env.n_states, dtype=int),
        "always order max": np.full(env.n_states, env.max_order),
        "(s=3, S=6) rule": env.order_up_to_policy(3, 6),
    }

    print(f"\n{'policy':<22} {'mean return':>12}  95% CI")
    for name, policy in policies.items():
        stats = evaluate_policy(env, policy, episodes=500, max_steps=HORIZON)
        lo, hi = stats["ci95"]
        print(f"{name:<22} {stats['mean_return']:12.2f}  [{lo:7.2f}, {hi:7.2f}]")

    print("\norder quantity by stock level:")
    print("  stock:", " ".join(f"{s:2d}" for s in range(env.n_states)))
    print("  order:", " ".join(f"{a:2d}" for a in agent.greedy_policy()))


if __name__ == "__main__":
    main()
