"""Train every tabular and policy-gradient agent on one grid world.

Run: ``python examples/compare_all.py``.
"""
import numpy as np

from rlkit.envs import GridWorld
from rlkit.algorithms import (
    A2C,
    DoubleQLearning,
    ExpectedSarsa,
    MonteCarloControl,
    QLearning,
    Reinforce,
    Sarsa,
)


def make_env():
    return GridWorld(rows=4, cols=4, start=(0, 0), goals=((3, 3),),
                     step_reward=-1.0, max_steps=100, seed=0)


def greedy_return(policy, max_steps=50):
    env = make_env()
    state = env.reset()
    total = 0.0
    for _ in range(max_steps):
        state, r, done, _ = env.step(int(policy[state]))
        total += r
        if done:
            break
    return total


def main():
    rows = []

    mc = MonteCarloControl(16, 4, gamma=1.0, epsilon=0.2, seed=0)
    mc.train(make_env(), episodes=3000)
    rows.append(("Monte-Carlo", greedy_return(mc.greedy_policy())))

    sarsa = Sarsa(16, 4, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    sarsa.train(make_env(), episodes=800)
    rows.append(("SARSA (TD)", greedy_return(sarsa.greedy_policy())))

    ql = QLearning(16, 4, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    ql.train(make_env(), episodes=800)
    rows.append(("Q-learning (TD)", greedy_return(ql.greedy_policy())))

    expected = ExpectedSarsa(16, 4, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    expected.train(make_env(), episodes=800)
    rows.append(("Expected SARSA", greedy_return(expected.greedy_policy())))

    double = DoubleQLearning(16, 4, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    double.train(make_env(), episodes=1500)  # two tables, so twice the episodes
    rows.append(("Double Q-learning", greedy_return(double.greedy_policy())))

    env = make_env()
    rf = Reinforce(16, 4, hidden=(64,), gamma=0.99, lr=0.02, seed=0)
    rf.train(make_env(), episodes=1500)
    rows.append(("REINFORCE", greedy_return(rf.greedy_policy(16, env.one_hot))))

    a2c = A2C(16, 4, hidden=(64,), gamma=0.99, actor_lr=0.02, critic_lr=0.05,
              entropy_coef=0.01, seed=0)
    a2c.train(make_env(), episodes=1500)
    rows.append(("A2C", greedy_return(a2c.greedy_policy(16, env.one_hot))))

    print(f"\n{'Algorithm':<18}{'Greedy return':>14}   (optimal = -5)")
    print("-" * 46)
    for name, ret in rows:
        print(f"{name:<18}{ret:>14.1f}")


if __name__ == "__main__":
    main()
