"""REINFORCE vs A2C on a 5x5 grid world with an obstacle.

Both are policy-gradient methods over a NumPy MLP. A2C adds a learned critic
baseline, which typically yields a smoother, lower-variance learning curve.
Run: ``python examples/train_policy_gradient.py``.
"""
import numpy as np

from rlkit.envs import GridWorld
from rlkit.algorithms import Reinforce, A2C


def make_env():
    return GridWorld(rows=5, cols=5, start=(0, 0), goals=((4, 4),),
                     obstacles=((2, 1), (2, 2), (2, 3)), step_reward=-1.0,
                     max_steps=100, seed=0)


def summarise(name, history):
    first = np.mean(history[:100])
    last = np.mean(history[-100:])
    print(f"{name:12s}  first-100 avg={first:7.2f}   last-100 avg={last:7.2f}")


if __name__ == "__main__":
    env = make_env()
    r = Reinforce(env.n_states, env.n_actions, hidden=(64,), gamma=0.99, lr=0.02, seed=0)
    hist_r = r.train(make_env(), episodes=1500, log_every=300)

    a = A2C(env.n_states, env.n_actions, hidden=(64,), gamma=0.99,
            actor_lr=0.02, critic_lr=0.05, entropy_coef=0.01, seed=0)
    hist_a = a.train(make_env(), episodes=1500, log_every=300)

    print()
    summarise("REINFORCE", hist_r)
    summarise("A2C", hist_a)
    print("\nA2C greedy policy:\n" + env.render_policy(a.greedy_policy(env.n_states, env.one_hot)))
