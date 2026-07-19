"""First-visit Monte-Carlo control on a 4x4 grid world.

Run: ``python examples/train_mc.py``.
"""
from rlkit.envs import GridWorld
from rlkit.algorithms import MonteCarloControl


if __name__ == "__main__":
    env = GridWorld(rows=4, cols=4, goals=((3, 3),), seed=0)
    agent = MonteCarloControl(env.n_states, env.n_actions, gamma=1.0, epsilon=0.2, seed=0)
    agent.train(env, episodes=3000, log_every=500)
    print("\nLearned greedy policy:\n" + env.render_policy(agent.greedy_policy()))
