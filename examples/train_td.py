"""Compare SARSA (on-policy) vs Q-learning (off-policy) on the cliff.

Q-learning learns the optimal (risky, cliff-edge) path; SARSA learns a safer
path because it accounts for the epsilon-greedy exploration that occasionally
pushes it off the edge. Run: ``python examples/train_td.py``.
"""
from rlkit.envs import CliffWalking
from rlkit.algorithms import Sarsa, QLearning


def run(agent_cls, name):
    env = CliffWalking(seed=0)
    agent = agent_cls(env.n_states, env.n_actions, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    agent.train(env, episodes=500, log_every=100)
    print(f"\n{name} greedy policy:\n{env.render_policy(agent.greedy_policy())}\n")


if __name__ == "__main__":
    run(Sarsa, "SARSA")
    run(QLearning, "Q-learning")
