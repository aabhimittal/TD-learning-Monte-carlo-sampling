"""Compare bootstrapping depth on the cliff: 1-step vs n-step vs SARSA(lambda).

Shows the bias/variance interpolation between one-step TD and Monte-Carlo.
Run: ``python examples/train_nstep.py``.
"""
import numpy as np

from rlkit.envs import CliffWalking
from rlkit.algorithms import Sarsa, NStepSarsa, SarsaLambda


def train(agent):
    return agent.train(CliffWalking(seed=0), episodes=500)


if __name__ == "__main__":
    agents = {
        "SARSA (n=1)": Sarsa(48, 4, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0),
        "4-step SARSA": NStepSarsa(48, 4, gamma=1.0, alpha=0.5, epsilon=0.1, n=4, seed=0),
        "SARSA(λ=0.9)": SarsaLambda(48, 4, gamma=1.0, alpha=0.5, epsilon=0.1, lam=0.9, seed=0),
    }
    print(f"{'Agent':<16}{'last-100 avg return':>22}")
    print("-" * 38)
    for name, agent in agents.items():
        hist = train(agent)
        print(f"{name:<16}{np.mean(hist[-100:]):>22.1f}")
