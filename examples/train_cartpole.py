"""Solve the continuous CartPole task with A2C and REINFORCE.

Both policy-gradient agents run over the raw 4-D observation via the NumPy
MLP -- no lookup table. Run: ``python examples/train_cartpole.py``.
"""
from rlkit.envs import CartPole
from rlkit.algorithms import A2C, Reinforce
from rlkit.utils import ascii_sparkline


def run(agent, name):
    hist = agent.train(CartPole(seed=0), episodes=600, max_steps=500, log_every=150)
    ev = agent.evaluate(CartPole(seed=1), episodes=20, max_steps=500)
    print(f"{name}: greedy eval over 20 eps = {ev:.1f} / 500")
    print("  train curve: " + ascii_sparkline(hist) + "\n")


if __name__ == "__main__":
    run(A2C(4, 2, hidden=(64,), gamma=0.99, actor_lr=0.01, critic_lr=0.02,
            entropy_coef=0.01, seed=0), "A2C")
    run(Reinforce(4, 2, hidden=(64,), gamma=0.99, lr=0.01, seed=0), "REINFORCE")
