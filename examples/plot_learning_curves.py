"""Train the tabular control agents and save a learning-curve figure.

Produces ``learning_curves.png`` (requires matplotlib:
``pip install -e '.[plot]'``). Run: ``python examples/plot_learning_curves.py``.
"""
from rlkit.envs import GridWorld
from rlkit.algorithms import MonteCarloControl, Sarsa, QLearning, NStepSarsa, SarsaLambda
from rlkit.utils import plot_learning_curves


def make_env():
    return GridWorld(rows=6, cols=6, start=(0, 0), goals=((5, 5),),
                     obstacles=((2, 2), (2, 3), (3, 2)), step_reward=-1.0,
                     max_steps=200, seed=0)


if __name__ == "__main__":
    episodes = 600
    curves = {
        "Monte-Carlo": MonteCarloControl(36, 4, gamma=1.0, epsilon=0.15, seed=0)
        .train(make_env(), episodes=episodes),
        "SARSA": Sarsa(36, 4, gamma=1.0, alpha=0.4, epsilon=0.1, seed=0)
        .train(make_env(), episodes=episodes),
        "Q-learning": QLearning(36, 4, gamma=1.0, alpha=0.4, epsilon=0.1, seed=0)
        .train(make_env(), episodes=episodes),
        "4-step SARSA": NStepSarsa(36, 4, gamma=1.0, alpha=0.4, epsilon=0.1, n=4, seed=0)
        .train(make_env(), episodes=episodes),
        "SARSA(λ=0.9)": SarsaLambda(36, 4, gamma=1.0, alpha=0.4, epsilon=0.1, lam=0.9, seed=0)
        .train(make_env(), episodes=episodes),
    }
    path = plot_learning_curves(curves, window=25,
                                title="Tabular control on a 6x6 grid world",
                                path="learning_curves.png")
    print(f"saved {path}")
