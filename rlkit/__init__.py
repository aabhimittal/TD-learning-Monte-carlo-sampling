"""rlkit -- a dependency-light reinforcement-learning teaching library.

Implements and contrasts the four canonical families of tabular / policy
algorithms on shared grid-world environments, using nothing heavier than
NumPy:

* Monte-Carlo control            (:class:`rlkit.algorithms.MonteCarloControl`)
* Temporal-difference control    (:class:`rlkit.algorithms.Sarsa`,
                                  :class:`rlkit.algorithms.QLearning`)
* REINFORCE policy gradient      (:class:`rlkit.algorithms.Reinforce`)
* Advantage Actor-Critic (A2C)   (:class:`rlkit.algorithms.A2C`)
"""
from .envs import GridWorld, CliffWalking, CartPole
from .algorithms import (
    MonteCarloControl,
    Sarsa,
    QLearning,
    NStepSarsa,
    SarsaLambda,
    TDPrediction,
    Reinforce,
    A2C,
)
from .utils import moving_average, ascii_sparkline, plot_learning_curves

__version__ = "0.2.0"

__all__ = [
    "GridWorld",
    "CliffWalking",
    "CartPole",
    "MonteCarloControl",
    "Sarsa",
    "QLearning",
    "NStepSarsa",
    "SarsaLambda",
    "TDPrediction",
    "Reinforce",
    "A2C",
    "moving_average",
    "ascii_sparkline",
    "plot_learning_curves",
]
