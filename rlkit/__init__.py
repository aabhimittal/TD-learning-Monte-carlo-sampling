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
from .envs import GridWorld, CliffWalking
from .algorithms import (
    MonteCarloControl,
    Sarsa,
    QLearning,
    TDPrediction,
    Reinforce,
    A2C,
)

__version__ = "0.1.0"

__all__ = [
    "GridWorld",
    "CliffWalking",
    "MonteCarloControl",
    "Sarsa",
    "QLearning",
    "TDPrediction",
    "Reinforce",
    "A2C",
]
