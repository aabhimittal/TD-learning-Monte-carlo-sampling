from .monte_carlo import MonteCarloControl
from .td import Sarsa, QLearning, TDPrediction
from .reinforce import Reinforce, discounted_returns
from .a2c import A2C

__all__ = [
    "MonteCarloControl",
    "Sarsa",
    "QLearning",
    "TDPrediction",
    "Reinforce",
    "A2C",
    "discounted_returns",
]
