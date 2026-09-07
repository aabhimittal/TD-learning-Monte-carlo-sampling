"""rlkit -- a dependency-light reinforcement-learning teaching library.

Implements and contrasts the canonical families of value-based and policy
algorithms on shared environments, using nothing heavier than NumPy:

* Monte-Carlo control            (:class:`rlkit.algorithms.MonteCarloControl`)
* Temporal-difference control    (:class:`rlkit.algorithms.Sarsa`,
                                  :class:`rlkit.algorithms.QLearning`,
                                  :class:`rlkit.algorithms.ExpectedSarsa`,
                                  :class:`rlkit.algorithms.DoubleQLearning`)
* n-step / eligibility traces    (:class:`rlkit.algorithms.NStepSarsa`,
                                  :class:`rlkit.algorithms.SarsaLambda`)
* Offline / batch RL             (:class:`rlkit.algorithms.BatchQLearning`,
                                  :class:`rlkit.algorithms.OffPolicyMonteCarlo`)
* REINFORCE policy gradient      (:class:`rlkit.algorithms.Reinforce`)
* Advantage Actor-Critic (A2C)   (:class:`rlkit.algorithms.A2C`)

Alongside the algorithms it ships the pieces a real deployment needs:
hyper-parameter :mod:`~rlkit.schedules`, experience :mod:`~rlkit.buffers`,
policy and off-policy :mod:`~rlkit.evaluation`, pickle-free checkpointing
(:mod:`~rlkit.persistence`) and two industrial environments -- inventory
control and predictive maintenance.
"""
from .envs import (
    GridWorld,
    CliffWalking,
    CartPole,
    InventoryControl,
    MachineMaintenance,
)
from .algorithms import (
    MonteCarloControl,
    Sarsa,
    QLearning,
    NStepSarsa,
    SarsaLambda,
    TDPrediction,
    DoubleQLearning,
    ExpectedSarsa,
    BatchQLearning,
    OffPolicyMonteCarlo,
    Reinforce,
    A2C,
)
from .buffers import ReplayBuffer, PrioritizedReplayBuffer, collect_transitions
from .schedules import ConstantSchedule, LinearSchedule, ExponentialSchedule
from .evaluation import evaluate_policy, off_policy_evaluation, effective_sample_size
from .utils import moving_average, ascii_sparkline, plot_learning_curves

__version__ = "0.3.0"

__all__ = [
    "GridWorld",
    "CliffWalking",
    "CartPole",
    "InventoryControl",
    "MachineMaintenance",
    "MonteCarloControl",
    "Sarsa",
    "QLearning",
    "NStepSarsa",
    "SarsaLambda",
    "TDPrediction",
    "DoubleQLearning",
    "ExpectedSarsa",
    "BatchQLearning",
    "OffPolicyMonteCarlo",
    "Reinforce",
    "A2C",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "collect_transitions",
    "ConstantSchedule",
    "LinearSchedule",
    "ExponentialSchedule",
    "evaluate_policy",
    "off_policy_evaluation",
    "effective_sample_size",
    "moving_average",
    "ascii_sparkline",
    "plot_learning_curves",
]
