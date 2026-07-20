# rlkit — Monte-Carlo · TD · REINFORCE · A2C

A small, dependency-light reinforcement-learning library that implements and
**contrasts the four canonical algorithm families** on shared grid-world
environments — using nothing heavier than **NumPy**. No Gym, no PyTorch, no
TensorFlow. Every update rule (including the policy-gradient backprop) is
written out explicitly so the maths stays readable.

| Family | Bootstraps? | Needs full episodes? | Learns a... | Class |
|---|---|---|---|---|
| **Monte-Carlo control** | no | yes | value function | `MonteCarloControl` |
| **TD control** | yes | no | value function | `Sarsa`, `QLearning` |
| **n-step / λ TD** | partial | no | value function | `NStepSarsa`, `SarsaLambda` |
| **REINFORCE** | no | yes | policy | `Reinforce` |
| **A2C** | yes (critic) | no* | policy + value | `A2C` |

\* the A2C here uses Monte-Carlo returns as the critic target for clarity, but
bootstraps through the learned baseline.

The tabular agents run on grid worlds; the policy-gradient agents additionally
solve a continuous **CartPole** (both reach the max score of 500) over the raw
observation via a NumPy MLP.

![Learning curves on a 6x6 grid world](docs/learning_curves.png)

## Why it exists

Most tutorials show these algorithms in isolation, each with its own
environment and framework. `rlkit` puts all five agents behind one tiny
Gym-like interface so you can run them head-to-head and *see* the trade-offs:
MC vs TD (episodic vs online), on-policy SARSA vs off-policy Q-learning
(safe vs optimal on the cliff), and value-based vs policy-gradient methods.

## Install

```bash
git clone https://github.com/aabhimittal/td-learning-monte-carlo-sampling
cd td-learning-monte-carlo-sampling
pip install -e .            # only dependency is numpy
pip install -e ".[test]"   # + pytest, to run the suite
pip install -e ".[plot]"   # + matplotlib, for learning-curve figures
```

## Quickstart

```python
from rlkit.envs import GridWorld
from rlkit.algorithms import QLearning

env = GridWorld(rows=4, cols=4, goals=((3, 3),), step_reward=-1.0)
agent = QLearning(env.n_states, env.n_actions, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
agent.train(env, episodes=800)

print(env.render_policy(agent.greedy_policy()))
# > > > v
# ^ > > v
# ^ ^ > v
# ^ ^ > G
```

Policy-gradient agents work over one-hot state features and the built-in
NumPy MLP:

```python
from rlkit.algorithms import A2C

agent = A2C(env.n_states, env.n_actions, hidden=(64,),
            actor_lr=0.02, critic_lr=0.05, entropy_coef=0.01, seed=0)
agent.train(env, episodes=1500)
policy = agent.greedy_policy(env.n_states, env.one_hot)
```

The same agents solve the continuous CartPole over its raw 4-D observation:

```python
from rlkit.envs import CartPole
from rlkit.algorithms import A2C

env = CartPole()
agent = A2C(env.n_features, env.n_actions, hidden=(64,),
            actor_lr=0.01, critic_lr=0.02, entropy_coef=0.01, seed=0)
agent.train(env, episodes=600, max_steps=500)
print(agent.evaluate(CartPole(seed=1)))   # -> 500.0 (the max score)
```

## Examples

```bash
python examples/train_mc.py               # first-visit Monte-Carlo control
python examples/train_td.py               # SARSA vs Q-learning on the cliff
python examples/train_nstep.py            # 1-step vs n-step vs SARSA(λ)
python examples/train_policy_gradient.py  # REINFORCE vs A2C on a grid world
python examples/train_cartpole.py         # REINFORCE & A2C solve CartPole
python examples/compare_all.py            # all agents, one summary table
python examples/plot_learning_curves.py   # save docs/learning_curves.png (needs matplotlib)
```

`compare_all.py` output (optimal greedy return on this 4×4 task is **-5**):

```
Algorithm          Greedy return   (optimal = -5)
----------------------------------------------
Monte-Carlo                 -5.0
SARSA (TD)                  -5.0
Q-learning (TD)             -5.0
REINFORCE                   -5.0
A2C                         -5.0
```

## The algorithms

### Monte-Carlo control (`rlkit/algorithms/monte_carlo.py`)
First-visit MC: play a full episode, compute the discounted return following
the first visit to each `(state, action)`, and average those returns into
`Q`. Policy improvement is ε-greedy. No bootstrapping, no environment model —
just sampled returns.

### Temporal-difference control (`rlkit/algorithms/td.py`)
One-step TD updates `Q(s,a) ← Q(s,a) + α·[target − Q(s,a)]` after **every
step**. The two agents differ only in the target:

* **SARSA** (on-policy): `target = r + γ·Q(s', a')` — the value of the action
  actually taken next. It learns the value of the ε-greedy policy it follows,
  so on the cliff it prefers the *safe* path away from the edge.
* **Q-learning** (off-policy): `target = r + γ·max_a Q(s', a)` — the greedy
  next value. It learns the *optimal* cliff-edge path regardless of
  exploration.

`TDPrediction` implements plain TD(0) state-value estimation for a fixed
policy.

### n-step and λ-returns (`rlkit/algorithms/td.py`)
Two ways to interpolate between one-step TD and Monte-Carlo:

* **`NStepSarsa`** accumulates `n` real rewards before bootstrapping off
  `Q[s_{t+n}, a_{t+n}]`. `n=1` is plain SARSA; large `n` approaches MC.
* **`SarsaLambda`** achieves the same interpolation *online* with accumulating
  eligibility traces: every visited `(s,a)` keeps a decaying trace and the TD
  error is broadcast to all of them. `λ=0` recovers one-step SARSA; `λ→1`
  (with `γ=1`) approaches MC. In practice it learns fastest on the grids here
  (see the plot above).

### REINFORCE (`rlkit/algorithms/reinforce.py`)
Monte-Carlo policy gradient. The policy is a softmax over a NumPy MLP. Using
the identity `d log π(a|s) / d logits = one_hot(a) − π`, the exact
score-function gradient is fed straight into the network's backward pass —
weighted by the (whitened) return. Optional entropy bonus.

### A2C (`rlkit/algorithms/a2c.py`)
Advantage Actor-Critic. A learned critic `V(s)` provides a state-dependent
baseline, so the actor is updated with the advantage `A_t = G_t − V(s_t)`
instead of the raw return — same expectation, far less variance. The critic
regresses toward the Monte-Carlo returns.

## Design

```
rlkit/
├── envs/          GridWorld + CliffWalking + CartPole (Gym-like reset/step)
├── nn/            NumPy MLP with manual backprop + Adam
├── algorithms/    monte_carlo · td (SARSA/Q/n-step/λ) · reinforce · a2c
└── utils.py       learning-curve smoothing, sparklines, matplotlib plotting
```

Environments share a common function-approximation interface — `env.features(obs)`
returns the feature vector (one-hot for grids, the raw 4-vector for CartPole) —
so the same `Reinforce`/`A2C` code runs on both discrete and continuous tasks.

The neural-net core (`rlkit/nn/mlp.py`) is a minimal reverse-mode stack:
`Linear`/`Tanh` layers implement `forward`/`backward`, the network accumulates
parameter gradients, and `Adam` consumes them. The policy-gradient algorithms
supply the upstream gradient directly, so there is no generic autograd to
trust — the learning signal is visible in ~10 lines per algorithm.

## Tests

```bash
pytest
```

The suite covers environment mechanics (walls, obstacles, goals, cliff resets,
truncation) and asserts that **every agent actually learns** a near-optimal
policy on a small grid.

## License

MIT — see [LICENSE](LICENSE).
