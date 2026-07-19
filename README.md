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
| **REINFORCE** | no | yes | policy | `Reinforce` |
| **A2C** | yes (critic) | no* | policy + value | `A2C` |

\* the A2C here uses Monte-Carlo returns as the critic target for clarity, but
bootstraps through the learned baseline.

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

## Examples

```bash
python examples/train_mc.py               # first-visit Monte-Carlo control
python examples/train_td.py               # SARSA vs Q-learning on the cliff
python examples/train_policy_gradient.py  # REINFORCE vs A2C
python examples/compare_all.py            # all five agents, one summary table
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
├── envs/          GridWorld + CliffWalking (Gym-like reset/step)
├── nn/            NumPy MLP with manual backprop + Adam
└── algorithms/    monte_carlo · td · reinforce · a2c
```

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
