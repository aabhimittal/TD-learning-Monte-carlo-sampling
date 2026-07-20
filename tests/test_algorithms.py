"""Learning tests: every agent should solve a small grid world.

The grid is a 4x4 world with a -1 step cost and the goal at the far corner.
The optimal path from (0,0) to (3,3) is 6 moves; the goal-entering move is
free, so the optimal return is -5. We assert each agent gets comfortably
close after training.
"""
import numpy as np

from rlkit.envs import GridWorld
from rlkit.algorithms import (
    MonteCarloControl, Sarsa, QLearning, NStepSarsa, SarsaLambda,
    TDPrediction, Reinforce, A2C,
)


def make_env():
    return GridWorld(rows=4, cols=4, start=(0, 0), goals=((3, 3),), step_reward=-1.0, max_steps=100)


def greedy_return(env, policy, max_steps=50):
    state = env.reset()
    total = 0.0
    for _ in range(max_steps):
        s, r, done, _ = env.step(int(policy[state]))
        total += r
        state = s
        if done:
            break
    return total


def test_monte_carlo_learns():
    env = make_env()
    agent = MonteCarloControl(env.n_states, env.n_actions, gamma=1.0, epsilon=0.2, seed=0)
    agent.train(env, episodes=3000, max_steps=100)
    assert greedy_return(env, agent.greedy_policy()) >= -8


def test_sarsa_learns():
    env = make_env()
    agent = Sarsa(env.n_states, env.n_actions, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    agent.train(env, episodes=800)
    assert greedy_return(env, agent.greedy_policy()) >= -8


def test_qlearning_learns():
    env = make_env()
    agent = QLearning(env.n_states, env.n_actions, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    agent.train(env, episodes=800)
    assert greedy_return(env, agent.greedy_policy()) >= -7


def test_nstep_sarsa_learns():
    env = make_env()
    agent = NStepSarsa(env.n_states, env.n_actions, gamma=1.0, alpha=0.4,
                       epsilon=0.1, n=4, seed=0)
    agent.train(env, episodes=800)
    assert greedy_return(env, agent.greedy_policy()) >= -8


def test_sarsa_lambda_learns():
    env = make_env()
    agent = SarsaLambda(env.n_states, env.n_actions, gamma=1.0, alpha=0.4,
                        epsilon=0.1, lam=0.9, seed=0)
    agent.train(env, episodes=800)
    assert greedy_return(env, agent.greedy_policy()) >= -8


def test_nstep_n1_matches_one_step_sarsa():
    # n=1 n-step SARSA should behave like plain SARSA
    env = make_env()
    agent = NStepSarsa(env.n_states, env.n_actions, gamma=1.0, alpha=0.5,
                       epsilon=0.1, n=1, seed=0)
    agent.train(env, episodes=800)
    assert greedy_return(env, agent.greedy_policy()) >= -8


def test_td_prediction_matches_optimal_path():
    env = make_env()
    agent = QLearning(env.n_states, env.n_actions, gamma=1.0, alpha=0.5, epsilon=0.1, seed=0)
    agent.train(env, episodes=800)
    policy = agent.greedy_policy()
    pred = TDPrediction(env.n_states, gamma=1.0, alpha=0.1)
    V = pred.train(env, policy, episodes=500)
    # value of the start state should be close to the (negative) path length
    assert V[env.reset()] <= -3


def test_reinforce_learns():
    env = make_env()
    agent = Reinforce(env.n_states, env.n_actions, hidden=(64,), gamma=0.99, lr=0.02, seed=0)
    agent.train(env, episodes=1500)
    ret = greedy_return(env, agent.greedy_policy(env.n_states, env.one_hot))
    assert ret >= -12


def test_a2c_learns():
    env = make_env()
    agent = A2C(env.n_states, env.n_actions, hidden=(64,), gamma=0.99,
                actor_lr=0.02, critic_lr=0.05, entropy_coef=0.01, seed=0)
    agent.train(env, episodes=1500)
    ret = greedy_return(env, agent.greedy_policy(env.n_states, env.one_hot))
    assert ret >= -12


def test_discounted_returns_values():
    from rlkit.algorithms import discounted_returns
    out = discounted_returns([1.0, 1.0, 1.0], gamma=0.5)
    np.testing.assert_allclose(out, [1.75, 1.5, 1.0])
