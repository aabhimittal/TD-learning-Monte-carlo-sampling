import numpy as np

from rlkit.envs import CartPole
from rlkit.algorithms import A2C, Reinforce


def test_cartpole_reset_and_shape():
    env = CartPole(seed=0)
    obs = env.reset()
    assert obs.shape == (4,)
    assert np.all(np.abs(obs) <= 0.05)
    assert env.n_actions == 2 and env.n_features == 4


def test_cartpole_step_reward_and_termination():
    env = CartPole(seed=0)
    env.reset()
    obs, r, done, info = env.step(1)
    assert r == 1.0
    assert obs.shape == (4,)
    assert isinstance(done, bool)


def test_cartpole_truncates_at_max_steps():
    env = CartPole(max_steps=5, seed=0)
    env.reset()
    done = False
    steps = 0
    # alternate actions to keep the pole up for a few steps
    while not done and steps < 100:
        _, _, done, _ = env.step(steps % 2)
        steps += 1
    assert steps <= 5


def test_a2c_solves_cartpole():
    agent = A2C(4, 2, hidden=(64,), gamma=0.99, actor_lr=0.01, critic_lr=0.02,
                entropy_coef=0.01, seed=0)
    agent.train(CartPole(seed=0), episodes=600, max_steps=500)
    assert agent.evaluate(CartPole(seed=1), episodes=10, max_steps=500) >= 195


def test_reinforce_learns_cartpole():
    agent = Reinforce(4, 2, hidden=(64,), gamma=0.99, lr=0.01, seed=0)
    agent.train(CartPole(seed=0), episodes=600, max_steps=500)
    # REINFORCE is higher-variance; require clear improvement over random (~22)
    assert agent.evaluate(CartPole(seed=1), episodes=10, max_steps=500) >= 100
