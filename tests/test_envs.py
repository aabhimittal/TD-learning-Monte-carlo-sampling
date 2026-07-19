import numpy as np

from rlkit.envs import GridWorld, CliffWalking


def test_gridworld_shapes_and_bounds():
    env = GridWorld(rows=3, cols=4)
    assert env.n_states == 12
    assert env.n_actions == 4
    s = env.reset()
    assert s == 0
    assert env.one_hot(s).shape == (12,)
    assert env.one_hot(s).sum() == 1.0


def test_walls_keep_agent_in_grid():
    env = GridWorld(rows=2, cols=2, start=(0, 0))
    env.reset()
    # move up from top row -> stay put
    s, r, done, _ = env.step(0)
    assert env.to_pos(s) == (0, 0)
    # move left from left column -> stay put
    s, r, done, _ = env.step(3)
    assert env.to_pos(s) == (0, 0)


def test_goal_terminates_with_reward():
    env = GridWorld(rows=1, cols=2, start=(0, 0), goals=((0, 1),), goal_reward=5.0)
    env.reset()
    s, r, done, _ = env.step(1)  # right into goal
    assert done
    assert r == 5.0
    assert env.is_terminal(s)


def test_obstacle_blocks():
    env = GridWorld(rows=1, cols=3, start=(0, 0), goals=((0, 2),), obstacles=((0, 1),))
    env.reset()
    s, r, done, _ = env.step(1)  # blocked by obstacle
    assert env.to_pos(s) == (0, 0)


def test_cliff_resets_to_start_with_penalty():
    env = CliffWalking()
    env.reset()
    s, r, done, _ = env.step(1)  # step right onto the cliff from start (3,0)->(3,1)
    assert r == -100.0
    assert env.to_pos(s) == env.start
    assert not done


def test_max_steps_truncation():
    env = GridWorld(rows=5, cols=5, goals=((4, 4),), max_steps=3)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(0)  # keep bumping the top wall
        steps += 1
    assert steps == 3
