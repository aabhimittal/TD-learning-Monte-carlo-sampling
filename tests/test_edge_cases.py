"""Edge cases that break RL agents in production rather than in textbooks.

Degenerate MDPs (one state, one action, terminal on arrival, no reachable
goal), misconfiguration, extreme reward scales, time-limit truncation,
non-stationary rewards, reproducibility and checkpoint round-trips. Each test
below corresponds to a failure that is easy to ship and hard to notice.
"""
import numpy as np
import pytest

from rlkit.algorithms import (
    A2C,
    DoubleQLearning,
    ExpectedSarsa,
    MonteCarloControl,
    NStepSarsa,
    QLearning,
    Reinforce,
    Sarsa,
    SarsaLambda,
    TDPrediction,
)
from rlkit.envs import GridWorld
from rlkit.schedules import LinearSchedule
from rlkit.utils import ascii_sparkline, moving_average

TABULAR_AGENTS = [MonteCarloControl, Sarsa, QLearning, NStepSarsa, SarsaLambda,
                  DoubleQLearning, ExpectedSarsa]
TD_AGENTS = [Sarsa, QLearning, NStepSarsa, SarsaLambda, DoubleQLearning,
             ExpectedSarsa]


def make_agent(cls, n_states, n_actions, **kwargs):
    """MC control takes no step size; everything else does."""
    if cls is MonteCarloControl:
        kwargs.pop("alpha", None)
    return cls(n_states, n_actions, **kwargs)


# --------------------------------------------------------------------------
# degenerate environments
# --------------------------------------------------------------------------
class TerminalOnFirstStep:
    """Every episode ends immediately -- a one-shot decision, not a trajectory."""

    n_states, n_actions = 1, 2

    def reset(self):
        return 0

    def step(self, action):
        return 0, 1.0 if action == 1 else -1.0, True, {"truncated": False}


class SingleActionTreadmill:
    """One state, one action, constant reward, ends only on the time limit."""

    n_states, n_actions = 1, 1

    def __init__(self, horizon=5, reward=1.0):
        self.horizon = horizon
        self.reward = reward
        self._steps = 0

    def reset(self):
        self._steps = 0
        return 0

    def step(self, action):
        self._steps += 1
        truncated = self._steps >= self.horizon
        return 0, self.reward, truncated, {"truncated": truncated}


@pytest.mark.parametrize("cls", TABULAR_AGENTS)
def test_agents_handle_episodes_that_end_on_the_first_step(cls):
    env = TerminalOnFirstStep()
    agent = make_agent(cls, env.n_states, env.n_actions, gamma=0.9, epsilon=0.3,
                       seed=0)
    history = agent.train(env, episodes=200, max_steps=10)
    assert len(history) == 200
    assert np.all(np.isfinite(agent.Q))
    assert agent.greedy_policy()[0] == 1  # the rewarding action


@pytest.mark.parametrize("cls", TABULAR_AGENTS)
def test_agents_handle_a_single_action_environment(cls):
    env = SingleActionTreadmill(horizon=4)
    agent = make_agent(cls, env.n_states, env.n_actions, gamma=0.9, epsilon=0.1,
                       seed=0)
    history = agent.train(env, episodes=50, max_steps=10)
    assert agent.greedy_policy().tolist() == [0]
    assert history[-1] == pytest.approx(4.0)


@pytest.mark.parametrize("cls", TD_AGENTS)
def test_agents_terminate_when_the_goal_is_unreachable(cls):
    """Walled-off goal: every episode truncates, and nothing hangs or diverges."""
    env = GridWorld(rows=4, cols=4, goals=((3, 3),), obstacles=((2, 3), (3, 2)),
                    step_reward=-1.0, max_steps=25)
    agent = make_agent(cls, env.n_states, env.n_actions, gamma=0.9, alpha=0.2,
                       epsilon=0.2, seed=0)
    history = agent.train(env, episodes=60, max_steps=40)
    assert len(history) == 60
    assert all(r == pytest.approx(-25.0) for r in history)  # always truncated
    assert np.all(np.isfinite(agent.Q))


def test_one_by_one_gridworld_where_start_is_the_goal():
    """A grid whose only cell is the goal: the first step must terminate."""
    env = GridWorld(rows=1, cols=1, start=(0, 0), goals=((0, 0),), max_steps=5)
    assert env.is_terminal(env.reset())
    state, reward, done, info = env.step(0)
    assert state == 0 and done and reward == pytest.approx(env.goal_reward)
    assert not info["truncated"]


# --------------------------------------------------------------------------
# hyper-parameter edge values and validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", TD_AGENTS)
def test_zero_step_size_learns_nothing(cls):
    env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-1.0, max_steps=20)
    agent = make_agent(cls, env.n_states, env.n_actions, gamma=0.9, alpha=0.0,
                       epsilon=0.5, seed=0)
    agent.train(env, episodes=50, max_steps=20)
    assert np.count_nonzero(agent.Q) == 0


def test_gamma_zero_makes_the_agent_purely_myopic():
    """With no discounting of the future, Q is exactly the immediate reward."""
    env = GridWorld(rows=2, cols=2, goals=((1, 1),), step_reward=-1.0,
                    goal_reward=5.0, max_steps=20)
    agent = QLearning(env.n_states, env.n_actions, gamma=0.0, alpha=1.0,
                      epsilon=1.0, seed=0)
    agent.train(env, episodes=300, max_steps=20)
    assert agent.Q[1, 2] == pytest.approx(5.0)   # (0,1) down -> goal
    assert agent.Q[0, 1] == pytest.approx(-1.0)  # (0,0) right -> not the goal


def test_epsilon_one_is_uniform_random_and_epsilon_zero_is_deterministic():
    agent = QLearning(4, 4, epsilon=1.0, seed=0)
    agent.Q[0] = [10.0, 0.0, 0.0, 0.0]
    counts = np.bincount([agent.act(0) for _ in range(4000)], minlength=4)
    assert counts.min() > 800   # every action still gets picked

    agent.epsilon = 0.0
    assert {agent.act(0) for _ in range(50)} == {0}


@pytest.mark.parametrize("cls", TD_AGENTS)
@pytest.mark.parametrize("kwargs", [
    {"gamma": 1.5}, {"gamma": -0.1}, {"gamma": float("nan")},
    {"alpha": -0.5}, {"alpha": 2.0}, {"epsilon": 1.2}, {"epsilon": -0.01},
])
def test_agents_reject_out_of_range_hyperparameters(cls, kwargs):
    with pytest.raises((ValueError, TypeError)):
        make_agent(cls, 4, 2, **kwargs)


@pytest.mark.parametrize("bad_size", [0, -3, 2.5, True])
def test_agents_reject_degenerate_table_sizes(bad_size):
    with pytest.raises((ValueError, TypeError)):
        QLearning(bad_size, 2)
    with pytest.raises((ValueError, TypeError)):
        QLearning(4, bad_size)


def test_nstep_and_lambda_validate_their_own_knobs():
    with pytest.raises(ValueError):
        NStepSarsa(4, 2, n=0)
    with pytest.raises(ValueError):
        SarsaLambda(4, 2, lam=1.5)


@pytest.mark.parametrize("env_cls_kwargs", [
    {"rows": 0}, {"cols": 0}, {"max_steps": 0},
])
def test_gridworld_rejects_degenerate_dimensions(env_cls_kwargs):
    with pytest.raises(ValueError):
        GridWorld(**env_cls_kwargs).reset()


def test_gridworld_rejects_invalid_actions():
    env = GridWorld(rows=2, cols=2)
    env.reset()
    for bad in (-1, 4, 99):
        with pytest.raises(ValueError):
            env.step(bad)


# --------------------------------------------------------------------------
# numerical robustness
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [1e-8, 1e6])
def test_extreme_reward_scales_stay_finite(scale):
    """Money-denominated rewards are not in [-1, 1]; nothing may overflow."""
    env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-scale,
                    goal_reward=scale, max_steps=30)
    agent = QLearning(env.n_states, env.n_actions, gamma=0.99, alpha=0.5,
                      epsilon=0.2, seed=0)
    agent.train(env, episodes=200, max_steps=30)
    assert np.all(np.isfinite(agent.Q))
    assert np.abs(agent.Q).max() < 1e12


def test_large_state_space_trains_without_blowing_up():
    """2500 states: the tables stay dense but tractable, and training runs."""
    env = GridWorld(rows=50, cols=50, goals=((49, 49),), step_reward=-1.0,
                    max_steps=60)
    agent = QLearning(env.n_states, env.n_actions, gamma=0.99, alpha=0.5,
                      epsilon=0.3, seed=0)
    agent.train(env, episodes=20, max_steps=60)
    assert agent.Q.shape == (2500, 4)
    assert np.all(np.isfinite(agent.Q))


def test_moving_average_and_sparkline_edge_inputs():
    assert moving_average([], window=5).size == 0
    assert moving_average([1.0, 2.0], window=99).tolist() == [1.0, 1.5]
    assert moving_average([1.0, 2.0], window=1).tolist() == [1.0, 2.0]
    assert ascii_sparkline([]) == ""
    assert len(ascii_sparkline([3.0])) == 1
    assert len(set(ascii_sparkline([2.0] * 10))) == 1  # flat line, no divide-by-zero
    assert len(ascii_sparkline(list(range(500)), width=20)) == 20


# --------------------------------------------------------------------------
# truncation vs termination
# --------------------------------------------------------------------------
def test_truncation_is_bootstrapped_not_treated_as_terminal():
    """The bug this guards: a time limit teaching the agent the world ends.

    On a treadmill paying +1 forever, the true value is ``1/(1-gamma) = 10``.
    An agent that treats the horizon as a terminal state learns the value of a
    5-step episode instead and badly under-estimates.
    """
    gamma = 0.9
    correct = QLearning(1, 1, gamma=gamma, alpha=0.1, epsilon=0.0, seed=0,
                        bootstrap_on_truncation=True)
    correct.train(SingleActionTreadmill(horizon=5), episodes=400, max_steps=50)
    naive = QLearning(1, 1, gamma=gamma, alpha=0.1, epsilon=0.0, seed=0,
                      bootstrap_on_truncation=False)
    naive.train(SingleActionTreadmill(horizon=5), episodes=400, max_steps=50)

    assert correct.Q[0, 0] == pytest.approx(10.0, rel=0.05)
    assert naive.Q[0, 0] < 0.8 * correct.Q[0, 0]


@pytest.mark.parametrize("cls", TD_AGENTS)
def test_every_td_agent_honours_the_truncation_flag(cls):
    env_kwargs = dict(gamma=0.9, alpha=0.1, epsilon=0.0, seed=0)
    correct = make_agent(cls, 1, 1, bootstrap_on_truncation=True, **env_kwargs)
    correct.train(SingleActionTreadmill(horizon=5), episodes=400, max_steps=50)
    naive = make_agent(cls, 1, 1, bootstrap_on_truncation=False, **env_kwargs)
    naive.train(SingleActionTreadmill(horizon=5), episodes=400, max_steps=50)
    assert correct.Q[0, 0] > naive.Q[0, 0]


def test_td_prediction_bootstraps_through_truncation_too():
    env = SingleActionTreadmill(horizon=5)
    predictor = TDPrediction(1, gamma=0.9, alpha=0.1)
    predictor.train(env, policy=np.zeros(1, dtype=int), episodes=400, max_steps=50)
    assert predictor.V[0] == pytest.approx(10.0, rel=0.05)


# --------------------------------------------------------------------------
# schedules in the training loop
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", TABULAR_AGENTS)
def test_exploration_schedule_is_advanced_once_per_episode(cls):
    env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-1.0, max_steps=20)
    agent = make_agent(cls, env.n_states, env.n_actions, gamma=0.9,
                       epsilon=LinearSchedule(1.0, 0.0, decay_steps=100), seed=0)
    assert agent.epsilon == pytest.approx(1.0)
    agent.train(env, episodes=100, max_steps=20)
    assert agent.epsilon == pytest.approx(0.0)


def test_step_size_schedule_anneals_and_a_float_still_works():
    agent = QLearning(4, 2, alpha=LinearSchedule(0.9, 0.1, decay_steps=10), seed=0)
    assert agent.alpha == pytest.approx(0.9)
    agent.train(SingleActionTreadmill(horizon=2), episodes=10, max_steps=5)
    assert agent.alpha == pytest.approx(0.1)
    agent.alpha = 0.25  # assigning a plain float replaces the schedule
    assert agent.alpha == pytest.approx(0.25)


# --------------------------------------------------------------------------
# non-stationarity
# --------------------------------------------------------------------------
class DriftingBandit:
    """A two-armed bandit whose better arm switches half way through."""

    n_states, n_actions = 1, 2

    def __init__(self):
        self.flipped = False

    def reset(self):
        return 0

    def step(self, action):
        good = 1 if self.flipped else 0
        return 0, 1.0 if action == good else 0.0, True, {"truncated": False}


def test_constant_step_size_tracks_a_regime_change_that_averaging_misses():
    """Concept drift: demand patterns move, and sample averages do not follow.

    A constant step size forgets exponentially and re-learns the new optimum;
    Monte-Carlo's running sample average keeps the pre-drift evidence forever
    and stays anchored to the arm that used to be better.
    """
    env = DriftingBandit()
    td = QLearning(1, 2, gamma=0.0, alpha=0.1, epsilon=0.2, seed=0)
    mc = MonteCarloControl(1, 2, gamma=0.0, epsilon=0.2, seed=0)
    td.train(env, episodes=600, max_steps=1)
    mc.train(env, episodes=600, max_steps=1)
    assert td.greedy_policy()[0] == 0 and mc.greedy_policy()[0] == 0

    env.flipped = True
    td.train(env, episodes=400, max_steps=1)
    mc.train(env, episodes=400, max_steps=1)

    assert td.greedy_policy()[0] == 1          # tracked the change
    assert td.Q[0, 0] < 0.5 < mc.Q[0, 0]       # MC is still anchored to the past


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", TABULAR_AGENTS)
def test_same_seed_reproduces_training_exactly(cls):
    def run(seed):
        env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-1.0,
                        max_steps=20)
        agent = make_agent(cls, env.n_states, env.n_actions, gamma=0.9, alpha=0.3,
                           epsilon=0.3, seed=seed)
        return agent.train(env, episodes=40, max_steps=20), agent.Q.copy()

    first_history, first_q = run(11)
    same_history, same_q = run(11)
    other_history, _ = run(12)
    assert first_history == same_history and np.array_equal(first_q, same_q)
    assert first_history != other_history


@pytest.mark.parametrize("cls", [Reinforce, A2C])
def test_policy_gradient_agents_are_reproducible(cls):
    def run(seed):
        env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-1.0,
                        max_steps=20)
        return cls(env.n_features, env.n_actions, seed=seed).train(
            env, episodes=20, max_steps=20)

    assert run(3) == run(3)


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", TABULAR_AGENTS)
def test_checkpoint_round_trip_preserves_the_policy(cls, tmp_path):
    env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-1.0, max_steps=20)
    agent = make_agent(cls, env.n_states, env.n_actions, gamma=0.9, alpha=0.3,
                       epsilon=0.2, seed=0)
    agent.train(env, episodes=100, max_steps=20)

    path = agent.save(tmp_path / "agent")
    assert path.endswith(".npz")
    restored = cls.load(path, seed=0)

    assert np.allclose(restored.Q, agent.Q)
    assert np.array_equal(restored.greedy_policy(), agent.greedy_policy())
    assert restored.gamma == agent.gamma and restored.n_actions == agent.n_actions
    restored.epsilon = 0.0
    agent.epsilon = 0.0
    assert [restored.act(s) for s in range(env.n_states)] == \
           [agent.act(s) for s in range(env.n_states)]


def test_restored_agent_keeps_training_from_where_it_stopped(tmp_path):
    env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-1.0, max_steps=20)
    agent = QLearning(env.n_states, env.n_actions, gamma=0.9, alpha=0.3,
                      epsilon=0.1, seed=0)
    agent.train(env, episodes=200, max_steps=20)
    before = agent.Q.copy()

    restored = QLearning.load(agent.save(tmp_path / "ckpt"), seed=1)
    restored.train(env, episodes=50, max_steps=20)
    assert not np.array_equal(restored.Q, before)   # it really kept learning
    assert np.all(np.isfinite(restored.Q))


def test_checkpoint_refuses_to_load_into_the_wrong_class(tmp_path):
    path = QLearning(4, 2, seed=0).save(tmp_path / "q")
    with pytest.raises(ValueError, match="QLearning"):
        Sarsa.load(path)


def test_td_prediction_checkpoint_round_trip(tmp_path):
    predictor = TDPrediction(4, gamma=0.9, alpha=0.2)
    predictor.V[:] = [1.0, 2.0, 3.0, 4.0]
    restored = TDPrediction.load(predictor.save(tmp_path / "v"))
    assert np.allclose(restored.V, predictor.V) and restored.gamma == 0.9
