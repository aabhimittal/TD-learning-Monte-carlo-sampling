"""Schedules and replay buffers, including the boundary cases that bite.

Most schedule/buffer bugs only appear at the edges: the first step, the step
after decay finishes, the moment the buffer wraps, a batch larger than the
data, or a sensor that emits ``nan``. Those are exactly the cases below.
"""
import numpy as np
import pytest

from rlkit.buffers import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    collect_transitions,
)
from rlkit.envs import GridWorld
from rlkit.schedules import (
    ConstantSchedule,
    ExponentialSchedule,
    LinearSchedule,
    as_schedule,
)


# --------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------
def test_linear_schedule_endpoints_and_clamping():
    sched = LinearSchedule(1.0, 0.1, decay_steps=100)
    assert sched(0) == pytest.approx(1.0)
    assert sched(50) == pytest.approx(0.55)
    assert sched(100) == pytest.approx(0.1)
    assert sched(10_000) == pytest.approx(0.1)  # holds after decay finishes


def test_linear_schedule_zero_decay_steps_jumps_immediately():
    """decay_steps=0 is the limit of the interpolation, not a divide-by-zero."""
    sched = LinearSchedule(1.0, 0.0, decay_steps=0)
    assert sched(0) == 0.0
    assert sched(5) == 0.0


def test_linear_schedule_can_increase():
    sched = LinearSchedule(0.0, 1.0, decay_steps=10)
    assert sched(5) == pytest.approx(0.5)
    assert sched(99) == pytest.approx(1.0)


def test_exponential_schedule_decays_and_floors():
    sched = ExponentialSchedule(1.0, decay_rate=0.5, end=0.1)
    assert sched(0) == pytest.approx(1.0)
    assert sched(2) == pytest.approx(0.25)
    assert sched(50) == pytest.approx(0.1)  # floored, never reaches 0


def test_exponential_schedule_decay_every_holds_value_between_drops():
    sched = ExponentialSchedule(1.0, decay_rate=0.5, decay_every=10)
    assert sched(9) == pytest.approx(1.0)
    assert sched(10) == pytest.approx(0.5)


def test_constant_schedule_and_coercion():
    assert ConstantSchedule(0.3)(1234) == pytest.approx(0.3)
    assert as_schedule(0.25)(0) == pytest.approx(0.25)
    sched = LinearSchedule(1.0, 0.0, 10)
    assert as_schedule(sched) is sched  # already a schedule: passed through


@pytest.mark.parametrize("factory", [
    lambda: LinearSchedule(float("nan"), 0.0, 10),
    lambda: LinearSchedule(1.0, 0.0, -5),
    lambda: ExponentialSchedule(1.0, decay_rate=0.0),
    lambda: ExponentialSchedule(1.0, decay_rate=1.5),
    lambda: ExponentialSchedule(0.1, decay_rate=0.5, end=0.5),  # end above start
    lambda: ExponentialSchedule(1.0, decay_rate=0.5, decay_every=0),
])
def test_invalid_schedules_rejected(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()


def test_negative_step_rejected():
    with pytest.raises(ValueError):
        LinearSchedule(1.0, 0.0, 10).value(-1)


# --------------------------------------------------------------------------
# replay buffer
# --------------------------------------------------------------------------
def test_buffer_evicts_oldest_when_full():
    buf = ReplayBuffer(capacity=3, seed=0)
    for i in range(5):
        buf.add(i, 0, float(i), i + 1, False)
    assert len(buf) == 3 and buf.is_full
    assert sorted(buf.all().states.tolist()) == [2, 3, 4]  # 0 and 1 evicted


def test_buffer_sampling_shapes_and_types():
    buf = ReplayBuffer(capacity=10, seed=0)
    for i in range(10):
        buf.add(i, i % 2, float(i), i + 1, i == 9)
    batch = buf.sample(4)
    assert batch.states.shape == batch.rewards.shape == (4,)
    assert batch.actions.dtype == np.int64 and batch.dones.dtype == bool


def test_buffer_sampling_without_replacement_is_distinct():
    buf = ReplayBuffer(capacity=10, seed=1)
    for i in range(10):
        buf.add(i, 0, 0.0, i, False)
    states = buf.sample(10, replace=False).states
    assert sorted(states.tolist()) == list(range(10))


def test_buffer_rejects_impossible_requests():
    buf = ReplayBuffer(capacity=5, seed=0)
    with pytest.raises(ValueError):
        buf.sample(1)          # empty
    with pytest.raises(ValueError):
        buf.all()              # empty
    buf.add(0, 0, 1.0, 1, False)
    with pytest.raises(ValueError):
        buf.sample(2, replace=False)  # more distinct samples than stored
    with pytest.raises(ValueError):
        buf.sample(0)                 # non-positive batch size


def test_buffer_rejects_non_finite_rewards():
    """A broken sensor should fail at ingest, not silently poison every batch."""
    buf = ReplayBuffer(capacity=4)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            buf.add(0, 0, bad, 1, False)
    assert len(buf) == 0


def test_buffer_clear_resets_write_position():
    buf = ReplayBuffer(capacity=2, seed=0)
    buf.add(0, 0, 0.0, 1, False)
    buf.clear()
    assert len(buf) == 0
    buf.add(7, 0, 0.0, 8, False)
    assert buf.all().states.tolist() == [7]


def test_buffer_holds_vector_states():
    """Continuous observations batch into a 2-D array, not an object array."""
    buf = ReplayBuffer(capacity=4, seed=0)
    for i in range(4):
        buf.add(np.zeros(3) + i, 0, 1.0, np.ones(3), False)
    assert buf.all().states.shape == (4, 3)


def test_buffer_sampling_is_reproducible():
    def draw():
        buf = ReplayBuffer(capacity=20, seed=42)
        for i in range(20):
            buf.add(i, 0, 0.0, i, False)
        return buf.sample(5).states.tolist()

    assert draw() == draw()


# --------------------------------------------------------------------------
# prioritised replay
# --------------------------------------------------------------------------
def test_prioritised_new_transitions_enter_at_max_priority():
    buf = PrioritizedReplayBuffer(capacity=4, alpha=1.0, seed=0)
    buf.add(0, 0, 0.0, 1, False, priority=5.0)
    buf.add(1, 0, 0.0, 2, False)  # no priority given -> inherits the max
    probs = buf._probabilities()
    assert probs[1] == pytest.approx(probs[0], rel=1e-3)


def test_prioritised_sampling_favours_high_priority():
    buf = PrioritizedReplayBuffer(capacity=2, alpha=1.0, seed=0)
    buf.add(0, 0, 0.0, 1, False, priority=0.0)
    buf.add(1, 0, 0.0, 2, False, priority=100.0)
    batch, _, _ = buf.sample(200)
    assert (batch.states == 1).mean() > 0.9


def test_prioritised_update_priorities_changes_the_distribution():
    buf = PrioritizedReplayBuffer(capacity=2, alpha=1.0, seed=0)
    buf.add(0, 0, 0.0, 1, False, priority=1.0)
    buf.add(1, 0, 0.0, 2, False, priority=1.0)
    buf.update_priorities([0], [50.0])
    _, idx, _ = buf.sample(200)
    assert (idx == 0).mean() > 0.9


def test_prioritised_weights_are_normalised_and_beta_zero_disables_them():
    buf = PrioritizedReplayBuffer(capacity=8, alpha=1.0, seed=0)
    for i in range(8):
        buf.add(i, 0, 0.0, i, False, priority=float(i + 1))
    _, _, weights = buf.sample(8, beta=0.6)
    assert weights.max() == pytest.approx(1.0) and np.all(weights > 0)
    _, _, flat = buf.sample(8, beta=0.0)
    assert np.allclose(flat, 1.0)


def test_prioritised_all_zero_priorities_falls_back_to_uniform():
    buf = PrioritizedReplayBuffer(capacity=3, alpha=1.0, epsilon=1e-12, seed=0)
    for i in range(3):
        buf.add(i, 0, 0.0, i, False, priority=0.0)
    probs = buf._probabilities()
    assert np.allclose(probs, 1 / 3)


def test_prioritised_rejects_bad_priority_updates():
    buf = PrioritizedReplayBuffer(capacity=2, seed=0)
    buf.add(0, 0, 0.0, 1, False)
    with pytest.raises(IndexError):
        buf.update_priorities([5], [1.0])
    with pytest.raises(ValueError):
        buf.update_priorities([0], [-1.0])
    with pytest.raises(ValueError):
        buf.update_priorities([0], [float("nan")])
    with pytest.raises(ValueError):
        buf.update_priorities([0, 0], [1.0])  # mismatched lengths


# --------------------------------------------------------------------------
# dataset collection
# --------------------------------------------------------------------------
def test_collect_transitions_marks_truncation_as_not_done():
    """A time limit must not be logged as a terminal state."""
    env = GridWorld(rows=4, cols=4, goals=((3, 3),), obstacles=((2, 3), (3, 2)),
                    step_reward=-1.0, max_steps=8)
    buf = collect_transitions(env, episodes=5, max_steps=50, seed=0)
    batch = buf.all()
    assert len(buf) == 40                 # 5 episodes x 8 truncated steps
    assert not batch.dones.any()          # goal unreachable: only truncations


def test_collect_transitions_respects_a_given_policy_and_buffer():
    env = GridWorld(rows=3, cols=3, goals=((2, 2),), step_reward=-1.0, max_steps=5)
    buf = ReplayBuffer(capacity=6, seed=0)
    collect_transitions(env, policy=lambda s: 1, episodes=4, max_steps=5, buffer=buf)
    assert len(buf) == 6 and buf.is_full   # capacity honoured across episodes
    assert set(buf.all().actions.tolist()) == {1}
