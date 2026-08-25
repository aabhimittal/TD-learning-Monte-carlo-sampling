"""Tabular grid-world environments with a Gym-like interface.

States are integers in ``[0, n_states)``. Actions are integers in
``[0, n_actions)`` mapping to up/right/down/left. The API is deliberately
minimal (``reset`` / ``step``) so both tabular and function-approximation
agents in this repo can share it.
"""
from __future__ import annotations

import numpy as np

from ..validation import check_finite, check_positive_int

# action id -> (d_row, d_col)
_MOVES = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}


class GridWorld:
    """A rectangular grid world.

    Parameters
    ----------
    rows, cols:
        Grid dimensions.
    start:
        ``(row, col)`` starting cell.
    goals:
        Iterable of ``(row, col)`` terminal goal cells (reward ``goal_reward``).
        Defaults to the bottom-right cell of whatever grid you asked for.
    obstacles:
        Iterable of impassable ``(row, col)`` cells; attempting to move into
        one leaves the agent in place.
    step_reward:
        Reward received on every non-terminal transition (usually negative to
        encourage short paths).
    goal_reward:
        Reward for entering a goal cell.
    max_steps:
        Episodes are truncated after this many steps.
    seed:
        Seed for the internal RNG (used only if stochasticity is added later).
    """

    n_actions = 4

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: tuple[int, int] = (0, 0),
        goals=None,
        obstacles=(),
        step_reward: float = -1.0,
        goal_reward: float = 0.0,
        max_steps: int = 100,
        seed: int | None = None,
    ):
        self.rows = check_positive_int("rows", rows)
        self.cols = check_positive_int("cols", cols)
        self.max_steps = check_positive_int("max_steps", max_steps)
        self.start = tuple(start)
        # Default to the far corner, which follows the grid's size instead of
        # silently landing outside it when rows/cols are changed.
        if goals is None:
            goals = ((self.rows - 1, self.cols - 1),)
        self.goals = {tuple(g) for g in goals}
        self.obstacles = {tuple(o) for o in obstacles}
        self.step_reward = check_finite("step_reward", step_reward)
        self.goal_reward = check_finite("goal_reward", goal_reward)
        self.rng = np.random.default_rng(seed)

        # A cell outside the grid, or a start buried in a wall, produces an
        # environment that "works" until an episode silently never terminates.
        for label, cells in (("start", [self.start]), ("goal", self.goals),
                             ("obstacle", self.obstacles)):
            for cell in cells:
                if len(cell) != 2 or not (0 <= cell[0] < self.rows
                                          and 0 <= cell[1] < self.cols):
                    raise ValueError(
                        f"{label} cell {cell} lies outside a {self.rows}x{self.cols} grid"
                    )
        if self.start in self.obstacles:
            raise ValueError(f"start cell {self.start} is also an obstacle")

        self.n_states = rows * cols
        # function-approximation interface (shared with CartPole): a state is
        # encoded as a one-hot vector of length n_features.
        self.n_features = self.n_states
        self._pos = start
        self._steps = 0

    # -- coordinate helpers -------------------------------------------------
    def to_state(self, pos: tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def to_pos(self, state: int) -> tuple[int, int]:
        return divmod(state, self.cols)

    def one_hot(self, state: int) -> np.ndarray:
        """Return a one-hot feature vector for a state (for function approx)."""
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[state] = 1.0
        return vec

    def features(self, state: int) -> np.ndarray:
        """Feature encoding used by the policy-gradient agents (one-hot)."""
        return self.one_hot(state)

    def is_terminal(self, state: int) -> bool:
        return self.to_pos(state) in self.goals

    # -- gym-like API -------------------------------------------------------
    def reset(self) -> int:
        self._pos = self.start
        self._steps = 0
        return self.to_state(self._pos)

    def step(self, action: int):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid action {action}")

        dr, dc = _MOVES[action]
        r, c = self._pos
        nr, nc = r + dr, c + dc

        # walls: stay in place if we would leave the grid or hit an obstacle
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (nr, nc) in self.obstacles:
            nr, nc = r, c

        self._pos = (nr, nc)
        self._steps += 1

        if (nr, nc) in self.goals:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False

        truncated = self._steps >= self.max_steps
        info = {"truncated": truncated and not done}
        return self.to_state(self._pos), reward, done or truncated, info

    # -- pretty printing ----------------------------------------------------
    def render_policy(self, policy) -> str:
        """Render a greedy ``policy`` (array of action ids) as ASCII arrows."""
        arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                pos = (r, c)
                if pos in self.goals:
                    row.append("G")
                elif pos in self.obstacles:
                    row.append("#")
                else:
                    row.append(arrows[int(policy[self.to_state(pos)])])
            lines.append(" ".join(row))
        return "\n".join(lines)


class CliffWalking(GridWorld):
    """The classic Sutton & Barto cliff-walking task (4x12 grid).

    Stepping onto any cliff cell along the bottom edge yields a reward of
    ``-100`` and teleports the agent back to the start. This env sharply
    separates the on-policy (SARSA) and off-policy (Q-learning) solutions,
    which makes it a good demonstrator for the TD algorithms in this repo.
    """

    def __init__(self, seed: int | None = None):
        super().__init__(
            rows=4,
            cols=12,
            start=(3, 0),
            goals=((3, 11),),
            step_reward=-1.0,
            goal_reward=0.0,
            max_steps=200,
            seed=seed,
        )
        self.cliff = {(3, c) for c in range(1, 11)}

    def step(self, action: int):
        dr, dc = _MOVES[action]
        r, c = self._pos
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c

        self._steps += 1

        if (nr, nc) in self.cliff:
            self._pos = self.start
            truncated = self._steps >= self.max_steps
            return self.to_state(self._pos), -100.0, truncated, {"truncated": truncated}

        self._pos = (nr, nc)
        if (nr, nc) in self.goals:
            return self.to_state(self._pos), self.goal_reward, True, {"truncated": False}

        truncated = self._steps >= self.max_steps
        return self.to_state(self._pos), self.step_reward, truncated, {"truncated": truncated}
