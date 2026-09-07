"""Periodic-review inventory control -- the canonical industrial MDP.

Every period (a day, a shift) you observe the on-hand stock, decide how much
to reorder, and then random demand arrives. Order too little and you lose
margin and goodwill; order too much and you pay to hold it. The optimal
policy for this class of problem is the textbook ``(s, S)`` rule -- reorder up
to ``S`` whenever stock drops below ``s`` -- which gives a known-good baseline
to check a learned policy against.

Why it is a useful test case for this library:

* rewards are **stochastic and continuous**, not the tidy -1 per step of a
  grid world, so it exercises averaging rather than pathfinding;
* the reward mixes several opposing cost terms, which is where reward-shaping
  mistakes show up;
* episodes end by **time limit**, never by reaching a goal, so it is the
  natural test for truncation-aware bootstrapping.

State is the on-hand inventory level (``0 .. capacity``); action is the number
of units to order (``0 .. max_order``), clipped to the remaining capacity.
"""
from __future__ import annotations

import numpy as np

from ..validation import (
    check_finite,
    check_non_negative_int,
    check_positive_int,
)


class InventoryControl:
    """Single-item inventory control with Poisson demand and lost sales.

    Parameters
    ----------
    capacity:
        Maximum units that fit in the warehouse.
    max_order:
        Largest single replenishment order (defaults to ``capacity``).
    demand_mean:
        Mean of the Poisson demand per period. ``0`` gives a deterministic
        no-demand environment, which is a useful degenerate case.
    sale_price, unit_cost, fixed_order_cost, holding_cost, shortage_cost:
        Revenue per unit sold, variable and fixed cost of ordering, cost of
        carrying one unit to the next period, and the goodwill penalty per
        unit of unmet demand.
    horizon:
        Periods per episode. The episode ends by truncation, never by
        termination.
    """

    def __init__(
        self,
        capacity: int = 10,
        max_order: int | None = None,
        demand_mean: float = 3.0,
        sale_price: float = 3.0,
        unit_cost: float = 1.0,
        fixed_order_cost: float = 2.0,
        holding_cost: float = 0.5,
        shortage_cost: float = 4.0,
        horizon: int = 30,
        seed: int | None = None,
    ):
        self.capacity = check_non_negative_int("capacity", capacity)
        self.max_order = (self.capacity if max_order is None
                          else check_non_negative_int("max_order", max_order))
        if self.max_order > self.capacity:
            raise ValueError(
                f"max_order ({self.max_order}) cannot exceed capacity ({self.capacity})"
            )
        self.demand_mean = check_finite("demand_mean", demand_mean)
        if self.demand_mean < 0:
            raise ValueError(f"demand_mean must be >= 0, got {self.demand_mean}")
        self.sale_price = check_finite("sale_price", sale_price)
        self.unit_cost = check_finite("unit_cost", unit_cost)
        self.fixed_order_cost = check_finite("fixed_order_cost", fixed_order_cost)
        self.holding_cost = check_finite("holding_cost", holding_cost)
        self.shortage_cost = check_finite("shortage_cost", shortage_cost)
        self.horizon = check_positive_int("horizon", horizon)
        self.rng = np.random.default_rng(seed)

        self.n_states = self.capacity + 1
        self.n_actions = self.max_order + 1
        self.n_features = self.n_states
        self._inventory = 0
        self._steps = 0

    # -- feature helpers (shared interface with the grid worlds) ------------
    def one_hot(self, state: int) -> np.ndarray:
        vec = np.zeros(self.n_states, dtype=np.float32)
        vec[int(state)] = 1.0
        return vec

    def features(self, state: int) -> np.ndarray:
        return self.one_hot(state)

    def is_terminal(self, state: int) -> bool:
        """No absorbing states: the episode only ever ends on the horizon."""
        return False

    # -- gym-like API -------------------------------------------------------
    def reset(self) -> int:
        self._inventory = 0
        self._steps = 0
        return self._inventory

    def sample_demand(self) -> int:
        """Draw one period of demand (Poisson, exposed for testing/planning)."""
        if self.demand_mean == 0:
            return 0
        return int(self.rng.poisson(self.demand_mean))

    def step(self, action: int):
        if not 0 <= action < self.n_actions:
            raise ValueError(f"invalid order quantity {action}")

        # Orders are clipped by the physical capacity of the warehouse: asking
        # for more than fits is not an error, it just cannot all be delivered.
        ordered = min(int(action), self.capacity - self._inventory)
        stock = self._inventory + ordered

        demand = self.sample_demand()
        sold = min(demand, stock)
        unmet = demand - sold
        end_inventory = stock - sold

        reward = (
            self.sale_price * sold
            - self.unit_cost * ordered
            - (self.fixed_order_cost if ordered > 0 else 0.0)
            - self.holding_cost * end_inventory
            - self.shortage_cost * unmet
        )

        self._inventory = end_inventory
        self._steps += 1
        truncated = self._steps >= self.horizon
        info = {
            "truncated": truncated,
            "demand": demand,
            "sold": sold,
            "unmet": unmet,
            "ordered": ordered,
        }
        return self._inventory, float(reward), truncated, info

    # -- baselines ----------------------------------------------------------
    def order_up_to_policy(self, reorder_point: int, order_up_to: int) -> np.ndarray:
        """The classic ``(s, S)`` policy as an action array, for comparison.

        Order up to ``S`` whenever on-hand stock is at or below ``s``; the
        quantity is clipped to ``max_order``, so a tight ``max_order`` makes
        the rule only approximately optimal -- which is exactly the kind of
        real-world constraint a learned policy can exploit.
        """
        reorder_point = check_non_negative_int("reorder_point", reorder_point)
        order_up_to = check_non_negative_int("order_up_to", order_up_to)
        if order_up_to > self.capacity:
            raise ValueError("order_up_to cannot exceed capacity")
        policy = np.zeros(self.n_states, dtype=np.int64)
        for stock in range(self.n_states):
            if stock <= reorder_point:
                policy[stock] = min(max(order_up_to - stock, 0), self.max_order)
        return policy
