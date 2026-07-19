"""A tiny reverse-mode MLP built on NumPy only.

This repo deliberately avoids a deep-learning framework so the policy-gradient
maths stays visible. Each layer implements ``forward``/``backward``; the
network accumulates parameter gradients that an :class:`Adam` optimizer then
consumes. Upstream gradients are supplied by the algorithm (e.g. the
REINFORCE score function ``one_hot(a) - pi``), so no generic autograd is
needed.
"""
from __future__ import annotations

import numpy as np


class Linear:
    """Fully connected layer ``y = x @ W + b`` with He initialisation."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        scale = np.sqrt(2.0 / in_dim)
        self.W = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float64)
        self.b = np.zeros(out_dim, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.dW[...] = self._x.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class Tanh:
    """Element-wise tanh activation."""

    def __init__(self):
        self._out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out ** 2)

    def params(self):
        return []

    def grads(self):
        return []


class MLP:
    """Sequential stack of :class:`Linear` layers with tanh hidden activations.

    ``hidden`` is a list of hidden-layer widths. The output layer is linear
    (raw logits or a scalar value depending on ``out_dim``).
    """

    def __init__(self, in_dim: int, hidden, out_dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        self.layers = []
        prev = in_dim
        for h in hidden:
            self.layers.append(Linear(prev, h, rng))
            self.layers.append(Tanh())
            prev = h
        self.layers.append(Linear(prev, out_dim, rng))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def grads(self):
        return [g for layer in self.layers for g in layer.grads()]


class Adam:
    """Adam optimizer operating in place on a fixed list of parameters."""

    def __init__(self, params, lr: float = 1e-2, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self, grads) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
