"""Checkpointing for tabular agents (``.npz``, no pickle).

Training runs get interrupted, and a policy that took an hour to learn should
survive the process that learned it. Every tabular agent in :mod:`rlkit`
mixes in :class:`SaveLoadMixin`, which serialises the declared value tables
and hyper-parameters to a compressed ``.npz`` file.

Deliberately pickle-free: checkpoints are plain arrays, so loading one can
never execute code. Two consequences worth knowing:

* schedules are stored as their *current* value, so a reloaded agent
  continues with a constant ``alpha`` / ``epsilon`` unless you reassign one;
* the RNG stream is not stored -- pass ``seed=`` to :meth:`SaveLoadMixin.load`
  if you need reproducible behaviour after resuming.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SaveLoadMixin"]


class SaveLoadMixin:
    """Save/load the arrays named in ``_SAVE_ARRAYS`` and ``_SAVE_SCALARS``."""

    #: names of ``np.ndarray`` attributes to persist
    _SAVE_ARRAYS: tuple[str, ...] = ()
    #: names of scalar (int/float/bool) attributes to persist
    _SAVE_SCALARS: tuple[str, ...] = ()

    def save(self, path) -> str:
        """Write a checkpoint; returns the actual path written (adds ``.npz``)."""
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        payload = {"_class": np.asarray(type(self).__name__)}
        for name in self._SAVE_ARRAYS:
            payload[f"array__{name}"] = np.asarray(getattr(self, name))
        for name in self._SAVE_SCALARS:
            payload[f"scalar__{name}"] = np.asarray(getattr(self, name))
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **payload)
        return path

    @classmethod
    def load(cls, path, seed: int | None = None):
        """Rebuild an agent from a checkpoint written by :meth:`save`."""
        with np.load(str(path), allow_pickle=False) as data:
            saved_class = str(data["_class"].item())
            if saved_class != cls.__name__:
                raise ValueError(
                    f"checkpoint holds a {saved_class}, but {cls.__name__}.load was "
                    f"called; use {saved_class}.load instead"
                )
            obj = cls.__new__(cls)
            obj._episode = 0
            obj.rng = np.random.default_rng(seed)
            for name in cls._SAVE_SCALARS:
                setattr(obj, name, data[f"scalar__{name}"].item())
            for name in cls._SAVE_ARRAYS:
                setattr(obj, name, np.array(data[f"array__{name}"]))
        return obj
