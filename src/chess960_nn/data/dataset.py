"""PyTorch Dataset over cached training shards.

Three flavours:

* ``ChessShardDataset`` (map-style): keeps the *current* shard in RAM and
  loads neighbouring shards on demand. Cheap for sequential / mildly
  random access. Random shuffling will be slow because every batch could
  touch many shards.

* ``InMemoryChessDataset`` (map-style): loads everything into RAM at
  construction time. Use for small datasets (<= ~5M positions).

* ``StreamingShardDataset`` (iterable): streams shards in random order,
  shuffles positions within each shard, yields single samples. Memory use
  bounded by one shard at a time. The right choice for multi-GB datasets.

Shard schema is defined in ``pipeline.write_shard``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from chess960_nn.data.pipeline import read_shard


class ChessShardDataset(Dataset):
    """Map-style dataset that lazy-loads one shard at a time.

    Best with sequential samplers or DataLoaders that keep accesses
    locally ordered (e.g. shuffled per-shard).
    """

    def __init__(self, shard_dir: Path | str):
        self.shard_dir = Path(shard_dir)
        self.shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")

        # Cache shard sizes (cheap - reads only the npz header).
        self.sizes: list[int] = []
        for shard in self.shards:
            with np.load(shard) as d:
                self.sizes.append(int(d["actions"].shape[0]))
        self.cum_sizes = np.cumsum(self.sizes)
        self.total = int(self.cum_sizes[-1])

        self._cur_idx: int = -1
        self._cur_data: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, idx: int) -> dict[str, np.ndarray]:
        if self._cur_idx == idx and self._cur_data is not None:
            return self._cur_data
        self._cur_data = read_shard(self.shards[idx])
        self._cur_idx = idx
        return self._cur_data

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        if not 0 <= index < self.total:
            raise IndexError(index)
        shard_idx = int(np.searchsorted(self.cum_sizes, index, side="right"))
        prev_total = int(self.cum_sizes[shard_idx - 1]) if shard_idx > 0 else 0
        offset = index - prev_total
        data = self._load_shard(shard_idx)
        state = torch.from_numpy(data["states"][offset]).float()
        action = int(data["actions"][offset])
        value = float(data["values"][offset])
        return state, action, value


class InMemoryChessDataset(Dataset):
    """Loads all shards into RAM at init. Fast random access; bounded by RAM."""

    def __init__(self, shard_dir: Path | str):
        self.shard_dir = Path(shard_dir)
        shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not shards:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")

        states_list, actions_list, values_list = [], [], []
        for shard in shards:
            d = read_shard(shard)
            states_list.append(d["states"])
            actions_list.append(d["actions"])
            values_list.append(d["values"])

        self.states = np.concatenate(states_list, axis=0)
        self.actions = np.concatenate(actions_list, axis=0)
        self.values = np.concatenate(values_list, axis=0)
        assert self.states.shape[0] == self.actions.shape[0] == self.values.shape[0]

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        state = torch.from_numpy(self.states[index]).float()
        action = int(self.actions[index])
        value = float(self.values[index])
        return state, action, value


class StreamingShardDataset(IterableDataset):
    """Iterable dataset: yields samples from shards in shuffled streaming order.

    For multi-GB datasets that don't fit in RAM. Memory bounded by one
    shard's positions plus a small buffer. Each epoch:

    1. Get list of shards (one slice per DataLoader worker, if multi-worker).
    2. Shuffle shard order (if ``shuffle``).
    3. For each shard: load it, shuffle indices, yield samples one by one.

    Use with a standard ``DataLoader`` and pass ``batch_size`` there; DO NOT
    set ``shuffle=True`` on the DataLoader (this dataset handles shuffling).
    """

    def __init__(
        self,
        shard_dir: Path | str,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No shards found in {self.shard_dir}")
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Cache total positions so __len__ is fast.
        self._total = 0
        for shard in self.shards:
            with np.load(shard) as d:
                self._total += int(d["actions"].shape[0])

    def __len__(self) -> int:
        return self._total

    def set_epoch(self, epoch: int) -> None:
        """Advance epoch counter to vary shuffling across epochs."""
        self._epoch = epoch

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            shards = list(self.shards)
            worker_id = 0
        else:
            shards = self.shards[worker.id :: worker.num_workers]
            worker_id = worker.id

        rng = np.random.default_rng(self.seed + self._epoch * 1009 + worker_id)
        if self.shuffle:
            rng.shuffle(shards)  # type: ignore[arg-type]

        for shard_path in shards:
            data = read_shard(shard_path)
            n = int(data["actions"].shape[0])
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            states = data["states"]
            actions = data["actions"]
            values = data["values"]
            for idx in order:
                yield (
                    torch.from_numpy(states[idx]).float(),
                    int(actions[idx]),
                    float(values[idx]),
                )
