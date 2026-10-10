"""Temporal graph snapshots.

Flows are sorted by start time, so every snapshot (time window) is a contiguous
slice of the flow arrays. Inside a snapshot, hosts are nodes and each flow is a
directed edge src -> dst carrying the flow's features and label.

A training/eval *sample* is a window of up to K consecutive snapshots ending at
snapshot t; the model classifies the edges of snapshot t and uses the earlier
snapshots only as history (no labels). Val/test targets may use history from any
split (it is past traffic); train targets use train history only, so with a blocked
split no val/test traffic is seen during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SPLITS = {"train": 0, "val": 1, "test": 2}


@dataclass
class Snapshot:
    num_raw: torch.Tensor  # (E, Fn) raw numeric features
    cat: torch.Tensor  # (E, Fc) encoded categorical features
    src: torch.Tensor  # (E,) local node index
    dst: torch.Tensor  # (E,) local node index
    hosts: torch.Tensor  # (N,) sorted global host ids of the local nodes
    y: torch.Tensor  # (E,) class index, -1 = class unseen in training
    y_bin: torch.Tensor  # (E,) 0 benign / 1 attack
    lag: int = 0  # how many windows before the target snapshot

    @property
    def n_nodes(self) -> int:
        return int(self.hosts.numel())

    def to(self, device) -> "Snapshot":
        return Snapshot(*(t.to(device, non_blocking=True) for t in
                          (self.num_raw, self.cat, self.src, self.dst, self.hosts, self.y, self.y_bin)),
                        lag=self.lag)


class SnapshotStore:
    def __init__(self, path: str, history_k: int = 1):
        blob = torch.load(path, weights_only=False)
        self.meta = blob["meta"]
        self.num_raw, self.cat = blob["num_raw"], blob["cat"]
        self.y, self.y_bin = blob["y"], blob["y_bin"]
        self.src, self.dst = blob["src"], blob["dst"]
        self.ptr, self.win = blob["snap_ptr"], blob["snap_window"]
        self.split = blob["snap_split"]
        self.k = history_k
        self._cache: dict[int, Snapshot] = {}

    def __len__(self) -> int:
        return len(self.win)

    @property
    def n_classes(self) -> int:
        return len(self.meta["classes"])

    def snapshot(self, i: int) -> Snapshot:
        if i not in self._cache:
            a, b = int(self.ptr[i]), int(self.ptr[i + 1])
            ends = torch.cat([self.src[a:b], self.dst[a:b]])
            hosts, inv = torch.unique(ends, sorted=True, return_inverse=True)
            e = b - a
            self._cache[i] = Snapshot(self.num_raw[a:b], self.cat[a:b], inv[:e], inv[e:], hosts,
                                      self.y[a:b], self.y_bin[a:b])
        return self._cache[i]

    def window(self, t: int, device=None) -> list[Snapshot]:
        """Snapshots [.., t] within the last K time windows, oldest first."""
        out = []
        w_t = int(self.win[t])
        train_target = int(self.split[t]) == SPLITS["train"]
        i = t
        while i >= 0 and w_t - int(self.win[i]) < self.k:
            if train_target and int(self.split[i]) != SPLITS["train"]:
                i -= 1  # never show val/test traffic to a training sample
                continue
            s = self.snapshot(i)
            s = Snapshot(s.num_raw, s.cat, s.src, s.dst, s.hosts, s.y, s.y_bin, lag=w_t - int(self.win[i]))
            out.append(s.to(device) if device is not None else s)
            i -= 1
        return out[::-1]

    def targets(self, split: str) -> list[int]:
        return torch.nonzero(self.split == SPLITS[split]).flatten().tolist()

    def flows(self, split: str) -> dict[str, torch.Tensor]:
        """All flows of a split as flat tensors (for flow-vector baselines)."""
        idx = self.targets(split)
        if not idx:
            raise ValueError(f"split {split!r} is empty")
        if idx[-1] - idx[0] + 1 == len(idx):  # one contiguous block (chronological split): views, no copy
            a, b = int(self.ptr[idx[0]]), int(self.ptr[idx[-1] + 1])
            return {"num_raw": self.num_raw[a:b], "cat": self.cat[a:b], "y": self.y[a:b], "y_bin": self.y_bin[a:b]}
        # blocked split: gather the split's snapshots, in time order
        rows = torch.cat([torch.arange(int(self.ptr[i]), int(self.ptr[i + 1])) for i in idx])
        return {"num_raw": self.num_raw[rows], "cat": self.cat[rows], "y": self.y[rows], "y_bin": self.y_bin[rows]}

    def class_counts(self, split: str = "train") -> torch.Tensor:
        y = self.flows(split)["y"]
        y = y[y >= 0]
        return torch.bincount(y, minlength=self.n_classes).float()
