"""Injection-proof detection: a detector whose context can only grow when flows are added, paired
with a trained graph model that only chooses the attack type (detect-attribute model).

A flow f of the target window is flagged when D_f > 0, with
    D_f = a(x_f) + sum_j g_j(x_f) * [c(src_f), c(dst_f)]_j,
where a and g >= 0 depend only on f's own features and c(v) summarises host v over the K windows
of the temporal context. Per window, the statistics of v are the max and the log-sum of the
per-flow suspicion q = sigmoid(a(x)) over v's outgoing and incoming flows, log(1 + degree) and
log(1 + distinct peers) in both directions. They are >= 0 (0 when v is silent) and can only grow
when flows are added, because q of an existing flow depends on its own features only; c combines
them with non-negative weights and a ReLU, and D is non-decreasing in c. Hence adding any number of
flows, to any window, from any (also spoofed or new) host, can never lower D_f of an existing flow.
tests/test_core.py checks this for random weights.

The attack type of a flagged flow is the most likely attack class of the attribution model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import Snapshot

N_STATS = 8


def host_stats(s: Snapshot, q: torch.Tensor) -> torch.Tensor:
    """(N, 8) monotone statistics of every host of snapshot s, given the suspicion q of its flows."""
    n = s.n_nodes
    zero = lambda: q.new_zeros(n)  # noqa: E731
    one = torch.ones_like(q)
    pair = torch.unique(s.src * n + s.dst)
    p1 = q.new_ones(len(pair))
    return torch.stack([
        zero().scatter_reduce(0, s.src, q, reduce="amax", include_self=True),
        zero().scatter_reduce(0, s.dst, q, reduce="amax", include_self=True),
        torch.log1p(zero().index_add(0, s.src, q)), torch.log1p(zero().index_add(0, s.dst, q)),
        torch.log1p(zero().index_add(0, s.src, one)), torch.log1p(zero().index_add(0, s.dst, one)),
        torch.log1p(zero().index_add(0, pair // n, p1)), torch.log1p(zero().index_add(0, pair % n, p1)),
    ], 1)


class MonoDetector(nn.Module):
    def __init__(self, in_dim: int, k: int, dcfg: dict):
        super().__init__()
        d, h = dcfg.get("hidden", 64), dcfg.get("context_dim", 16)
        self.k, self.context = k, dcfg.get("context", True)
        self.enc = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.own = nn.Linear(d, 1)
        self.gate = nn.Linear(d, 2 * h)
        # softplus(w_raw) >= 0 and softplus(gate) start small, so that training begins close to the
        # flow-only detector instead of flagging every flow of a busy host
        nn.init.constant_(self.gate.bias, -4.0)
        self.w_raw = nn.Parameter(torch.full((k * N_STATS, h), -5.0) + 0.1 * torch.randn(k * N_STATS, h))
        self.b = nn.Parameter(torch.zeros(h))

    def forward_inputs(self, window: list[Snapshot], xs: list[torch.Tensor]) -> torch.Tensor:
        """Detection logit D for every flow of the last snapshot (inputs xs of every snapshot)."""
        t = window[-1]
        h_t = self.enc(xs[-1])
        own = self.own(h_t).squeeze(-1)
        if not self.context:
            return own
        seq = own.new_zeros(t.n_nodes, self.k, N_STATS)
        for s, x in zip(window, xs):
            q = torch.sigmoid(own if s is t else self.own(self.enc(x)).squeeze(-1))
            st = host_stats(s, q)
            pos = torch.searchsorted(s.hosts, t.hosts).clamp(max=s.hosts.numel() - 1)
            ok = s.hosts[pos] == t.hosts
            seq[ok, self.k - 1 - s.lag] = st[pos[ok]]
        c = F.relu(seq.reshape(t.n_nodes, -1) @ F.softplus(self.w_raw) + self.b)
        g = F.softplus(self.gate(h_t))
        return own + (g * torch.cat([c[t.src], c[t.dst]], 1)).sum(1)


class DetectAttribute(nn.Module):
    """Detector + attribution model. Returns logits whose argmax is "benign" when D <= 0 and
    otherwise the attribution model's most likely attack class, so evaluate.py, the attacks and
    the metrics treat it like any other model."""

    def __init__(self, transform: dict, n_classes: int, cfg: dict):
        super().__init__()
        from .model import build_model  # local import: model.build_model builds this class
        mcfg = cfg["model"]
        self.typer = build_model({"transform": transform, "classes": list(range(n_classes))},
                                 {**cfg, "model": mcfg["typer"]})
        self.tf = self.typer.tf
        self.detector = MonoDetector(self.tf.out_dim, cfg["graph"]["history_k"], mcfg.get("detector", {}))

    def forward(self, window, num_override=None, x_override=None):
        t = window[-1]
        if x_override is not None:
            x = x_override
        elif num_override is not None:
            x = self.tf(num_override, t.cat)
        else:
            x = self.tf(t.num_raw, t.cat)
        return self.forward_inputs(window, [self.tf(s.num_raw, s.cat) for s in window[:-1]] + [x])

    def forward_inputs(self, window, xs, history_grad=False):
        d = self.detector.forward_inputs(window, xs)
        typ = self.typer.forward_inputs(window, xs, history_grad).log_softmax(1)[:, 1:]
        typ = typ - typ.max(1, keepdim=True).values  # best attack class -> 0
        return torch.cat([torch.zeros_like(d)[:, None], d[:, None] + typ], 1)
