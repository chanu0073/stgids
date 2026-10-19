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
With detector.channels = m > 1, m - 1 further learned per-flow scores in [0, 1] (functions of the
flow's own features) are summarised the same way, which keeps the guarantee. tests/test_core.py
checks it for random weights.

The attack type of a flagged flow is the most likely attack class of the attribution model: a trained
graph model (STGIDS), or a FlowTyper that sees only the flow's own features, in which case injected
flows can change neither the detection nor the type of an existing flow (predictions can only move
from benign to an attack class). With detector.fusion = "or", a flow is also flagged when the
attribution model itself predicts an attack; flows flagged by the detector keep the guarantee.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import Snapshot


def host_stats(s: Snapshot, q: torch.Tensor) -> torch.Tensor:
    """(N, 4m + 4) monotone statistics of every host of snapshot s, given m evidence channels
    q (E, m) >= 0 of its flows: per channel the max and the log-sum over outgoing and incoming
    flows, then log(1 + degree) and log(1 + distinct peers) in both directions."""
    q = q[:, None] if q.dim() == 1 else q
    n, m = s.n_nodes, q.shape[1]
    zero = q.new_zeros(n, m)
    pair = torch.unique(s.src * n + s.dst)
    count = lambda idx, w: torch.log1p(q.new_zeros(n).index_add(0, idx, w))[:, None]  # noqa: E731
    return torch.cat([
        zero.scatter_reduce(0, s.src[:, None].expand(-1, m), q, reduce="amax", include_self=True),
        zero.scatter_reduce(0, s.dst[:, None].expand(-1, m), q, reduce="amax", include_self=True),
        torch.log1p(zero.index_add(0, s.src, q)), torch.log1p(zero.index_add(0, s.dst, q)),
        count(s.src, q.new_ones(len(q))), count(s.dst, q.new_ones(len(q))),
        count(pair // n, q.new_ones(len(pair))), count(pair % n, q.new_ones(len(pair))),
    ], 1)


class MonoDetector(nn.Module):
    def __init__(self, in_dim: int, k: int, dcfg: dict):
        super().__init__()
        d, h = dcfg.get("hidden", 64), dcfg.get("context_dim", 16)
        self.k, self.context = k, dcfg.get("context", True)
        # evidence channels: the suspicion sigmoid(a) plus channels - 1 learned per-flow scores in
        # [0, 1] (e.g. "scan-like flow"), all computed from the flow's own features only
        self.channels = dcfg.get("channels", 1)
        self.n_stats = 4 * self.channels + 4
        self.enc = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.own = nn.Linear(d, 1)
        if self.channels > 1:
            self.evid = nn.Linear(d, self.channels - 1)
        self.gate = nn.Linear(d, 2 * h)
        # softplus(w_raw) >= 0 and softplus(gate) start small, so that training begins close to the
        # flow-only detector instead of flagging every flow of a busy host
        nn.init.constant_(self.gate.bias, -4.0)
        self.w_raw = nn.Parameter(torch.full((k * self.n_stats, h), -5.0) + 0.1 * torch.randn(k * self.n_stats, h))
        self.b = nn.Parameter(torch.zeros(h))

    def evidence(self, own: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        q = torch.sigmoid(own)[:, None]
        return torch.cat([q, torch.sigmoid(self.evid(h))], 1) if self.channels > 1 else q

    def forward_inputs(self, window: list[Snapshot], xs: list[torch.Tensor]) -> torch.Tensor:
        """Detection logit D for every flow of the last snapshot (inputs xs of every snapshot)."""
        t = window[-1]
        h_t = self.enc(xs[-1])
        own = self.own(h_t).squeeze(-1)
        if not self.context:
            return own
        seq = own.new_zeros(t.n_nodes, self.k, self.n_stats)
        for s, x in zip(window, xs):
            if s is t:
                q = self.evidence(own, h_t)
            else:
                h = self.enc(x)
                q = self.evidence(self.own(h).squeeze(-1), h)
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
        dcfg = mcfg.get("detector", {})
        self.detector = MonoDetector(self.tf.out_dim, cfg["graph"]["history_k"], dcfg)
        self.fusion = dcfg.get("fusion", "detector")
        if self.fusion not in ("detector", "or"):
            raise ValueError(f"unknown detector.fusion {self.fusion!r}")

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
        lg = self.typer.forward_inputs(window, xs, history_grad)
        if self.fusion == "or":  # also flag what the attribution model flags (> 0 iff it predicts an attack)
            d = torch.maximum(d, lg[:, 1:].max(1).values - lg[:, 0])
        typ = lg.log_softmax(1)[:, 1:]
        typ = typ - typ.max(1, keepdim=True).values  # best attack class -> 0
        return torch.cat([torch.zeros_like(d)[:, None], d[:, None] + typ], 1)


class FlowTyper(nn.Module):
    """Attribution model that sees only each flow's own features: a trained neural flow baseline of
    baselines.py (e.g. "mlp_adv", the adversarially trained DNN). Injected flows cannot change its
    output for an existing flow."""

    def __init__(self, transform: dict, n_classes: int, net: str):
        super().__init__()
        from .baselines import NETS  # local imports: baselines imports model, model builds this class
        from .model import FeatureTransform
        self.tf = FeatureTransform(transform)
        self.net = NETS[net](self.tf.out_dim, n_classes)

    def forward(self, window, num_override=None, x_override=None):
        t = window[-1]
        x = x_override if x_override is not None else self.tf(
            num_override if num_override is not None else t.num_raw, t.cat)
        return self.net(x)

    def forward_inputs(self, window, xs, history_grad=False):
        return self.net(xs[-1])
