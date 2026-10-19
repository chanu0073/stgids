"""Models: feature transform, edge-aware attention GNN, temporal encoder, classifier.

Pure PyTorch (no PyTorch Geometric needed); message passing uses index_add_ and
a scatter softmax.

STGIDS forward, for a window of snapshots [t-K+1, ..., t]:
    1. FeatureTransform: raw flow features -> log1p -> standardise -> selected columns
    2. SpatialEncoder per snapshot: edge-aware multi-head attention over the
       host graph -> node embeddings h and edge embeddings e     (SSGMHAN / E-GraphSAGE)
    3. TemporalEncoder: for each host of snapshot t, the sequence of its node
       embeddings over the K snapshots -> Transformer (or GRU)   (ST-GNN / Euler)
    4. EdgeClassifier: [z_src, z_dst, e] -> softmax over classes for every flow of t
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graphs import Snapshot


# --------------------------------------------------------------------------- helpers
def scatter_softmax(score: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of score (E, H) over entries sharing the same index (E,)."""
    h = score.shape[1]
    idx = index[:, None].expand(-1, h)
    mx = torch.full((n, h), float("-inf"), device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, idx, score, reduce="amax", include_self=True)
    ex = torch.exp(score - mx[index])
    den = torch.zeros((n, h), device=score.device, dtype=score.dtype).index_add_(0, index, ex)
    return ex / (den[index] + 1e-16)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros((n, src.shape[1]), device=src.device, dtype=src.dtype).index_add_(0, index, src)
    cnt = torch.zeros(n, device=src.device, dtype=src.dtype).index_add_(
        0, index, torch.ones_like(index, dtype=src.dtype))
    return out / cnt.clamp(min=1)[:, None]


def degree(index: torch.Tensor, n: int, dtype) -> torch.Tensor:
    return torch.zeros(n, device=index.device, dtype=dtype).index_add_(
        0, index, torch.ones_like(index, dtype=dtype))


class FeatureTransform(nn.Module):
    """Differentiable version of the preprocessing, so attacks can act on raw flow values."""

    def __init__(self, transform: dict):
        super().__init__()
        self.register_buffer("log_mask", torch.as_tensor(np.asarray(transform["log_mask"]), dtype=torch.bool))
        self.register_buffer("mu", torch.as_tensor(np.asarray(transform["mu"]), dtype=torch.float32))
        self.register_buffer("sigma", torch.as_tensor(np.asarray(transform["sigma"]), dtype=torch.float32))
        self.register_buffer("selected", torch.as_tensor(np.asarray(transform["selected"]), dtype=torch.long))

    @property
    def out_dim(self) -> int:
        return int(self.selected.numel())

    def forward(self, num_raw: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        logged = torch.log1p(num_raw.clamp(min=0))
        num = torch.where(self.log_mask, logged, num_raw)
        x = (torch.cat([num, cat.to(num.dtype)], dim=1) - self.mu) / self.sigma
        return x[:, self.selected]


# --------------------------------------------------------------------------- spatial
class EdgeAwareAttention(nn.Module):
    """Multi-head attention where keys/values combine the neighbour and the connecting edge.

    Messages flow in both directions of each flow; a learned direction embedding
    tells the layer whether it is looking at an outgoing or incoming flow.
    """

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0, "hidden must be divisible by heads"
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ek, self.ev = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.direction = nn.Parameter(torch.zeros(2, d))
        self.o = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.edge_mlp = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e, src, dst):
        n = h.shape[0]
        s2, d2 = torch.cat([src, dst]), torch.cat([dst, src])
        e2 = torch.cat([e + self.direction[0], e + self.direction[1]])
        q = self.q(h)[d2].view(-1, self.h, self.dh)
        k = (self.k(h)[s2] + self.ek(e2)).view(-1, self.h, self.dh)
        v = (self.v(h)[s2] + self.ev(e2)).view(-1, self.h, self.dh)
        alpha = scatter_softmax((q * k).sum(-1) / math.sqrt(self.dh), d2, n)
        alpha = self.drop(alpha)
        agg = torch.zeros((n, self.h, self.dh), device=h.device, dtype=h.dtype)
        agg.index_add_(0, d2, alpha[..., None] * v)
        h = self.n1(h + self.drop(self.o(agg.reshape(n, -1))))
        h = self.n2(h + self.drop(self.ffn(h)))
        e = self.n3(e + self.drop(self.edge_mlp(torch.cat([h[src], h[dst], e], dim=1))))
        return h, e


class SpatialEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.edge_in = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        # hosts have no own features in NetFlow: start from their in/out flow statistics
        self.node_in = nn.Sequential(nn.Linear(2 * in_dim + 2, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList(EdgeAwareAttention(d, heads, dropout) for _ in range(layers))

    def forward(self, x, src, dst, n):
        node0 = torch.cat([scatter_mean(x, src, n), scatter_mean(x, dst, n),
                           torch.log1p(degree(src, n, x.dtype))[:, None],
                           torch.log1p(degree(dst, n, x.dtype))[:, None]], dim=1)
        h, e = self.node_in(node0), self.edge_in(x)
        for layer in self.layers:
            h, e = layer(h, e, src, dst)
        return h, e


# --------------------------------------------------------------------------- temporal
class TemporalEncoder(nn.Module):
    def __init__(self, d: int, k: int, kind: str, heads: int, layers: int, dropout: float):
        super().__init__()
        self.kind, self.k = kind, k
        if kind == "transformer":
            self.pos = nn.Parameter(torch.randn(k, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        elif kind == "gru":
            self.absent = nn.Parameter(torch.zeros(d))
            self.enc = nn.GRU(d + 1, d, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        elif kind != "none":
            raise ValueError(f"unknown temporal encoder {kind!r}")

    def forward(self, seq: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """seq (N, K, d), valid (N, K) bool; last position = current snapshot (always valid)."""
        if self.kind == "none":
            return seq[:, -1]
        if self.kind == "transformer":
            out = self.enc(seq + self.pos[None], src_key_padding_mask=~valid)
            return out[:, -1]
        x = torch.where(valid[..., None], seq, self.absent.expand_as(seq))
        out, _ = self.enc(torch.cat([x, valid[..., None].float()], dim=-1))
        return out[:, -1]


def gather_history(hosts_t: torch.Tensor, snaps: list[tuple[torch.Tensor, torch.Tensor, int]],
                   k: int, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (N_t, K, d) sequences of node embeddings for the hosts of the target snapshot."""
    n = hosts_t.numel()
    seq = hosts_t.new_zeros((n, k, d), dtype=snaps[-1][1].dtype)
    valid = torch.zeros((n, k), dtype=torch.bool, device=hosts_t.device)
    for hosts, h, lag in snaps:
        pos = torch.searchsorted(hosts, hosts_t).clamp(max=hosts.numel() - 1)
        ok = hosts[pos] == hosts_t
        slot = k - 1 - lag
        seq[ok, slot] = h[pos[ok]]
        valid[ok, slot] = True
    return seq, valid


# --------------------------------------------------------------------------- full models
class STGIDS(nn.Module):
    def __init__(self, transform: dict, n_classes: int, mcfg: dict, k: int):
        super().__init__()
        self.tf = FeatureTransform(transform)
        d = mcfg["hidden"]
        self.k, self.d, self.detach_history = k, d, mcfg.get("detach_history", True)
        self.spatial = SpatialEncoder(self.tf.out_dim, d, mcfg["heads"], mcfg["gnn_layers"], mcfg["dropout"])
        self.temporal = TemporalEncoder(d, k, mcfg["temporal"], mcfg["heads"],
                                        mcfg["temporal_layers"], mcfg["dropout"])
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(mcfg["dropout"]),
                                  nn.Linear(d, n_classes))

    def inputs(self, s: Snapshot) -> torch.Tensor:
        return self.tf(s.num_raw, s.cat)

    def forward(self, window: list[Snapshot], num_override: torch.Tensor | None = None,
                x_override: torch.Tensor | None = None) -> torch.Tensor:
        """Logits for every flow of the last snapshot in `window`.

        num_override replaces the raw numeric features of the target snapshot
        (constrained attacks); x_override replaces its transformed model input
        (unconstrained attacks).
        """
        target = window[-1]
        if x_override is not None:
            x = x_override
        elif num_override is not None:
            x = self.tf(num_override, target.cat)
        else:
            x = self.inputs(target)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.detach_history):
            xs = [self.inputs(s) for s in window[:-1]]
        return self.forward_inputs(window, xs + [x])

    def forward_inputs(self, window: list[Snapshot], xs: list[torch.Tensor],
                       history_grad: bool = False) -> torch.Tensor:
        """Logits for every flow of the last snapshot, given the model inputs of every snapshot
        (used by injection attacks; history_grad lets gradients reach the history inputs)."""
        target = window[-1]
        hist = []
        for s, x in zip(window[:-1], xs[:-1]):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (history_grad or not self.detach_history)):
                h, _ = self.spatial(x, s.src, s.dst, s.n_nodes)
            hist.append((s.hosts, h, s.lag))
        h, e = self.spatial(xs[-1], target.src, target.dst, target.n_nodes)
        hist.append((target.hosts, h, 0))
        seq, valid = gather_history(target.hosts, hist, self.k, self.d)
        z = self.temporal(seq, valid)
        return self.head(torch.cat([z[target.src], z[target.dst], e], dim=1))


class EGraphSAGE(nn.Module):
    """Baseline following Lo et al. (NOMS 2022): mean-aggregated edge features, no temporal part."""

    def __init__(self, transform: dict, n_classes: int, mcfg: dict, k: int = 1):
        super().__init__()
        self.tf = FeatureTransform(transform)
        d, f = mcfg["hidden"], self.tf.out_dim
        self.layers = nn.ModuleList()
        in_node = f
        for _ in range(mcfg["gnn_layers"]):
            self.layers.append(nn.Linear(in_node + f, d))
            in_node = d
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(mcfg["dropout"]),
                                  nn.Linear(d, n_classes))
        self.drop = nn.Dropout(mcfg["dropout"])

    def forward(self, window, num_override=None, x_override=None):
        t = window[-1]
        x = x_override if x_override is not None else self.tf(
            num_override if num_override is not None else t.num_raw, t.cat)
        return self.forward_inputs(window, [x])

    def forward_inputs(self, window, xs, history_grad=False):
        t, x = window[-1], xs[-1]  # no temporal part: only the current snapshot is used
        n = t.n_nodes
        h = torch.ones((n, x.shape[1]), device=x.device, dtype=x.dtype)
        s2, d2 = torch.cat([t.src, t.dst]), torch.cat([t.dst, t.src])
        x2 = torch.cat([x, x])
        for layer in self.layers:
            agg = scatter_mean(x2, d2, n)
            h = self.drop(F.relu(layer(torch.cat([h, agg], dim=1))))
        return self.head(torch.cat([h[t.src], h[t.dst]], dim=1))


def build_model(meta: dict, cfg: dict) -> nn.Module:
    arch = cfg["model"]["arch"]
    if arch == "detect_attribute":
        from .detector import DetectAttribute
        return DetectAttribute(meta["transform"], len(meta["classes"]), cfg)
    if arch == "flow":  # flow baseline used as the attribution model of a detect-attribute model
        from .detector import FlowTyper
        return FlowTyper(meta["transform"], len(meta["classes"]), cfg["model"]["net"])
    cls = {"stgids": STGIDS, "egraphsage": EGraphSAGE}[arch]
    return cls(meta["transform"], len(meta["classes"]), cfg["model"], cfg["graph"]["history_k"])


# --------------------------------------------------------------------------- self-supervised pretraining
class DGIHead(nn.Module):
    """Deep Graph Infomax on edge embeddings (as in Anomal-E): real vs feature-shuffled graph."""

    def __init__(self, d: int):
        super().__init__()
        self.w = nn.Bilinear(d, d, 1)

    def loss(self, spatial: SpatialEncoder, x, src, dst, n) -> torch.Tensor:
        _, e_pos = spatial(x, src, dst, n)
        _, e_neg = spatial(x[torch.randperm(x.shape[0], device=x.device)], src, dst, n)
        summary = torch.sigmoid(e_pos.mean(0, keepdim=True))
        lp = self.w(e_pos, summary.expand_as(e_pos)).squeeze(-1)
        ln = self.w(e_neg, summary.expand_as(e_neg)).squeeze(-1)
        return F.binary_cross_entropy_with_logits(torch.cat([lp, ln]),
                                                  torch.cat([torch.ones_like(lp), torch.zeros_like(ln)]))
