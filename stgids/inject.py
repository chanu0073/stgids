"""Problem-space flow injection: the attacker ADDS flows; it cannot remove or alter flows it does
not own (that would need access to the flow exporter or the collector).

Attacker hosts are the sources of the attack flows of the target window. Each of them injects
`beta` flows into a window, copied from benign flows of that window (same destination server,
same features): traffic that an ordinary client could plausibly send. With mode "now" only the
target window is changed; with mode "hist" also the K-1 earlier windows that the temporal encoder
uses as history (history shaping).

The adaptive attack then optimises the model inputs of the injected flows with PGD, inside the
per-feature range of benign training flows, so that the original attack flows look benign. It can
be combined with a feature attack on the attack flows themselves (any space of adversarial.py).
Metrics are always computed on the original flows only.
"""
from __future__ import annotations

import torch

from .adversarial import make_space, no_cudnn
from .graphs import Snapshot


def add_flows(s: Snapshot, src_g: torch.Tensor, dst_g: torch.Tensor, num_raw: torch.Tensor,
              cat: torch.Tensor) -> Snapshot:
    """Append flows given by global host ids; unseen hosts become new nodes (hosts stay sorted)."""
    if len(src_g) == 0:
        return s
    hosts = torch.unique(torch.cat([s.hosts, src_g, dst_g]))
    src = torch.searchsorted(hosts, torch.cat([s.hosts[s.src], src_g]))
    dst = torch.searchsorted(hosts, torch.cat([s.hosts[s.dst], dst_g]))
    zero = torch.zeros(len(src_g), dtype=s.y.dtype, device=s.y.device)
    return Snapshot(torch.cat([s.num_raw, num_raw]), torch.cat([s.cat, cat]), src, dst, hosts,
                    torch.cat([s.y, zero]), torch.cat([s.y_bin, zero]), lag=s.lag)


def attacker_hosts(s: Snapshot) -> torch.Tensor:
    """Global ids of the hosts that send attack flows in snapshot s."""
    return torch.unique(s.hosts[s.src[s.y_bin == 1]])


def inject_templates(s: Snapshot, attackers: torch.Tensor, beta: int,
                     gen: torch.Generator) -> tuple[Snapshot, int]:
    """Every attacker host sends `beta` copies of random benign flows of s to their destinations."""
    benign = torch.nonzero(s.y_bin == 0).squeeze(1)
    if beta == 0 or len(benign) == 0 or len(attackers) == 0:
        return s, 0
    pick = benign[torch.randint(len(benign), (len(attackers) * beta,), generator=gen, device=gen.device)
                  .to(benign.device)]
    src_g = attackers.repeat_interleave(beta)
    dst_g = s.hosts[s.dst[pick]]
    keep = src_g != dst_g  # an attacker host does not send flows to itself
    return add_flows(s, src_g[keep], dst_g[keep], s.num_raw[pick[keep]], s.cat[pick[keep]]), int(keep.sum())


def inject_window(window: list[Snapshot], beta: int, mode: str,
                  gen: torch.Generator) -> tuple[list[Snapshot], list[int]]:
    """Inject into the target window ("now") or into every window of the context ("hist").
    Returns the new window and the number of injected flows per snapshot (appended last)."""
    if mode not in ("now", "hist"):
        raise ValueError(f"unknown injection mode {mode!r}")
    attackers = attacker_hosts(window[-1])
    out, counts = [], []
    for i, s in enumerate(window):
        k = 0
        if i == len(window) - 1 or mode == "hist":
            s, k = inject_templates(s, attackers, beta, gen)
        out.append(s)
        counts.append(k)
    return out, counts


def benign_input_range(model, store, n: int = 300_000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature min / max of the model inputs of benign training flows (box for injected flows)."""
    fl = store.flows("train")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(fl["y"]), generator=g)[:n]
    idx = idx[fl["y_bin"][idx] == 0]
    dev = next(model.parameters()).device
    lo = hi = None
    with torch.no_grad():
        for i in range(0, len(idx), 50_000):
            b = idx[i:i + 50_000]
            x = model.tf(fl["num_raw"][b].to(dev), fl["cat"][b].to(dev))
            lo = x.min(0).values if lo is None else torch.minimum(lo, x.min(0).values)
            hi = x.max(0).values if hi is None else torch.maximum(hi, x.max(0).values)
    return lo, hi


@no_cudnn
def attack_window(model, window: list[Snapshot], counts: list[int], lo: torch.Tensor, hi: torch.Tensor,
                  steps: int = 20, step: float = 0.25, feature: tuple | None = None,
                  objective: str = "evasion") -> list[torch.Tensor]:
    """Adaptive injection: PGD on the inputs of the injected flows (and, if `feature` =
    (threat, threat_model, eps) is given, jointly on the attack flows of the target window).
    objective "evasion" maximises the log-probability that the original attack flows are classified
    as benign; "untargeted" maximises their loss on the true class (any wrong class, e.g. another
    attack type). Returns the model inputs of every snapshot of the attacked window."""
    if objective not in ("evasion", "untargeted"):
        raise ValueError(f"unknown objective {objective!r}")
    tgt = window[-1]
    e = len(tgt.y) - counts[-1]  # original flows of the target come first
    att = torch.zeros(len(tgt.y), dtype=torch.bool, device=tgt.y.device)
    att[:e] = tgt.y_bin[:e] == 1
    with torch.no_grad():
        xs = [model.tf(s.num_raw, s.cat) for s in window]
    if not att.any():
        return xs
    space = None
    if feature is not None:
        threat, tm, eps = feature
        space = make_space(threat, tm, model.tf, tgt.num_raw[:e], tgt.cat[:e], att[:e], eps)
        u = space.project(space.init(True))
    deltas = [torch.zeros(k, x.shape[1], device=x.device) for x, k in zip(xs, counts)]

    def inputs(deltas, u):
        out = []
        for i, (x, d, k) in enumerate(zip(xs, deltas, counts)):
            if i == len(xs) - 1 and space is not None:
                orig = space.apply(u)
                orig = model.tf(orig, tgt.cat[:e]) if space.kind == "num" else orig
                x = torch.cat([orig, x[e:]])
            if k:
                x = torch.cat([x[: len(x) - k], (x[len(x) - k:] + d).clamp(lo, hi)])
            out.append(x)
        return out

    for _ in range(steps):
        deltas = [d.detach().requires_grad_(k > 0) for d, k in zip(deltas, counts)]
        params = [d for d, k in zip(deltas, counts) if k]
        if space is not None:
            u = u.detach().requires_grad_(True)
            params.append(u)
        if not params:
            break
        logits = model.forward_inputs(window, inputs(deltas, u if space is not None else None),
                                      history_grad=True)
        logp = logits.log_softmax(1)[att]
        if objective == "evasion":
            gain = logp[:, 0].sum()
        else:  # classes unseen in training (label -1) count as evasion targets
            y = tgt.y[att]
            gain = -logp.gather(1, y.clamp(min=0)[:, None]).squeeze(1).where(y >= 0, -logp[:, 0]).sum()
        grads = torch.autograd.grad(gain, params, allow_unused=True)
        with torch.no_grad():
            gi = iter(grads)
            for d, k in zip(deltas, counts):
                if k:
                    g = next(gi)
                    if g is not None:
                        d += step * g.sign()
            if space is not None:
                g = next(gi)
                if g is not None:
                    u = space.project(u + (2.5 / steps) * g.sign() * space.active)
    with torch.no_grad():
        if space is not None:
            u = space.project(u.detach(), final=True)
        return [x.detach() for x in inputs([d.detach() for d in deltas], u if space is not None else None)]
