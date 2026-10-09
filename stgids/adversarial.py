"""Adversarial attacks under a realistic (constrained) and an unconstrained threat model.

Constrained threat model (used for adversarial training and the main robustness results):
  * only flows sent by the attacker (true label = attack) are modified;
  * only attacker-controllable features change (src->dst bytes, packets, timing,
    packet length); protocol, ports and TCP flags stay fixed;
  * changes are additive only (padding bytes, extra packets, delays);
  * budget per feature: delta <= eps * max(x, abs_floor), capped by physical limits (MTU);
  * dependent features (e.g. throughput) are recomputed with a ratio rule;
  * validity: integers stay integers, bytes >= packets * min header size, etc.
Unconstrained threat model (what most papers report): L-inf eps on every
standardised model input of the attack flows, no validity checks.

All attacks optimise a variable u that is mapped to a perturbation by the
"space" object: constrained delta = u * hi with u in [0, 1]; unconstrained
delta = u * eps with u in [-1, 1].

Attacks: FGSM, PGD (Madry et al. 2018), DeepFool (batched approximation: each
flow uses its own gradient row, cross-flow effects through the graph are
ignored in the step size), and a Square-Attack-style score-based random search
(black-box: uses only output scores, no gradients).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .utils import LOG

Forward = Callable[[str, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- threat model
class ThreatModel:
    def __init__(self, tcfg: dict, num_names: list[str], verbose: bool = True):
        col = {n: i for i, n in enumerate(num_names)}
        names = [n for n in tcfg["controllable"] if n in col]
        missing = [n for n in tcfg["controllable"] if n not in col]
        if missing and verbose:
            LOG.warning(f"controllable features not in data (ignored): {missing}")
        if not names:
            raise ValueError("none of the controllable features exist in the data")
        self.names = names
        self.cols = torch.tensor([col[n] for n in names])
        pos = {n: i for i, n in enumerate(names)}
        self.is_int = torch.tensor([n in set(tcfg.get("integer", [])) for n in names])
        self.floor = torch.tensor([float(tcfg.get("abs_floor", {}).get(n, 1.0)) for n in names])
        self.ub = torch.tensor([float(tcfg.get("upper_bound", {}).get(n, float("inf"))) for n in names])
        self.deps = []
        for dep, powers in (tcfg.get("dependent") or {}).items():
            if dep in col and all(p in pos for p in powers):
                self.deps.append((col[dep], [(pos[p], float(w)) for p, w in powers.items()]))
        self.ge = []
        for a, b, c in tcfg.get("ge_constraints") or []:
            ref = lambda n: ("c", pos[n]) if n in pos else (("f", col[n]) if n in col else None)  # noqa: E731
            ra, rb = ref(a), ref(b)
            if ra and rb and (ra[0] == "c" or rb[0] == "c"):
                self.ge.append((ra, rb, float(c)))
        self.objective = tcfg.get("objective", "evasion")


class ConstrainedSpace:
    kind = "num"
    vertices = (0.0, 1.0)

    def __init__(self, tm: ThreatModel, num_raw: torch.Tensor, mask: torch.Tensor, eps: float):
        dev = num_raw.device
        self.tm, self.num_raw = tm, num_raw
        self.cols = tm.cols.to(dev)
        self.is_int, floor, ub = tm.is_int.to(dev), tm.floor.to(dev), tm.ub.to(dev)
        self.x0 = num_raw[:, self.cols]
        budget = eps * torch.maximum(self.x0, floor)
        room = (ub - self.x0).clamp(min=0)
        self.hi = torch.minimum(budget, room) * mask[:, None].float()
        self.active = self.hi > 0

    def init(self, random: bool) -> torch.Tensor:
        u = torch.rand_like(self.hi) if random else torch.zeros_like(self.hi)
        return u * self.active

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        x = self.x0 + u * self.hi
        out = self.num_raw.clone()
        out[:, self.cols] = x
        for dep, powers in self.tm.deps:
            ratio = torch.ones_like(x[:, 0])
            for p, w in powers:
                ratio = ratio * ((x[:, p] + 1) / (self.x0[:, p] + 1)) ** w
            out[:, dep] = self.num_raw[:, dep] * ratio
        return out

    @torch.no_grad()
    def project(self, u: torch.Tensor, final: bool = False) -> torch.Tensor:
        delta = u.clamp(0, 1) * self.hi
        if final:
            delta = torch.where(self.is_int, torch.floor(delta), delta)
        x = self.x0 + delta
        cap = self.x0 + self.hi
        for _ in range(2):
            for ra, rb, c in self.tm.ge:
                val = lambda r, src: src[:, r[1]] if r[0] == "c" else self.num_raw[:, r[1]]  # noqa: E731
                va, vb = val(ra, x), val(rb, x)
                va0, vb0 = val(ra, self.x0), val(rb, self.x0)
                viol = (va < c * vb - 1e-6) & (va0 >= c * vb0 - 1e-6)
                if not viol.any():
                    continue
                if ra[0] == "c":
                    need = c * vb
                    if final and self.is_int[ra[1]]:
                        need = torch.ceil(need - 1e-6)
                    new_a = torch.minimum(torch.maximum(va, need), cap[:, ra[1]])
                    x[:, ra[1]] = torch.where(viol, new_a, va)
                    va = x[:, ra[1]]
                    viol = viol & (va < c * vb - 1e-6)
                if rb[0] == "c" and viol.any():
                    new_b = va / c
                    if final and self.is_int[rb[1]]:
                        new_b = torch.floor(new_b + 1e-6)
                    new_b = torch.maximum(new_b, self.x0[:, rb[1]])
                    x[:, rb[1]] = torch.where(viol, new_b, vb)
        delta = x - self.x0
        return torch.where(self.active, delta / self.hi.clamp(min=1e-12), torch.zeros_like(delta))


class UnconstrainedSpace:
    kind = "x"
    vertices = (-1.0, 1.0)

    def __init__(self, x0: torch.Tensor, mask: torch.Tensor, eps: float):
        self.x0, self.eps = x0.detach(), eps
        self.active = mask[:, None].expand_as(x0).clone()

    def init(self, random: bool) -> torch.Tensor:
        u = (torch.rand_like(self.x0) * 2 - 1) if random else torch.zeros_like(self.x0)
        return u * self.active

    def apply(self, u):
        return self.x0 + u * self.eps

    @torch.no_grad()
    def project(self, u, final: bool = False):
        return u.clamp(-1, 1) * self.active


def make_space(threat: str, tm: ThreatModel, transform, num_raw, cat, mask, eps):
    if threat == "constrained":
        return ConstrainedSpace(tm, num_raw, mask, eps)
    if threat == "unconstrained":
        with torch.no_grad():
            x0 = transform(num_raw, cat)
        return UnconstrainedSpace(x0, mask, eps)
    raise ValueError(f"unknown threat model {threat!r}")


# --------------------------------------------------------------------------- objectives
def _gain(logits, y, mask, objective, benign=0):
    """Scalar the attacker maximises (PGD / FGSM)."""
    logp = F.log_softmax(logits, dim=1)
    if objective == "evasion":
        return logp[mask, benign].sum()
    seen = mask & (y >= 0)
    unseen = mask & (y < 0)
    return F.nll_loss(logp[seen], y[seen], reduction="sum") + logp[unseen, benign].sum()


def _margin(logits, y, objective, benign=0):
    """Per-flow margin the attacker maximises (> 0 means the attack succeeded)."""
    other = logits.clone()
    other[:, benign] = float("-inf")
    m_evasion = logits[:, benign] - other.max(1).values
    if objective == "evasion":
        return m_evasion
    yy = y.clamp(min=0)
    other = logits.clone()
    other.scatter_(1, yy[:, None], float("-inf"))
    m = other.max(1).values - logits.gather(1, yy[:, None]).squeeze(1)
    return torch.where(y >= 0, m, m_evasion)


# --------------------------------------------------------------------------- attacks
def no_cudnn(fn):
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            return fn(*args, **kwargs)
    return wrapper


@no_cudnn
def pgd(fwd: Forward, space, y, mask, steps: int, objective: str, rand_init: bool = True,
        alpha: float | None = None) -> torch.Tensor:
    alpha = alpha if alpha is not None else 2.5 / max(steps, 1)
    u = space.project(space.init(rand_init))
    for _ in range(steps):
        u = u.detach().requires_grad_(True)
        gain = _gain(fwd(space.kind, space.apply(u)), y, mask, objective)
        (g,) = torch.autograd.grad(gain, u)
        u = space.project(u.detach() + alpha * g.sign() * space.active)
    return space.apply(space.project(u, final=True)).detach()


def fgsm(fwd: Forward, space, y, mask, objective: str) -> torch.Tensor:
    return pgd(fwd, space, y, mask, steps=1, objective=objective, rand_init=False, alpha=1.0)

@no_cudnn
def deepfool(fwd: Forward, space, y, mask, iters: int, objective: str,
             overshoot: float = 0.02, benign: int = 0) -> torch.Tensor:
    u = space.init(False)
    for _ in range(iters):
        u = u.detach().requires_grad_(True)
        logits = fwd(space.kind, space.apply(u))
        pred = logits.argmax(1)
        if objective == "evasion":
            active = mask & (pred != benign)
            diff = logits[:, benign] - logits.gather(1, pred[:, None]).squeeze(1)
        else:
            active = mask & (y >= 0) & (pred == y)
            other = logits.clone()
            other.scatter_(1, y.clamp(min=0)[:, None], float("-inf"))
            diff = other.max(1).values - logits.gather(1, y.clamp(min=0)[:, None]).squeeze(1)
        if not active.any():
            break
        (w,) = torch.autograd.grad(diff[active].sum(), u)
        w = w * space.active
        gap = (-diff).clamp(min=0).detach()
        step = (gap + 1e-4)[:, None] * w / (w.pow(2).sum(1, keepdim=True) + 1e-12)
        u = space.project(u.detach() + (1 + overshoot) * step * active[:, None])
    return space.apply(space.project(u.detach(), final=True)).detach()


@torch.no_grad()
def square(fwd: Forward, space, y, mask, iters: int, objective: str, p_init: float = 0.3,
           seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device=space.active.device).manual_seed(seed)
    lo, hi = space.vertices
    rnd = lambda: torch.rand(space.active.shape, generator=gen, device=space.active.device)  # noqa: E731
    best_u = space.project(space.init(False) if lo == 0 else torch.where(rnd() < 0.5, lo, hi) * space.active)
    best = _margin(fwd(space.kind, space.apply(best_u)), y, objective)
    for i in range(iters):
        frac = i / max(iters, 1)
        p = p_init / (2 ** sum(frac > b for b in (0.1, 0.25, 0.5, 0.8)))
        choose = (rnd() < p) & space.active
        cand = space.project(torch.where(choose, torch.where(rnd() < 0.5, lo, hi), best_u))
        m = _margin(fwd(space.kind, space.apply(cand)), y, objective)
        better = (m > best) & mask
        best_u[better] = cand[better]
        best[better] = m[better]
    return space.apply(space.project(best_u, final=True))


def run_attack(name: str, fwd: Forward, space, y, mask, ecfg: dict, objective: str, seed: int = 0):
    if name == "fgsm":
        return fgsm(fwd, space, y, mask, objective)
    if name == "pgd":
        return pgd(fwd, space, y, mask, ecfg["pgd_steps"], objective)
    if name == "deepfool":
        return deepfool(fwd, space, y, mask, ecfg["deepfool_iters"], objective)
    if name == "square":
        return square(fwd, space, y, mask, ecfg["square_iters"], objective, seed=seed)
    raise ValueError(f"unknown attack {name!r}")


GRADIENT_FREE = {"square"}
