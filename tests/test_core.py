"""Unit tests: run with `python -m pytest tests -q`."""
import numpy as np
import torch

from stgids.adversarial import ConstrainedSpace, ThreatModel, UnconstrainedSpace, pgd, square
from stgids.config import DEFAULTS
from stgids.data import chronological_split
from stgids.losses import FocalLoss
from stgids.model import scatter_softmax

NAMES = ["IN_BYTES", "IN_PKTS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "LONGEST_FLOW_PKT",
         "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_AVG_THROUGHPUT", "OUT_BYTES", "PROTOCOL"]


def _flows(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    pkts = torch.randint(1, 50, (n,), generator=g).float()
    size = torch.randint(40, 1400, (n,), generator=g).float()
    dur = torch.randint(0, 5000, (n,), generator=g).float()
    x = torch.stack([pkts * size, pkts, dur, dur * 0.8, torch.clamp(size * 1.1, max=1500).floor(),
                     dur / pkts, dur / pkts * 1.5, pkts * size * 8 / (dur / 1000 + 1e-3),
                     pkts * size, torch.full((n,), 6.0)], 1)
    return x.floor()


def _space(eps=0.5, n=64):
    tm = ThreatModel(DEFAULTS["threat_model"], NAMES, verbose=False)
    x = _flows(n)
    mask = torch.arange(n) % 2 == 0  # even rows = attack flows
    return tm, x, mask, ConstrainedSpace(tm, x, mask, eps)


def test_constrained_projection_keeps_flows_valid():
    tm, x, mask, sp = _space()
    u = sp.project(torch.rand_like(sp.hi) * 3 - 1, final=True)  # wild values, then project
    adv = sp.apply(u)
    col = {n: i for i, n in enumerate(NAMES)}
    d = adv - x
    assert torch.all(d[~mask] == 0), "benign flows must not change"
    ctrl = tm.cols
    assert torch.all(d[:, ctrl] >= -1e-4), "changes must be additive"
    for fixed in ["OUT_BYTES", "PROTOCOL"]:
        assert torch.all(d[:, col[fixed]] == 0), f"{fixed} is not attacker-controllable"
    for n in DEFAULTS["threat_model"]["integer"]:
        v = adv[:, col[n]]
        assert torch.allclose(v, v.round()), f"{n} must stay integer"
    assert torch.all(adv[:, col["LONGEST_FLOW_PKT"]] <= torch.maximum(x[:, col["LONGEST_FLOW_PKT"]],
                                                                       torch.tensor(1500.0)))
    ok0 = x[:, col["IN_BYTES"]] >= 40 * x[:, col["IN_PKTS"]]
    ok1 = adv[:, col["IN_BYTES"]] >= 40 * adv[:, col["IN_PKTS"]]
    assert torch.all(ok1[ok0]), "bytes >= packets * header must hold after perturbation"


def test_budget_respected():
    tm, x, mask, sp = _space(eps=0.2)
    u = sp.project(torch.ones_like(sp.hi), final=True)
    d = sp.apply(u)[:, tm.cols] - x[:, tm.cols]
    assert torch.all(d <= sp.hi + 1e-3)


def test_dependent_feature_follows_ratio():
    tm, x, mask, sp = _space()
    u = torch.zeros_like(sp.hi)
    u[:, 0] = 1.0  # only IN_BYTES grows
    adv = sp.apply(u)
    i_thr, i_b = NAMES.index("SRC_TO_DST_AVG_THROUGHPUT"), NAMES.index("IN_BYTES")
    expected = x[:, i_thr] * (adv[:, i_b] + 1) / (x[:, i_b] + 1)
    assert torch.allclose(adv[:, i_thr], expected, rtol=1e-5)


def test_pgd_and_square_increase_attacker_gain():
    torch.manual_seed(0)
    n, d, c = 256, 6, 3
    x = torch.randn(n, d)
    w = torch.randn(d, c)
    fwd = lambda kind, t: t @ w  # noqa: E731
    y = (x @ w).argmax(1)
    mask = y != 0
    sp = UnconstrainedSpace(x, mask, eps=1.0)
    clean_benign = ((x @ w).argmax(1)[mask] == 0).float().mean()
    for adv in (pgd(fwd, sp, y, mask, 10, "evasion"), square(fwd, sp, y, mask, 50, "evasion")):
        assert torch.all((adv - x)[~mask] == 0)
        assert (adv - x).abs().max() <= 1.0 + 1e-5
        assert ((adv @ w).argmax(1)[mask] == 0).float().mean() > clean_benign


def test_scatter_softmax_sums_to_one():
    score = torch.randn(10, 2)
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3])
    a = scatter_softmax(score, idx, 4)
    sums = torch.zeros(4, 2).index_add_(0, idx, a)
    assert torch.allclose(sums, torch.ones(4, 2), atol=1e-6)


def test_focal_loss_ignores_unseen_and_reduces_to_ce():
    logits, y = torch.randn(8, 3), torch.tensor([0, 1, 2, -1, 0, 1, 2, -1])
    ce = torch.nn.functional.cross_entropy(logits[y >= 0], y[y >= 0])
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), ce)


def test_chronological_split_never_splits_a_window():
    w = np.repeat(np.arange(20), 5)
    s = chronological_split(w, [0.7, 0.15, 0.15])
    for win in np.unique(w):
        assert len(set(s[w == win])) == 1
    assert np.all(np.diff(s) >= 0), "later windows must never go to an earlier split"
