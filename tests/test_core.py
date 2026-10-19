"""Unit tests: run with `python -m pytest tests -q`."""
import numpy as np
import torch

from stgids.adversarial import ConstrainedSpace, ThreatModel, UnconstrainedSpace, pgd, square
from stgids.config import DEFAULTS
from stgids.data import chronological_split
from stgids.detector import DetectAttribute, MonoDetector
from stgids.graphs import Snapshot
from stgids.inject import add_flows
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


def test_blocked_split_puts_every_block_in_all_splits():
    w = np.repeat(np.arange(90), 3)  # 90 windows -> 3 blocks of 30
    s = chronological_split(w, [0.7, 0.15, 0.15], block_windows=30)
    for win in np.unique(w):
        assert len(set(s[w == win])) == 1, "a window must never be split"
    per_window = s[::3].reshape(3, 30)
    for block in per_window:
        assert list(np.bincount(block, minlength=3)) == [21, 5, 4]
        assert np.all(np.diff(block) >= 0), "inside a block: train, then val, then test"


def test_store_with_blocked_split(tmp_path):
    from stgids.graphs import SnapshotStore
    split = np.array([0, 0, 1, 2, 0, 0, 1, 2])  # two blocks of 4 snapshots, one window each
    sizes = np.array([3, 2, 2, 1, 3, 2, 2, 1])
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    n = int(ptr[-1])
    torch.save({"num_raw": torch.arange(n, dtype=torch.float32)[:, None], "cat": torch.zeros(n, 0),
                "y": torch.zeros(n, dtype=torch.long), "y_bin": torch.zeros(n, dtype=torch.long),
                "src": torch.zeros(n, dtype=torch.long), "dst": torch.ones(n, dtype=torch.long),
                "snap_ptr": torch.as_tensor(ptr), "snap_window": torch.arange(8),
                "snap_split": torch.as_tensor(split), "meta": {"classes": ["Benign"]}}, tmp_path / "p.pt")
    store = SnapshotStore(str(tmp_path / "p.pt"), history_k=6)
    rows = store.flows("train")["num_raw"].flatten().tolist()
    expect = [r for i in np.flatnonzero(split == 0) for r in range(ptr[i], ptr[i + 1])]
    assert rows == expect, "train flows must be exactly the train snapshots' flows"
    # train target 4: history may only contain train snapshots (0, 1), never val/test (2, 3)
    assert [s.lag for s in store.window(4)] == [4, 3, 0]
    # test target 7: any past snapshot within K windows is allowed
    assert [s.lag for s in store.window(7)] == [5, 4, 3, 2, 1, 0]


F_IN = 5  # numeric features of the toy snapshots (identity feature transform, no categoricals)
TOY_TRANSFORM = {"log_mask": np.zeros(F_IN, bool), "mu": np.zeros(F_IN), "sigma": np.ones(F_IN),
                 "selected": np.arange(F_IN)}


def _snapshot(g, lag, n_flows=30, n_hosts=8, host_pool=20):
    hosts = torch.randperm(host_pool, generator=g)[:n_hosts].sort().values
    y = torch.randint(0, 4, (n_flows,), generator=g)
    return Snapshot(torch.randn(n_flows, F_IN, generator=g), torch.zeros(n_flows, 0, dtype=torch.uint8),
                    torch.randint(n_hosts, (n_flows,), generator=g), torch.randint(n_hosts, (n_flows,), generator=g),
                    hosts, y, (y > 0).long(), lag=lag)


def _injected(s, g, m=15, host_pool=25):  # random flows, also from / to hosts never seen before
    return add_flows(s, torch.randint(host_pool, (m,), generator=g), torch.randint(host_pool, (m,), generator=g),
                     torch.randn(m, F_IN, generator=g) * 5, torch.zeros(m, 0, dtype=torch.uint8))


def test_add_flows_keeps_existing_flows():
    g = torch.Generator().manual_seed(0)
    s = _snapshot(g, 0)
    new = _injected(s, g)
    e = len(s.y)
    assert torch.all(new.hosts[1:] > new.hosts[:-1]), "hosts must stay sorted and unique"
    assert torch.equal(new.hosts[new.src[:e]], s.hosts[s.src]) and torch.equal(new.hosts[new.dst[:e]], s.hosts[s.dst])
    assert torch.equal(new.num_raw[:e], s.num_raw) and torch.equal(new.y[:e], s.y)
    assert torch.all(new.y_bin[e:] == 0)


def test_detector_never_decreases_when_flows_are_added():
    g = torch.Generator().manual_seed(1)
    for trial in range(20):
        torch.manual_seed(trial)
        channels = 1 if trial % 2 else 4  # also with extra learned evidence channels
        det = MonoDetector(F_IN, 3, {"hidden": 8, "context_dim": 4, "channels": channels}).double()  # float64
        with torch.no_grad():  # random, also large weights: the guarantee must not depend on training
            for prm in det.parameters():
                prm.add_(torch.randn_like(prm) * 2)
        window = [_snapshot(g, lag) for lag in (2, 1, 0)]
        d0 = det.forward_inputs(window, [s.num_raw.double() for s in window])
        for hist_too in (False, True):
            new = [_injected(s, g) if (hist_too or s is window[-1]) else s for s in window]
            d1 = det.forward_inputs(new, [s.num_raw.double() for s in new])[: len(d0)]
            assert torch.all(d1 >= d0 - 1e-9 * d0.abs().clamp(min=1)), "adding flows lowered a detection score"


def test_detect_attribute_flags_exactly_when_detector_is_positive():
    g = torch.Generator().manual_seed(2)
    torch.manual_seed(0)
    typer = {"arch": "stgids", "hidden": 8, "heads": 2, "gnn_layers": 1, "dropout": 0.0,
             "temporal": "transformer", "temporal_layers": 1, "detach_history": True}
    model = DetectAttribute(TOY_TRANSFORM, 4, {"graph": {"history_k": 3}, "model": {
        "arch": "detect_attribute", "typer": typer, "detector": {"hidden": 8, "context_dim": 4}}}).eval()
    window = [_snapshot(g, lag) for lag in (2, 1, 0)]
    with torch.no_grad():
        xs = [model.tf(s.num_raw, s.cat) for s in window]
        model.detector.own.bias -= model.detector.forward_inputs(window, xs).median()  # both decisions occur
        d = model.detector.forward_inputs(window, xs)
        z = model(window)
        typ = model.typer(window)[:, 1:].argmax(1) + 1
    assert (d > 0).any() and (d <= 0).any()
    assert torch.equal(z.argmax(1), torch.where(d > 0, typ, torch.zeros_like(typ)))


def test_flow_types_and_or_fusion_only_add_alerts_under_injection():
    """With a flow-feature type head, injected flows can move an existing flow only from benign to an
    attack class: never hide a flagged flow and never change its attack type (also with OR fusion)."""
    g = torch.Generator().manual_seed(3)
    for fusion in ("detector", "or"):
        torch.manual_seed(0)
        model = DetectAttribute(TOY_TRANSFORM, 4, {"graph": {"history_k": 3}, "model": {
            "arch": "detect_attribute", "typer": {"arch": "flow", "net": "mlp"},
            "detector": {"hidden": 8, "context_dim": 4, "channels": 3, "fusion": fusion}}}).double().eval()
        window = [_snapshot(g, lag) for lag in (2, 1, 0)]
        new = [_injected(s, g, m=40) for s in window]
        xs = [s.num_raw.double() for s in window]
        with torch.no_grad():
            for prm in model.detector.parameters():
                prm.add_(torch.randn_like(prm))
            # shift both models so that each flags only part of the flows (otherwise nothing can change)
            model.detector.own.bias -= model.detector.forward_inputs(window, xs).quantile(0.7)
            lg = model.typer.forward_inputs(window, xs)
            model.typer.net.net[-1].bias[0] += (lg[:, 1:].max(1).values - lg[:, 0]).quantile(0.7)
            p0 = model.forward_inputs(window, xs).argmax(1)
            p1 = model.forward_inputs(new, [s.num_raw.double() for s in new]).argmax(1)[: len(p0)]
        assert (p0 == 0).any() and (p0 != 0).any()
        changed = p1 != p0
        assert torch.all(p0[changed] == 0), "a flagged flow was hidden or relabelled by injected flows"
