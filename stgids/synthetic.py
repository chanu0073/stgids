"""Generate a small synthetic NetFlow-v3-style CSV for smoke-testing the pipeline.

The numbers are NOT meant to be realistic enough for research results - they only
exercise every code path (graphs, imbalance, attacks) quickly. Use the real
NF-UNSW-NB15-v3 / NF-CSE-CIC-IDS2018-v3 files for the paper.

    python -m stgids.synthetic --out data/synthetic_nf_v3.csv --flows 40000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SYN, ACK, PSH, FIN, RST = 2, 16, 8, 1, 4


def _flows(rng, n, t, src, dst, dport, proto, in_pkts, pkt_size, out_ratio, dur_ms, cflags, sflags, l7, label):
    in_pkts = np.maximum(1, in_pkts).astype(np.int64)
    in_bytes = np.maximum(in_pkts * 40, (in_pkts * pkt_size).astype(np.int64))
    out_pkts = np.maximum(0, np.round(in_pkts * out_ratio)).astype(np.int64)
    out_bytes = np.where(out_pkts > 0, np.maximum(out_pkts * 40, (out_pkts * pkt_size * 1.5).astype(np.int64)), 0)
    dur = np.maximum(0, dur_ms).astype(np.int64)
    dur_in = (dur * rng.uniform(0.6, 1.0, n)).astype(np.int64)
    dur_out = (dur * rng.uniform(0.3, 1.0, n)).astype(np.int64)
    longest = np.minimum(1500, np.maximum(40, (pkt_size * rng.uniform(1.0, 2.0, n)))).astype(np.int64)
    shortest = np.maximum(40, (pkt_size * rng.uniform(0.3, 1.0, n))).astype(np.int64)
    shortest = np.minimum(shortest, longest)
    secs = np.maximum(dur, 1) / 1000.0
    iat_avg = np.where(in_pkts > 1, dur_in / np.maximum(in_pkts - 1, 1), 0)
    tot_pkts = in_pkts + out_pkts
    size = (in_bytes + out_bytes) / np.maximum(tot_pkts, 1)
    bins = np.zeros((n, 5), dtype=np.int64)
    b = np.digitize(size, [128, 256, 512, 1024])
    bins[np.arange(n), b] = tot_pkts
    return pd.DataFrame({
        "FLOW_START_MILLISECONDS": t.astype(np.int64),
        "FLOW_END_MILLISECONDS": (t + dur).astype(np.int64),
        "IPV4_SRC_ADDR": src, "L4_SRC_PORT": rng.integers(32768, 60999, n),
        "IPV4_DST_ADDR": dst, "L4_DST_PORT": dport,
        "PROTOCOL": proto, "L7_PROTO": l7,
        "IN_BYTES": in_bytes, "IN_PKTS": in_pkts, "OUT_BYTES": out_bytes, "OUT_PKTS": out_pkts,
        "TCP_FLAGS": cflags | sflags, "CLIENT_TCP_FLAGS": cflags, "SERVER_TCP_FLAGS": sflags,
        "FLOW_DURATION_MILLISECONDS": dur, "DURATION_IN": dur_in, "DURATION_OUT": dur_out,
        "MIN_TTL": np.full(n, 63), "MAX_TTL": np.full(n, 64),
        "LONGEST_FLOW_PKT": longest, "SHORTEST_FLOW_PKT": shortest,
        "MIN_IP_PKT_LEN": shortest, "MAX_IP_PKT_LEN": longest,
        "SRC_TO_DST_SECOND_BYTES": in_bytes / secs, "DST_TO_SRC_SECOND_BYTES": out_bytes / secs,
        "RETRANSMITTED_IN_BYTES": 0, "RETRANSMITTED_IN_PKTS": 0,
        "RETRANSMITTED_OUT_BYTES": 0, "RETRANSMITTED_OUT_PKTS": 0,
        "SRC_TO_DST_AVG_THROUGHPUT": (in_bytes * 8 / secs).astype(np.int64),
        "DST_TO_SRC_AVG_THROUGHPUT": (out_bytes * 8 / secs).astype(np.int64),
        "NUM_PKTS_UP_TO_128_BYTES": bins[:, 0], "NUM_PKTS_128_TO_256_BYTES": bins[:, 1],
        "NUM_PKTS_256_TO_512_BYTES": bins[:, 2], "NUM_PKTS_512_TO_1024_BYTES": bins[:, 3],
        "NUM_PKTS_1024_TO_1514_BYTES": bins[:, 4],
        "TCP_WIN_MAX_IN": np.where(proto == 6, 64240, 0), "TCP_WIN_MAX_OUT": np.where(proto == 6, 65160, 0),
        "ICMP_TYPE": 0, "ICMP_IPV4_TYPE": 0,
        "DNS_QUERY_ID": np.where(dport == 53, rng.integers(0, 65535, n), 0),
        "DNS_QUERY_TYPE": np.where(dport == 53, 1, 0), "DNS_TTL_ANSWER": np.where(dport == 53, 300, 0),
        "FTP_COMMAND_RET_CODE": 0,
        "SRC_TO_DST_IAT_MIN": (iat_avg * 0.5).astype(np.int64), "SRC_TO_DST_IAT_MAX": (iat_avg * 1.8).astype(np.int64),
        "SRC_TO_DST_IAT_AVG": iat_avg.astype(np.int64), "SRC_TO_DST_IAT_STDDEV": (iat_avg * 0.3).astype(np.int64),
        "DST_TO_SRC_IAT_MIN": (iat_avg * 0.5).astype(np.int64), "DST_TO_SRC_IAT_MAX": (iat_avg * 2).astype(np.int64),
        "DST_TO_SRC_IAT_AVG": iat_avg.astype(np.int64), "DST_TO_SRC_IAT_STDDEV": (iat_avg * 0.4).astype(np.int64),
        "Label": int(label != "Benign"), "Attack": label,
    })


def generate(n_flows: int = 40000, hours: float = 3.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    span = hours * 3600 * 1000
    t0 = 1_700_000_000_000
    clients = np.array([f"10.0.0.{i}" for i in range(1, 61)])
    servers = np.array([f"10.0.1.{i}" for i in range(1, 11)])
    attackers = np.array([f"172.16.0.{i}" for i in range(1, 6)])
    svc_port = np.array([80, 443, 22, 53, 3306])
    svc_l7 = np.array([7.0, 91.0, 92.0, 5.0, 20.0])
    parts = []

    # benign: clients talk to services all the time
    n = int(n_flows * 0.80)
    s = rng.integers(0, len(svc_port), n)
    parts.append(_flows(rng, n, t0 + rng.uniform(0, span, n), rng.choice(clients, n), rng.choice(servers, n),
                        svc_port[s], np.where(svc_port[s] == 53, 17, 6), rng.lognormal(2.0, 0.8, n),
                        rng.lognormal(6.0, 0.6, n), rng.uniform(0.8, 1.5, n), rng.lognormal(6.0, 1.2, n),
                        np.full(n, SYN | ACK | PSH | FIN), np.full(n, SYN | ACK | PSH | FIN), svc_l7[s], "Benign"))

    def episodes(k, length_ms):
        starts = rng.uniform(0, span - length_ms, k)
        return starts, length_ms

    def burst(label, n, k, length_ms, make):
        st, ln = episodes(k, length_ms)
        ep = rng.integers(0, k, n)
        t = t0 + st[ep] + rng.uniform(0, ln, n)
        parts.append(make(n, t, ep, label))

    # DoS: many small flows at one server during short bursts
    burst("DoS", int(n_flows * 0.08), 12, 90_000, lambda n, t, ep, l: _flows(
        rng, n, t, attackers[ep % 5], servers[ep % 10], np.full(n, 80), np.full(n, 6),
        rng.integers(1, 4, n), rng.uniform(40, 80, n), rng.uniform(0, 0.5, n), rng.uniform(0, 30, n),
        np.full(n, SYN), np.full(n, SYN | ACK | RST), np.full(n, 7.0), l))
    # Reconnaissance: port scan, one tiny flow per port
    burst("Reconnaissance", int(n_flows * 0.06), 12, 120_000, lambda n, t, ep, l: _flows(
        rng, n, t, attackers[(ep + 1) % 5], servers[(ep * 3) % 10], rng.integers(1, 10000, n), np.full(n, 6),
        np.ones(n), rng.uniform(40, 60, n), np.ones(n), rng.uniform(0, 5, n),
        np.full(n, SYN), np.full(n, RST | ACK), np.zeros(n), l))
    # Brute force: repeated SSH logins, normal-looking sizes
    burst("BruteForce", int(n_flows * 0.04), 10, 300_000, lambda n, t, ep, l: _flows(
        rng, n, t, attackers[(ep + 2) % 5], servers[(ep * 7) % 10], np.full(n, 22), np.full(n, 6),
        rng.integers(10, 22, n), rng.uniform(80, 140, n), rng.uniform(0.9, 1.2, n), rng.uniform(800, 2500, n),
        np.full(n, SYN | ACK | PSH | FIN), np.full(n, SYN | ACK | PSH | FIN), np.full(n, 92.0), l))
    # Exploits: few large requests to web servers
    burst("Exploits", int(n_flows * 0.015), 10, 200_000, lambda n, t, ep, l: _flows(
        rng, n, t, attackers[(ep + 3) % 5], servers[(ep * 5) % 10], np.full(n, 80), np.full(n, 6),
        rng.integers(5, 30, n), rng.uniform(600, 1400, n), rng.uniform(0.3, 0.8, n), rng.lognormal(6.5, 0.8, n),
        np.full(n, SYN | ACK | PSH), np.full(n, SYN | ACK | PSH | FIN), np.full(n, 7.0), l))
    # Backdoor: rare, periodic beacons from an internal host to an external address
    nb = max(60, int(n_flows * 0.004))
    t = t0 + np.sort(rng.uniform(0, span, nb))
    parts.append(_flows(rng, nb, t, np.full(nb, "10.0.0.61"), np.full(nb, "172.16.0.9"), np.full(nb, 4444),
                        np.full(nb, 6), rng.integers(3, 6, nb), rng.uniform(90, 120, nb), np.full(nb, 1.0),
                        rng.uniform(50, 150, nb), np.full(nb, SYN | ACK | PSH), np.full(nb, SYN | ACK | PSH),
                        np.zeros(nb), "Backdoor"))

    df = pd.concat(parts, ignore_index=True).sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="data/synthetic_nf_v3.csv")
    p.add_argument("--flows", type=int, default=40000)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    df = generate(a.flows, a.hours, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} flows to {a.out}\n{df['Attack'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
