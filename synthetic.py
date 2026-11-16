"""Schema-faithful SIMULATED stand-ins for UNSW-NB15 and CIC-IDS2017.

IMPORTANT: this is NOT real traffic.  It writes CSV files that use each dataset's
NATIVE column names, units and label strings (including the well-known quirks:
leading spaces in CIC headers, 'Backdoors', microsecond durations, inf/NaN rates,
negative durations, duplicates, hex ports) so that the *real* loader, cleaning and
alignment code is exercised.  Class-conditional distributions differ between the two
generators on purpose, which creates covariate/concept shift for the cross-dataset
experiments.  Replace the CSVs in datasets/raw/ with the real files to run on real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WK_PORTS = [80, 443, 53, 22, 21, 25, 445, 110]  # all < 1024 (well-known class)
EPOCH0_UNSW = 1421927414  # arbitrary capture start (epoch seconds)
EPOCH0_CIC = 1499155800

# Each spec: log-normal (mu, sigma) in natural-log space for duration (s), packets, mean packet size (B);
# proto = P(tcp, udp, icmp); port = P(well-known, registered, ephemeral); sess = number of sessions.
UNSW_SPEC = {
    "Normal": dict(share=0.775, dur=(-1.2, 1.8), ip=(1.7, 1.0), op=(1.8, 1.0), isz=(4.6, 0.9), osz=(5.5, 1.1), proto=(0.72, 0.26, 0.02), port=(0.60, 0.25, 0.15), tail=0.0, sess=900, attacker=False),
    "Generic": dict(share=0.070, dur=(-3.0, 0.5), ip=(0.9, 0.3), op=(0.9, 0.3), isz=(4.3, 0.2), osz=(4.5, 0.3), proto=(0.0, 1.0, 0.0), port=(0.9, 0.05, 0.05), tail=0.0, sess=40, attacker=True),
    "Exploits": dict(share=0.050, dur=(-0.5, 0.5), ip=(2.4, 0.3), op=(2.3, 0.3), isz=(5.7, 0.2), osz=(6.3, 0.25), proto=(0.95, 0.05, 0.0), port=(0.50, 0.40, 0.10), tail=0.0, sess=50, attacker=True),
    "Fuzzers": dict(share=0.030, dur=(-1.5, 0.8), ip=(1.4, 0.5), op=(1.2, 0.5), isz=(4.9, 0.4), osz=(4.8, 0.5), proto=(0.9, 0.1, 0.0), port=(0.4, 0.4, 0.2), tail=0.0, sess=30, attacker=True),
    "DoS": dict(share=0.030, dur=(-2.0, 0.5), ip=(3.2, 0.35), op=(0.4, 0.5), isz=(4.2, 0.2), osz=(3.8, 0.5), proto=(0.55, 0.40, 0.05), port=(0.85, 0.10, 0.05), tail=0.0, sess=30, attacker=True),
    "Reconnaissance": dict(share=0.025, dur=(-3.5, 0.6), ip=(0.7, 0.3), op=(0.3, 0.4), isz=(3.9, 0.15), osz=(3.8, 0.5), proto=(0.80, 0.15, 0.05), port=(0.30, 0.50, 0.20), tail=0.0, sess=30, attacker=True),
    "Analysis": dict(share=0.008, dur=(-1.0, 0.7), ip=(1.0, 0.4), op=(0.8, 0.5), isz=(4.5, 0.4), osz=(4.4, 0.5), proto=(0.9, 0.1, 0.0), port=(0.3, 0.5, 0.2), tail=0.0, sess=15, attacker=True),
    "Backdoors": dict(share=0.012, dur=(0.0, 0.4), ip=(1.6, 0.2), op=(1.5, 0.2), isz=(4.9, 0.15), osz=(5.0, 0.2), proto=(0.85, 0.15, 0.0), port=(0.30, 0.35, 0.35), tail=0.0, sess=20, attacker=True),
}

CIC_SPEC = {
    "BENIGN": dict(share=0.830, dur=(0.5, 2.5), ip=(1.6, 1.3), op=(1.5, 1.4), isz=(4.4, 1.0), osz=(5.2, 1.4), proto=(0.88, 0.11, 0.01), port=(0.72, 0.12, 0.16), tail=0.08, sess=900, attacker=False),
    "DoS Hulk": dict(share=0.030, dur=(1.5, 0.6), ip=(2.6, 0.4), op=(1.0, 0.4), isz=(4.5, 0.25), osz=(6.0, 0.4), proto=(1.0, 0.0, 0.0), port=(0.97, 0.02, 0.01), tail=0.0, sess=30, attacker=True),
    "DDoS": dict(share=0.025, dur=(-0.5, 0.6), ip=(2.9, 0.4), op=(0.8, 0.5), isz=(4.3, 0.2), osz=(5.0, 0.5), proto=(1.0, 0.0, 0.0), port=(0.98, 0.01, 0.01), tail=0.0, sess=30, attacker=True),
    "PortScan": dict(share=0.040, dur=(-4.0, 0.8), ip=(0.4, 0.3), op=(0.2, 0.4), isz=(3.8, 0.1), osz=(3.5, 0.5), proto=(0.98, 0.02, 0.0), port=(0.35, 0.45, 0.20), tail=0.0, sess=30, attacker=True),
    "Web Attack - Brute Force": dict(share=0.008, dur=(0.2, 0.7), ip=(2.0, 0.4), op=(2.0, 0.4), isz=(5.9, 0.3), osz=(7.0, 0.4), proto=(1.0, 0.0, 0.0), port=(0.98, 0.01, 0.01), tail=0.0, sess=10, attacker=True),
    "Web Attack - XSS": dict(share=0.004, dur=(0.2, 0.7), ip=(2.0, 0.4), op=(2.0, 0.4), isz=(5.9, 0.3), osz=(7.0, 0.4), proto=(1.0, 0.0, 0.0), port=(0.98, 0.01, 0.01), tail=0.0, sess=10, attacker=True),
    "Web Attack - Sql Injection": dict(share=0.003, dur=(0.2, 0.7), ip=(2.0, 0.4), op=(2.0, 0.4), isz=(5.9, 0.3), osz=(7.0, 0.4), proto=(1.0, 0.0, 0.0), port=(0.98, 0.01, 0.01), tail=0.0, sess=8, attacker=True),
    "Bot": dict(share=0.012, dur=(1.2, 0.5), ip=(1.4, 0.25), op=(1.5, 0.25), isz=(4.7, 0.18), osz=(5.4, 0.22), proto=(0.90, 0.10, 0.0), port=(0.30, 0.50, 0.20), tail=0.0, sess=15, attacker=True),
    "FTP-Patator": dict(share=0.025, dur=(-1.0, 0.5), ip=(1.6, 0.3), op=(1.6, 0.3), isz=(4.3, 0.2), osz=(4.4, 0.3), proto=(1.0, 0.0, 0.0), port=(1.0, 0.0, 0.0), tail=0.0, sess=15, attacker=True),
    "SSH-Patator": dict(share=0.023, dur=(0.3, 0.5), ip=(2.4, 0.3), op=(2.4, 0.3), isz=(4.6, 0.2), osz=(4.8, 0.3), proto=(1.0, 0.0, 0.0), port=(1.0, 0.0, 0.0), tail=0.0, sess=15, attacker=True),
}


def _ln(rng, n, mu_sd):
    return np.exp(rng.normal(mu_sd[0], mu_sd[1], n))


def _sample_class(rng: np.random.Generator, n: int, p: dict) -> dict:
    dur, ip, op = _ln(rng, n, p["dur"]), _ln(rng, n, p["ip"]), _ln(rng, n, p["op"])
    if p["tail"] > 0:  # a share of long-lived benign flows (CIC has many)
        m = rng.random(n) < p["tail"]
        dur[m] *= np.exp(3.5)
        ip[m] *= np.exp(1.2)
        op[m] *= np.exp(1.2)
    in_pkts = np.maximum(1, np.floor(ip)).astype(int)
    out_pkts = np.maximum(0, np.floor(op)).astype(int)
    isz = np.clip(_ln(rng, n, p["isz"]), 40, 1500)
    osz = np.clip(_ln(rng, n, p["osz"]), 40, 1500)
    in_bytes = np.round(in_pkts * isz).astype(int)
    out_bytes = np.round(out_pkts * osz).astype(int)
    dur = np.where(in_pkts + out_pkts <= 1, 0.0, np.maximum(dur, 1e-6))
    proto = rng.choice(3, size=n, p=np.array(p["proto"]) / sum(p["proto"]))
    pc = rng.choice(3, size=n, p=np.array(p["port"]) / sum(p["port"]))
    dport = np.where(pc == 0, rng.choice(WK_PORTS, size=n), np.where(pc == 1, rng.integers(1024, 49152, n), rng.integers(49152, 65535, n)))
    return dict(dur=dur, in_pkts=in_pkts, out_pkts=out_pkts, in_bytes=in_bytes, out_bytes=out_bytes, proto=proto, dport=dport)


def _generate(spec: dict, n_total: int, rng: np.random.Generator, ip_prefix_benign: str, ip_prefix_attack: str, epoch0: int) -> pd.DataFrame:
    names = list(spec)
    counts = np.array([int(round(spec[k]["share"] * n_total)) for k in names])
    counts[0] += n_total - counts.sum()
    frames = []
    for name, n in zip(names, counts):
        p = spec[name]
        d = pd.DataFrame(_sample_class(rng, int(n), p))
        # sessions: attackers use a few IPs, repeated over many 10-minute buckets
        n_sess = p["sess"]
        sess_bucket = rng.integers(0, 288, n_sess)  # 2 days of 10-min buckets
        if p["attacker"]:
            sess_ip = [f"{ip_prefix_attack}{rng.integers(1, 5)}" for _ in range(n_sess)]
        else:
            sess_ip = [f"{ip_prefix_benign}{rng.integers(0, 40)}.{rng.integers(1, 250)}" for _ in range(n_sess)]
        weights = rng.dirichlet(np.ones(n_sess) * 2.0)
        sid = rng.choice(n_sess, size=int(n), p=weights)
        d["srcip"] = np.array(sess_ip)[sid]
        d["ts"] = epoch0 + sess_bucket[sid] * 600 + rng.integers(0, 600, int(n))
        d["raw_label"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)


def generate_unsw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = _generate(UNSW_SPEC, n, rng, "59.166.", "175.45.176.", EPOCH0_UNSW)
    attack = d["raw_label"] != "Normal"
    ttl = np.where(attack, rng.choice([31, 62, 254], size=len(d), p=[0.05, 0.15, 0.80]), rng.choice([31, 62, 254], size=len(d), p=[0.25, 0.70, 0.05]))
    dur = d["dur"].to_numpy()
    out = pd.DataFrame(
        {
            "srcip": d["srcip"],
            "dsport": d["dport"].astype(object),
            "proto": np.array(["tcp", "udp", "icmp"])[d["proto"]],
            "state": rng.choice(["FIN", "CON", "INT", "REQ", "RST"], size=len(d)),
            "dur": dur,
            "sbytes": d["in_bytes"],
            "dbytes": d["out_bytes"],
            "spkts": d["in_pkts"],
            "dpkts": d["out_pkts"],
            "sttl": ttl,
            "sload": np.where(dur > 0, d["in_bytes"] * 8 / np.maximum(dur, 1e-9), 0.0),
            "smean": np.round(d["in_bytes"] / np.maximum(d["in_pkts"], 1)).astype(int),
            "dmean": np.round(d["out_bytes"] / np.maximum(d["out_pkts"], 1)).astype(int),
            "Stime": d["ts"],
            "attack_cat": d["raw_label"].replace({"Fuzzers": " Fuzzers "}),
        }
    )
    out["Label"] = (d["raw_label"] != "Normal").astype(int)
    # injected quality problems
    dup = out.sample(frac=0.010, random_state=seed)
    bad = rng.random(len(out)) < 0.001
    out.loc[bad, "dsport"] = "0x000b"  # hex ports exist in real UNSW-NB15 CSVs
    return pd.concat([out, dup], ignore_index=True).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def generate_cic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = _generate(CIC_SPEC, n, rng, "192.168.", "172.16.0.", EPOCH0_CIC)
    dur_s = d["dur"].to_numpy()
    fwd_payload = np.maximum(0, d["in_bytes"] - 54 * d["in_pkts"])  # CIC lengths are payload-only
    bwd_payload = np.maximum(0, d["out_bytes"] - 54 * d["out_pkts"])
    tot_pkts = d["in_pkts"] + d["out_pkts"]
    tot_pay = fwd_payload + bwd_payload
    with np.errstate(divide="ignore", invalid="ignore"):
        bps = np.where(dur_s > 0, tot_pay / dur_s, np.where(tot_pay > 0, np.inf, np.nan))
        pps = np.where(dur_s > 0, tot_pkts / dur_s, np.inf)
    ts = pd.to_datetime(d["ts"], unit="s")
    out = pd.DataFrame(
        {
            "Flow ID": [f"{a}-{b}" for a, b in zip(d["srcip"], d["dport"])],
            " Source IP": d["srcip"],
            " Destination Port": d["dport"],
            " Protocol": np.array([6, 17, 1])[d["proto"]],
            " Timestamp": ts.dt.strftime("%d/%m/%Y %H:%M"),
            " Flow Duration": np.round(dur_s * 1e6).astype("int64"),  # microseconds
            " Total Fwd Packets": d["in_pkts"],
            " Total Backward Packets": d["out_pkts"],
            "Total Length of Fwd Packets": fwd_payload,
            " Total Length of Bwd Packets": bwd_payload,
            "Flow Bytes/s": bps,
            " Flow Packets/s": pps,
            " Flow IAT Mean": np.where(tot_pkts > 1, dur_s * 1e6 / np.maximum(tot_pkts - 1, 1), 0.0),
            " Fwd Header Length": 32 * d["in_pkts"],
            " SYN Flag Count": (rng.random(len(d)) < 0.3).astype(int),
            " Average Packet Size": np.where(tot_pkts > 0, tot_pay / np.maximum(tot_pkts, 1), 0.0),
            " Label": d["raw_label"],
        }
    )
    bad = rng.random(len(out)) < 0.004
    out.loc[bad, " Flow Duration"] = -1  # negative durations exist in real CIC-IDS2017
    nan_rows = rng.random(len(out)) < 0.001
    out.loc[nan_rows, "Flow Bytes/s"] = np.nan
    dup = out.sample(frac=0.020, random_state=seed)
    return pd.concat([out, dup], ignore_index=True).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)


def write_raw_datasets(raw_dir: Path, cfg: dict) -> dict:
    raw_dir = Path(raw_dir)
    (raw_dir / "UNSW-NB15").mkdir(parents=True, exist_ok=True)
    (raw_dir / "CIC-IDS2017").mkdir(parents=True, exist_ok=True)
    seed = cfg["dataset_seed"]
    u = generate_unsw(cfg["n_flows"]["UNSW-NB15"], seed)
    c = generate_cic(cfg["n_flows"]["CIC-IDS2017"], seed + 1)
    u.to_csv(raw_dir / "UNSW-NB15" / "UNSW_NB15_SIMULATED.csv", index=False)
    c.to_csv(raw_dir / "CIC-IDS2017" / "CIC_IDS2017_SIMULATED.csv", index=False)
    return {"UNSW-NB15": len(u), "CIC-IDS2017": len(c)}


if __name__ == "__main__":
    from src.config import RAW_DIR, load_config

    print(write_raw_datasets(RAW_DIR, load_config()))
