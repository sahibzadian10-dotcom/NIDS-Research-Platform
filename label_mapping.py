"""Common class set mapping. Every decision is documented in the returned table."""
from __future__ import annotations

import re

import pandas as pd

# (normalised raw label) -> (common class, note)
UNSW_MAP = {
    "normal": ("Benign", "exact"),
    "dos": ("DoS", "exact"),
    "reconnaissance": ("Recon", "exact"),
    "exploits": ("Exploit", "approximate: UNSW 'Exploits' vs CIC web attacks are not the same technique"),
    "backdoor": ("Botnet", "approximate: backdoor C2 behaviour treated as bot-like"),
    "backdoors": ("Botnet", "approximate: label spelled 'Backdoors' in some UNSW files"),
}
UNSW_DROP = {
    "generic": "no CIC counterpart",
    "fuzzers": "no CIC counterpart",
    "analysis": "no CIC counterpart",
    "shellcode": "no CIC counterpart",
    "worms": "no CIC counterpart",
}
CIC_MAP = {
    "benign": ("Benign", "exact"),
    "dos hulk": ("DoS", "exact"),
    "ddos": ("DoS", "merged with DoS (volumetric DoS family)"),
    "portscan": ("Recon", "exact"),
    "web attack - brute force": ("Exploit", "approximate: web attacks merged into Exploit"),
    "web attack - xss": ("Exploit", "approximate: web attacks merged into Exploit"),
    "web attack - sql injection": ("Exploit", "approximate: web attacks merged into Exploit"),
    "bot": ("Botnet", "exact"),
    "dos goldeneye": ("DoS", "merged with DoS (application-layer DoS family)"),
    "dos slowloris": ("DoS", "merged with DoS (application-layer DoS family)"),
    "dos slowhttptest": ("DoS", "merged with DoS (application-layer DoS family)"),
}
CIC_DROP = {
    "ftp-patator": "no UNSW counterpart (brute force)",
    "ssh-patator": "no UNSW counterpart (brute force)",
    "infiltration": "no UNSW counterpart",
    "heartbleed": "no UNSW counterpart",
}


def _key(text) -> str:
    """Lower-case and collapse every run of non-alphanumerics to one space.

    Real CIC-IDS2017 files write 'Web Attack \x96 Brute Force' (a cp1252 en dash that decodes to U+FFFD or a dash,
    depending on the reader), so punctuation must not matter for matching.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


UNSW_MAP = {_key(k): v for k, v in UNSW_MAP.items()}
UNSW_DROP = {_key(k): v for k, v in UNSW_DROP.items()}
CIC_MAP = {_key(k): v for k, v in CIC_MAP.items()}
CIC_DROP = {_key(k): v for k, v in CIC_DROP.items()}
MAPS = {"UNSW-NB15": (UNSW_MAP, UNSW_DROP), "CIC-IDS2017": (CIC_MAP, CIC_DROP)}


def normalise(s: pd.Series) -> pd.Series:
    return s.map(_key)


def map_labels(raw: pd.Series, dataset: str) -> tuple[pd.Series, pd.DataFrame]:
    keep, drop = MAPS[dataset]
    norm = normalise(raw)
    mapped = norm.map(lambda k: keep[k][0] if k in keep else None)
    rows = []
    for k, cnt in norm.value_counts().items():
        if k in keep:
            rows.append({"raw_label": k, "common_class": keep[k][0], "rows": int(cnt), "decision": keep[k][1]})
        else:
            rows.append({"raw_label": k, "common_class": "(dropped)", "rows": int(cnt), "decision": drop.get(k, "unmapped label")})
    return mapped, pd.DataFrame(rows)
