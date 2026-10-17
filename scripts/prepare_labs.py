"""Build one harmonised CSV per lab in data/labs/ from the raw public files.

Usage:
  venv/bin/python scripts/prepare_labs.py --brfss PATH/LLCP2023.XPT --pns-dir DIR --knhanes PATH/HN24_ALL.sav --nhanes-dir DIR

Columns in every output: age, male, group, hypertension, stroke, heart_attack, sbp, dbp, weight
(0/1 for conditions, blank = missing / not asked / don't know). Raw data stays out of git.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "labs"
COLS = ["age", "male", "group", "hypertension", "stroke", "heart_attack", "sbp", "dbp", "weight"]


def yn(s, yes, no):
    return np.where(s == yes, 1.0, np.where(s == no, 0.0, np.nan))


def brfss(path):
    keep = ["_AGE80", "SEXVAR", "_IMPRACE", "BPHIGH6", "CVDSTRK3", "CVDINFR4", "_LLCPWT"]
    b = pd.concat(ch[keep] for ch in pd.read_sas(path, format="xport", chunksize=50000, encoding="latin1"))
    return pd.DataFrame({
        "age": b._AGE80, "male": (b.SEXVAR == 1).astype(int),
        "group": b._IMPRACE.map({1: "White", 2: "Black", 3: "Asian", 4: "AIAN", 5: "Hispanic", 6: "Other"}),
        "hypertension": yn(b.BPHIGH6, 1, 3), "stroke": yn(b.CVDSTRK3, 1, 2), "heart_attack": yn(b.CVDINFR4, 1, 2),
        "sbp": np.nan, "dbp": np.nan, "weight": b._LLCPWT})


def pns(d):
    d = Path(d)
    x = pd.read_excel(d / "dicionario_PNS_microdados_2019.xls", header=None, engine="xlrd")
    pos = {}
    for _, r in x.iterrows():
        if pd.notna(r[0]) and pd.notna(r[2]) and str(r[0]).strip().isdigit():
            pos[str(r[2]).strip()] = (int(r[0]), int(r[1]))
    v = ["C006", "C008", "C009", "Q00201", "Q06306", "Q06307", "Q068", "V00291"]
    p = pd.read_fwf(d / "PNS_2019.txt", colspecs=[(pos[k][0] - 1, pos[k][0] - 1 + pos[k][1]) for k in v],
                    names=v, dtype=str, header=None, encoding="latin1")
    for c in p:
        p[c] = pd.to_numeric(p[c].str.strip(), errors="coerce")
    p = p[p.V00291.notna() & p.Q00201.notna()]
    # Q06307 (infarction) is only asked when Q06306 (any heart disease) = yes.
    mi = np.where(p.Q06307 == 1, 1.0, np.where((p.Q06307 == 2) | (p.Q06306 == 2), 0.0, np.nan))
    return pd.DataFrame({
        "age": p.C008, "male": (p.C006 == 1).astype(int),
        "group": p.C009.map({1: "White", 2: "Black", 3: "Asian", 4: "Brown", 5: "Indigenous"}),
        "hypertension": yn(p.Q00201, 1, 2), "stroke": yn(p.Q068, 1, 2), "heart_attack": mi,
        "sbp": np.nan, "dbp": np.nan, "weight": p.V00291})


def knhanes(path):
    import pyreadstat
    v = ["age", "sex", "DI1_dg", "DI3_dg", "DI5_dg", "HE_sbp", "HE_dbp", "wt_itvex"]
    try:
        k, _ = pyreadstat.read_sav(path, usecols=v)
    except Exception:
        k, _ = pyreadstat.read_sav(path, usecols=v, encoding="cp949")
    # 0 = no, 1 = yes; 8/9 (not applicable / unknown) -> missing
    return pd.DataFrame({
        "age": k.age, "male": (k.sex == 1).astype(int), "group": "Korean",
        "hypertension": yn(k.DI1_dg, 1, 0), "stroke": yn(k.DI3_dg, 1, 0), "heart_attack": yn(k.DI5_dg, 1, 0),
        "sbp": k.HE_sbp, "dbp": k.HE_dbp, "weight": k.wt_itvex})


def nhanes(d):
    d = Path(d)
    rd = lambda n: pd.read_sas(next(p for p in d.iterdir() if p.name.lower() == f"{n.lower()}.xpt"), format="xport")
    n = (rd("DEMO_J")[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "WTMEC2YR"]]
         .merge(rd("BPQ_J")[["SEQN", "BPQ020"]], on="SEQN", how="left")
         .merge(rd("MCQ_J")[["SEQN", "MCQ160E", "MCQ160F"]], on="SEQN", how="left")
         .merge(rd("BPX_J")[["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"]], on="SEQN", how="left"))
    return pd.DataFrame({
        "age": n.RIDAGEYR, "male": (n.RIAGENDR == 1).astype(int),
        "group": n.RIDRETH3.map({1: "Hispanic", 2: "Hispanic", 3: "White", 4: "Black", 6: "Asian", 7: "Other"}),
        "hypertension": yn(n.BPQ020, 1, 2), "stroke": yn(n.MCQ160F, 1, 2), "heart_attack": yn(n.MCQ160E, 1, 2),
        "sbp": n[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1),
        "dbp": n[["BPXDI1", "BPXDI2", "BPXDI3"]].replace(0, np.nan).mean(axis=1), "weight": n.WTMEC2YR})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brfss"); ap.add_argument("--pns-dir"); ap.add_argument("--knhanes"); ap.add_argument("--nhanes-dir")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {"US_BRFSS_2023": (brfss, a.brfss), "BRAZIL_PNS_2019": (pns, a.pns_dir),
            "KOREA_KNHANES_2024": (knhanes, a.knhanes), "US_NHANES_2017": (nhanes, a.nhanes_dir)}
    for lab, (fn, src) in jobs.items():
        if not src:
            print(f"skip {lab} (no source given)"); continue
        df = fn(src)[COLS]
        df = df[df.age >= 18]
        df.to_csv(OUT / f"{lab}.csv", index=False)
        print(f"{lab}: {len(df):,} rows -> {OUT / (lab + '.csv')}")
