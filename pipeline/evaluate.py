"""Jury -> scorecard -> conformal licence -> stress, plus baselines. Reads a run dir from run_extract.py.

usage: python evaluate.py --run <dir> [--alpha 0.05] [--seed 7]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURES, IDS, CAP, TOL, W, IS_NUM, PROXY, D

def ci95(p, n):
    se = np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)); return f"{100*p:5.1f}% ±{100*1.96*se:4.1f} (n={n})"

# ----------------------------------------------------------------------------- jury
def reliability(T):
    """No ground-truth facts on real CVs -> reliability = leave-one-out agreement with the others' consensus."""
    n, r, d = T.shape; RHO = np.zeros((r, d))
    for k in range(r):
        others = np.delete(T, k, axis=1)
        cons = np.nanmean(others, 1)
        ok = np.where(IS_NUM[None], np.abs(T[:, k] - cons) <= TOL[None], T[:, k] == np.round(cons))
        valid = ~np.isnan(T[:, k]) & ~np.isnan(cons)
        RHO[k] = np.where(valid.sum(0) > 0, (ok & valid).sum(0) / np.maximum(valid.sum(0), 1), 0.5)
    return np.clip(RHO, 0.05, 1.0)

def vote(T, RHO):
    cnt = (~np.isnan(T)).sum(1)
    w = np.where(np.isnan(T), 0.0, RHO[None])
    C_num = np.nansum(np.nan_to_num(T) * w, 1) / np.maximum(w.sum(1), 1e-9)
    spread = np.nanmax(T, 1) - np.nanmin(T, 1)
    agree_num = spread <= TOL[None]
    C_bin = (np.nansum(np.nan_to_num(T) * w, 1) / np.maximum(w.sum(1), 1e-9) > 0.5).astype(float)
    agree_bin = np.nanmax(T, 1) == np.nanmin(T, 1)
    C = np.where(IS_NUM[None], C_num, C_bin)
    agree = np.where(IS_NUM[None], agree_num, agree_bin)
    U = (cnt < 2) | ~agree
    C = np.where(cnt == 0, np.nan, C)
    return C, agree, U, cnt

# ----------------------------------------------------------------------------- engine
def score(C, w):
    F = np.clip(np.nan_to_num(C, nan=0.0) / CAP[None], 0, 1)
    return F @ w / w.sum(), F

def conformal_threshold(r, alpha):
    n = len(r); k = int(np.ceil((n + 1) * (1 - alpha))); return np.sort(r)[min(k, n) - 1]

def licence(s, y_cal_mask, y, groups, alpha):
    """Per-group thresholds on the calibration half, envelope = max, applied to everyone."""
    r = np.where(y == 1, 1 - s, s)
    ts = {g: conformal_threshold(r[y_cal_mask & (groups == g)], alpha) for g in np.unique(groups)}
    t = max(ts.values())
    in1, in0 = (1 - s) <= t, s <= t
    decided = in1 ^ in0
    decision = np.where(decided, in1.astype(int), 2)          # 2 = refer
    covered = np.where(y == 1, in1, in0)
    return decision, covered, t, ts

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    T = np.load(os.path.join(a.run, "T.npy")); meta = pd.read_csv(os.path.join(a.run, "meta.csv"))
    info = json.load(open(os.path.join(a.run, "run.json")))
    y = meta.label.values; n = len(y)
    readers = info["readers"]
    print(f"run: {info['dataset']}\n{n} real CVs, readers {readers}, target {info['target']}\n")

    # ------------------------------------------------------------- split (frozen by seed)
    perm = rng.permutation(n); cal = np.zeros(n, bool); cal[perm[: n // 2]] = True
    # a NON-protected group to demonstrate per-group calibration: CV length (short vs long)
    raw = json.load(open(os.path.join(a.run, "raw.json")))
    groups = (meta.ID.index % 2).values  # placeholder replaced below
    lengths = np.array([sum(len(str(v)) for v in raw[str(i)].values()) for i in range(n)])
    groups = (lengths > np.median(lengths)).astype(int)  # 0 = short, 1 = long

    # ------------------------------------------------------------- jury
    RHO = reliability(T[cal]) if cal.sum() >= 6 else np.full((len(readers), D), 0.8)
    C, agree, U, cnt = vote(T, RHO)
    print("JURY")
    print("  reader reliability (leave-one-out agreement, calibration half):")
    for k, name in enumerate(readers):
        print(f"    {name:7s} " + " ".join(f"{IDS[j][:9]:>9s}={RHO[k,j]:.2f}" for j in range(D)))
    print(f"  fields found by >=2 readers: {100*(cnt>=2).mean():.0f}%   unanimous/agreed: {100*agree[cnt>=2].mean():.0f}% of those")
    print(f"  unresolved (-> human) per field: " + " ".join(f"{IDS[j][:9]}={100*U[:,j].mean():.0f}%" for j in range(D)))
    print(f"  CVs with any unresolved WEIGHTED field: {100*(U[:, W>0].any(1)).mean():.0f}%\n")

    # ------------------------------------------------------------- scorecard + licence
    s, F = score(C, W)
    thr = np.quantile(s[cal], 1 - y[cal].mean())              # pick threshold on calibration half to match base rate
    hard = (s > thr).astype(int)
    print("SCORECARD (weights from rubric, no training)")
    print(f"  accuracy vs category label, test half : {ci95((hard[~cal]==y[~cal]).mean(), (~cal).sum())}")
    print(f"  precision {100*(y[~cal][hard[~cal]==1]==1).mean():.0f}%  recall {100*(hard[~cal][y[~cal]==1]==1).mean():.0f}%")
    print("  contribution of each weighted feature (mean over shortlisted): " +
          " ".join(f"{IDS[j][:9]}={np.mean(F[hard==1][:, j])*W[j]/W.sum():.2f}" for j in range(D) if W[j] > 0))
    dec, cov, t, ts = licence(s, cal, y, groups, a.alpha)
    n_min = int(np.ceil(1 / a.alpha)) - 1
    n_g = {g: int((cal & (groups == g)).sum()) for g in (0, 1)}
    print(f"\nCONFORMAL LICENCE  alpha={a.alpha}  thresholds per group {{short:{ts[0]:.3f}, long:{ts[1]:.3f}}} -> envelope {t:.3f}")
    if min(n_g.values()) < n_min:
        print(f"  WARNING: alpha={a.alpha} needs >= {n_min} calibration cases per group; have {n_g}. "
              f"Smallest honest alpha with this data = {1/(min(n_g.values())+1):.2f}. Guarantee below is NOT valid at this n.")
    tm = ~cal
    for g, name in [(0, "short CVs"), (1, "long CVs ")]:
        m = tm & (groups == g); d_ = dec[m] != 2
        wrong_all = ((dec[m] != 2) & (dec[m] != y[m])).mean()
        wrong_dec = ((dec[m][d_] != y[m][d_]).mean() if d_.any() else float('nan'))
        print(f"  {name}: coverage {ci95(cov[m].mean(), m.sum())}  wrong(all)={100*wrong_all:4.1f}%  wrong|decided={100*wrong_dec:4.1f}%  referred={100*(~d_).mean():4.1f}%")
    m = tm; d_ = dec[m] != 2
    print(f"  overall  : coverage {ci95(cov[m].mean(), m.sum())}  wrong(all)={100*((dec[m]!=2)&(dec[m]!=y[m])).mean():4.1f}%  referred={100*(~d_).mean():4.1f}%  accuracy|decided={100*(dec[m][d_]==y[m][d_]).mean():.1f}%")

    # ------------------------------------------------------------- baselines
    print("\nBASELINES (test half)")
    kw = T[:, readers.index("regex")] if "regex" in readers else None
    if kw is not None:
        s_kw = np.nan_to_num(kw[:, IDS.index("prog_langs")]) + 3 * np.nan_to_num(kw[:, IDS.index("infra_skills")]) + np.nan_to_num(kw[:, IDS.index("it_certs")])
        t_kw = np.quantile(s_kw[cal], 1 - y[cal].mean()); d_kw = (s_kw > t_kw).astype(int)
        print(f"  keyword ATS (regex reader alone)      : accuracy {ci95((d_kw[tm]==y[tm]).mean(), tm.sum())}")
    jp = os.path.join(a.run, "judge.npy")
    if os.path.exists(jp):
        J = np.load(jp); ok = (J >= 0).all(1)
        maj = (J.sum(1) >= 2).astype(int)
        incons = (J.min(1) != J.max(1))[ok].mean()
        jm = info.get("judge_model", "LLM")
        print(f"  LLM judge ({jm}, majority of 3)   : accuracy {ci95((maj[tm & ok]==y[tm & ok]).mean(), (tm & ok).sum())}   rerun inconsistency {100*incons:.1f}% of CVs got different answers")
    print(f"  ours (scorecard, hard threshold)      : accuracy {ci95((hard[tm]==y[tm]).mean(), tm.sum())}")
    print(f"  ours (with licence, decided only)     : accuracy {100*(dec[m][d_]==y[m][d_]).mean():.1f}%  on {100*d_.mean():.0f}% of cases; rest referred")

    # ------------------------------------------------------------- stress
    k = 10_000
    spread = np.where(cnt >= 2, np.nanmax(T, 1) - np.nanmin(T, 1), TOL[None])
    P = np.clip(F[:, None, :] + rng.uniform(-1, 1, (n, k, D)) * (spread / CAP)[:, None, :], 0, 1)
    flip = rng.random((n, k, PROXY.sum())) < 0.5
    P[..., PROXY] = np.where(flip, 1 - P[..., PROXY], P[..., PROXY])
    S = np.einsum("nkd,d->nk", P, W) / W.sum()
    stab = ((S > thr) == (s > thr)[:, None]).mean(1)
    print(f"\nSTRESS  {n*k:,} what-ifs: median stability {np.median(stab):.3f}; least stable 10% of CVs have stability < {np.quantile(stab, 0.1):.2f}")
    print(f"  proxy-only flips move the score by max {np.abs(np.einsum('nkd,d->nk', P[..., PROXY], W[PROXY])).max():.1e} (weight 0 => 0)")
    unstable = np.argsort(stab)[:3]
    print("  three least stable CVs (score near the threshold, jury disagreement wide): idx", unstable.tolist(), "scores", np.round(s[unstable], 3).tolist())

    if "usage_tokens" in info and info["usage_tokens"].get("claude"):
        i_t, o_t = info["usage_tokens"]["claude"]
        cost = i_t / 1e6 * 0.10 + o_t / 1e6 * 0.50
        print(f"\nCOST  Claude Haiku 5.5 reader: {i_t:,} in / {o_t:,} out tokens = ${cost:.3f} total = ${cost/n*100:.3f} per 100 CVs; local + regex readers = $0")

if __name__ == "__main__":
    main()
