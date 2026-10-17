"""Deterministic statistics on aggregates only. No AI here: these numbers set the verdict ceiling."""
from __future__ import annotations

import math

from scipy import stats as st


def ci(log_est: float, se: float) -> tuple[float, float, float]:
    return math.exp(log_est), math.exp(log_est - 1.96 * se), math.exp(log_est + 1.96 * se)


def p_from_z(log_est: float, se: float) -> float:
    return float(2 * st.norm.sf(abs(log_est / se))) if se > 0 else float("nan")


def dersimonian_laird(logs: list[float], ses: list[float]) -> dict:
    """Random-effects meta-analysis of log estimates."""
    k = len(logs)
    if k == 0:
        return {}
    if k == 1:
        e, lo, hi = ci(logs[0], ses[0])
        return {"k": 1, "log": logs[0], "se": ses[0], "estimate": e, "ci_low": lo, "ci_high": hi,
                "Q": 0.0, "Q_p": 1.0, "I2": 0.0, "p": p_from_z(logs[0], ses[0])}
    w = [1 / s ** 2 for s in ses]
    fe = sum(wi * yi for wi, yi in zip(w, logs)) / sum(w)
    Q = sum(wi * (yi - fe) ** 2 for wi, yi in zip(w, logs))
    df = k - 1
    C = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr = [1 / (s ** 2 + tau2) for s in ses]
    re = sum(wi * yi for wi, yi in zip(wr, logs)) / sum(wr)
    se = math.sqrt(1 / sum(wr))
    e, lo, hi = ci(re, se)
    return {"k": k, "log": re, "se": se, "estimate": e, "ci_low": lo, "ci_high": hi, "tau2": tau2,
            "Q": Q, "Q_p": float(st.chi2.sf(Q, df)), "I2": max(0.0, (Q - df) / Q) if Q > 0 else 0.0,
            "p": p_from_z(re, se)}


def mantel_haenszel(group: dict[str, tuple[int, int]], ref: dict[str, tuple[int, int]]) -> dict:
    """MH odds ratio of 'a vs b' in group relative to reference, over shared strata.

    group/ref map stratum label -> (a, b). Variance: Robins-Breslow-Greenland.
    """
    shared = sorted(set(group) & set(ref))
    R = S = PR = PS_QR = QS = 0.0
    for s in shared:
        A, B = group[s]
        C, D = ref[s]
        N = A + B + C + D
        if N == 0:
            continue
        r, s_ = A * D / N, B * C / N
        P, Qv = (A + D) / N, (B + C) / N
        R += r; S += s_; PR += P * r; PS_QR += P * s_ + Qv * r; QS += Qv * s_
    if R == 0 or S == 0:
        return {"strata_used": shared, "estimate": None}
    log = math.log(R / S)
    var = PR / (2 * R ** 2) + PS_QR / (2 * R * S) + QS / (2 * S ** 2)
    se = math.sqrt(var)
    e, lo, hi = ci(log, se)
    return {"strata_used": shared, "log": log, "se": se, "estimate": e, "ci_low": lo, "ci_high": hi,
            "p": p_from_z(log, se)}


def ratio(log_a: float, se_a: float, log_b: float, se_b: float) -> dict:
    """Ratio of two estimates (a / b), treating them as independent (conservative if they share a reference)."""
    log, se = log_a - log_b, math.sqrt(se_a ** 2 + se_b ** 2)
    e, lo, hi = ci(log, se)
    return {"log": log, "se": se, "estimate": e, "ci_low": lo, "ci_high": hi, "p": p_from_z(log, se)}


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


# ---------------------------------------------------------------- verdict ceilings (the rules)
def ceiling_for_effect(est: dict | None, null: float = 1.0) -> tuple[str, str]:
    """Is there an effect in the stated direction (estimate > null)?"""
    if not est or est.get("estimate") is None:
        return "inconclusive", "No estimate could be released (gate refusal or no shared strata)."
    lo, hi = est["ci_low"], est["ci_high"]
    if lo > null:
        return "supported", f"95% CI {lo:.2f}-{hi:.2f} lies entirely above {null:g}."
    if hi < null:
        return "not_supported", f"95% CI {lo:.2f}-{hi:.2f} lies entirely below {null:g} (opposite direction)."
    return "inconclusive", f"95% CI {lo:.2f}-{hi:.2f} includes {null:g}."


def ceiling_for_pooled(meta: dict, n_labs_ok: int, null: float = 1.0) -> tuple[str, str]:
    """Pooled claim across labs: needs >= 2 contributing labs and acceptable heterogeneity."""
    if n_labs_ok < 2:
        return "inconclusive", f"Only {n_labs_ok} lab could contribute; a cross-lab claim needs at least 2."
    if meta.get("Q_p", 1) < 0.10 and meta.get("I2", 0) >= 0.5:
        return "inconsistent", f"Labs disagree (I2={meta['I2']:.0%}, Q p={meta['Q_p']:.2g}); pooling would hide this."
    return ceiling_for_effect(meta, null)


def ceiling_for_transfer(r: dict) -> tuple[str, str]:
    """Claim: 'the external effect size applies to the local group'. Can only be refuted, never proven."""
    if r.get("estimate") is None:
        return "inconclusive", "Ratio could not be computed."
    if r["ci_low"] > 1 or r["ci_high"] < 1:
        return "not_supported", f"Effect sizes differ: ratio {r['estimate']:.2f} (95% CI {r['ci_low']:.2f}-{r['ci_high']:.2f}) excludes 1."
    return "inconclusive", (f"Ratio {r['estimate']:.2f} (95% CI {r['ci_low']:.2f}-{r['ci_high']:.2f}) includes 1: no evidence of a "
                            "difference, but similarity is not proven.")
