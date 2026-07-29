from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats

_RNG = np.random.default_rng(0)

import os, sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *
from SHapRAG.utils import *

# ----------------------------------------------------------------------------- 
# bootstrap
# -----------------------------------------------------------------------------


def bootstrap_ci(
    x: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for `stat` of a single sample.

    Returns (point_estimate, lo, hi).  NaNs are dropped (questions where the
    model produced no answer, which you already exclude in App. E).
    """
    rng = rng or _RNG
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1) if _accepts_axis(stat) else np.array(
        [stat(x[i]) for i in idx]
    )
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(x)), float(lo), float(hi))


def paired_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """CI for mean(x - y) over paired observations (resample question indices).

    This is the number to put in the rebuttal table: "FACILE - Spex =
    +0.25 [0.21, 0.29]".  It is immune to the overlapping-marginal-CI fallacy.
    """
    rng = rng or _RNG
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"unpaired shapes {x.shape} vs {y.shape}")
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    if d.size == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(d.mean()), float(lo), float(hi))


def _accepts_axis(fn) -> bool:
    return fn in (np.mean, np.median, np.sum, np.std)


# -----------------------------------------------------------------------------
# paired tests + multiplicity control
# -----------------------------------------------------------------------------


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> dict:
    """Paired Wilcoxon signed-rank + rank-biserial effect size."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    d = x[m] - y[m]
    d_nz = d[d != 0]
    if d_nz.size < 6:
        return dict(n=int(d.size), stat=np.nan, p=np.nan, effect=np.nan,
                    note="too few non-tied pairs")
    res = stats.wilcoxon(d_nz, alternative="two-sided", zero_method="wilcox")
    # rank-biserial correlation: (R+ - R-) / (R+ + R-)
    ranks = stats.rankdata(np.abs(d_nz))
    r_pos, r_neg = ranks[d_nz > 0].sum(), ranks[d_nz < 0].sum()
    effect = (r_pos - r_neg) / (r_pos + r_neg)
    return dict(n=int(d.size), stat=float(res.statistic), p=float(res.pvalue),
                effect=float(effect), mean_diff=float(d.mean()),
                win_rate=float((d > 0).mean()))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family = one metric x all baselines)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def min_detectable_effect(sd_of_differences: float, n: int, power: float = 0.8,
                          alpha: float = 0.05) -> float:
    """Smallest paired mean difference detectable at given n (normal approx).

    Use this to answer "is 100 questions enough?" quantitatively rather than
    conceding the point.  With sd(diff)=0.15 and n=100, MDE ~= 0.042 nDCG:
    every gap you report in Table 8/11/12 is far above it.
    """
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * sd_of_differences / np.sqrt(n))


def required_n(sd_of_differences: float, target_effect: float,
               power: float = 0.8, alpha: float = 0.05) -> int:
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(((z_a + z_b) * sd_of_differences / target_effect) ** 2))


# -----------------------------------------------------------------------------
# result-table plumbing (matches the notebooks' `all_results` schema)
# -----------------------------------------------------------------------------

_BUDGET_RE = re.compile(r"^(?P<method>.+?)_(?P<budget>\d+)$")


def split_method_budget(key: str) -> tuple[str, int | None]:
    """'FACILE_128' -> ('FACILE', 128); 'Exact-FSII' -> ('Exact-FSII', None)."""
    m = _BUDGET_RE.match(key)
    if not m:
        return key, None
    return m.group("method"), int(m.group("budget"))


@dataclass
class ResultTable:
    """Long-format view of the notebooks' per-question results.

    rows: list of dicts with keys
        qid, metric, method, budget, k, value
    """

    rows: list[dict]

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_all_results(cls, all_results: Iterable[dict],
                         k_values: Sequence[int] = (1, 2, 3, 4, 5)) -> "ResultTable":
        rows: list[dict] = []
        for r in all_results:
            qid = r.get("query_index")
            for metric, payload in r.get("metrics", {}).items():
                if not isinstance(payload, dict):
                    continue
                for key, val in payload.items():
                    method, budget = split_method_budget(key)
                    if np.isscalar(val):
                        rows.append(dict(qid=qid, metric=metric, method=method,
                                         budget=budget, k=None, value=float(val)))
                    else:  # per-k list, e.g. Recall / topk_probability
                        for k, v in zip(k_values, np.atleast_1d(val)):
                            rows.append(dict(qid=qid, metric=metric, method=method,
                                             budget=budget, k=int(k),
                                             value=float(v)))
        return cls(rows)

    @classmethod
    def from_pickle(cls, path: str, **kw) -> "ResultTable":
        with open(path, "rb") as fh:
            return cls.from_all_results(pickle.load(fh), **kw)

    # -- selection ------------------------------------------------------------

    def series(self, metric: str, method: str, budget: int | None = None,
               k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (qids, values) sorted by qid, so two calls are index-aligned."""
        sel = [r for r in self.rows
               if r["metric"] == metric and r["method"] == method
               and (budget is None or r["budget"] == budget)
               and (k is None or r["k"] == k)]
        sel.sort(key=lambda r: (r["qid"] is None, r["qid"]))
        return (np.array([r["qid"] for r in sel]),
                np.array([r["value"] for r in sel], dtype=float))

    def methods(self, metric: str) -> list[str]:
        return sorted({r["method"] for r in self.rows if r["metric"] == metric})

    def _aligned(self, metric, a, b, budget, k):
        qa, va = self.series(metric, a, budget, k)
        qb, vb = self.series(metric, b, budget, k)
        common = np.intersect1d(qa, qb)
        return (va[np.isin(qa, common)], vb[np.isin(qb, common)])

    # -- reporting ------------------------------------------------------------

    def summary(self, metric: str, budget: int | None = None,
                k: int | None = None, alpha: float = 0.05) -> dict[str, tuple]:
        out = {}
        for m in self.methods(metric):
            _, v = self.series(metric, m, budget, k)
            if v.size:
                out[m] = bootstrap_ci(v, alpha=alpha)
        return out

    def paired_test(self, metric: str, method: str, baseline: str,
                    budget: int | None = None, k: int | None = None) -> dict:
        va, vb = self._aligned(metric, method, baseline, budget, k)
        res = wilcoxon_paired(va, vb)
        res["diff_ci"] = paired_bootstrap_ci(va, vb)
        res["mde"] = min_detectable_effect(np.nanstd(va - vb, ddof=1), va.size) \
            if va.size > 1 else np.nan
        return res

    def all_paired_tests(self, metric: str, reference: str = "FACILE",
                         budget: int | None = None, k: int | None = None) -> dict:
        others = [m for m in self.methods(metric)
                  if m != reference and not m.startswith("Exact")]
        raw = {m: self.paired_test(metric, reference, m, budget, k) for m in others}
        adj = holm([raw[m]["p"] for m in others])
        for m, a in zip(others, adj):
            raw[m]["p_holm"] = float(a)
        return raw

    def latex(self, metric: str, budget: int | None = None, k: int | None = None,
              reference: str = "FACILE", decimals: int = 3) -> str:
        """Emit a rebuttal-ready table: mean [CI], plus paired diff vs reference."""
        summ = self.summary(metric, budget, k)
        tests = self.all_paired_tests(metric, reference, budget, k)
        f = f"{{:.{decimals}f}}"
        lines = [r"\begin{tabular}{lccc}", r"\toprule",
                 r"Method & mean [95\% CI] & $\Delta$ vs " + reference
                 + r" [95\% CI] & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
        for m, (pt, lo, hi) in summ.items():
            if m == reference:
                lines.append(f"{m} & {f.format(pt)} [{f.format(lo)}, "
                             f"{f.format(hi)}] & --- & --- \\\\")
            elif m in tests:
                d, dlo, dhi = tests[m]["diff_ci"]
                p = tests[m]["p_holm"]
                ps = "<0.001" if p < 1e-3 else f"{p:.3f}"
                lines.append(f"{m} & {f.format(pt)} [{f.format(lo)}, "
                             f"{f.format(hi)}] & {f.format(d)} [{f.format(dlo)},"
                             f" {f.format(dhi)}] & {ps} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    # -- variance decomposition ----------------------------------------------

    def variance_report(self, metric: str, method: str,
                        budget: int | None = None) -> dict:
        """Between-question sd, which is what actually drives the CI width.

        Report this next to the CI.  It pre-empts "results are sensitive to
        dataset and model choice" by showing how much of the spread is
        question-level noise vs. a systematic dataset effect.
        """
        _, v = self.series(metric, method, budget)
        return dict(n=int(v.size), mean=float(np.nanmean(v)),
                    sd=float(np.nanstd(v, ddof=1)),
                    sem=float(stats.sem(v, nan_policy="omit")),
                    iqr=float(np.subtract(*np.nanpercentile(v, [75, 25]))))


if __name__ == "__main__":
    # self-test on synthetic paired data with a known effect
    rng = np.random.default_rng(1)
    n = 100
    facile = np.clip(rng.normal(0.93, 0.05, n), 0, 1)
    spex = np.clip(facile - rng.normal(0.25, 0.15, n), 0, 1)
    rows = []
    for q in range(n):
        rows.append(dict(qid=q, metric="NDCG_inter", method="FACILE",
                         budget=128, k=None, value=facile[q]))
        rows.append(dict(qid=q, metric="NDCG_inter", method="Spex",
                         budget=128, k=None, value=spex[q]))
    rt = ResultTable(rows)
    print("summary:", rt.summary("NDCG_inter", budget=128))
    t = rt.paired_test("NDCG_inter", "FACILE", "Spex", budget=128)
    print("paired:", {k: t[k] for k in ("n", "p", "mean_diff", "win_rate",
                                        "diff_ci", "mde")})
    print("MDE at n=100, sd=0.15:", min_detectable_effect(0.15, 100))
    print("n needed for 0.02 effect:", required_n(0.15, 0.02))
    print(rt.latex("NDCG_inter", budget=128))
