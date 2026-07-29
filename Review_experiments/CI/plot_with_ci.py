from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import pickle
import re
from collections import defaultdict
from typing import Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, ndcg_score

try:
    from facile_stats import bootstrap_ci, holm, paired_bootstrap_ci, wilcoxon_paired
except ImportError:  # keep this file runnable on its own
    def bootstrap_ci(x, n_boot=10_000, alpha=0.05, rng=None):
        rng = rng or np.random.default_rng(0)
        x = np.asarray([v for v in x if np.isfinite(v)], float)
        if x.size == 0:
            return (np.nan, np.nan, np.nan)
        if x.size == 1:
            return (float(x[0]),) * 3
        idx = rng.integers(0, x.size, size=(n_boot, x.size))
        b = x[idx].mean(axis=1)
        lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(x.mean()), float(lo), float(hi))

    def paired_bootstrap_ci(x, y, n_boot=10_000, alpha=0.05, rng=None):
        rng = rng or np.random.default_rng(0)
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        d = x[m] - y[m]
        if d.size == 0:
            return (np.nan, np.nan, np.nan)
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        b = d[idx].mean(axis=1)
        lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (float(d.mean()), float(lo), float(hi))

    def wilcoxon_paired(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        d = (x[m] - y[m])
        d = d[d != 0]
        if d.size < 6:
            return dict(n=int(m.sum()), p=np.nan, mean_diff=np.nan,
                        win_rate=np.nan)
        res = stats.wilcoxon(d)
        return dict(n=int(m.sum()), p=float(res.pvalue),
                    mean_diff=float(d.mean()), win_rate=float((d > 0).mean()))

    def holm(p):
        p = np.asarray(p, float)
        order = np.argsort(p)
        m, adj, run = p.size, np.empty(p.size), 0.0
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * p[i])
            adj[i] = min(1.0, run)
        return adj


# -----------------------------------------------------------------------------
# figure style -- deliberately not the submitted one
# -----------------------------------------------------------------------------

COLORS = {"FACILE": "#DD6B07", "ContextCite": "#0f91ee", "Spex": "#D10505",
          "ProxySpex": "#760adb", "Shapiq": "#018d01", "Exact-Shapley": "#888780",
          "Exact-FSII": "#04B49D"}
MARKERS = {"FACILE": "o", "ContextCite": "p", "Spex": "s", "ProxySpex": "^",
           "Shapiq": "D", "Exact-Shapley": "v", "Exact-FSII": ">"}

STYLE = {
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "lines.markersize": 5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
"""The submitted figures set `font.size` to 24 on small multi-panel plots, which
is why 3ptb calls them hard to read: labels collide and several panels lose
their x-axis label entirely. 11pt with per-panel labels fits the NeurIPS column
without shrinking."""


def color_of(m):
    return COLORS.get(m, "#5F5E5A")


def marker_of(m):
    return MARKERS.get(m, ".")


# -----------------------------------------------------------------------------
# loading and reshaping
# -----------------------------------------------------------------------------

_BUDGET_RE = re.compile(r"^(?P<m>.+?)_(?P<b>\d+)$")


def split_method_budget(key: str) -> tuple[str, int | None]:
    m = _BUDGET_RE.match(key)
    return (m.group("m"), int(m.group("b"))) if m else (key, None)


def load_results(path: str) -> list[dict]:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "all_results" in obj:
        obj = obj["all_results"]
    if not isinstance(obj, list):
        raise TypeError(f"expected a list of per-question dicts, got {type(obj)}")
    return obj


# --- ground-truth providers ---------------------------------------------------


def gt_provider_stored(all_results):
    def f(i):
        g = all_results[i].get("gt")
        return list(g) if g is not None else None
    return f


def gt_provider_prefix(all_results, csv_path, len_col="len_gt"):
    lens = _csv_lookup(csv_path, all_results, len_col, int)
    def f(i):
        L = lens.get(i)
        return list(range(L)) if L else None
    return f


def gt_provider_csv(all_results, csv_path, col="gt_paragraphs"):
    vals = _csv_lookup(csv_path, all_results, col, _as_list)
    return lambda i: vals.get(i)


def _as_list(v):
    return list(ast.literal_eval(v)) if isinstance(v, str) else list(v)


def _csv_lookup(csv_path, all_results, col, cast):
    """Join CSV rows to results by question text.

    Duplicate question rows are fine as long as they agree on the value -- this
    CSV is three exports concatenated, so most questions appear more than once.
    A question whose duplicates DISAGREE is dropped rather than guessed at.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "len_gt" in df.columns:
        df = df[df["len_gt"].astype(str) != "len_gt"]
    q = df["question"].astype(str).str.strip()
    grouped = defaultdict(set)
    for text, val in zip(q, df[col]):
        grouped[text].add(str(val))
    by_q, conflicting = {}, 0
    for text, vals in grouped.items():
        if len(vals) == 1:
            by_q[text] = next(iter(vals))
        else:
            conflicting += 1
    out, missing = {}, 0
    for i, rw in enumerate(all_results):
        t = str(rw.get("query", "")).strip()
        if t in by_q:
            out[i] = cast(by_q[t])
        else:
            missing += 1
    if conflicting or missing:
        logging.warning(
            "ground truth from CSV column %r: %d resolved, %d questions absent "
            "from the CSV, %d question texts whose duplicate rows disagree",
            col, len(out), missing, conflicting)
    return out


def diagnose_gt(all_results, csv_path) -> None:
    """Score each candidate answer key against the exact reference."""
    cands = {"stored": gt_provider_stored(all_results)}
    if csv_path:
        cands["prefix"] = gt_provider_prefix(all_results, csv_path)
        cands["csv"] = gt_provider_csv(all_results, csv_path)
    print("\nground-truth diagnostic -- overlap of Exact-Shapley top-|gt| with each key")
    print("(the key the run actually used should be clearly highest)\n")
    for name, f in cands.items():
        hits, n = [], 0
        for i, rw in enumerate(all_results):
            g = f(i)
            if not g:
                continue
            sv = np.asarray(rw["methods"].get("Exact-Shapley", []), float)
            if sv.size == 0:
                continue
            top = set(np.argsort(-sv)[:len(g)])
            hits.append(len(top & set(g)) / len(g))
            n += 1
        m = np.mean(hits) if hits else float("nan")
        print(f"  {name:8s} resolved {n:4d}/{len(all_results)} questions  "
              f"mean overlap {m:.3f}")
    print()


# --- per-question metric extraction -------------------------------------------


def extract(all_results, gt_fn: Callable[[int], list[int] | None] | None = None,
            ks: Sequence[int] = (1, 3, 5),
            reference: str = "Exact-Shapley") -> list[dict]:
    """Long-format rows: qid, metric, method, budget, value."""
    rows: list[dict] = []

    def add(qid, metric, key, value):
        if value is None or not np.isfinite(value):
            return
        m, b = split_method_budget(key)
        rows.append(dict(qid=qid, metric=metric, method=m, budget=b,
                         value=float(value)))

    for qid, rw in enumerate(all_results):
        methods = rw.get("methods", {})
        ref = np.asarray(methods.get(reference, []), float)
        gt = gt_fn(qid) if gt_fn else None

        for key, attr in methods.items():
            a = np.asarray(attr, float)
            if a.size == 0 or not np.all(np.isfinite(a)):
                continue
            if ref.size == a.size and not key.startswith("Exact"):
                shifted = ref - ref.min()
                if shifted.max() > 0:
                    for k in ks:
                        add(qid, f"ndcg@{k}", key,
                            ndcg_score(shifted[None, :], a[None, :],
                                       k=min(k, a.size)))
                rho, _ = stats.spearmanr(ref, a)
                add(qid, "spearman_vs_exact", key, rho)
            if gt:
                y = np.isin(np.arange(a.size), gt).astype(int)
                if 0 < y.sum() < y.size:
                    add(qid, "pr_auc", key, average_precision_score(y, a))
                    for k in ks:
                        top = set(np.argsort(-a)[:k])
                        add(qid, f"recall@{k}", key,
                            len(top & set(gt)) / len(gt))

        for metric, payload in rw.get("metrics", {}).items():
            if not isinstance(payload, dict):
                continue
            for key, val in payload.items():
                if isinstance(val, dict):          # topk_probability: {k: v}
                    for k, v in val.items():
                        add(qid, f"{_pretty(metric)}@{k}", key, _f(v))
                elif isinstance(val, (list, tuple, np.ndarray)):  # Recall list
                    for k, v in zip(range(1, len(val) + 1), np.ravel(val)):
                        add(qid, f"stored_{_pretty(metric)}@{k}", key, _f(v))
                else:
                    add(qid, _pretty(metric), key, _f(val))
    return rows


def _pretty(m: str) -> str:
    return {"topk_probability": "topk_drop", "R2": "r2_util",
            "Delta_R2": "r2_delta", "LDS": "lds", "Recall": "recall"}.get(m, m)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- aggregation --------------------------------------------------------------


def series(rows, metric, method, budget):
    sel = [r for r in rows if r["metric"] == metric and r["method"] == method
           and r["budget"] == budget]
    sel.sort(key=lambda r: r["qid"])
    return np.array([r["qid"] for r in sel]), np.array([r["value"] for r in sel])


def aligned(rows, metric, a, b, budget):
    qa, va = series(rows, metric, a, budget)
    qb, vb = series(rows, metric, b, budget)
    common = np.intersect1d(qa, qb)
    return va[np.isin(qa, common)], vb[np.isin(qb, common)], common.size


def summarise(rows, metric, alpha=0.05):
    """{(method, budget): (mean, lo, hi, n)} for one metric."""
    out = {}
    cells = {(r["method"], r["budget"]) for r in rows if r["metric"] == metric}
    for m, b in sorted(cells, key=lambda c: (c[0], -1 if c[1] is None else c[1])):
        _, v = series(rows, metric, m, b)
        if v.size:
            pt, lo, hi = bootstrap_ci(v, alpha=alpha)
            out[(m, b)] = (pt, lo, hi, int(v.size))
    return out


def paired_table(rows, metric, reference="FACILE"):
    """Paired difference of `reference` minus each other method, per budget."""
    budgets = sorted({r["budget"] for r in rows
                      if r["metric"] == metric and r["budget"] is not None})
    others = sorted({r["method"] for r in rows if r["metric"] == metric
                     and r["method"] != reference
                     and not r["method"].startswith("Exact")})
    table = {}
    pvals, keys = [], []
    for b in budgets:
        for o in others:
            va, vb, n = aligned(rows, metric, reference, o, b)
            if n < 6:
                continue
            t = wilcoxon_paired(va, vb)
            t["diff_ci"] = paired_bootstrap_ci(va, vb)
            t["n"] = n
            d = np.asarray(va, float) - np.asarray(vb, float)
            d = d[np.isfinite(d)]
            t["win"] = float(np.mean(d > 0))
            t["tie"] = float(np.mean(d == 0))
            t["loss"] = float(np.mean(d < 0))
            nz = d[d != 0]
            t["win_excl_ties"] = float(np.mean(nz > 0)) if nz.size else np.nan
            t["median_diff"] = float(np.median(d))
            table[(b, o)] = t
            pvals.append(t["p"] if np.isfinite(t["p"]) else 1.0)
            keys.append((b, o))
    for k, p in zip(keys, holm(pvals) if pvals else []):
        table[k]["p_holm"] = float(p)
    return table


# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------


def plot_budget_curve(rows, metric, out_dir, reference="FACILE",
                      ylabel=None, title=None, logx=True):
    """Two stacked panels: absolute values with CI bands, paired difference below."""
    summ = summarise(rows, metric)
    if not summ:
        return None
    budgeted = defaultdict(list)
    flat = {}
    for (m, b), v in summ.items():
        (budgeted[m].append((b, v)) if b is not None else flat.setdefault(m, v))
    if not budgeted:
        return None

    with plt.rc_context(STYLE):
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(5.2, 5.0), sharex=True,
            gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.12))

        for m, pts in sorted(budgeted.items()):
            pts.sort(key=lambda t: t[0])
            x = np.array([p[0] for p in pts], float)
            y = np.array([p[1][0] for p in pts])
            lo = np.array([p[1][1] for p in pts])
            hi = np.array([p[1][2] for p in pts])
            c = color_of(m)
            ax.plot(x, y, marker=marker_of(m), color=c, label=m, zorder=3)
            ax.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0, zorder=2)

        for m, (pt, lo, hi, n) in flat.items():
            c = color_of(m)
            ax.axhline(pt, color=c, ls="--", lw=1.2, alpha=0.9, label=m)
            ax.axhspan(lo, hi, color=c, alpha=0.10, linewidth=0)

        ptab = paired_table(rows, metric, reference)
        if ptab:
            per_method = defaultdict(list)
            for (b, o), t in ptab.items():
                per_method[o].append((b, t))
            for o, pts in sorted(per_method.items()):
                pts.sort(key=lambda t: t[0])
                x = np.array([p[0] for p in pts], float)
                d = np.array([p[1]["diff_ci"][0] for p in pts])
                lo = np.array([p[1]["diff_ci"][1] for p in pts])
                hi = np.array([p[1]["diff_ci"][2] for p in pts])
                ax2.plot(x, d, marker=marker_of(o), color=color_of(o),
                         label=f"{reference} - {o}", zorder=3)
                ax2.fill_between(x, lo, hi, color=color_of(o), alpha=0.18,
                                 linewidth=0, zorder=2)
                for xi, (_, t) in zip(x, pts):
                    if t.get("p_holm", 1) < 0.05:
                        ax2.plot(xi, 0, marker="*", color="#444441",
                                 markersize=7, zorder=4, clip_on=False)
        ax2.axhline(0, color="#888780", lw=1.0, zorder=1)

        ax.set_ylabel(ylabel or metric)
        ax2.set_ylabel(f"paired\n{chr(0x0394)}", labelpad=2)
        ax2.set_xlabel("budget (LLM evaluations)")
        if logx:
            xs = sorted({b for _, b in summ if b is not None})
            ax2.set_xscale("log", base=2)
            ax2.set_xticks(xs)
            ax2.set_xticklabels([str(v) for v in xs])
            ax2.minorticks_off()
        n_any = max(v[3] for v in summ.values())
        ax.set_title(title or f"{metric}  (n = {n_any} questions, 95% bootstrap CI)",
                     pad=8)
        ax.legend(frameon=False, loc="best", ncol=1)
        ax2.legend(frameon=False, loc="best", fontsize=9)

        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"{metric.replace('@', '_at_')}")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png")
        plt.close(fig)
    return stem + ".pdf"


def plot_by_k(rows, metric_prefix, out_dir, budget, ylabel=None):
    """Grouped bars over k at one budget, with CI error bars (Recall@k, Top-k drop)."""
    metrics = sorted({r["metric"] for r in rows
                      if r["metric"].startswith(metric_prefix + "@")},
                     key=lambda s: int(s.split("@")[1]))
    if not metrics:
        return None
    methods = sorted({r["method"] for r in rows if r["metric"] in metrics
                      and r["budget"] == budget})
    if not methods:
        return None

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        width = 0.8 / max(1, len(methods))
        xs = np.arange(len(metrics))
        n_any = 0
        for j, m in enumerate(methods):
            pts, los, his = [], [], []
            for met in metrics:
                _, v = series(rows, met, m, budget)
                pt, lo, hi = bootstrap_ci(v) if v.size else (np.nan,) * 3
                pts.append(pt); los.append(pt - lo); his.append(hi - pt)
                n_any = max(n_any, v.size)
            ax.bar(xs + j * width - 0.4 + width / 2, pts, width * 0.92,
                   yerr=[np.abs(los), np.abs(his)], capsize=3,
                   color=color_of(m), alpha=0.85, label=m,
                   error_kw=dict(lw=1.0, ecolor="#444441"))
        ax.set_xticks(xs)
        ax.set_xticklabels([m.split("@")[1] for m in metrics])
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel or metric_prefix)
        ax.set_title(f"{metric_prefix} at budget {budget}  "
                     f"(n = {n_any}, 95% CI)", pad=8)
        ax.legend(frameon=False)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"{metric_prefix}_by_k_budget{budget}")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png")
        plt.close(fig)
    return stem + ".pdf"


# -----------------------------------------------------------------------------
# text output
# -----------------------------------------------------------------------------


def latex_table(rows, metric, reference="FACILE", decimals=3) -> str:
    """Mean [CI], paired difference [CI], Holm p, and win/tie/loss counts.

    The win/tie/loss column matters more than it looks.  For PR-AUC on this
    pickle the mean difference is positive and highly significant at every
    budget, yet the raw "win rate" is below 0.5 -- because 40-83% of questions
    are exact ties (both methods rank the gold documents identically).  Among
    non-tied questions FACILE wins 62-88%.  Reporting the bare win rate would
    look like a loss; reporting only the mean would hide that most questions
    are indistinguishable.  Report all three.
    """
    summ = summarise(rows, metric)
    ptab = paired_table(rows, metric, reference)
    f = f"{{:.{decimals}f}}"
    out = [r"\begin{tabular}{llcccc}", r"\toprule",
           r"Method & Budget & mean [95\% CI] & $\Delta$ vs " + reference
           + r" & $p_{\mathrm{Holm}}$ & W/T/L \\", r"\midrule"]
    for (m, b), (pt, lo, hi, n) in summ.items():
        cell = f"{f.format(pt)} [{f.format(lo)}, {f.format(hi)}]"
        if m == reference or b is None or (b, m) not in ptab:
            out.append(f"{m} & {b if b else '--'} & {cell} & --- & --- & --- \\\\")
        else:
            t = ptab[(b, m)]
            d, dlo, dhi = t["diff_ci"]
            p = t.get("p_holm", np.nan)
            ps = "<0.001" if p < 1e-3 else f"{p:.3f}"
            wtl = (f"{int(round(t['win']*t['n']))}/"
                   f"{int(round(t['tie']*t['n']))}/"
                   f"{int(round(t['loss']*t['n']))}")
            out.append(f"{m} & {b} & {cell} & {f.format(d)} "
                       f"[{f.format(dlo)}, {f.format(dhi)}] & {ps} & {wtl} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def dump_csv(rows, path):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "metric", "method",
                                           "budget", "value"])
        w.writeheader()
        w.writerows(rows)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

NICE = {
    "pr_auc": "PR-AUC (marginal)",
    "ndcg@5": "nDCG@5 to Exact-Shapley",
    "ndcg@3": "nDCG@3 to Exact-Shapley",
    "ndcg@1": "nDCG@1 to Exact-Shapley",
    "r2_util": r"$R^2_{\mathrm{util}}$",
    "r2_delta": r"$R^2_{\Delta}$",
    "lds": "LDS (Spearman)",
    "spearman_vs_exact": "Spearman to Exact-Shapley",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle", nargs="?", default="results.pkl")
    ap.add_argument("--out", default="figures_ci")
    ap.add_argument("--csv", default=None, help="question CSV, for --gt csv/prefix")
    ap.add_argument("--gt", default="stored",
                    choices=["stored", "prefix", "csv", "none", "diagnose"])
    ap.add_argument("--reference", default="FACILE")
    ap.add_argument("--budget-for-k", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_results = load_results(args.pickle)
    logging.info("loaded %d questions from %s", len(all_results), args.pickle)

    if args.gt == "diagnose":
        diagnose_gt(all_results, args.csv)
        return

    gt_fn = None
    if args.gt == "stored":
        gt_fn = gt_provider_stored(all_results)
        logging.warning("using the `gt` field stored in the pickle; confirm it "
                        "matches the document order the run actually used")
    elif args.gt == "prefix":
        gt_fn = gt_provider_prefix(all_results, args.csv)
    elif args.gt == "csv":
        gt_fn = gt_provider_csv(all_results, args.csv)

    rows = extract(all_results, gt_fn)
    os.makedirs(args.out, exist_ok=True)
    dump_csv(rows, os.path.join(args.out, "per_question_metrics.csv"))

    metrics = sorted({r["metric"] for r in rows})
    budgets = sorted({r["budget"] for r in rows if r["budget"] is not None})
    bk = args.budget_for_k or (budgets[len(budgets) // 2] if budgets else None)
    logging.info("metrics: %s", ", ".join(metrics))

    made, tables = [], {}
    for m in metrics:
        if "@" in m and m.split("@")[0] in ("recall", "topk_drop",
                                            "stored_recall"):
            continue
        p = plot_budget_curve(rows, m, args.out, args.reference,
                              ylabel=NICE.get(m, m))
        if p:
            made.append(p)
            tables[m] = latex_table(rows, m, args.reference)

    for prefix, ylab in (("recall", "Recall@k"),
                         ("topk_drop", "Top-k drop"),
                         ("stored_recall", "Recall@k (stored)")):
        if bk is not None:
            p = plot_by_k(rows, prefix, args.out, bk, ylabel=ylab)
            if p:
                made.append(p)

    with open(os.path.join(args.out, "tables.tex"), "w") as fh:
        for m, t in tables.items():
            fh.write(f"% ---- {m}\n{t}\n\n")
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump({m: {f"{k[0]}@{k[1]}": v for k, v in summarise(rows, m).items()}
                   for m in metrics}, fh, indent=2)

    print(f"\nwrote {len(made)} figures to {args.out}/")
    for p in made:
        print("  ", p)
    print(f"   {args.out}/per_question_metrics.csv")
    print(f"   {args.out}/tables.tex")
    print(f"   {args.out}/summary.json")


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# diagnostic: why the comparison is paired
# -----------------------------------------------------------------------------


def plot_paired_diagnostic(rows, metric, out_dir, budget,
                           reference="FACILE", baseline="ContextCite",
                           ylabel=None):
    """Two panels showing the pairing itself.

    Left  -- one dot per question, baseline on x, reference on y, identity line.
             Dots hug the diagonal because both methods see the same question:
             the per-question difficulty is shared, and pairing removes it.
    Right -- histogram of the 240 per-question differences, with mean and its
             bootstrap CI.  The test is entirely about whether this histogram
             is centred away from zero.

    Worth putting in the appendix: it is the clearest possible answer to
    "are these differences within noise?".
    """
    va, vb, n = aligned(rows, metric, reference, baseline, budget)
    if n < 3:
        return None
    d = va - vb
    pt, lo, hi = paired_bootstrap_ci(va, vb)
    rho = float(np.corrcoef(va, vb)[0, 1])

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))

        lim = [min(va.min(), vb.min()), max(va.max(), vb.max())]
        pad = 0.03 * (lim[1] - lim[0] or 1)
        lim = [lim[0] - pad, lim[1] + pad]
        ax1.plot(lim, lim, color="#888780", lw=1.0, zorder=1)
        ax1.scatter(vb, va, s=14, alpha=0.5, color=color_of(reference),
                    edgecolors="none", zorder=2)
        ax1.set_xlim(lim); ax1.set_ylim(lim); ax1.set_aspect("equal")
        ax1.set_xlabel(f"{baseline}")
        ax1.set_ylabel(f"{reference}")
        ax1.set_title(f"per question  (r = {rho:.2f})", pad=6)

        ax2.hist(d, bins=40, color=color_of(reference), alpha=0.75,
                 edgecolor="none")
        ax2.axvline(0, color="#888780", lw=1.0)
        ax2.axvline(pt, color="#444441", lw=1.6)
        ax2.axvspan(lo, hi, color="#444441", alpha=0.18, linewidth=0)
        ax2.set_xlabel(f"{reference} - {baseline}")
        ax2.set_ylabel("questions")
        ax2.set_title(f"mean {pt:+.3f}  [{lo:+.3f}, {hi:+.3f}]", pad=6)

        fig.suptitle(f"{ylabel or metric} at budget {budget}  (n = {n})",
                     y=1.02)
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir,
                            f"paired_{metric.replace('@', '_at_')}_b{budget}")
        fig.savefig(stem + ".pdf")
        fig.savefig(stem + ".png")
        plt.close(fig)
    return stem + ".pdf"
