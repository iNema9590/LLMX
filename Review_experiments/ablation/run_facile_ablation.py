#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Sequence
import sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *
from SHapRAG.utils import *

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facile_ablation import (ABLATIONS, AblationSpec, prune_interactions,
                             run_ablation_grid, structured_sample, to_long)
from facile_stats import (bootstrap_ci, holm, paired_bootstrap_ci,
                          wilcoxon_paired)
from fm_numpy import make_fm_factory

try:
    from facile_eval_fix import HeldOutEvaluator
except ImportError:
    HeldOutEvaluator = None

LOG = logging.getLogger("ablation")

DEFAULT_BUDGETS = (32, 64, 128, 264, 528, 728)
METRICS = ("NDCG5_marg", "NDCG5_inter", "spearman_inter", "R2_util", "R2_delta")
REFERENCE = "FACILE (full)"


# =============================================================================
# spec selection
# =============================================================================

GROUPS = {
    "sampling": ("FACILE (full)", "-- core set", "-- neighbor expansion",
                 "-- kernel weighting", "-- both (uniform only)",
                 "size-uniform fill"),
    "rank": ("FACILE (full)", "rank sel: r2_util", "rank fixed 1",
             "rank fixed 2", "rank fixed 4", "rank fixed 8", "rank oracle"),
    "pruning": ("FACILE (full)", "+ pruning top-20%", "+ pruning MAD"),
}


def select_specs(groups: Sequence[str]) -> list[AblationSpec]:
    if "all" in groups:
        return list(ABLATIONS)
    wanted, seen = [], set()
    for g in groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g!r}; choose from "
                             f"{sorted(GROUPS)} or 'all'")
        for name in GROUPS[g]:
            if name not in seen:
                seen.add(name)
                wanted.append(name)
    by_name = {s.name: s for s in ABLATIONS}
    return [by_name[n] for n in wanted if n in by_name]


# =============================================================================
# harnesses
# =============================================================================


class SyntheticHarness:
    """A degree-2 game with known parameters, for --dry-run.

    The point is to exercise the whole pipeline -- sampling, rank selection,
    pruning, aggregation, printing -- without a GPU, and to sanity-check that
    the ablation recovers something sensible on a game where the ground truth
    is known in closed form.  For a degree-2 FM game:

        exact Shapley  phi_i = w_i + 0.5 * sum_{j != i} F_ij
        exact order-2 FSII of pair (i, j) = F_ij

    both exactly, so the references need no approximation.  Interactions are
    generated low-rank by default; pass rank_true > n to break the low-rank
    assumption and watch FACILE degrade (that is 3ptb's Q1).
    """

    def __init__(self, n_items=10, rank_true=2, noise=0.02, seed=0):
        rng = np.random.default_rng(seed)
        self.n_items = n_items
        self.utility_mode = "synthetic"
        self.w = rng.normal(0, 1.0, n_items)
        V = rng.normal(0, 0.6, (n_items, max(1, rank_true)))
        self.F = V @ V.T
        np.fill_diagonal(self.F, 0.0)
        self.noise = noise
        self._rng = np.random.default_rng(seed + 1)
        self._cache: dict[tuple, float] = {}
        self.n_calls = 0

    def get_utility(self, z, mode=None):
        z = tuple(int(b) for b in z)
        if z not in self._cache:
            a = np.asarray(z, float)
            val = float(self.w @ a + 0.5 * a @ self.F @ a)
            self._cache[z] = val + self._rng.normal(0, self.noise)
            self.n_calls += 1
        return self._cache[z]

    def exact_shapley(self):
        return self.w + 0.5 * self.F.sum(axis=1)

    def exact_fsii_matrix(self):
        return self.F.copy()


def build_real_harness(row_docs, query, model, tokenizer, accelerator,
                       cache_path, utility_mode):
    from rag_shap import ContextAttribution  # noqa: F401  (or SHapRAG)

    return ContextAttribution(
        items=row_docs, query=query,
        prepared_model=model, prepared_tokenizer=tokenizer,
        accelerator=accelerator, utility_cache_path=cache_path,
        utility_mode=utility_mode, verbose=False,
    )


def import_context_attribution(module_name: str):
    """Import ContextAttribution from whichever package layout you have."""
    import importlib
    for name in (module_name, "SHapRAG", "SHapRAG.rag_shap", "rag_shap"):
        if not name:
            continue
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "ContextAttribution"):
                LOG.info("using ContextAttribution from %s", name)
                return mod.ContextAttribution
        except ImportError:
            continue
    raise SystemExit(
        "could not import ContextAttribution. Pass --module <name> or run this "
        "script from the directory that contains your SHapRAG package.")


# =============================================================================
# exact references
# =============================================================================


def exact_references(harness, skip_interactions=False):
    """(exact Shapley vector, exact order-2 FSII matrix).

    Both come from the full 2^N utility table; the second call is free because
    `get_utility` caches.
    """
    n = harness.n_items
    if isinstance(harness, SyntheticHarness):
        return harness.exact_shapley(), harness.exact_fsii_matrix()

    sv = np.asarray(harness._calculate_exact(method="SV"), float).ravel()
    F = np.zeros((n, n))
    if not skip_interactions:
        _, interaction_terms, _ = harness.compute_exact_faith(
            max_order=2, method="FSII")
        for pattern, coef in interaction_terms.items():
            if len(pattern) == 2:
                i, j = pattern
                F[i, j] = F[j, i] = float(coef)
    return sv, F


# =============================================================================
# aggregation + printing
# =============================================================================


def _cell(rows, metric, method, budget):
    v = [r["value"] for r in rows
         if r["metric"] == metric and r["method"] == method
         and r["budget"] == budget]
    return np.asarray(v, float)


def _paired(rows, metric, a, b, budget):
    da = {r["qid"]: r["value"] for r in rows if r["metric"] == metric
          and r["method"] == a and r["budget"] == budget}
    db = {r["qid"]: r["value"] for r in rows if r["metric"] == metric
          and r["method"] == b and r["budget"] == budget}
    common = sorted(set(da) & set(db))
    return (np.array([da[q] for q in common]),
            np.array([db[q] for q in common]))


def print_metric_table(rows, metric, budgets, specs, reference=REFERENCE,
                       ranks=None, out=sys.stdout):
    names = [s.name for s in specs]
    print(f"\n{'=' * 100}", file=out)
    print(f"{metric}", file=out)
    print("=" * 100, file=out)

    for b in budgets:
        present = [n for n in names if _cell(rows, metric, n, b).size]
        if not present:
            continue
        print(f"\n  budget {b}", file=out)
        print(f"  {'cell':<26} {'mean [95% CI]':<26} "
              f"{'delta vs full':<24} {'p_Holm':>8} {'W/T/L':>12} {'rank':>5}",
              file=out)
        print("  " + "-" * 106, file=out)

        pvals, keys, cache = [], [], {}
        for name in present:
            if name == reference:
                continue
            va, vb = _paired(rows, metric, reference, name, b)
            if va.size < 6:
                continue
            t = wilcoxon_paired(va, vb)
            d = va - vb
            t["diff_ci"] = paired_bootstrap_ci(va, vb)
            t["wtl"] = (int((d > 0).sum()), int((d == 0).sum()),
                        int((d < 0).sum()))
            cache[name] = t
            pvals.append(t["p"] if np.isfinite(t["p"]) else 1.0)
            keys.append(name)
        for name, p in zip(keys, holm(pvals) if pvals else []):
            cache[name]["p_holm"] = float(p)

        for name in present:
            v = _cell(rows, metric, name, b)
            pt, lo, hi = bootstrap_ci(v)
            rk = ""
            if ranks and (name, b) in ranks:
                rk = f"{np.mean(ranks[(name, b)]):.1f}"
            if name == reference:
                print(f"  {name:<26} {pt:.4f} [{lo:.4f},{hi:.4f}]  "
                      f"{'--':<24} {'--':>8} {'--':>12} {rk:>5}", file=out)
            elif name in cache:
                t = cache[name]
                dd, dlo, dhi = t["diff_ci"]
                p = t.get("p_holm", np.nan)
                ps = "<0.001" if p < 1e-3 else f"{p:.3f}"
                star = " *" if p < 0.05 else "  "
                w, ti, l = t["wtl"]
                print(f"  {name:<26} {pt:.4f} [{lo:.4f},{hi:.4f}]  "
                      f"{dd:+.4f} [{dlo:+.4f},{dhi:+.4f}]{star}"
                      f"{ps:>8} {f'{w}/{ti}/{l}':>12} {rk:>5}", file=out)
            else:
                print(f"  {name:<26} {pt:.4f} [{lo:.4f},{hi:.4f}]", file=out)


def print_component_summary(rows, metric, budgets, specs, reference=REFERENCE,
                            out=sys.stdout):
    """Compact view: how much each removed component costs, per budget.

    Positive = the full method is better = the component earns its place.
    This is the table to put in the rebuttal; the per-budget detail goes in the
    appendix.
    """
    names = [s.name for s in specs if s.name != reference]
    print(f"\n{'=' * 100}", file=out)
    print(f"component contribution -- {metric}   "
          f"(full method minus ablated cell; positive = component helps)",
          file=out)
    print("=" * 100, file=out)
    head = "  " + f"{'cell':<26}" + "".join(f"{('M=' + str(b)):>16}"
                                            for b in budgets)
    print(head, file=out)
    print("  " + "-" * (26 + 16 * len(budgets)), file=out)
    for name in names:
        line = f"  {name:<26}"
        for b in budgets:
            va, vb = _paired(rows, metric, reference, name, b)
            if va.size < 6:
                line += f"{'--':>16}"
                continue
            d, lo, hi = paired_bootstrap_ci(va, vb)
            sig = "*" if (lo > 0 or hi < 0) else " "
            line += f"{d:+.4f}{sig:>2}".rjust(16)
        print(line, file=out)
    print("\n  * = 95% CI excludes zero (per-cell, before multiplicity "
          "correction; see the per-metric tables for Holm-adjusted p)", file=out)


# =============================================================================
# checkpointing
# =============================================================================


def save_rows(path, rows):
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def load_rows(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# =============================================================================
# main
# =============================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FACILE component ablation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="synthetic degree-2 game, no model or data needed")
    p.add_argument("--report-only", action="store_true",
                   help="re-print tables from --out without running anything")

    p.add_argument("--csv", default=None, help="question CSV")
    p.add_argument("--docs-col", default="reordered_paragraphs")
    p.add_argument("--query-col", default="question")
    p.add_argument("--n-docs", type=int, default=10)
    p.add_argument("--n-questions", type=int, default=None,
                   help="default 50 for a real run, 12 for --dry-run")
    p.add_argument("--start", type=int, default=0)

    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--module", default=None,
                   help="module exporting ContextAttribution")
    p.add_argument("--utility-mode", default="log-perplexity")
    p.add_argument("--cache-dir", default="utility_cache")

    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--groups", nargs="+", default=["all"],
                   help="sampling | rank | pruning | all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fm-iters", type=int, default=None,
                   help="ALS iterations (default 300; 60 in --dry-run, where "
                        "fit time dominates -- in a real run the LLM calls do)")
    p.add_argument("--fm-backend", default="auto",
                   choices=["auto", "fastfm", "numpy"],
                   help="'auto' uses fastFM if importable, else the bundled "
                        "numpy ALS solver (identical model class)")
    p.add_argument("--skip-interactions", action="store_true")
    p.add_argument("--heldout", type=int, default=0,
                   help="if >0 and facile_eval_fix is importable, also compute "
                        "leakage-free R2_util / R2_delta on this many coalitions")

    p.add_argument("--out", default="facile_ablation_run")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run-rank", type=int, default=2,
                   help="true interaction rank in --dry-run; set > n-docs to "
                        "break the low-rank assumption")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "rows.jsonl")
    specs = select_specs(args.groups)
    budgets = sorted(args.budgets)

    # ---------------------------------------------------------------- report
    if args.report_only:
        rows = load_rows(rows_path)
        if not rows:
            raise SystemExit(f"no rows in {rows_path}")
        report(rows, budgets, specs, args)
        return

    existing = load_rows(rows_path) if args.resume else []
    done_qids = {r["qid"] for r in existing}
    if args.resume and done_qids:
        LOG.info("resuming: %d questions already done", len(done_qids))
    elif os.path.exists(rows_path):
        os.remove(rows_path)

    print(f"\nspecs   : {len(specs)}  ({', '.join(s.name for s in specs)})")
    print(f"budgets : {budgets}")
    print(f"fits    : {len(specs) * len(budgets)} per question")
    print(f"output  : {rows_path}\n")

    fm_iters = args.fm_iters or (60 if args.dry_run else 300)
    fm_factory = make_fm_factory(args.fm_backend, n_iter=fm_iters,
                                 seed=args.seed)
    LOG.info("FM backend: %s (n_iter=%d)", fm_factory.backend, fm_iters)

    raw_rows, rank_log = [], defaultdict(list)

    # ------------------------------------------------------------- dry run
    if args.dry_run:
        n_q = args.n_questions or 12
        LOG.info("dry run: %d synthetic questions, n_docs=%d, true rank=%d",
                 n_q, args.n_docs, args.dry_run_rank)
        for qid in range(n_q):
            h = SyntheticHarness(n_items=args.n_docs,
                                 rank_true=args.dry_run_rank,
                                 seed=1000 + qid)
            sv, F = exact_references(h)
            rr = run_ablation_grid(h, qid, budgets, sv, F, specs=specs,
                                   evaluator=None, seed=args.seed,
                                   fm_factory=fm_factory)
            for r in rr:
                rank_log[(r["method"], r["budget"])].append(r["rank"])
            raw_rows += rr
            long = to_long(rr, METRICS)
            save_rows(rows_path, long)
            if (qid + 1) % 5 == 0:
                LOG.info("  %d/%d questions", qid + 1, n_q)
        rows = load_rows(rows_path)
        report(rows, budgets, specs, args, ranks=rank_log)
        return

    # ------------------------------------------------------------- real run
    if not args.csv:
        raise SystemExit("--csv is required unless --dry-run is set")

    import ast
    import pandas as pd
    import torch
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ContextAttribution = import_context_attribution(args.module)

    df = pd.read_csv(args.csv)
    if "len_gt" in df.columns:            # drop repeated header rows
        df = df[df["len_gt"].astype(str) != "len_gt"].reset_index(drop=True)

    LOG.info("loading %s", args.model)
    accelerator = Accelerator(mixed_precision="fp16")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float16)
    if tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    model = accelerator.prepare(model)
    accelerator.unwrap_model(model).eval()

    os.makedirs(args.cache_dir, exist_ok=True)
    lo, hi = args.start, args.start + (args.n_questions or 50)
    t0 = time.time()
    n_run = 0

    for qid in range(lo, min(hi, len(df))):
        if qid in done_qids:
            continue
        docs = df[args.docs_col].iloc[qid]
        if isinstance(docs, str):
            docs = ast.literal_eval(docs)
        docs = list(docs)[:args.n_docs]
        query = str(df[args.query_col].iloc[qid])
        if len(docs) < 2:
            LOG.warning("q%d has %d documents, skipping", qid, len(docs))
            continue

        cache_path = os.path.join(args.cache_dir, f"utilities_q_idx{qid}.pkl")
        try:
            harness = ContextAttribution(
                items=docs, query=query, prepared_model=model,
                prepared_tokenizer=tok, accelerator=accelerator,
                utility_cache_path=cache_path, utility_mode=args.utility_mode,
                verbose=False)
        except TypeError:                  # older signature without `verbose`
            harness = ContextAttribution(
                items=docs, query=query, prepared_model=model,
                prepared_tokenizer=tok, accelerator=accelerator,
                utility_cache_path=cache_path, utility_mode=args.utility_mode)

        if not accelerator.is_main_process:
            continue

        try:
            sv, F = exact_references(harness, args.skip_interactions)
        except Exception as exc:
            LOG.error("q%d exact references failed: %s", qid, exc)
            continue

        evaluator = None
        if args.heldout and HeldOutEvaluator is not None:
            train = {s.name: structured_sample(
                harness.n_items, max(budgets), core=s.core,
                neighbors=s.neighbors, fill=s.fill, seed=args.seed)
                for s in specs}
            try:
                evaluator = HeldOutEvaluator(harness, train,
                                             n_test=args.heldout)
            except Exception as exc:
                LOG.warning("q%d held-out evaluator unavailable: %s", qid, exc)

        try:
            rr = run_ablation_grid(harness, qid, budgets, sv, F, specs=specs,
                                   evaluator=evaluator, seed=args.seed,
                                   fm_factory=fm_factory)
        except Exception as exc:
            LOG.exception("q%d ablation failed: %s", qid, exc)
            continue

        for r in rr:
            rank_log[(r["method"], r["budget"])].append(r["rank"])
        raw_rows += rr
        save_rows(rows_path, to_long(rr, METRICS))

        try:
            harness.save_utility_cache(cache_path)
        except Exception:
            pass

        n_run += 1
        el = time.time() - t0
        LOG.info("q%d done  (%d run, %.1f min elapsed, %.1f min/question)",
                 qid, n_run, el / 60, el / 60 / max(1, n_run))

    with open(os.path.join(args.out, "ranks.json"), "w") as fh:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in rank_log.items()}, fh,
                  indent=2)

    rows = load_rows(rows_path)
    report(rows, budgets, specs, args, ranks=rank_log)


def report(rows, budgets, specs, args, ranks=None):
    qids = sorted({r["qid"] for r in rows})
    metrics = [m for m in METRICS
               if any(r["metric"] == m for r in rows)]
    print(f"\n{'#' * 100}")
    print(f"# FACILE ablation -- {len(qids)} questions, "
          f"{len(specs)} cells, budgets {budgets}")
    print(f"# metrics: {', '.join(metrics)}")
    print("#" * 100)

    txt_path = os.path.join(args.out, "report.txt")
    with open(txt_path, "w") as fh:
        for sink in (sys.stdout, fh):
            for m in metrics:
                print_metric_table(rows, m, budgets, specs, ranks=ranks,
                                   out=sink)
            for m in metrics:
                print_component_summary(rows, m, budgets, specs, out=sink)

    # LaTeX for the headline metric
    try:
        from facile_stats import ResultTable
        rt = ResultTable([dict(r, k=None) for r in rows])
        tex_path = os.path.join(args.out, "ablation_tables.tex")
        with open(tex_path, "w") as fh:
            for m in metrics:
                for b in budgets:
                    fh.write(f"% ---- {m} @ budget {b}\n")
                    fh.write(rt.latex(m, budget=b, reference=REFERENCE) + "\n\n")
        print(f"\nwrote {tex_path}")
    except Exception as exc:
        LOG.warning("LaTeX emission skipped: %s", exc)

    print(f"wrote {txt_path}")
    print(f"wrote {os.path.join(args.out, 'rows.jsonl')}  "
          f"(long format: qid, metric, method, budget, value)")


if __name__ == "__main__":
    main()
