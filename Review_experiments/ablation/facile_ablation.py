from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score, r2_score
from sklearn.model_selection import KFold
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *
from SHapRAG.utils import *

# -----------------------------------------------------------------------------
# component-wise coalition sampling
# -----------------------------------------------------------------------------


def structured_sample(
    n: int,
    budget: int,
    core: bool = True,
    neighbors: bool = True,
    fill: str = "kernelshap",
    seed: int = 42,
) -> list[tuple]:
    """FACILE's sampler with every component individually switchable.

    core      : include the empty set and the grand coalition
    neighbors : include all N singletons and all N leave-one-out sets
    fill      : "kernelshap" | "uniform" | "size_uniform" | "none"

    Returns exactly `budget` distinct coalitions when possible.  If
    core+neighbors already exceed `budget` the structured part is truncated in a
    deterministic, interleaved order (singleton_0, LOO_0, singleton_1, ...) so
    the ablation stays budget-faithful.

    BUDGET NOTE FOR THE PAPER: FACILE's structured core costs 2N+2 evaluations,
    so the method is only defined for M >= 2N+2 (22 at N=10).  State this.  It
    also means the N=10 setting is the only one where a budget of 32 is even
    reachable; at N=20 the smallest FACILE budget is 42 and the "32" column
    would not exist.  Reviewer UmNy's Q1 about scaling in N is really a question
    about this constraint.
    """
    rng = np.random.default_rng(seed)
    empty, full = tuple([0] * n), tuple([1] * n)

    structured: list[tuple] = []
    if core:
        structured += [empty, full]
    if neighbors:
        for i in range(n):
            s = [0] * n
            s[i] = 1
            l = [1] * n
            l[i] = 0
            structured += [tuple(s), tuple(l)]

    out: list[tuple] = []
    seen: set[tuple] = set()
    for v in structured:
        if len(out) >= budget:
            break
        if v not in seen:
            seen.add(v)
            out.append(v)

    if fill == "none" or len(out) >= budget:
        return out

    sizes = np.arange(1, n)
    if fill == "kernelshap":
        w = (n - 1) / (sizes * (n - sizes))
        pmf = w / w.sum()
    elif fill == "size_uniform":
        pmf = np.ones(n - 1) / (n - 1)
    elif fill == "uniform":
        pmf = None
    else:
        raise ValueError(fill)

    guard = 0
    while len(out) < budget and guard < 500 * budget:
        guard += 1
        if pmf is None:
            v = tuple(int(b) for b in rng.integers(0, 2, n))
        else:
            s = int(rng.choice(sizes, p=pmf))
            idx = rng.choice(n, size=s, replace=False)
            z = np.zeros(n, dtype=int)
            z[idx] = 1
            v = tuple(int(b) for b in z)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# -----------------------------------------------------------------------------
# rank selection: delta-based vs utility-based vs fixed
# -----------------------------------------------------------------------------


def _hamming1_pairs(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.int8)
    if X.shape[0] < 2:
        return np.empty((0, 2), dtype=int)
    d = (X[:, None, :] != X[None, :, :]).sum(-1)
    iu = np.triu_indices(X.shape[0], k=1)
    mask = d[iu] == 1
    return np.stack([iu[0][mask], iu[1][mask]], axis=1)


def select_rank(
    X: np.ndarray,
    y: np.ndarray,
    candidate_ranks: Sequence[int] = (1, 2, 4, 8),
    criterion: str = "r2_delta",
    n_splits: int = 4,
    seed: int = 42,
    fm_factory: Callable | None = None,
) -> tuple[int, dict]:
    """Cross-validated rank selection.

    criterion:
      "r2_delta" -- FACILE's proposed criterion (R^2 on Hamming-1 utility gaps)
      "r2_util"  -- standard criterion (R^2 on utilities).  THE MISSING BASELINE.
      "mse_util" -- equivalent to r2_util up to monotone transform, kept for
                    completeness
      "oracle_shapley" -- upper bound: pick the rank that maximises nDCG against
                    exact Shapley.  Report it as a ceiling so the reader can see
                    how much rank selection is leaving on the table.

    Deterministic: `KFold(shuffle=True, random_state=seed)`.  The submitted code
    passes no random_state, which makes the chosen rank -- and hence every
    FACILE number in the paper -- irreproducible.
    """
    if fm_factory is None:
        from fastFM import als  # noqa: F401  (only needed at run time)

        def fm_factory(rank):
            return als.FMRegression(n_iter=200, rank=rank, l2_reg_w=0.01,
                                    l2_reg_V=0.1, random_state=seed)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    kf = KFold(n_splits=min(n_splits, max(2, X.shape[0] // 4)),
               shuffle=True, random_state=seed)

    scores: dict[int, float] = {}
    diag: dict[int, dict] = {}
    for r in candidate_ranks:
        fold_util, fold_delta = [], []
        for tr, va in kf.split(X):
            m = fm_factory(r)
            m.fit(csr_matrix(X[tr]), y[tr])
            p = m.predict(csr_matrix(X[va]))
            fold_util.append(r2_score(y[va], p))
            pairs = _hamming1_pairs(X[va])
            if pairs.shape[0] >= 2:
                i, j = pairs[:, 0], pairs[:, 1]
                fold_delta.append(r2_score(y[va][i] - y[va][j], p[i] - p[j]))
        diag[r] = dict(
            r2_util=float(np.mean(fold_util)) if fold_util else np.nan,
            r2_delta=float(np.mean(fold_delta)) if fold_delta else np.nan,
            n_delta_folds=len(fold_delta),
        )
        scores[r] = diag[r].get(criterion, np.nan)

    # If every fold lacked Hamming-1 pairs (the failure the submitted code hits
    # silently with `continue`), fall back to utility fit and SAY SO.
    if all(not np.isfinite(v) for v in scores.values()):
        scores = {r: diag[r]["r2_util"] for r in candidate_ranks}
        for r in diag:
            diag[r]["fallback_to_r2_util"] = True

    best = max(scores, key=lambda r: (scores[r] if np.isfinite(scores[r]) else -np.inf))
    return best, diag


# -----------------------------------------------------------------------------
# interaction pruning (the component the paper describes but does not implement)
# -----------------------------------------------------------------------------


def prune_interactions(
    F: np.ndarray,
    w: np.ndarray,
    mode: str = "topq",
    q: float = 0.2,
    tau: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the pruned interaction set C and recompute attributions.

    mode:
      "none"  -- no pruning (what the code currently does)
      "topq"  -- keep the ceil(q * N(N-1)/2) pairs with largest |F_ij|
      "thresh"-- keep pairs with |F_ij| >= tau (tau defaults to 1 MAD)
      "hard"  -- keep pairs whose removal changes phi by more than 1% (slow)

    Returns (F_pruned, phi_pruned, mask).  phi = w + 0.5 * sum_{j in C} F_ij,
    matching Sec. 4.3 rather than Algorithm 1 line 27.

    Reporting this as an ablation row lets you answer "does pruning matter?"
    honestly.  If it does not change nDCG (likely, since FM interactions are
    already low-rank and shrunk by l2_reg_V), say so and drop the component --
    that is a stronger paper than one that lists a component with no effect.
    """
    n = F.shape[0]
    F = np.array(F, dtype=float)
    np.fill_diagonal(F, 0.0)
    iu = np.triu_indices(n, k=1)
    vals = np.abs(F[iu])

    if mode == "none":
        mask = np.ones_like(F, dtype=bool)
        np.fill_diagonal(mask, False)
    elif mode == "topq":
        k = max(1, int(np.ceil(q * vals.size)))
        if k >= vals.size:
            keep = np.ones_like(vals, dtype=bool)
        else:
            thresh = np.partition(vals, -k)[-k]
            keep = vals >= thresh
        mask = np.zeros_like(F, dtype=bool)
        mask[iu[0][keep], iu[1][keep]] = True
        mask |= mask.T
    elif mode == "thresh":
        if tau is None:
            med = np.median(vals)
            tau = med + 1.4826 * np.median(np.abs(vals - med))
        mask = np.abs(F) >= tau
        np.fill_diagonal(mask, False)
    else:
        raise ValueError(mode)

    Fp = np.where(mask, F, 0.0)
    phi = np.asarray(w, float) + 0.5 * Fp.sum(axis=1)
    return Fp, phi, mask


# -----------------------------------------------------------------------------
# the grid
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AblationSpec:
    name: str
    core: bool = True
    neighbors: bool = True
    fill: str = "kernelshap"
    criterion: str = "r2_delta"
    fixed_rank: int | None = None
    prune: str = "none"
    note: str = ""


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("FACILE (full)", note="all components"),
    # --- sampling ------------------------------------------------------------
    AblationSpec("-- core set", core=False,
                 note="drop empty + grand coalition"),
    AblationSpec("-- neighbor expansion", neighbors=False,
                 note="pure kernel sampling, the KernelSHAP-style baseline"),
    AblationSpec("-- kernel weighting", fill="uniform",
                 note="neighbors + uniform fill"),
    AblationSpec("-- both (uniform only)", core=False, neighbors=False,
                 fill="uniform", note="ContextCite's sampler with an FM head"),
    AblationSpec("size-uniform fill", fill="size_uniform",
                 note="isolates the kernel shape from the coverage effect"),
    # --- rank selection ------------------------------------------------------
    AblationSpec("rank sel: r2_util", criterion="r2_util",
                 note="THE MISSING BASELINE for Sec. 4.3's claim"),
    AblationSpec("rank fixed 1", fixed_rank=1),
    AblationSpec("rank fixed 2", fixed_rank=2),
    AblationSpec("rank fixed 4", fixed_rank=4),
    AblationSpec("rank fixed 8", fixed_rank=8),
    AblationSpec("rank oracle", criterion="oracle_shapley",
                 note="ceiling, not a method"),
    # --- pruning -------------------------------------------------------------
    AblationSpec("+ pruning top-20%", prune="topq"),
    AblationSpec("+ pruning MAD", prune="thresh"),
)


def run_ablation_grid(
    harness,
    qid: int,
    budgets: Sequence[int],
    exact_shapley: np.ndarray,
    exact_fsii: np.ndarray,
    specs: Sequence[AblationSpec] = ABLATIONS,
    evaluator=None,
    seed: int = 42,
    fm_factory: Callable | None = None,
) -> list[dict]:
    """Run every ablation cell at every budget for one question.

    Returns long-format rows for `facile_stats.ResultTable`, so CIs and paired
    Wilcoxon tests over ablation cells come for free -- which matters, because
    the whole point of the ablation is to say which component differences are
    real and which are noise.

    Cost: zero new LLM calls if the 2^N utility cache for this question is warm
    (it is, since you compute exact Shapley and exact FSII).
    """
    if fm_factory is None:
        from fastFM import als

        def fm_factory(rank):
            return als.FMRegression(n_iter=1000, rank=rank, l2_reg_w=0.01,
                                    l2_reg_V=0.1, random_state=seed)

    n = harness.n_items
    rows: list[dict] = []
    iu = np.triu_indices(n, k=1)
    fsii_pairs = np.asarray(exact_fsii)[iu]

    for spec in specs:
        for M in budgets:
            coalitions = structured_sample(
                n, M, core=spec.core, neighbors=spec.neighbors,
                fill=spec.fill, seed=seed,
            )
            X = np.array(coalitions, dtype=np.float64)
            y = np.array([harness.get_utility(tuple(int(b) for b in v),
                                             mode=harness.utility_mode)
                          for v in coalitions], dtype=float)

            if spec.fixed_rank is not None:
                rank, diag = spec.fixed_rank, {}
            elif spec.criterion == "oracle_shapley":
                best, best_score = None, -np.inf
                for r in (1, 2, 4, 8):
                    m = fm_factory(r)
                    m.fit(csr_matrix(X), y)
                    F_ = m.V_.T @ m.V_
                    np.fill_diagonal(F_, 0.0)
                    phi_ = m.w_ + 0.5 * F_.sum(axis=1)
                    s = _ndcg(exact_shapley, phi_, 5)
                    if s > best_score:
                        best, best_score = r, s
                rank, diag = best, {"oracle_ndcg": best_score}
            else:
                rank, diag = select_rank(X, y, criterion=spec.criterion,
                                         seed=seed, fm_factory=fm_factory)

            model = fm_factory(rank)
            model.fit(csr_matrix(X), y)
            w, V = model.w_, model.V_.T
            F = V @ V.T
            np.fill_diagonal(F, 0.0)
            F, phi, _ = prune_interactions(F, w, mode=spec.prune)

            rec = dict(qid=qid, method=spec.name, budget=M,
                       n_coalitions=len(coalitions), rank=rank)
            rec["NDCG5_marg"] = _ndcg(exact_shapley, phi, 5)
            rec["NDCG5_inter"] = _ndcg(np.abs(fsii_pairs), np.abs(F[iu]), 5)
            rec["spearman_inter"] = _spearman(fsii_pairs, F[iu])
            if evaluator is not None:
                rec["R2_util"] = evaluator.r2({spec.name: phi},
                                              {spec.name: model})[spec.name]
                rec["R2_delta"] = evaluator.delta_r2({spec.name: phi},
                                                     {spec.name: model})[spec.name]
            rec.update({f"diag_{k}": v for k, v in diag.items()
                        if np.isscalar(v)})
            rows.append(rec)
    return rows


def to_long(rows: Sequence[dict],
            metrics: Sequence[str] = ("NDCG5_marg", "NDCG5_inter",
                                      "spearman_inter", "R2_util", "R2_delta"),
            ) -> list[dict]:
    """Reshape `run_ablation_grid` output for `facile_stats.ResultTable`."""
    out = []
    for r in rows:
        for m in metrics:
            if m in r and r[m] is not None and np.isfinite(r[m]):
                out.append(dict(qid=r["qid"], metric=m, method=r["method"],
                                budget=r["budget"], k=None, value=float(r[m])))
    return out


# -----------------------------------------------------------------------------
# small metric helpers (shifted nDCG, as in Sec. 5.2.2)
# -----------------------------------------------------------------------------


def _ndcg(true_scores: np.ndarray, pred_scores: np.ndarray, k: int) -> float:
    t = np.asarray(true_scores, float).ravel()
    p = np.asarray(pred_scores, float).ravel()
    if t.size != p.size or t.size == 0:
        return np.nan
    t = t - t.min()  # Sec. 5.2.2: shift to non-negative
    if t.max() == 0:
        return np.nan
    return float(ndcg_score(t[None, :], p[None, :], k=min(k, t.size)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(np.asarray(a, float).ravel(), np.asarray(b, float).ravel())
    return float(r)


if __name__ == "__main__":
    # --- sampler component check (no model needed) ---------------------------
    n, M = 10, 64
    for spec in ABLATIONS[:6]:
        c = structured_sample(n, M, core=spec.core, neighbors=spec.neighbors,
                              fill=spec.fill)
        sizes = np.array([sum(v) for v in c])
        print(f"{spec.name:26s} n={len(c):3d} distinct={len(set(c)):3d} "
              f"has_empty={tuple([0]*n) in c} has_full={tuple([1]*n) in c} "
              f"n_singletons={(sizes==1).sum():2d} n_LOO={(sizes==n-1).sum():2d} "
              f"mean_size={sizes.mean():.2f}")

    print("\nbudget floor: FACILE needs M >= 2N+2 =", 2 * n + 2)
    c = structured_sample(n, 16)
    print("  at M=16 the structured core is truncated to", len(c), "coalitions")

    # --- pruning check -------------------------------------------------------
    rng = np.random.default_rng(0)
    V = rng.normal(0, 1, (n, 3))
    F = V @ V.T
    np.fill_diagonal(F, 0.0)
    w = rng.normal(0, 1, n)
    for mode in ("none", "topq", "thresh"):
        Fp, phi, mask = prune_interactions(F, w, mode=mode)
        kept = int(mask[np.triu_indices(n, 1)].sum())
        print(f"prune={mode:7s} kept {kept:2d}/{n*(n-1)//2} pairs  "
              f"max|dphi|={np.abs(phi - (w + 0.5*F.sum(1))).max():.4f}")
