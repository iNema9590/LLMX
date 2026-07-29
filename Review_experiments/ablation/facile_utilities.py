from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Callable, Sequence

import numpy as np
import os, sys
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from SHapRAG import *
from SHapRAG.utils import *


# -----------------------------------------------------------------------------
# text-overlap scoring (SQuAD-style, so it is comparable to the QA literature)
# -----------------------------------------------------------------------------


def normalise_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalise_answer(pred) == normalise_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    p = normalise_answer(pred).split()
    g = normalise_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec, rec = n_same / len(p), n_same / len(g)
    return 2 * prec * rec / (prec + rec)


# -----------------------------------------------------------------------------
# registering new utility modes on an existing harness
# -----------------------------------------------------------------------------


def install_utilities(
    harness,
    embedder: Callable[[Sequence[str]], np.ndarray] | None = None,
    max_new_tokens: int = 32,
    gold_answer: str | None = None,
) -> None:
    """Extend `harness._compute_response_metric` with generative utilities.

    `embedder`: callable mapping a list of strings to an (n, d) array, e.g.
        st = SentenceTransformer("all-MiniLM-L6-v2")
        embedder = lambda xs: st.encode(xs, normalize_embeddings=True)
    Only needed for "gen-similarity".

    `gold_answer`: the dataset's gold answer.  If None, the grand-coalition
    greedy response (`harness.target_response`) is used as the reference, which
    keeps the experiment self-contained but measures *self-consistency* rather
    than correctness -- state which one you report.

    IMPORTANT PROMPT BUG THIS EXPOSES.  `_llm_generate_response` builds the
    prompt as
        "You are a helpful assistant. You use the provided context ...
         ###context: {c}. ###question: {q}."
    while `_compute_response_metric` builds it as
        "You are a helpful assistant. Answer ONLY from the given CONTEXT.
         If the answer is not in context, say 'Insufficient information'.
         Otherwise answer 'Yes' or 'No'.
         ### Context:\n{c}\n\n### Question:\n{q}"
    So the response being scored was produced under a *different* prompt from
    the one used to score it, and the scoring prompt additionally instructs the
    model to answer 'Yes' or 'No' -- a leftover from a yes/no task that has no
    business in multi-hop QA and that pushes probability mass away from the
    target answer for every coalition.  Fix both prompts to be identical
    (`SHARED_PROMPT` below) and re-run; this is cheap and it removes an
    obvious attack surface.
    """
    original = harness._compute_response_metric
    ref_answer = gold_answer if gold_answer is not None else harness.target_response
    gen_cache: dict[tuple, str] = {}

    def _generate(context_str: str) -> str:
        key = (context_str,)
        if key not in gen_cache:
            gen_cache[key] = harness._llm_generate_response(
                context_str=context_str, max_new_tokens=max_new_tokens)
        return gen_cache[key]

    def patched(context_str: str, mode: str, response: str | None = None) -> float:
        if mode == "neg-perplexity":
            # the quantity the paper claims: -exp(-mean log p)
            lp_sum = original(context_str, "log-prob", response)
            n_tok = max(1, _n_answer_tokens(harness, response or ref_answer))
            # original("log-prob") is baseline-subtracted; undo it so we report a
            # true perplexity rather than a ratio
            lp_empty = original("", "log-prob", response)
            total_lp = lp_sum + lp_empty  # == log P(R | Q, S)
            return -math.exp(-total_lp / n_tok)

        if mode == "logprob-raw":
            return original(context_str, "log-prob", response) + \
                original("", "log-prob", response)

        if mode == "gen-f1":
            return token_f1(_generate(context_str), ref_answer)

        if mode == "gen-em":
            return exact_match(_generate(context_str), ref_answer)

        if mode == "gen-similarity":
            if embedder is None:
                raise RuntimeError("gen-similarity needs an `embedder`")
            gen = _generate(context_str)
            E = np.asarray(embedder([gen, ref_answer]), dtype=float)
            a, b = E[0], E[1]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            return float(a @ b / denom) if denom else 0.0

        return original(context_str, mode, response)

    harness._compute_response_metric = patched
    harness._generation_cache = gen_cache


def _n_answer_tokens(harness, response: str) -> int:
    return int(harness.tokenizer(response, add_special_tokens=False,
                                 return_tensors="pt").input_ids.shape[1])


SHARED_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided "
    "context, in as few words as possible. If the answer is not in the context, "
    "say 'Insufficient information'.\n\n"
    "### Context:\n{context}\n\n### Question:\n{question}"
)
"""Use this string in BOTH `_llm_generate_response` and
`_compute_response_metric` so the scored response and the scoring distribution
come from the same prompt.  Note it drops the stray 'answer Yes or No'."""


# -----------------------------------------------------------------------------
# the agreement experiment
# -----------------------------------------------------------------------------


def _spearman(a, b):
    from scipy.stats import spearmanr
    r, _ = spearmanr(np.ravel(a), np.ravel(b))
    return float(r)


def _kendall(a, b):
    from scipy.stats import kendalltau
    r, _ = kendalltau(np.ravel(a), np.ravel(b))
    return float(r)


def topk_overlap(a, b, k: int) -> float:
    a, b = np.ravel(np.asarray(a)), np.ravel(np.asarray(b))
    ta = set(np.argsort(-np.abs(a))[:k])
    tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / k


def sign_agreement(A, B, tau_rel: float = 0.1) -> dict:
    """Fraction of pairs whose interaction SIGN agrees between two settings.

    Used for two reviewer questions at once:
      * 3ptb Q4 / item 5 -- signs across utilities, LLM backbones, or
        correct/incorrect response groups: pass the two interaction matrices.
      * HpdH -- does the interaction ranking change under a different utility.

    Pairs whose magnitude is below tau_rel * (max |.|) in *either* setting are
    counted separately as `weak`, because calling a sign flip on a near-zero
    interaction is meaningless and inflates disagreement.
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = A[iu], B[iu]
    ta = tau_rel * (np.abs(a).max() or 1.0)
    tb = tau_rel * (np.abs(b).max() or 1.0)
    strong = (np.abs(a) >= ta) & (np.abs(b) >= tb)
    if strong.sum() == 0:
        return dict(n_strong=0, sign_agree=np.nan, weak_frac=1.0)
    agree = (np.sign(a[strong]) == np.sign(b[strong])).mean()
    return dict(n_strong=int(strong.sum()), sign_agree=float(agree),
                weak_frac=float(1 - strong.mean()))


def utility_agreement(
    harness,
    utilities: Sequence[str],
    attribute_fn: Callable[[object, str, int, int], tuple[np.ndarray, np.ndarray]],
    budget: int = 128,
    seed: int = 42,
    k: int = 3,
) -> dict:
    """Compare attributions computed under different utility functions.

    `attribute_fn(harness, utility_mode, budget, seed) -> (phi, F)` should run
    FACILE end to end with `harness.utility_mode` temporarily set.  Example:

        def attribute_fn(h, mode, M, seed):
            old, h.utility_mode = h.utility_mode, mode
            try:
                phi, F, _ = h.compute_wss(num_samples=M, seed=seed,
                                          sampling_method="bf_kernelshap",
                                          sur_type="fm_tuning",
                                          selection_metric="r2_delta")
                return np.asarray(phi), np.asarray(F)
            finally:
                h.utility_mode = old

    Returns a dict keyed by (utility_a, utility_b).  Aggregate across questions
    with `facile_stats.bootstrap_ci` and report a matrix of Spearman
    correlations -- if marginal rankings agree at rho > 0.8 but interaction
    signs agree at only 0.6, say so plainly.  A negative result here is far
    better for the paper than no result, because it converts an unexamined
    assumption into a measured, bounded caveat.
    """
    res = {}
    computed = {}
    for u in utilities:
        computed[u] = attribute_fn(harness, u, budget, seed)

    for ua, ub in [(a, b) for i, a in enumerate(utilities)
                   for b in utilities[i + 1:]]:
        pa, Fa = computed[ua]
        pb, Fb = computed[ub]
        n = len(pa)
        iu = np.triu_indices(n, k=1)
        res[(ua, ub)] = dict(
            marg_spearman=_spearman(pa, pb),
            marg_kendall=_kendall(pa, pb),
            marg_top_k=topk_overlap(pa, pb, k),
            inter_spearman=_spearman(np.abs(Fa[iu]), np.abs(Fb[iu])),
            inter_top_k=topk_overlap(Fa[iu], Fb[iu], k),
            **{f"sign_{key}": val
               for key, val in sign_agreement(Fa, Fb).items()},
        )
    return res


def correctness_split_agreement(per_question: dict[int, tuple[np.ndarray, np.ndarray]],
                                is_correct: dict[int, bool]) -> dict:
    """Interaction-sign statistics split by response correctness (3ptb Q4).

    per_question: {qid: (phi, F)};  is_correct: {qid: bool}
    Reports, for each group, the fraction of strong interactions that are
    positive -- so you can state whether incorrect answers are associated with
    more negative (interfering / redundant) interaction structure.  That is a
    concrete, testable claim, and it would be the beginning of the downstream
    application HpdH asks for (hallucination detection).
    """
    out = {}
    for label, want in (("correct", True), ("incorrect", False)):
        fr = []
        for qid, (phi, F) in per_question.items():
            if is_correct.get(qid) is not want:
                continue
            n = F.shape[0]
            iu = np.triu_indices(n, k=1)
            v = F[iu]
            thr = 0.1 * (np.abs(v).max() or 1.0)
            strong = v[np.abs(v) >= thr]
            if strong.size:
                fr.append(float((strong > 0).mean()))
        out[label] = dict(n=len(fr),
                          frac_positive=float(np.mean(fr)) if fr else np.nan,
                          sd=float(np.std(fr, ddof=1)) if len(fr) > 1 else np.nan)
    return out


if __name__ == "__main__":
    print("token_f1('Die schone Lurette', 'die schone lurette') =",
          token_f1("Die schone Lurette", "die schone lurette"))
    print("token_f1('1960', '1 September 2000')               =",
          round(token_f1("1960", "1 September 2000"), 3))
    print("exact_match('The falcon', 'falcon')                =",
          exact_match("The falcon", "falcon"))

    rng = np.random.default_rng(0)
    n = 10
    V = rng.normal(0, 1, (n, 2))
    A = V @ V.T
    np.fill_diagonal(A, 0)
    B = A + rng.normal(0, 0.3, (n, n))
    B = (B + B.T) / 2
    np.fill_diagonal(B, 0)
    print("\nsign_agreement(A, A + noise):", sign_agreement(A, B))
    print("topk_overlap of marginals:",
          topk_overlap(A.sum(1), B.sum(1), 3))
