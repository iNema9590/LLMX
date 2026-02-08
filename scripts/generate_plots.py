#!/usr/bin/env python3
"""
generate_all_figures.py

Cleaned & organized version of your plotting pipeline.
Usage: python generate_all_figures.py
"""

import os
import re
import ast
import pickle
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ndcg_score, average_precision_score
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator


DATA_CSV = Path("../data/sampled_musique.csv")
# DATA_CSV = Path("../data/sampled_hotpot.csv")
UTILITY_CACHE_BASE_DIR_ROOT = Path(f"../Experiment_data/{DATA_CSV.stem}")

# MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  # change as needed
# MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"

FIGURE_BASE = Path("../Figures") /DATA_CSV.stem/ MODEL_PATH.split("/")[1].split("-")[0]

# plotting aesthetics   
METHOD_COLORS = {
    "FACILE": "#DD6B07",
    "Spex": "#D10505",
    "Shapiq": "#018d01",
    "ProxySpex": "#760adb",
    "Exact-Shapley": "#f2f27a",
    "Exact-FSII": "#04B49D",

    "ContextCite": "#0f91ee",
    "default": "#0d0aaf",
}

METHOD_MARKERS = {
    "FACILE": "o",
    "Spex": "s",
    "Shapiq": "D",
    "ProxySpex": "^",
    "Exac-Shapley": "p",
    "Exac-FBII": "<",
    "Exac-FSII": ">",
    "ContextCite": "v",
    "default": ".",
}

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(METHOD_COLORS.values()))
plt.rcParams.update({'font.size': 32})
plt.rcParams['axes.labelsize'] = 40
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 13

K_VALUES = [1, 2, 3, 4, 5]
FIXED_BUDGET = 264

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def family_of(method_key: str) -> str:
    """Map a method key to a color family."""
    if method_key.startswith("FACILE"):
        return "FACILE"
    elif method_key.startswith("Spex"):
        return "Spex"
    elif method_key.startswith("Shapiq"):
        return "Shapiq"
    elif method_key.startswith("ProxySpex"):
        return "ProxySpex"
    elif method_key.startswith("ContextCite"):
        return "ContextCite"
    elif method_key in ("Exact-FSII", "Exact-Shapley", "LOO", "ARC-JSD"):
        return method_key
    return "default"


def get_color(method_key: str) -> str:
    return METHOD_COLORS.get(family_of(method_key), METHOD_COLORS["default"])

def get_marker(method_key: str) -> str:
    return METHOD_MARKERS.get(family_of(method_key), METHOD_MARKERS["default"])


def ensure_dirs():
    FIGURE_BASE.mkdir(parents=True, exist_ok=True)
    logging.info("Figures will be saved to: %s", FIGURE_BASE.resolve())


def load_inputs():
    logging.info("Loading CSV and pickles...")
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"{DATA_CSV} not found.")
    dfin = pd.read_csv(DATA_CSV)
    # convert paragraphs string to list if needed
    dfin["paragraphs"] = dfin["paragraphs"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    utility_cache_base_dir = UTILITY_CACHE_BASE_DIR_ROOT / MODEL_PATH.split("/")[1]
    results_pkl = utility_cache_base_dir / "results.pkl"
    extras_pkl = utility_cache_base_dir / "extras.pkl"
    if not results_pkl.exists() or not extras_pkl.exists():
        raise FileNotFoundError(f"Pickles not found in {utility_cache_base_dir}")
    with open(results_pkl, "rb") as f:
        all_results = pickle.load(f)
    with open(extras_pkl, "rb") as f:
        extras = pickle.load(f)
    logging.info("Loaded %d experiments in all_results, %d in extras", len(all_results), len(extras))
    return dfin, all_results, extras


def GT(dfin, i):
    """Ground-truth indices for query i based on id_type column (2hop, 3hop, 4hop)."""
    t = dfin["id_type"].iloc[i] if "id_type" in dfin.columns else None
    if t == "2hop":
        return [0, 1]
    if t == "3hop":
        return [0, 1, 2]
    if t == "4hop":
        return [0, 1, 2, 3]
    # fallback: empty
    return []

# def GT(dfin, i):
#     if dfin["len_gt"][i]==2:
#         return [0,1]
#     elif dfin["len_gt"][i]==3:
#         return [0,1,2]
#     elif dfin["len_gt"][i]==4:
#         return [0,1,2,3]

# ---------------------
# 1. Marginal metrics summary
# ---------------------
def scale_pair_preserve_order(ref_arr, att_arr):
    """
    Scale reference and attribution together to [0,1] using the combined min/max so
    ranking relationships between them stay consistent.
    Returns ref_scaled, att_scaled (1D numpy arrays).
    """
    ref = np.asarray(ref_arr).reshape(-1)
    att = np.asarray(att_arr).reshape(-1)
    combined = np.concatenate([ref, att])
    minv, maxv = combined.min(), combined.max()
    if maxv == minv:
        # avoid division by zero
        ref_s = np.zeros_like(ref, dtype=float)
        att_s = np.zeros_like(att, dtype=float)
    else:
        ref_s = (ref - minv) / (maxv - minv)
        att_s = (att - minv) / (maxv - minv)
    return ref_s, att_s


def compute_marginal_ndcg_vs_budget(all_results, save_path):
    logging.info("Computing marginal NDCG vs budget...")
    # collect NDCG per method across experiments
    spearmans = {i: [] for i in all_results[0]['methods'] if "Exact" not in i}
    for method_res in all_results:
        for method, attribution in method_res['methods'].items():
            if "Exact" not in method:
                ref, att = scale_pair_preserve_order(method_res['methods']["Exact-Shapley"], attribution)
                # compute NDCG with k=5 (as in original code)
                try:
                    spear = ndcg_score([ref], [att], k=5)
                except Exception:
                    spear = 0.0
                spearmans[method].append(spear)

    # parse methods and budgets
    parsed = {}
    budgets = set()
    for key, values in spearmans.items():
        avg_val = float(np.mean(values)) if values else 0.0
        match = re.match(r"(.+?)_(\d+)$", key)
        if match:
            method, budget = match.groups()
            budget = int(budget)
            budgets.add(budget)
            parsed.setdefault(method, {})[budget] = avg_val
        else:
            parsed.setdefault(key, {})[None] = avg_val

    budgets = sorted(budgets) if budgets else [0, 1]
    plt.figure(figsize=(9, 7))
    for method, results in parsed.items():
        if None in results:
            plt.hlines(results[None], xmin=min(budgets), xmax=max(budgets),
                       linestyles='--', label=method, colors=get_color(method), linewidth=3)
        else:
            xs = sorted(results.keys())
            ys = [results[b] for b in xs]
            plt.plot(xs, ys, marker=get_marker(method), label=method, color=get_color(method), 
                    markersize= 13, linewidth=3)
    plt.xlabel("Budget")
    plt.ylabel("NDCG@5")
    plt.xscale('log', base=2)
    plt.xticks([32, 128, 512], labels=[str(t) for t in [32, 128, 512]])
    plt.gca().xaxis.set_major_locator(LogLocator(base=2))
    plt.grid(True)
    plt.tight_layout()
    out = save_path / "ndcg_vs_budget_marginal.pdf"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s", out)


def compute_ndcg_per_k(all_results, save_path):
    logging.info("Computing NDCG@k for K_VALUES: %s", K_VALUES)
    spearmans_per_k = {k: {i: [] for i in all_results[0]['methods'] if "Exact" not in i} for k in K_VALUES}
    for method_res in all_results:
        for method, attribution in method_res['methods'].items():
            if "Exact" in method:
                continue
            ref, att = scale_pair_preserve_order(method_res['methods']["Exact-Shapley"], attribution)
            for k in K_VALUES:
                try:
                    score = ndcg_score([ref], [att], k=k)
                except Exception:
                    score = 0.0
                spearmans_per_k[k][method].append(score)

    avg_spearmans_per_k = {
        k: {m: float(np.mean(vals)) for m, vals in methods_scores.items() if len(vals) > 0}
        for k, methods_scores in spearmans_per_k.items()
    }

    # Build parsed per k by families/budgets like original script
    parsed_per_k = {}
    all_budgets = set()
    for k, method_dict in avg_spearmans_per_k.items():
        parsed_per_k[k] = {}
        for key, avg_val in method_dict.items():
            match = re.match(r"(.+?)_(\d+)$", key)
            if match:
                method, budget = match.groups()
                budget = int(budget)
                all_budgets.add(budget)
                parsed_per_k[k].setdefault(method, {})[budget] = avg_val
            else:
                parsed_per_k[k].setdefault(key, {})[None] = avg_val

    # fixed budget plot (FIXED_BUDGET)
    fixed_results = {}
    for k in K_VALUES:
        for method, results in parsed_per_k[k].items():
            if FIXED_BUDGET in results:
                fixed_results.setdefault(method, {})[k] = results[FIXED_BUDGET]
            elif None in results:
                fixed_results.setdefault(method, {})[k] = results[None]
    pd.DataFrame(fixed_results).to_csv(save_path / f"ndcg@k{FIXED_BUDGET}.csv")
    plt.figure(figsize=(9, 7))
    for method, k_dict in fixed_results.items():
        ks_sorted = sorted(k_dict.keys())
        ys = [k_dict[kk] for kk in ks_sorted]
        plt.plot(ks_sorted, ys, marker=get_marker(method), label=method, color=get_color(method),
                markersize= 13, linewidth=3)
    plt.xlabel("k")
    plt.ylabel("NDCG")
    plt.grid(True)
    plt.tight_layout()
    out = save_path / f"ndcg_vs_k_fixed_budget_{FIXED_BUDGET}.pdf"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s", out)
                

def average_precision_for_query(scores, true_indices):
    # Reuse sklearn AP for binary relevance if possible; otherwise implement fallback
    true_set = set(true_indices)
    if len(true_set) == 0:
        return 0.0
    ranked = np.argsort(-np.asarray(scores))
    num_relevant = len(true_set)
    num_hits = 0
    precisions = []
    recalls = []
    for rank, idx in enumerate(ranked, start=1):
        if idx in true_set:
            num_hits += 1
        precision = num_hits / rank
        recall = num_hits / num_relevant
        precisions.append(precision)
        recalls.append(recall)
    if len(recalls) == 0:
        return 0.0
    # compute AP (simple trapezoid on recall-precision curve)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]
    recalls_ext = np.concatenate(([0.0], recalls))
    precisions_ext = np.concatenate(([precisions[0]], precisions))
    ap = 0.0
    for i in range(1, len(recalls_ext)):
        delta_r = recalls_ext[i] - recalls_ext[i - 1]
        ap += delta_r * precisions_ext[i]
    return float(ap)


def compute_prauc(all_results, dfin, save_path):
    logging.info("Computing PR-AUC / mean average precision per method...")
    # map_from_scores equivalent using sklearn where possible
    aps = {i: [] for i in all_results[0]["methods"].keys()}
    for i, rw in enumerate(all_results):
        for method in aps:
            scores = np.asarray(rw["methods"][method])
            try:
                gt = GT(dfin, i)
                ap = average_precision_score(
                    np.isin(np.arange(len(scores)), gt).astype(int),
                    scores
                ) if len(gt) > 0 else 0.0
            except Exception:
                # fallback to custom implementation
                ap = average_precision_for_query(scores, GT(dfin, i))
            aps[method].append(float(ap))

    # separate constant/budgeted methods
    constant_methods_list = ['Exact-Shapley', 'Exact-FSII']  # modify if needed
    budgeted_data = defaultdict(list)
    constant_data = {}
    for method, values in aps.items():
        mean_val = float(np.mean(values)) if values else 0.0
        if method in constant_methods_list:
            constant_data[method] = mean_val
        else:
            # parse family and budget
            parts = method.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                family = "_".join(parts[:-1])
                budget = int(parts[-1])
                budgeted_data[family].append((budget, mean_val))
            else:
                constant_data[method] = mean_val

    plt.figure(figsize=(9, 7))
    for family, items in budgeted_data.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        budgets = [b for b, _ in items_sorted]
        means = [m for _, m in items_sorted]
        plt.plot(budgets, means, marker=get_marker(family), label=family, color=get_color(family),
                markersize= 13, linewidth=3)

    plt.xlabel("Budget")
    plt.ylabel("PR-AUC (mean AP)")
    plt.xscale('log', base=2)
    plt.xticks([32, 128, 512], labels=[str(t) for t in [32, 128, 512]])
    plt.gca().xaxis.set_major_locator(LogLocator(base=2))
    plt.grid(True)
    plt.tight_layout()
    out = save_path / "prauc_vs_budget_marginal.pdf"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s", out)


# ---------------------
# 2. SURROGATE / metrics summary
# ---------------------
def summarize_and_print(all_results, k_values=[1, 2, 3, 4, 5]):
    """
    Summarize surrogate metrics (LDS, R2, Delta_R2, topk_probability, Recall).
    Returns pd.DataFrame
    """
    table_data = defaultdict(lambda: defaultdict(list))
    for res in all_results:
        metrics = res.get("metrics", {})
        for method_name, lds_val in metrics.get("LDS", {}).items():
            table_data[method_name]["LDS"].append(lds_val)
        for method_name, r2_val in metrics.get("R2", {}).items():
            table_data[method_name]["R2"].append(r2_val)
        for method_name, delta_val in metrics.get("Delta_R2", {}).items():
            table_data[method_name]["Delta_R2"].append(delta_val)
        for method_name, k_dict in metrics.get("topk_probability", {}).items():
            for k in k_values:
                if k in k_dict:
                    table_data[method_name][f"topk_probability_k{k}"].append(k_dict[k])
        for method_name, k_list in metrics.get("Recall", {}).items():
            for idx, k in enumerate(k_values):
                if idx < len(k_list):  # k_list is [recall@1, recall@2, ..., recall@k_max]
                    col_name = f"Recall@{k}"
                    table_data[method_name][col_name].append(k_list[idx])


    # compute averages and stds
    avg_table = {}
    for method, metric_dict in table_data.items():
        avg_table[method] = {metric: (float(np.nanmean(values)) if values else np.nan)
                             for metric, values in metric_dict.items()}
        for metric in ["LDS", "R2", "Delta_R2"]:
            if metric in metric_dict:
                avg_table[method][f"{metric}_std"] = float(np.nanstd(metric_dict[metric]))
    df_summary = pd.DataFrame.from_dict(avg_table, orient="index").sort_index()
    logging.info("=== Metrics Summary Across All Queries ===\n%s", df_summary.to_string(float_format='%.4f'))
    return df_summary


def plot_surrogate_metrics(df_summary, save_path):
    """Plot R2, Delta_R2, LDS, and Recall@k metrics vs budget."""
    logging.info("Plotting surrogate metrics...")
    
    # Separate constant and budgeted methods
    constant_methods = ['Exact-Shapley', 'Exact-FSII', 'Exact-FBII', 'LOO', 'ARC-JSD']
    df_reset = df_summary.reset_index().rename(columns={'index': 'method'})
    df_budgeted = df_reset[~df_reset['method'].isin(constant_methods)].copy()
    df_const = df_reset[df_reset['method'].isin(constant_methods)]
    
    # Extract family and budget for budgeted methods
    df_budgeted['family'] = df_budgeted['method'].apply(
        lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x
    )
    df_budgeted['budget'] = df_budgeted['method'].apply(
        lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else None
    )
    df_budgeted = df_budgeted.sort_values(by=['family', 'budget'])

    # Function to plot a single metric
    def plot_metric(metric, ylabel):
        plt.figure(figsize=(9, 7))
        
        # Plot budgeted families
        families = df_budgeted['family'].unique()
        for fam in families:
            subset = df_budgeted[df_budgeted['family'] == fam]
            if metric in subset.columns:
                subset_clean = subset.dropna(subset=[metric])
                if not subset_clean.empty:
                    
                    plt.plot(subset_clean['budget'], subset_clean[metric], 
                                marker=get_marker(fam), label=fam, color=get_color(fam),
                                markersize= 13, linewidth=3)
        
        # Plot constant methods as horizontal lines
        # for _, row in df_const.iterrows():
        #     if metric in row.index and pd.notna(row[metric]):
        #         plt.axhline(y=row[metric], linestyle='--', label=row['method'], 
        #                    color=get_color(row['method']))
        
        plt.xlabel("Budget")
        plt.ylabel(ylabel)
        plt.xscale('log', base=2)
        plt.xticks([32, 128, 512], labels=[str(t) for t in [32, 128, 512]])
        plt.gca().xaxis.set_major_locator(LogLocator(base=2))
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        out = save_path / f"{metric.lower()}_vs_budget.pdf"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        logging.info("Saved %s", out)

    # Plot R2, Delta_R2, LDS
    plot_metric("R2", "R²")
    plot_metric("Delta_R2", "ΔR²")
    plot_metric("LDS", "LDS")
    
    # Plot Recall@k vs k at fixed budget
    df_budgeted_fixed = df_budgeted[df_budgeted['budget'] == FIXED_BUDGET]
    
    if not df_budgeted_fixed.empty:
        recall_metrics = [f"Recall@{k}" for k in K_VALUES]
        k_values = list(K_VALUES)
        
        plt.figure(figsize=(9, 7))
        
        # Plot budgeted families at fixed budget
        for fam in df_budgeted_fixed['family'].unique():
            subset = df_budgeted_fixed[df_budgeted_fixed['family'] == fam]
            if not subset.empty:
                row = subset.iloc[0]
                recalls = [row[col] for col in recall_metrics 
                          if col in row.index and pd.notna(row[col])]
                
                if recalls:
                    plt.plot(k_values[:len(recalls)], recalls, marker=get_marker(fam), 
                            label=fam, color=get_color(fam), markersize= 13, linewidth=3)
        
        # Plot constant methods
        # for _, row in df_const.iterrows():
        # #     recalls = [row[col] for col in recall_metrics 
        # #               if col in row.index and pd.notna(row[col])]
        # #     if recalls:
        # #         avg_recall = float(np.nanmean(recalls))
        #         plt.axhline(y=avg_recall, linestyle='--', label=row['method'], 
        #                    color=get_color(row['method']))
        
        plt.xlabel("k")
        plt.ylabel("Recall@k")
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        out = save_path / f"recall_at_k_{FIXED_BUDGET}.pdf"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        logging.info("Saved %s", out)
    else:
        logging.warning(f"No methods found at budget {FIXED_BUDGET} for recall plotting")

    if not df_budgeted_fixed.empty:
        topk_metrics = [f"topk_probability_k{k}" for k in K_VALUES]
        k_values = list(K_VALUES)
        
        plt.figure(figsize=(9, 7))
        
        # Plot budgeted families at fixed budget
        for fam in df_budgeted_fixed['family'].unique():
            subset = df_budgeted_fixed[df_budgeted_fixed['family'] == fam]
            if not subset.empty:
                row = subset.iloc[0]
                topk_vals = [row[col] for col in topk_metrics 
                            if col in row.index and pd.notna(row[col])]
                
                if topk_vals:
                    plt.plot(k_values[:len(topk_vals)], topk_vals, marker=get_marker(fam), 
                            label=fam, color=get_color(fam), markersize= 13, linewidth=3)
        
        plt.xlabel('k')
        plt.ylabel('Top-k Removal Drop')
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        out = save_path / f"topk_removal_{FIXED_BUDGET}.pdf"
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        logging.info("Saved %s", out)
    else:
        logging.warning(f"No methods found at budget {FIXED_BUDGET} for top-k plotting")
# ---------------------
# 3. Interaction quality (RR, NDCG over pairs)
# ---------------------
def compute_rr_at_k(interaction, ground_truth, k):
    """
    Recovery@k for top-k interaction pairs.
    interaction: dict {(i,j): value} or square matrix (list/ndarray)
    ground_truth: set of indices
    """
    if k <= 0:
        return 0.0
    if isinstance(interaction, (list, np.ndarray)):
        mat = np.array(interaction)
        pairs = {(i, j): float(mat[i][j]) for i in range(mat.shape[0]) for j in range(mat.shape[1]) if i != j}
    elif isinstance(interaction, dict):
        pairs = interaction
    else:
        return 0.0
    sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)
    rr_sum = 0.0
    for i in range(min(k, len(sorted_pairs))):
        pair_indices = set(sorted_pairs[i][0])
        rr_sum += len(ground_truth & pair_indices) / (len(pair_indices) if len(pair_indices) else 1)
    return rr_sum / k


def extract_budget(key):
    match = re.search(r'_(\d+)$', key)
    return int(match.group(1)) if match else None


def extract_family(key):
    if key.startswith("FM"):
        return 'FM'
    
    elif key.startswith("Spex"):
        return "Spex"

    elif key.startswith("ProxySpex"):
        return "ProxySpex"

    elif key.startswith("Shapiq"):
        return "Shapiq"

    elif key.startswith("FACILE"):
        return "FACILE"


def pairs_from_exact(extras):
    for exp in extras:
        exact = exp.get('Exact-FSII')
        if exact and isinstance(exact, dict):
            return sorted(exact.keys())
    for exp in extras:
        for v in exp.values():
            if isinstance(v, dict) and v:
                return sorted(v.keys())
    return []


def vector_for_pairs(val, pairs):
    if isinstance(val, (list, np.ndarray)):
        mat = np.array(val)
        return [abs(mat[i][j]) if 0 <= i < mat.shape[0] and 0 <= j < mat.shape[1] else 0.0 for (i, j) in pairs]
    elif isinstance(val, dict):
        return [abs(val.get(pair, 0.0)) for pair in pairs]
    else:
        return [0.0 for _ in pairs]


def interaction_rr_and_ndcg(extras, dfin, save_path):
    logging.info("Computing interaction RR and NDCG...")
    # RR@5 over budgets
    rr_by_method_budget = defaultdict(lambda: defaultdict(list))
    for i, exp in enumerate(extras):
        gt = set(GT(dfin, i))
        for method, interaction in exp.items():
            budget = extract_budget(method)
            family = extract_family(method)
            if budget is not None and family is not None:
                rr = compute_rr_at_k(interaction, gt, k=5)
                rr_by_method_budget[family][budget].append(rr)
    rr_avg = {fam: {b: float(np.mean(vals)) for b, vals in bd.items()} for fam, bd in rr_by_method_budget.items()}

    plt.figure(figsize=(9, 7))
    for family, budget_rrs in rr_avg.items():
        budgets = sorted(budget_rrs.keys())
        values = [budget_rrs[b] for b in budgets]
        plt.plot(budgets, values, marker=get_marker(family), label=family, color=get_color(family),
                markersize= 13, linewidth=3)
    plt.xlabel('Budget')
    plt.ylabel('RR@5')
    plt.xscale('log', base=2)
    plt.xticks([32, 128, 512], labels=[str(t) for t in [32, 128, 512]])
    plt.gca().xaxis.set_major_locator(LogLocator(base=2))
    plt.grid(True)
    plt.tight_layout()
    out = save_path / "rr_at_4.pdf"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s", out)

    # NDCG for interactions using canonical pair ordering
    pairs = pairs_from_exact(extras)
    if not pairs:
        logging.warning("No pair ordering could be inferred from Exact-FSII or other dicts in extras. Skipping interaction NDCG.")
        return

    per_method_ndcg = defaultdict(list)
    for exp in extras:
        exact = exp.get('Exact-FSII', {})
        exact_vec = vector_for_pairs(exact, pairs)
        if np.allclose(exact_vec, 0.0):
            continue
        for method, val in exp.items():
            if method == 'Exact-FSII':
                continue
            vec = vector_for_pairs(val, pairs)
            try:
                score = ndcg_score([exact_vec], [vec], k=5)
                per_method_ndcg[method].append(score)
            except Exception:
                continue

    avg_ndcg = {m: float(np.mean(scores)) for m, scores in per_method_ndcg.items() if scores}

    family_budget = defaultdict(lambda: defaultdict(list))
    for method, score in avg_ndcg.items():
        budget = extract_budget(method)
        family = extract_family(method)
        if budget is not None and family is not None:
            family_budget[family][budget].append(score)

    family_budget_avg = {fam: {b: float(np.mean(vals)) for b, vals in bd.items()} for fam, bd in family_budget.items()}

    plt.figure(figsize=(9, 7))
    for fam, bd in family_budget_avg.items():
        xs = sorted(bd.keys())
        ys = [bd[x] for x in xs]
        plt.plot(xs, ys, marker=get_marker(fam), label=fam, color=get_color(fam),
                markersize= 13, linewidth=3)
    # constant lines for known constant methods if present
    constant_methods = ['Exact-FSII', 'Exact-Shapley', 'LOO', 'ARC-JSD']
    for cm in constant_methods:
        if cm in avg_ndcg:
            plt.axhline(y=avg_ndcg[cm], linestyle='--', label=cm)
    plt.xlabel('Budget')
    plt.ylabel('NDCG@5 (interaction)')
    plt.xscale('log', base=2)
    plt.xticks([32, 128, 512], labels=[str(t) for t in [32, 128, 512]])
    plt.gca().xaxis.set_major_locator(LogLocator(base=2))
    plt.grid(True)
    plt.tight_layout()
    out = save_path / "ndcg_at_4_interactions.pdf"
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    logging.info("Saved %s", out)

    # === Save K-based metrics for the fixed budget ===
    FIXED_BUDGET = 264
    K_VALUES = [1, 2, 3, 4, 5]

    logging.info(f"Saving RR and NDCG metrics for K={K_VALUES} at budget={FIXED_BUDGET}...")

    rows = []  # store CSV rows

    # compute RR and NDCG for each experiment and family
    for fam in family_budget_avg.keys():  # loop families
        for k in K_VALUES:
            rr_scores = []
            ndcg_scores = []

            for i, exp in enumerate(extras):
                gt = set(GT(dfin, i))

                # process RR@k for the family at fixed budget
                for method, interaction in exp.items():
                    budget = extract_budget(method)
                    family = extract_family(method)
                    if family == fam and budget == FIXED_BUDGET:
                        try:
                            rr_scores.append(compute_rr_at_k(interaction, gt, k=k))
                        except Exception:
                            pass

                # process NDCG@k
                exact = exp.get("Exact-FSII", {})
                exact_vec = vector_for_pairs(exact, pairs)
                if np.allclose(exact_vec, 0.0):
                    continue

                for method, val in exp.items():
                    budget = extract_budget(method)
                    family = extract_family(method)
                    if family == fam and budget == FIXED_BUDGET:
                        try:
                            vec = vector_for_pairs(val, pairs)
                            ndcg = ndcg_score([exact_vec], [vec], k=k)
                            ndcg_scores.append(ndcg)
                        except Exception:
                            pass

            if rr_scores or ndcg_scores:
                rows.append({
                    "family": fam,
                    "k": k,
                    "rr": np.mean(rr_scores) if rr_scores else None,
                    "ndcg": np.mean(ndcg_scores) if ndcg_scores else None
                })

    # write output CSV
    csv_path = save_path / "rr_ndcg_k_metrics.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["family", "k", "rr", "ndcg"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Saved RR/NDCG per K to %s", csv_path)


def main():
    ensure_dirs()
    dfin, all_results, extras = load_inputs()
    compute_marginal_ndcg_vs_budget(all_results, FIGURE_BASE)
    compute_ndcg_per_k(all_results, FIGURE_BASE)
    compute_prauc(all_results, dfin, FIGURE_BASE)
    df_summary = summarize_and_print(all_results, k_values=K_VALUES)
    plot_surrogate_metrics(df_summary, FIGURE_BASE)
    interaction_rr_and_ndcg(extras, dfin, FIGURE_BASE)
    logging.info("All done. Figures are under %s", FIGURE_BASE.resolve())


if __name__ == "__main__":
    main()
