import itertools
import json
import math
import os
import pickle
import random
import warnings
from typing import Optional, Tuple
from itertools import product
from collections import defaultdict
from weakref import ref
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import functools
import spectralexplain as spex
import shapiq
import numpy as np
from scipy.special import comb
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from fastFM import als
from shapiq import SHAPIQ
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from scipy.stats import beta as beta_dist
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .Weightedfm import*

class ContextAttribution:

    def __init__(self, items: list[str], query: str,
                 prepared_model: AutoModelForCausalLM,
                 prepared_tokenizer: AutoTokenizer,
                 accelerator: Accelerator = None,
                 verbose: bool = True,
                 utility_cache_path: str = None,
                 utility_mode: str=None):
        
        self.accelerator = accelerator if accelerator else Accelerator()
        self.items = items
        self.query = query
        self.utility_mode = utility_mode
        self.model = prepared_model
        self.tokenizer = prepared_tokenizer
        self.verbose = verbose
        self.n_items = len(items)
        self.device = self.accelerator.device

        if not items: raise ValueError("items list cannot be empty")
        
        # Nested cache for multiple utility types
        self.utility_cache = defaultdict(dict)
        self.full_budget=pow(2,self.n_items)
        # self.scaler = StandardScaler()
        # Model and tokenizer setup
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


        self._factorials = {k: math.factorial(k) for k in range(self.n_items + 1)}
        
        # Step 1: Main process attempts to load the cache.
        loaded_cache_on_main = None
        if utility_cache_path and os.path.exists(utility_cache_path):
            if self.accelerator.is_main_process:
                if self.verbose: print(f"Main Process: Attempting to load utility cache from {utility_cache_path}...")
                try:
                    with open(utility_cache_path, "rb") as f:
                        loaded_cache_on_main = pickle.load(f)
                    if not isinstance(loaded_cache_on_main, (dict, defaultdict)):
                        print("Warning: Loaded cache is not a dictionary. Ignoring.")
                        loaded_cache_on_main = None
                    elif self.verbose:
                        print(f"Successfully loaded {len(loaded_cache_on_main)} cached utility entries.")
                except Exception as e:
                    if self.verbose: print(f"Warning: Failed to load cache from {utility_cache_path}: {e}")
                    loaded_cache_on_main = None
        
        # Step 2: Main process puts its result (loaded cache or None) into a list for broadcasting.
        # All other processes have None.
        object_to_broadcast = [loaded_cache_on_main] if self.accelerator.is_main_process else [None]

        # Step 3: Broadcast the object from the main process to all others.
        broadcast_object_list(object_to_broadcast, from_process=0)
        
        # Step 4: All processes now have the same cache object.
        loaded_cache_from_broadcast = object_to_broadcast[0]
        
        # Initialize the instance's cache.
        if loaded_cache_from_broadcast:
            self.utility_cache = defaultdict(dict, loaded_cache_from_broadcast)
        else:
            # If nothing was loaded, it remains an empty defaultdict.
            self.utility_cache = defaultdict(dict)

        # Synchronize all processes to ensure the cache is set before proceeding.
        self.accelerator.wait_for_everyone()
        
        # --- Target Response Generation (Main process generates, then broadcasts) ---
        target_response_obj = [None]
        if self.accelerator.is_main_process:
            target_response_obj[0] = self._llm_generate_response(context_str="\n\n".join(self.items))
        
        broadcast_object_list(target_response_obj, from_process=0)
        self.target_response = target_response_obj[0]

    def get_utility(self, subset_tuple: tuple, mode: str="log-perplexity") -> float:
        """Gatekeeper for utility values. Returns from cache or computes if not present."""
        if mode in self.utility_cache.get(subset_tuple, {}):
            return self.utility_cache[subset_tuple][mode]
        
        # Compute the utility if not found in cache
        # print(f"Computing utility for subset {subset_tuple} in mode '{mode}'...")
        context_str = self._get_ablated_context_from_vector(np.array(subset_tuple))
        utility = self._compute_response_metric(context_str=context_str, mode=mode)
        
        # Store in the nested cache structure
        self.utility_cache[subset_tuple][mode] = utility
        return utility

    def save_utility_cache(self, file_path: str):
        """
        Saves the current state of the utility cache to a file.
        This operation is only performed by the main process to prevent race conditions.
        """
        if self.accelerator.is_main_process:
            if self.verbose:
                print(f"Main Process: Saving {len(self.utility_cache)} utility entries to {file_path}...")
            
            # Convert defaultdict to a standard dict for safer pickling/reloading
            cache_to_save = dict(self.utility_cache)
            
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(cache_to_save, f)
                if self.verbose:
                    print("Save complete.")
            except Exception as e:
                print(f"Error: Failed to save utility cache to {file_path}. Reason: {e}")
        
        # It's good practice to wait for the main process to finish writing
        # before other processes might proceed to exit or do other things.
        self.accelerator.wait_for_everyone()
    
    # --- LLM Interaction Methods ---
    def _llm_generate_response(self, context_str: str, max_new_tokens: int = 100) -> str:
        # ... (same as previous implementation)
        messages = [{"role": "system", "content": """You are a helpful assistant. You use the provided context to answer
                questions in few words. Avoid using your own knowledge or make assumptions.
                """}]
        if context_str:
            messages.append({"role": "user", "content": f"###context: {context_str}. ###question: {self.query}."})
        else:
            messages.append({"role": "user", "content": self.query})
        chat_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokenized = self.tokenizer(chat_text, return_tensors="pt", padding=True)
        input_ids, attention_mask = tokenized["input_ids"].to(self.device), tokenized["attention_mask"].to(self.device)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            outputs_gen = unwrapped_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False,temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else unwrapped_model.config.eos_token_id
            )
        generated_ids = outputs_gen.sequences if hasattr(outputs_gen, 'sequences') else outputs_gen
        response_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_text = response_text.lstrip().removeprefix("assistant").lstrip(": \n").strip()
        del input_ids, attention_mask, generated_ids, unwrapped_model, outputs_gen; torch.cuda.empty_cache()
        return cleaned_text

    def _compute_response_metric(self, context_str: str, mode: str, response: str = None) -> float:
        if response is None:
            response = self.target_response

        answer_ids = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        num_answer_tokens = answer_ids.shape[1]
        if num_answer_tokens == 0:
            return 0.0

        def _compute_logprob_for_context(c_str: str):
            sys_msg = {
                "role": "system",
                "content": """You use the provided context to answer
                            questions in few words. Avoid using your own knowledge or make assumptions."""
            }
            user_content = f"### Context:\n{c_str}\n\n### Question:\n{self.query}" if c_str else self.query
            messages = [sys_msg, {"role": "user", "content": user_content}]
            prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.device)
            full_input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
            prompt_len = prompt_ids.shape[1]

            with torch.no_grad():
                logits = self.model(input_ids=full_input_ids).logits

            shift_logits = logits[..., prompt_len - 1:-1, :].contiguous()
            log_probs_all = F.log_softmax(shift_logits, dim=-1)
            answer_log_probs = torch.gather(log_probs_all, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
            total_log_prob = answer_log_probs.sum()

            # cleanup
            del logits, shift_logits, log_probs_all, answer_log_probs
            torch.cuda.empty_cache()

            return total_log_prob

        if mode in ['log-prob', 'raw-prob', 'logit-prob', 'log-perplexity']:
            # compute log-probs with context and with empty context
            log_prob_with = _compute_logprob_for_context(context_str)
            log_prob_empty = _compute_logprob_for_context("")

            # normalize by length if needed
            if mode == 'log-prob':
                final_metric = log_prob_with-log_prob_empty
            elif mode == 'raw-prob':
                prob_with = torch.exp(log_prob_with)
                prob_empty = torch.exp(log_prob_empty)
                final_metric = prob_with - prob_empty
            elif mode == 'logit-prob':
                prob_with = torch.exp(log_prob_with)
                prob_empty = torch.exp(log_prob_empty)
                logit_with = logit(prob_with) if 0.0 < prob_with < 1.0 else (float('inf') if prob_with >= 1.0 else -float('inf'))
                logit_empty = logit(prob_empty) if 0.0 < prob_empty < 1.0 else (float('inf') if prob_empty >= 1.0 else -float('inf'))
                final_metric = logit_with - logit_empty
            elif mode == 'log-perplexity':
                final_metric = (log_prob_with - log_prob_empty) / num_answer_tokens

            return final_metric.item() if isinstance(final_metric, torch.Tensor) else final_metric

        elif mode == 'divergence_utility':
            # keep your divergence-based utility as-is
            if not hasattr(self, '_baseline_distributions'):
                if self.verbose and self.accelerator.is_main_process:
                    print("  (Divergence Utility) Caching baseline token distributions for full context...")
                full_context = "\n\n".join(self.items)
                self._baseline_distributions = self._get_response_token_distributions(full_context, response)

            baseline_distributions = self._baseline_distributions
            if baseline_distributions.nelement() == 0:
                return 0.0

            ablated_distributions = self._get_response_token_distributions(context_str, response)
            total_jsd = 0.0
            if ablated_distributions.nelement() > 0 and len(baseline_distributions) == len(ablated_distributions):
                for token_idx in range(len(baseline_distributions)):
                    p, q = baseline_distributions[token_idx], ablated_distributions[token_idx]
                    total_jsd += self._jensen_shannon_divergence(p, q)
            else:
                return 0.0  # Return low utility if distributions are invalid/mismatched

            beta_param = 1.0
            utility = math.exp(-beta_param * total_jsd)
            return utility
        else:
            raise ValueError(f"Invalid mode for _compute_response_metric: '{mode}'")

    def _calculate_exact(self, method: str) -> np.ndarray:
        explainer = shapiq.game_theory.exact.ExactComputer(
            n_players=self.n_items,
            game=self._make_value_function(self.utility_mode, scale=True)
        )
        if method == 'SV':
            values = explainer('SV')
        elif method == 'BV':
            values = explainer('BV')
        return values.values[1:]
    
    def compute_shapley_interaction_index_pairs_matrix(self) -> np.ndarray:
        n = self.n_items
        interaction_matrix = np.zeros((n, n), dtype=float)

        item_indices = list(range(n))
        pbar_pairs = tqdm(
            list(itertools.combinations(item_indices, 2)),
            desc=f"Pairwise Interactions (mode={self.utility_mode})",
            disable=not self.verbose,
        )

        for i, j in pbar_pairs:  # i < j guaranteed
            interaction_sum_for_pair_ij = 0.0
            remaining_indices = [idx for idx in item_indices if idx != i and idx != j]
            num_subsets = 2 ** len(remaining_indices)

            for k_s in range(num_subsets):
                # build subset S from bits of k_s
                v_S_np = np.zeros(n, dtype=int)
                for bit_pos, idx in enumerate(remaining_indices):
                    if (k_s >> bit_pos) & 1:
                        v_S_np[idx] = 1
                v_S_tuple = tuple(v_S_np)

                # construct S ∪ {i}, S ∪ {j}, S ∪ {i,j}
                v_S_union_i_tuple = tuple(v_S_np | (np.arange(n) == i))
                v_S_union_j_tuple = tuple(v_S_np | (np.arange(n) == j))
                v_S_union_ij_tuple = tuple(v_S_np | (np.arange(n) == i) | (np.arange(n) == j))

                # fetch utilities through cache/computation
                util_S = self.get_utility(v_S_tuple, mode=self.utility_mode)
                util_S_i = self.get_utility(v_S_union_i_tuple, mode=self.utility_mode)
                util_S_j = self.get_utility(v_S_union_j_tuple, mode=self.utility_mode)
                util_S_ij = self.get_utility(v_S_union_ij_tuple, mode=self.utility_mode)

                delta_ij_S = util_S_ij - util_S_i - util_S_j + util_S

                if n == 2:
                    weight = 1.0
                else:
                    size_S = sum(v_S_np)
                    numerator = self._factorials[size_S] * self._factorials[n - size_S - 2]
                    denominator = self._factorials[n - 1]
                    weight = numerator / denominator

                interaction_sum_for_pair_ij += weight * delta_ij_S

            interaction_matrix[i, j] = interaction_sum_for_pair_ij
            interaction_matrix[j, i] = interaction_sum_for_pair_ij  # symmetry

        return interaction_matrix

    def _get_response_token_distributions(self, context_str: str, response: str) -> torch.Tensor:
        # ... (same as previous implementation)
        answer_ids = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        L = answer_ids.shape[1];
        if L == 0: return torch.tensor([], device=self.device)
        messages = [{
            "role": "system",
            "content": """You are a helpful assistant. You use the provided context to answer
                    questions in few words. Avoid using your own knowledge or make assumptions."""
        },
                    {"role": "user", "content": f"###context: {context_str}. ###question: {self.query}." if context_str else self.query}]
        prompt_str = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.device)
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1); prompt_len = prompt_ids.shape[1]
        with torch.no_grad(): logits = self.model(input_ids=input_ids).logits
        shifted_logits = logits[..., prompt_len - 1:-1, :].contiguous()
        distributions = F.softmax(shifted_logits, dim=-1).squeeze(0)
        del logits, shifted_logits, prompt_ids, answer_ids, input_ids; torch.cuda.empty_cache()
        return distributions

    @staticmethod
    def _jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> float:
        # ... (same as previous implementation)
        p, q = p + epsilon, q + epsilon; p /= p.sum(); q /= q.sum()
        m = 0.5 * (p + q)
        return 0.5 * (F.kl_div(m.log(), p, reduction='sum') + F.kl_div(m.log(), q, reduction='sum')).item()
    

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    def _generate_sampled_ablations(
        self,
        num_samples: int,
        sampling_method: str = "uniform",
        seed: int = None,
        pair_fraction=0.75
    ) -> list[tuple]:

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n = self.n_items
        sampled_tuples_set = set()

        empty = tuple([0] * n)
        full = tuple([1] * n)

        # Always include empty and full
        if num_samples >= 1:
            sampled_tuples_set.add(empty)
        if num_samples >= 2:
            sampled_tuples_set.add(full)

        remaining_to_sample = num_samples - len(sampled_tuples_set)
        if remaining_to_sample <= 0:
            return list(sampled_tuples_set)

        # ------------------------------------------------------------------
        # Helper functions
        # ------------------------------------------------------------------

        def hamming_neighbors(base_tuple, k):
            """Generate all tuples at Hamming distance k from base_tuple."""
            indices = range(n)
            for combo in itertools.combinations(indices, k):
                v = list(base_tuple)
                for idx in combo:
                    v[idx] = 1 - v[idx]
                yield tuple(v)

        def kernelshap_sample_one():
            sizes = np.arange(1, n)
            weights = (n - 1) / (sizes * (n - sizes))
            probabilities = weights / weights.sum()

            z = np.random.choice(sizes, p=probabilities)
            indices = np.random.choice(n, size=z, replace=False)
            v = np.zeros(n, dtype=int)
            v[indices] = 1
            return tuple(v)

        # ------------------------------------------------------------------
        # Sampling strategies
        # ------------------------------------------------------------------

        if sampling_method == "uniform":
            while len(sampled_tuples_set) < num_samples:
                sampled_tuples_set.add(tuple(np.random.randint(0, 2, n)))

        elif sampling_method == "kernelshap":
            while len(sampled_tuples_set) < num_samples:
                sampled_tuples_set.add(kernelshap_sample_one())

        elif sampling_method == "bf":
            # Deterministic BFS over Hamming distance
            for k in range(1, n + 1):
                for v in hamming_neighbors(empty, k):
                    if len(sampled_tuples_set) >= num_samples:
                        break
                    sampled_tuples_set.add(v)

                for v in hamming_neighbors(full, k):
                    if len(sampled_tuples_set) >= num_samples:
                        break
                    sampled_tuples_set.add(v)

                if len(sampled_tuples_set) >= num_samples:
                    break

        elif sampling_method == "bf_uniform":
            # First: Hamming distance 1 neighbors
            for v in hamming_neighbors(empty, 1):
                sampled_tuples_set.add(v)
            for v in hamming_neighbors(full, 1):
                sampled_tuples_set.add(v)

            # Fill rest uniformly
            while len(sampled_tuples_set) < num_samples:
                v = tuple(np.random.randint(0, 2, n))
                sampled_tuples_set.add(v)

        elif sampling_method == "bf_kernelshap":
            # First: Hamming distance 1 neighbors
            for v in hamming_neighbors(empty, 1):
                sampled_tuples_set.add(v)
            for v in hamming_neighbors(full, 1):
                sampled_tuples_set.add(v)

            # Fill rest using KernelSHAP distribution
            while len(sampled_tuples_set) < num_samples:
                sampled_tuples_set.add(kernelshap_sample_one())

        else:
            raise ValueError("Please input a valid sampling method")

        return list(sampled_tuples_set)


    def _train_surrogate(self, ablations: list[tuple], utilities: list[float], sur_type="linear",rank=None, candidate_ranks=None, pair_indices=None, selection_metric=None) -> Tuple[object, np.ndarray, Optional[np.ndarray]]:
        """Internal method to train a surrogate model on utility data."""
        # utilities_scaled=self.scaler.transform(np.array(utilities).reshape(-1,1)).flatten()
        X_train = np.array(ablations)
        y_train = np.array(utilities)
        if sur_type == "linear":
            model = Lasso(alpha=0.01, fit_intercept=True, random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            return model, model.coef_, None

        elif sur_type == "fm":
            X_train_fm = csr_matrix(X_train)
            model = als.FMRegression(
                n_iter=2000,
                rank=rank,
                l2_reg_w=0.1,
                l2_reg_V=0.1,
                random_state=42
            )
            model.fit(X_train_fm, y_train)

            w, V = model.w_, model.V_.T
            F = V @ V.T
            np.fill_diagonal(F, 0.0)
            attr = w + 0.5 * F.sum(axis=1)

            return model, attr, F
        
        elif sur_type == "fm_tuning":
            X_train_fm = csr_matrix(X_train)

            # --- Rank tuning if rank not provided ---
            n_splits = 4
            kf = KFold(n_splits=n_splits, shuffle=True)
            results = {}

            for r in candidate_ranks:
                fold_metrics = defaultdict(list)
                for train_idx, val_idx in kf.split(X_train_fm):
                    X_tr, X_val = X_train_fm[train_idx], X_train_fm[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    model = als.FMRegression(
                        n_iter=200,
                        rank=r,
                        l2_reg_w=0.01,
                        l2_reg_V=0.1,
                        random_state=42
                    )
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                    pairs = pairs_hamming_1_bitmask(np.asarray(X_train[val_idx]).astype(np.uint8))
                    if len(pairs)==0:
                        continue

                    i, j = pairs[:, 0], pairs[:, 1]

                    pred_deltas = preds[i] - preds[j]
                    true_deltas = y_val[i] - y_val[j]
                    weights = np.abs(true_deltas)
                    weights = weights / (weights.mean() + 1e-8)               
                    
                    # r2_util = model.score(X_val, y_val)
                    r2_delta = r2_score(true_deltas, pred_deltas, sample_weight=weights)
                    # mse_util = mean_squared_error(y_val, preds)
                    # mse_delta = mean_squared_error(true_deltas,pred_deltas, sample_weight=weights)
                    # fold_metrics["r2_util"].append(r2_util)
                    fold_metrics["r2_delta"].append(r2_delta)
                    # fold_metrics["mse_util"].append(mse_util)
                    # fold_metrics["mse_delta"].append(mse_delta)
                # Average R² across folds
                results[r] = {k: np.mean(v) for k, v in fold_metrics.items()}

            # Pick rank with maximum R² instead of minimum MSE
            maximize_metrics = {"r2_delta"}
            # minimize_metrics = {"mse_util",  "mse_delta"}
            best_by_metric = {}

            for metric in maximize_metrics:
                best_by_metric[metric] = max(
                    results, key=lambda r: results[r][metric]
                )

            # for metric in minimize_metrics:
            #     best_by_metric[metric] = min(
            #         results, key=lambda r: results[r][metric]
            #     )
            
            print(
                f"  Best rank by {selection_metric:>10}: "
                f"rank={best_by_metric[selection_metric]}"
            )

            # --- Train final model with best rank ---
            model = als.FMRegression(
                n_iter=1000,
                rank=best_by_metric[selection_metric],
                l2_reg_w=0.01,
                l2_reg_V=0.1,
                random_state=42
            )
            model.fit(X_train_fm, y_train)

            w, V = model.w_, model.V_.T
            F = V @ V.T
            np.fill_diagonal(F, 0.0)
            attr = w + 0.5 * F.sum(axis=1)

            return model, attr, F
        
        elif sur_type == "myfm_tuning":

            # --- Rank tuning if rank not provided ---
            n_splits = 4
            kf = KFold(n_splits=n_splits, shuffle=False)
            results = {}

            # for r in candidate_ranks:
            # fold_r2 = []
            pair_indices=np.arange(len(X_train)).reshape(-1, 2)
            for train_idx, val_idx in kf.split(pair_indices):
                X_tr, X_val = X_train[pair_indices[train_idx].flatten()], X_train[pair_indices[val_idx].flatten()]
                y_tr, y_val = y_train[pair_indices[train_idx].flatten()], y_train[pair_indices[val_idx].flatten()]
            #         model = FactorizationMachine(
            #                 rank=r,
            #                 alpha=0.01,        # L1 for w
            #                 lambda_w=0.001,    # L2 for w
            #                 lambda_v=0.01,     # L2 for V
            #                 n_iter=500,
            #                 fit_intercept=True,
            #                 random_state=42,
            #                 lr=0.005
            #             )
        
            #         model.fit(X_tr, y_tr)
            #         preds = model.predict(X_val)
            #         pred_deltas = []
            #         true_deltas = []
                    
            #         for i in range(0, len(preds), 2):
            #             idx_with = i
            #             idx_without = i + 1

            #             pred_delta = preds[idx_with] - preds[idx_without]
            #             true_delta = y_val[idx_with] - y_val[idx_without]

            #             pred_deltas.append(pred_delta)
            #             true_deltas.append(true_delta)
                    
            #         pred_deltas = np.array(pred_deltas)
            #         true_deltas = np.array(true_deltas)

            #         y_norm = (y_val - y_val.mean()) / y_val.std() + 1e-8
            #         y_pred_norm = (preds - preds.mean()) / preds.std() + 1e-8
            #         true_deltas_norm = (true_deltas - true_deltas.mean()) / true_deltas.std() + 1e-8
            #         pred_deltas_norm = (pred_deltas - pred_deltas.mean()) / pred_deltas.std() + 1e-8

            #         # ------------------------------------------------------------------
            #         # 2. MSE on normalized utilities and deltas
            #         # ------------------------------------------------------------------
            #         # X_val is a numpy array here
            #         s = X_val.sum(1)[1::2]
            #         S = X_val.sum(1)
            #         shapley_kernel_imp = 1/(s * (self.n_items - s) + 1e-6)
            #         weights=shapley_kernel_imp / shapley_kernel_imp.sum()
            #         mse = np.mean(weights*true_deltas_norm*(true_deltas_norm - pred_deltas_norm) ** 2)

            #         Shapley_kernel_imp = 1/(S * (self.n_items - S) + 1e-6)
            #         Weights=Shapley_kernel_imp / Shapley_kernel_imp.sum()
            #         MSE = np.mean(Weights*(y_norm - y_pred_norm) ** 2)
            #         # ------------------------------------------------------------------
            #         # 3. Pairwise logistic loss (ranking)
            #         #    Convert ONCE to torch, no tensor re-wrapping inside
            #         # ------------------------------------------------------------------
            #         # pred_t = torch.from_numpy(pred_deltas_norm)
            #         # true_t = torch.from_numpy(true_deltas_norm)

            #         prl = pairwise_logistic_loss(pred_deltas_norm, true_deltas_norm).item()
            #         fold_r2.append(0.7*mse + 0.3*MSE)

            #     # Average R² across folds
            #     results[r] = np.mean(prl+mse+MSE)
            ranks =list(range(1,10))
            betas = [0.0, 0.25, 0.5, 0.75, 1.0]

            best_score = np.inf
            best_params = None

            for r, beta in product(ranks, betas):
                model = FactorizationMachine(
                    rank=r,
                    beta=beta,
                    alpha=0.01,
                    lambda_w=0.0,
                    lambda_v=0.0,
                    n_iter=500,
                    fit_intercept=True,
                    random_state=42,
                    lr=0.005
                )
                
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                pred_deltas = []
                true_deltas = []
                
                for i in range(0, len(preds), 2):
                    idx_with = i
                    idx_without = i + 1

                    pred_delta = preds[idx_with] - preds[idx_without]
                    true_delta = y_val[idx_with] - y_val[idx_without]

                    pred_deltas.append(pred_delta)
                    true_deltas.append(true_delta)
                
                pred_deltas = np.array(pred_deltas)
                true_deltas = np.array(true_deltas)

                # Scale y_val and preds to be between 0 and 1
                # scaler = MinMaxScaler()
                # preds = scaler.fit_transform(preds.reshape(-1, 1)).flatten()
                # y_val = scaler.fit_transform(y_val.reshape(-1, 1)).flatten()


                y_norm = (y_val - y_val.mean()) / y_val.std() + 1e-8
                y_pred_norm = (preds - preds.mean()) / preds.std() + 1e-8
                true_deltas_norm = (true_deltas - true_deltas.mean()) / true_deltas.std() + 1e-8
                pred_deltas_norm = (pred_deltas - pred_deltas.mean()) / pred_deltas.std() + 1e-8

                s = X_val.sum(1)[1::2]
                S = X_val.sum(1)
                shapley_kernel_imp = 1/(s * (self.n_items - s) + 1e-6)
                weights=shapley_kernel_imp / shapley_kernel_imp.sum()
                mse = np.mean((true_deltas_norm - pred_deltas_norm) ** 2)

                Shapley_kernel_imp = 1/(S * (self.n_items - S) + 1e-6)
                Weights=Shapley_kernel_imp / Shapley_kernel_imp.sum()
                MSE = np.mean((y_norm - y_pred_norm) ** 2)
                # print(f"Rank={r}, Beta={beta} => MSE on deltas: {mse:.4f}, MSE on utilities: {MSE:.4f}")
                score = beta*mse + (1-beta)*MSE
                # score = ndcg_score([y_val], [preds])

                if score < best_score:
                    best_score = score
                    best_params = {"rank": r, "beta": beta}
                    # best_model = model

            print("Best params:", best_params)


            # Pick rank with maximum R² instead of minimum MSE
            # best_rank = min(results, key=results.get)
            # print(f"Selected rank={best_rank} (Combined loss={results[best_rank]:.4f})")

            # --- Train final model with best rank ---
            model = FactorizationMachine(
                            rank=best_params["rank"],
                            alpha=0.01,        # L1 for w
                            lambda_w=0.0,    # L2 for w
                            lambda_v=0.0,     # L2 for V
                            n_iter=500,
                            fit_intercept=True,
                            random_state=42,
                            lr=0.005,
                            beta=best_params["beta"]
                        )
            model.fit(X_train, y_train)

            attr, F = model.get_shapley_attributions()

            return model, attr, F
       

    def compute_contextcite(self, num_samples: int, seed: int = None):

        if not self.accelerator.is_main_process:
            # Return None or empty array on non-main processes
            return np.array([])

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Generate a list of subset tuples to evaluate
        # ContextCite uses uniform sampling of subsets.
        sampled_tuples = self._generate_sampled_ablations(
            num_samples, 
            sampling_method="uniform", 
            seed=seed
        )

        # Compute utilities on-demand for the sampled subsets
        utilities_for_samples = [self.get_utility(v_tuple, mode=self.utility_mode) for v_tuple in sampled_tuples]

        # Filter out any samples where utility computation failed
        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
        if len(valid_indices) < len(sampled_tuples):
            print(f"Warning: {len(sampled_tuples) - len(valid_indices)} utility computations failed. Training surrogate on {len(valid_indices)} samples.")

        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]

        if not utilities_for_train:
            print("Warning: No valid utilities could be computed for ContextCite. Returning empty weights.")
            return np.array([])

        model, weights, _ = self._train_surrogate(
            sampled_tuples_for_train, 
            utilities_for_train,
            sur_type="linear"
        )
        return weights, model


    def compute_wss(self, num_samples: int, seed: int = None, sampling_method=None, sur_type=None ,rank=None, candidate_ranks=[1,2,4,8], selection_metric=None):
  
        # Generate subsets and compute utilities
        sampled_tuples = self._generate_sampled_ablations(num_samples, sampling_method=sampling_method, seed=seed)
        # expand sampled_tuples to include a pair after each subset by flipping randomly one bit
        # sampled_tuples_expanded = []
        # for v_tuple in sampled_tuples:
        #     sampled_tuples_expanded.append(v_tuple)
        #     v_np = np.array(v_tuple)
        #     flip_idx = np.random.randint(0, self.n_items)
        #     v_np[flip_idx] = 1 - v_np[flip_idx]
            # sampled_tuples_expanded.append(tuple(v_np))

        utilities_for_samples = [self.get_utility(tuple(v_tuple), mode=self.utility_mode) for v_tuple in sampled_tuples]

        # Train surrogate model
        model, attr, F = self._train_surrogate(
            sampled_tuples, 
            utilities_for_samples, 
            sur_type=sur_type, rank=rank, candidate_ranks=candidate_ranks, selection_metric=selection_metric
        )
        return attr, F, model


    def compute_facilehp(self, num_samples: int, seed: int = None):
        
        rank_candidates = [1,2, 4, 8]
        beta_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Step 1: Sample subsets
        sampled_tuples = self._generate_sampled_ablations(num_samples // 2, sampling_method="kernelshap", seed=seed)

        # Step 2: Expand tuples by flipping one random bit
        sampled_tuples_expanded = []
        rng = np.random.default_rng(seed)
        for v_tuple in sampled_tuples:
            sampled_tuples_expanded.append(v_tuple)
            v_np = np.array(v_tuple)
            flip_idx = rng.integers(0, self.n_items)
            v_np[flip_idx] = 1 - v_np[flip_idx]
            sampled_tuples_expanded.append(tuple(v_np))

        # Step 3: Compute utilities
        utilities_for_samples = [self.get_utility(tuple(v_tuple), mode=self.utility_mode)
                                for v_tuple in sampled_tuples_expanded]
        
        X = np.array(sampled_tuples_expanded, dtype=np.float32) 
        y = np.array(utilities_for_samples, dtype=np.float32) 
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Step 4: Prepare data
        n_splits = 4
        kf = KFold(n_splits=n_splits, shuffle=False)

        best_loss = np.inf
        best_rank = None
        best_beta = None

        # Grid search with cross-validation
        for rank in rank_candidates:
            for beta in beta_candidates:

                fold_losses = []

                for train_idx, val_idx in kf.split(X):
                    X_train = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
                    y_train = torch.tensor(y[train_idx], dtype=torch.float32, device=device)

                    X_val = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
                    y_val = torch.tensor(y[val_idx], dtype=torch.float32, device=device)

                    model = TorchFactorizationMachine(
                        p=X.shape[1],
                        rank=rank,
                        alpha=0,
                        lambda_w=0.1,
                        lambda_v=0.1,
                        beta=beta,
                        device=device
                    ).to(device)

                    # Train on train fold
                    train_fm(
                        model,
                        X_train,
                        y_train,
                        lr=1e-2,
                        n_iter=50,
                        verbose=False
                    )

                    # Validation loss
                    val_loss = model.loss(X_val, y_val).detach().cpu().item()
                    fold_losses.append(val_loss)

                mean_val_loss = np.mean(fold_losses)

                if mean_val_loss < best_loss:
                    best_loss = mean_val_loss
                    best_rank = rank
                    best_beta = beta

        # --------------------------------------------------
        # Re-train best model on full dataset
        # --------------------------------------------------
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        best_model = TorchFactorizationMachine(
            p=X.shape[1],
            rank=best_rank,
            alpha=0.0,
            lambda_w=0.1,
            lambda_v=0.1,
            beta=best_beta,
            device=device
        ).to(device)

        train_fm(
            best_model,
            X_t,
            y_t,
            lr=1e-2,
            n_iter=100,
            verbose=False
        )

        best_shapley, best_F = best_model.get_shapley_attributions()

        print(
            f"Best CV loss: {best_loss:.4f}, "
            f"rank={best_rank}, beta={best_beta}"
        )

        return best_shapley, best_F, best_model
    
    
    def compute_facile(self, X, y, all_pairs,loss_type, rank):
         
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = TorchFactorizationMachine(
            p=X.shape[1],
            rank=rank,
            alpha=0.01,
            lambda_w=0.01,
            lambda_v=0.01,
            loss_type=loss_type,
            device=device,
        ).to(device)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        train_fm(
            model,
            X_t,
            y_t,
            lr=1e-2,
            n_iter= 100,
            verbose=True,
            all_pairs=all_pairs
        )
        shapley_values, F = model.get_shapley_attributions()

        return shapley_values.numpy(), F.numpy(), model

    def _make_value_function(self, utility_mode: str, scale=False):
        """
        Returns a callable mapping binary subset vectors -> utility,
        using cached results when available.
        """
        def value_function(subsets: list[np.ndarray | list | tuple]) -> np.ndarray:
            results = []
            for subset in subsets:
                subset_tuple = tuple(int(x) for x in subset)
                results.append(self.get_utility(subset_tuple, mode=utility_mode))
            # if scale:
            #     self.scaler.fit(np.array(results).reshape(-1,1))
            # results = self.scaler.transform(np.array(results).reshape(-1,1)).flatten()
            return np.array(results, dtype=float)

        return value_function

    def _run_spex(self, method: str, sample_budget: int, max_order: int, utility_mode: str):
        """
        Internal runner for SPEX methods.
        Returns (attributions, interactions).
        """
        if not self.accelerator.is_main_process:
            return np.zeros(self.n_items), {}

        value_function = self._make_value_function(utility_mode)
        approximator = shapiq.SPEX(n=self.n_items, index=method, max_order=max_order)
        
        moebius_interactions = approximator.approximate(budget=sample_budget, game=value_function)
        print(f"SPEX approximation completed.")
        attribution = np.zeros(self.n_items)
        interaction_terms = {}

        for pattern, coef in moebius_interactions.dict_values.items():
            order = len(pattern)
            if order == 1:
                attribution[pattern] = coef
            elif order == 2:
                interaction_terms[pattern] = coef

        return attribution, interaction_terms, moebius_interactions.dict_values

    def compute_exact_faith(self, max_order: int, aggregate: bool = True, method: str = "FSII"):
  
        explainer = shapiq.game_theory.exact.ExactComputer(
            n_players=self.n_items,
            game=self._make_value_function(self.utility_mode)
        )
        if method == 'FSII':
            interaction_values = explainer.compute_fii('FSII', max_order)
        elif method == 'FBII':
            interaction_values = explainer.compute_fii('FBII', max_order)

        n = self.n_items
        attribution = np.zeros(n)         # main + optional split interactions
        main_effects = np.zeros(n)        # store only main effects
        interaction_terms = {}            # (tuple of players) -> value

        for pattern, coef in interaction_values.dict_values.items():
            order = len(pattern)
            if order == 1:
                main_effects[pattern[0]] = coef
                attribution[pattern[0]] += coef
            elif order ==2:
                interaction_terms[pattern] = coef
                # if aggregate:  # split equally among participants
                #     share = coef / order
                #     for p in pattern:
                #         attribution[p] += share

        return main_effects, interaction_terms, interaction_values.dict_values

    def _run_proxyspex(self, method: str, sample_budget: int, max_order: int, utility_mode: str):
        """
        Internal runner for SPEX methods.
        Returns (attributions, interactions).
        """
        if not self.accelerator.is_main_process:
            return np.zeros(self.n_items), {}

        value_function = self._make_value_function(utility_mode)
        approximator = shapiq.ProxySPEX(n=self.n_items, index=method, max_order=max_order)
        
        moebius_interactions = approximator.approximate(budget=sample_budget, game=value_function)
        print(f"SPEX approximation completed.")
        attribution = np.zeros(self.n_items)
        interaction_terms = {}

        for pattern, coef in moebius_interactions.dict_values.items():
            order = len(pattern)
            if order == 1:
                attribution[pattern] = coef
            elif order == 2:
                interaction_terms[pattern] = coef

        return attribution, interaction_terms, moebius_interactions.dict_values

    def compute_shapiq(self, budget, method: str = "FSII"):
        explainer = SHAPIQ(n=self.n_items, index=method, max_order=1, top_order=False, random_state=42)
        main_effects=explainer(game=self._make_value_function(self.utility_mode), budget=budget).dict_values

        explainer2 = SHAPIQ(n=self.n_items, index=method, max_order=2, top_order=False, random_state=42)
        interaction_terms=explainer2(game=self._make_value_function(self.utility_mode), budget=budget).dict_values
        interaction_values = interaction_terms|main_effects
        return np.array(list(main_effects.values())), interaction_terms, interaction_values

    def compute_spex(self, sample_budget: int, max_order: int, method: str = "FSII"):
        """Compute attribution scores using SPEX (FSII method)."""
        return self._run_spex(method, sample_budget, max_order, self.utility_mode)

    def compute_proxyspex(self, sample_budget: int, max_order: int, method: str = "FBII"):
        """Compute attribution scores using SPEX (FBII method)."""
        return self._run_proxyspex(method, sample_budget, max_order, self.utility_mode)
    # --------------------------------------------------------------------------
    # Helper & Internal Methods
    # --------------------------------------------------------------------------
    
    def compute_jsd_for_ablated_indices(self, ablated_indices) -> float:

        # Create ablated context by removing specified documents
        ablated_items = [item for j, item in enumerate(self.items) if j not in ablated_indices]
        ablated_context_str = "\n\n".join(ablated_items)
        
        # Get baseline distributions (full context)
        full_context_str = "\n\n".join(self.items)
        baseline_distributions = self._get_response_token_distributions(
            context_str=full_context_str,
            response=self.target_response
        )
        
        # Get distributions for ablated context
        ablated_distributions = self._get_response_token_distributions(
            context_str=ablated_context_str,
            response=self.target_response
        )

        total_jsd = 0.0
        # Sum JSD scores over all tokens
        if ablated_distributions.nelement() > 0:
            for token_idx in range(len(baseline_distributions)):
                p = baseline_distributions[token_idx]
                q = ablated_distributions[token_idx]
                token_jsd = self._jensen_shannon_divergence(p, q)
                total_jsd += token_jsd
                
        return total_jsd
    
    def lds(self, results_dict, n_eval_util, models):
        eval_subsets = self._generate_sampled_ablations(n_eval_util, sampling_method='kernelshap', seed=2)

        # eval_subsets, _ = sample_paired_coalitions(self.n_items, n_eval_util,seed=seed)
        # sampled_tuples_expanded = []
        # for v_tuple in eval_subsets:
        #     sampled_tuples_expanded.append(v_tuple)
        #     v_np = np.array(v_tuple)
        #     flip_idx = np.random.randint(0, self.n_items)
        #     v_np[flip_idx] = 1 - v_np[flip_idx]
        #     sampled_tuples_expanded.append(tuple(v_np))
        X_all = np.array(eval_subsets)
        exact_utilities = [self.get_utility(tuple(v_tuple), mode=self.utility_mode) for v_tuple in eval_subsets]
        X_all_sparse = csr_matrix(X_all)
        lds = {}
        # Predict effects for all subsets using surrogates
        for method_name, scores in results_dict.items():
            if "FACILE" in method_name:
                model = models[method_name]                
                predicted_effect = model.predict(X_all_sparse)
            elif "II" in method_name or "pex" in method_name in method_name:
                predicted_effect = np.zeros(len(X_all))
                for i, x in enumerate(X_all):
                    for loc, coef in models[method_name].items():
                        if all(x[l] == 1 for l in loc):
                            predicted_effect[i] += coef
            elif "ContextCite" in method_name:
                predicted_effect=models[method_name].predict(X_all)
            
            # elif "Facile" in method_name:
            #     model=models[method_name]
            #     predicted_effect= model.predict(torch.tensor(X_all, dtype=torch.float32, device="cuda"))
            else:
                predicted_effect = [np.dot(scores, i) for i in X_all]

            try:
                spearman,_ = spearmanr(exact_utilities, predicted_effect)
            except Exception:
                spearman = float('nan')
            lds[method_name] = spearman
        return lds
    
    def r2(self, results_dict, n_eval_util, models):
        # eval_subsets, _ = sample_paired_coalitions(self.n_items, n_eval_util, seed=seed)
        eval_subsets = self._generate_sampled_ablations(n_eval_util, sampling_method='kernelshap', seed=2)

        # eval_subsets, _ = sample_paired_coalitions(self.n_items, n_eval_util,seed=seed)
        # sampled_tuples_expanded = []
        # for v_tuple in eval_subsets:
        #     sampled_tuples_expanded.append(v_tuple)
        #     v_np = np.array(v_tuple)
        #     flip_idx = np.random.randint(0, self.n_items)
        #     v_np[flip_idx] = 1 - v_np[flip_idx]
        #     sampled_tuples_expanded.append(tuple(v_np))
        X_all = np.array(eval_subsets)
        exact_utilities = [self.get_utility(tuple(v_tuple), mode=self.utility_mode) for v_tuple in eval_subsets]
        X_all_sparse = csr_matrix(X_all)
        r2_scores={}
        for method_name, scores in results_dict.items():
            if "FACILE" in method_name:
                model = models[method_name]
                predicted_effect = model.predict(X_all_sparse)

            elif "ContextCite" in method_name:
                predicted_effect=models[method_name].predict(X_all)
            
            # elif "Facile" in method_name:
            #     model=models[method_name]
            #     predicted_effect= model.predict(torch.tensor(X_all, dtype=torch.float32, device="cuda"))

            elif "II" in method_name or "pex" in method_name in method_name:
                predicted_effect = np.zeros(len(X_all))
                for i, x in enumerate(X_all):
                    for loc, coef in models[method_name].items():
                        # print(loc)
                        if all(x[l] == 1 for l in loc):
                            predicted_effect[i] += coef

            else:
                predicted_effect=[np.dot(scores, i) for i in X_all]

            r2_scores[method_name]=r2_score(exact_utilities, predicted_effect)
        return r2_scores


    def recall_at_k(self, gtset_k, results_dict, k_val ):
        recall={}
        for method_name, scores in results_dict.items():
            rec=[]
            for i in k_val:
                topk= np.array(scores).argsort()[-i:]
                rec.append(len(set(gtset_k).intersection(topk))/len(gtset_k))
            recall[method_name]=rec
        return recall
        
    def delta_r2(self, results_dict, num_samples, models=None):

        # Generate evaluation samples using KernelSHAP distribution
        # sampled_tuples, _ = sample_paired_coalitions(n=self.n_items, num_samples=num_samples, seed=seed)
        eval_subsets = self._generate_sampled_ablations(num_samples, sampling_method='kernelshap', seed=2)

        # eval_subsets, _ = sample_paired_coalitions(self.n_items, n_eval_util,seed=seed)
        sampled_tuples_expanded = []
        for v_tuple in eval_subsets:
            sampled_tuples_expanded.append(v_tuple)
            v_np = np.array(v_tuple)
            flip_idx = np.random.randint(0, self.n_items)
            v_np[flip_idx] = 1 - v_np[flip_idx]
            sampled_tuples_expanded.append(tuple(v_np))
        X=np.array(sampled_tuples_expanded)
        y =np.array( [self.get_utility(tuple(v_tuple), mode=self.utility_mode) for v_tuple in sampled_tuples_expanded])
        # Build delta pairs
        all_indecies = pairs_hamming_1_bitmask(X)

        true_delta = y[all_indecies[:,0]] - y[all_indecies[:,1]]

            
        delta_r2_scores = {}
        
        # For each method, compute predicted deltas and calculate R²
        for method_name, scores in results_dict.items():
            
            # Handle factorization machine models
            if "FACILE" in method_name:
                # Prepare all S vectors and S\{i} vectors in batch
                
                model = models[method_name]
                
                X_sparse= csr_matrix(X)
                y_pred = model.predict(X_sparse)
                pred_deltas = y_pred[all_indecies[:,0]] - y_pred[all_indecies[:,1]]
            
            # Handle interaction-based models
            elif "II" in method_name or "pex" in method_name in method_name:
                y_pred = np.zeros(len(X))
                for i, x in enumerate(X):
                    for loc, coef in models[method_name].items():
                        if all(x[l] == 1 for l in loc):
                            y_pred[i] += coef
                pred_deltas = y_pred[all_indecies[:,0]] - y_pred[all_indecies[:,1]]

            elif "ContextCite" in method_name:
                model=models[method_name]
                y_pred= model.predict(X)
                pred_deltas = y_pred[all_indecies[:,0]] - y_pred[all_indecies[:,1]]

            # elif "Facile" in method_name:
            #     model=models[method_name]
            #     y_pred= model.predict(torch.tensor(X, dtype=torch.float32, device="cuda"))
            #     pred_deltas = y_pred[all_indecies[:,0]] - y_pred[all_indecies[:,1]]
            
            # Handle linear attribution models
            else:
                pred_deltas = np.dot(X[all_indecies[:,0]], scores)- np.dot(X[all_indecies[:,1]], scores)
            
            delta_r2_scores[method_name] = r2_score(true_delta, pred_deltas)
                        
        return delta_r2_scores
    
    
    def evaluate_topk_performance(self, results_dict, models, k_values:list):
    
        n_docs = self.n_items
        evaluation_results = {}
        # Get full context utility based on type
        full_utility = self.get_utility(tuple([1] * n_docs), mode=self.utility_mode)

        for method_name, scores in results_dict.items():
            # Skip non-attribution results
            method_drops = {}
            for k in k_values:
                if k > n_docs:
                    continue
                # Get indices of top k documents
                # if "FM_Weights" in method_name:
                #     topk_indices=self.compute_exhaustive_top_k(k, np.argsort(scores)[-10:], model=models[method_name])
                # elif "Exact" in method_name:
                #     topk_indices=self.compute_exhaustive_top_k(k, np.argsort(scores)[-10:])
                # else:
                topk_indices = np.argsort(scores)[-k:]
                # Ensure topk_indices is a 1-dimensional array of integers
                # If topk_indices could be a scalar for k=1 or similar, convert it to an array
                if not isinstance(topk_indices, np.ndarray):
                    topk_indices = np.array([topk_indices])
                elif topk_indices.ndim > 1:
                    topk_indices = topk_indices.flatten()
                # Create ablation vector without top k
                ablation_vector = np.ones(n_docs, dtype=int)
                ablation_vector[topk_indices] = 0
                # Compute utility without top k
                util_without_topk = self.get_utility(tuple(ablation_vector), mode=self.utility_mode)
                # Calculate utility drop
                if util_without_topk != -float('inf') and full_utility != -float('inf'):
                    drop = full_utility - util_without_topk
                else:
                    drop = float('nan')
                method_drops[k] = drop
            evaluation_results[method_name] = method_drops
        return evaluation_results

    def precision(self, gtset_k, inf_scores):
        k=len(gtset_k)
        topk= np.array(inf_scores).argsort()[-k:]
        prec= len(set(gtset_k).intersection(topk))/k
        return prec
    

def logit(p, eps=1e-7):
    """Safe logit calculation with clamping to avoid numerical instability"""
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))


def shapley_kernel_weight(ablations: np.ndarray) -> np.ndarray:

    n_features = ablations.shape[1]
    coalition_sizes = ablations.sum(axis=1)
    
    # Avoid division by zero for empty or full sets
    weights = np.zeros_like(coalition_sizes, dtype=float)
    mask = (coalition_sizes > 0) & (coalition_sizes < n_features)
    
    weights[mask] = ((n_features - 1) / 
                     (comb(n_features, coalition_sizes[mask]) * 
                      coalition_sizes[mask] * (n_features - coalition_sizes[mask])))
    
    return weights



def soft_threshold(x: float, threshold: float) -> float:
    """Soft thresholding operator for LASSO."""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0



class TorchFactorizationMachine(nn.Module):
    def __init__(
        self,
        p: int,
        rank: int = 0,
        alpha: float = 0.01,
        lambda_w: float = 0.01,
        lambda_v: float = 0.01,
        fit_intercept: bool = True,
        loss_type: str = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.fit_intercept = fit_intercept
        self.device = device
        self.loss_type = loss_type

        self.w = nn.Parameter(torch.zeros(p, device=device))

        if fit_intercept:
            self.w0 = nn.Parameter(torch.zeros(1, device=device))
        else:
            self.register_parameter("w0", None)

        if rank > 0:
            self.V = nn.Parameter(
                0.01 * torch.randn(p, rank, device=device)
            )
        else:
            self.register_parameter("V", None)

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        y = X @ self.w

        if self.fit_intercept:
            y = y + self.w0

        if self.rank > 0:
            XV = X @ self.V
            interaction = 0.5 * torch.sum(
                XV ** 2 - (X ** 2) @ (self.V ** 2),
                dim=1
            )
            y = y + interaction

        return y

    # ------------------------------------------------------------------
    def loss(self, X: torch.Tensor, y: torch.Tensor, all_pairs: list) -> torch.Tensor:
        y_pred = self.forward(X)
        pairs = torch.tensor(all_pairs, device=X.device)
        if self.loss_type == "util_mse":
            mse = torch.mean((y - y_pred) ** 2)
        elif self.loss_type == "delta_mse":
            # return all possible pairs with Hamming distance 1
           

            # Calculate the MSE over the deltas between pairs
            i, j = pairs[:, 0], pairs[:, 1]

            pred_deltas = y_pred[i] - y_pred[j]
            true_deltas = y[i] - y[j]
            weights = torch.abs(true_deltas)
            weights = weights / (weights.mean() + 1e-8)
            mse = torch.mean(weights*(true_deltas - pred_deltas) ** 2)
                    
        # Regularization
        reg = (
            self.alpha * torch.sum(torch.abs(self.w)) +
            0.5 * self.lambda_w * torch.sum(self.w ** 2)
        )

        if self.rank > 0:
            reg = reg + 0.5 * self.lambda_v * torch.sum(self.V ** 2)

        return mse + reg

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_shapley_attributions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rank == 0 or self.V is None:
            return self.w.detach().clone(), None

        F = self.V @ self.V.T
        F.fill_diagonal_(0.0)

        attr = self.w + 0.5 * F.sum(dim=1)
        return attr.detach().cpu(), F.detach().cpu()
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict with the trained FM model.
        Returns CPU NumPy array.
        """
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred.detach().cpu().numpy()



def train_fm(
    model: TorchFactorizationMachine,
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 1e-2,
    n_iter: int = 200,
    verbose: bool = True,
    all_pairs: list = None
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # train half of the itterations with util_mse then the rest with delta_mse
    for it in range(n_iter):
        optimizer.zero_grad()
        loss = model.loss(X, y, all_pairs)
        loss.backward()
        optimizer.step()

# def train_fm(
#     model: TorchFactorizationMachine,
#     X: torch.Tensor,
#     y: torch.Tensor,
#     lr: float = 1e-2,
#     n_iter: int = 200,
#     verbose: bool = True,
#     all_pairs: list = None
# ):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     half = n_iter // 2

#     # ------------------------
#     # Phase 1: util_mse
#     # ------------------------
#     model.loss_type = "util_mse"
#     for it in range(half):
#         optimizer.zero_grad()
#         loss = model.loss(X, y, all_pairs)
#         loss.backward()
#         optimizer.step()

#         # if verbose and it % 20 == 0:
#         #     print(f"[util_mse] Iter {it:4d} | loss={loss.item():.6f}")

#     # ------------------------
#     # Phase 2: delta_mse
#     # ------------------------
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     model.loss_type = "delta_mse"
#     for it in range(half, n_iter):
#         optimizer.zero_grad()
#         loss = model.loss(X, y, all_pairs)
#         loss.backward()
#         optimizer.step()





def pairs_hamming_1_bitmask(vectors):
    masks = [int("".join(map(str, v)), 2) for v in vectors]
    result = []

    for i, j in itertools.combinations(range(len(masks)), 2):
        x = masks[i] ^ masks[j]
        if x and (x & (x - 1)) == 0:  # power of two → one differing bit
            result.append((i, j))
    # print(f"Found {len(result)} pairs with Hamming distance 1.")
    return np.array(result)

def pairs_hamming_1_torch(X):
    """
    X: torch.Tensor (N, d), values in {0,1} or {0.0,1.0}
    returns: list of (i, j, diff_idx)
    """
    N, d = X.shape

    masks = (X << torch.arange(d, device=X.device)).sum(dim=1)

    result = []
    for i in range(N):
        xor = masks[i] ^ masks[i+1:]
        valid = (xor != 0) & ((xor & (xor - 1)) == 0)

        js = torch.nonzero(valid, as_tuple=False).flatten() + i + 1
        for j in js.tolist():
            result.append((i, j))

    return result

def pairwise_logistic_loss(pred_deltas, true_deltas):
    """
    Pairwise logistic (RankNet) loss.
    """
    # Labels: no gradients needed
    true_deltas = true_deltas.detach()

    # Pairwise comparisons
    true_i = true_deltas.unsqueeze(1)   # [n, 1]
    true_j = true_deltas.unsqueeze(0)   # [1, n]
    mask = (true_i > true_j).float()    # [n, n]

    pred_i = pred_deltas.unsqueeze(1)   # [n, 1]
    pred_j = pred_deltas.unsqueeze(0)   # [1, n]
    pred_diff = pred_i - pred_j         # [n, n]

    # Numerically stable logistic loss
    loss_mat = F.softplus(-pred_diff)

    loss = (mask * loss_mat).sum()
    num_pairs = mask.sum()

    if num_pairs > 0:
        loss = loss / num_pairs

    return loss



def sample_paired_coalitions(n, num_samples, seed=None):
    """
    Sampling method with reproducible randomness
    """
    rng = np.random.default_rng(seed)

    num_pairs = num_samples // 2
    sampled_tuples = []
    pair_indices = []
    
    # KernelSHAP distribution over |S|
    sizes = np.arange(1, n)
    weights = (n - 1) / (sizes * (n - sizes))
    probabilities = weights / weights.sum()
    
    sampled_sizes = rng.choice(
        sizes,
        size=num_pairs,
        p=probabilities
    )
    
    for k in sampled_sizes:
        # sample player i
        i = rng.integers(0, n)
        
        # sample S ⊆ N \ {i}, |S| = k
        others = [j for j in range(n) if j != i]
        chosen = rng.choice(
            others,
            size=min(k, len(others)),
            replace=False
        )
        
        S_without_i = np.zeros(n, dtype=int)
        S_without_i[chosen] = 1
        
        S_with_i = S_without_i.copy()
        S_with_i[i] = 1
        
        idx_with = len(sampled_tuples)
        sampled_tuples.append(S_with_i)
        
        idx_without = len(sampled_tuples)
        sampled_tuples.append(S_without_i)
        
        pair_indices.append((idx_with, idx_without))
    
    return sampled_tuples, np.array(pair_indices)