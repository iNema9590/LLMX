import itertools
import json
import math
import os
import pickle
import random
import warnings
from collections import defaultdict
from scipy.stats import spearmanr
import functools
import spectralexplain as spex
import shapiq
from scipy.special import comb
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
# from fastFM.sgd import FMRegression as FMRegressionSGD   
from fastFM import als
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from scipy.stats import beta as beta_dist
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class ContextAttribution:

    def __init__(self, items: list[str], query: str,
                 prepared_model: AutoModelForCausalLM,
                 prepared_tokenizer: AutoTokenizer,
                 accelerator: Accelerator = None,
                 verbose: bool = True,
                 utility_cache_path: str = None):
        
        self.accelerator = accelerator if accelerator else Accelerator()
        self.items = items
        self.query = query
        self.model = prepared_model
        self.tokenizer = prepared_tokenizer
        self.verbose = verbose
        self.n_items = len(items)
        self.device = self.accelerator.device

        if not items: raise ValueError("items list cannot be empty")
        
        # Nested cache for multiple utility types
        self.utility_cache = defaultdict(dict)
        
        # Model and tokenizer setup
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # Note: This line might modify a model config shared across processes.
            # It's generally safe but good to be aware of.
            # self.model.config.pad_token_id = self.model.config.eos_token_id

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

    def get_utility(self, subset_tuple: tuple, mode: str) -> float:
        """Gatekeeper for utility values. Returns from cache or computes if not present."""
        if mode in self.utility_cache.get(subset_tuple, {}):
            return self.utility_cache[subset_tuple][mode]
        
        # Compute the utility if not found in cache
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
        # --- CORRECTED: Added main process guard ---
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
                "content": """You are a helpful assistant. You use the provided context to answer
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
                final_metric = log_prob_with - log_prob_empty
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


    def _calculate_shapley(self, mode: str = "logit-prob") -> np.ndarray:
        shapley_values = np.zeros(self.n_items)
        n = self.n_items
        factorials_local = self._factorials  # Already initialized in __init__

        pbar_desc = f"Calculating Shapley (mode={mode})"
        # Show tqdm only on main process for this CPU-bound calculation
        pbar_enabled = self.verbose and self.accelerator.is_main_process and n > 10  # heuristic threshold

        item_indices_iterator = range(n)
        if pbar_enabled:
            item_indices_iterator = tqdm(item_indices_iterator, desc=pbar_desc, leave=False)

        # Iterate over all items
        for i in item_indices_iterator:
            shap_i = 0.0
            # Iterate through all subsets of items that exclude i
            for subset_int in range(1 << n):
                if (subset_int >> i) & 1:
                    continue  # skip subsets that already contain i

                # Build subset as tuple of 0/1 indicators
                s_tuple = tuple((subset_int >> j) & 1 for j in range(n))
                s_size = sum(s_tuple)

                # Get utility of S
                s_util = self.get_utility(s_tuple, mode=mode)
                if s_util == -float("inf"):
                    continue  # skip failed utility

                # Get utility of S ∪ {i}
                s_union_i_list = list(s_tuple)
                s_union_i_list[i] = 1
                s_union_i_tuple = tuple(s_union_i_list)
                s_union_i_util = self.get_utility(s_union_i_tuple, mode=mode)

                # Marginal contribution
                marginal_contribution = s_union_i_util - s_util
                weight = (factorials_local[s_size] * factorials_local[n - s_size - 1]) / factorials_local[n]
                shap_i += weight * marginal_contribution

            shapley_values[i] = shap_i

        return shapley_values


    def compute_shapley_interaction_index_pairs_matrix(self, mode: str = "logit-prob") -> np.ndarray:
        n = self.n_items
        interaction_matrix = np.zeros((n, n), dtype=float)

        item_indices = list(range(n))
        pbar_pairs = tqdm(
            list(itertools.combinations(item_indices, 2)),
            desc=f"Pairwise Interactions (mode={mode})",
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
                util_S = self.get_utility(v_S_tuple, mode=mode)
                util_S_i = self.get_utility(v_S_union_i_tuple, mode=mode)
                util_S_j = self.get_utility(v_S_union_j_tuple, mode=mode)
                util_S_ij = self.get_utility(v_S_union_ij_tuple, mode=mode)

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
    
    # --- Public Compute Methods ---


    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    # --------------------------------------------------------------------------
    # Sampling and Approximation Methods (Efficient)
    # --------------------------------------------------------------------------

    def _generate_sampled_ablations(self, num_samples: int, seed: int = None) -> list[tuple]:
        """Generates a list of random subset tuples (coalitions)."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        n = self.n_items
        sampled_tuples_set = set()

        # Always include the empty and full sets, as they are crucial for KernelSHAP
        if num_samples >= 1: sampled_tuples_set.add(tuple([0] * n))
        if num_samples >= 2: sampled_tuples_set.add(tuple([1] * n))
        
        # Sample the rest uniformly
        while len(sampled_tuples_set) < num_samples:
            v_tuple = tuple(np.random.randint(0, 2, n))
            sampled_tuples_set.add(v_tuple)

        return list(sampled_tuples_set)
    
    def _calculate_shap_kernel_weights(self, ablations: list[tuple]) -> np.ndarray:
        n_features = self.n_items
        weights = []
        for v_tuple in ablations:
            k = sum(v_tuple)
            # Rather than infinite, use finite clipping and avoid zero/NaN
            if k == 0 or k == n_features:
                # Give them a stable but not infinite weight (they're usually important)
                weight = 1e3
            else:
                weight = (n_features - 1) / (comb(n_features, k) * k * (n_features - k))
            weights.append(weight)
        weights = np.array(weights, dtype=float)
        # numerical safety: replace inf/nan and clip extremes
        weights = np.nan_to_num(weights, posinf=1e6, neginf=0.0)
        # Optionally normalize to have mean 1 so scale doesn't blow up optimizers
        weights = weights / np.mean(weights)
        return weights

    def compute_contextcite(self, num_samples: int, seed: int = 42, utility_mode="logit-prob"):

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
            seed=seed
        )

        # Compute utilities on-demand for the sampled subsets
        pbar = tqdm(sampled_tuples, desc="Computing utilities for ContextCite", disable=not self.verbose)
        utilities_for_samples = [self.get_utility(v_tuple, mode=utility_mode) for v_tuple in pbar]

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

    def compute_wss(self, num_samples: int, seed: int = 42, 
                    attribution_method="kernelshap", # Renamed for clarity
                    sur_type="fm", rank: int=None, utility_mode="logit-prob"):
        """
        Computes attributions using a weighted surrogate model.
        Returns: (main_effects, interaction_effects, model)
        """
        if not self.accelerator.is_main_process: 
            return (np.zeros(self.n_items), np.zeros(self.n_items), None)
        
        # Step 1: Generate UNIFORM random subsets
        # The 'sampling' parameter is removed from the call as it's always uniform now.
        sampled_tuples = self._generate_sampled_ablations(num_samples, seed=seed)
        
        # Step 2: Compute utilities for these subsets
        # pbar = tqdm(sampled_tuples, desc=f"Computing utilities for {sur_type} surrogate", disable=not self.verbose)
        utilities_for_samples = [self.get_utility(v_tuple, mode=utility_mode) for v_tuple in sampled_tuples]

        # Filter invalid utilities
        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u is not None and u != -float('inf')]
        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]
        
        # Step 3: Calculate weights for the valid samples
        weights_for_train = None
        if attribution_method == "kernelshap":
            # print("Calculating KernelSHAP weights for training...")
            weights_for_train = self._calculate_shap_kernel_weights(sampled_tuples_for_train)
        
        # Step 4: Train the surrogate model with the calculated weights
        model, attr, F = self._train_surrogate(
            sampled_tuples_for_train, 
            utilities_for_train, 
            sur_type=sur_type, 
            rank=rank,
            sample_weights=weights_for_train
        )
        
        return attr, F, model

    def _train_surrogate(self, ablations: list[tuple], utilities: list[float], 
                         sur_type="linear", rank=None, alpha=0.01, 
                         sample_weights: np.ndarray = None): # <-- Added parameter
        """Internal method to train a surrogate model on utility data."""
        X_train = np.array(ablations)
        y_train = np.array(utilities)

        if sur_type == "linear":
            model = Lasso(alpha=alpha, fit_intercept=True, random_state=42, max_iter=2000)
            # Pass weights to the fit method
            model.fit(X_train, y_train, sample_weight=sample_weights)
            return model, model.coef_, None
        
        if sur_type == "fm":
            # set defaults
            if rank is None:
                rank = min(8, X_train.shape[1] // 2)
            model, attr, F = train_fm_torch(X_train, y_train, sample_weights=sample_weights, rank=rank, n_iter=1500, lr=1e-3, l2_reg=1e-4, device=str(self.device))
            return model, attr, F
        
        elif sur_type == "full_poly2":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X_train)
            model = Ridge(alpha=alpha, fit_intercept=True)
            # Pass weights to the fit method
            model.fit(X_poly, y_train, sample_weight=sample_weights)
            
            n = X_train.shape[1]
            linear, pairs = model.coef_[:n], np.zeros((n, n))
            idx = n
            for i in range(n):
                for j in range(i + 1, n):
                    pairs[i, j] = pairs[j, i] = model.coef_[idx]
                    idx += 1
            importance = linear + 0.5 * pairs.sum(axis=1)
            return model, importance, pairs

    def compute_tmc_shap(self, num_iterations_max: int, performance_tolerance: float, 
                        max_unique_lookups: int, seed: int = None, 
                        shared_cache: dict = None, utility_mode="logit-prob"):
        """
        Computes Shapley values using Truncated Monte Carlo sampling.
        
        This version uses a provided shared_cache to store and retrieve utilities,
        and manages its own lookup budget independently of the cache's total size.
        It runs on the main process.
        """
        if not self.accelerator.is_main_process:
            return np.zeros(self.n_items)
            
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Use the provided shared cache or the instance's own cache if none is given.
        cache = shared_cache if shared_cache is not None else self.utility_cache
        
        # This method's own counter for its budget.
        lookups_made_by_this_call = 0

        shapley_values = np.zeros(self.n_items)
        marginal_counts = np.zeros(self.n_items, dtype=int)
        
        # Nested function to handle on-demand utility calls while tracking budget.
        def get_utility_with_budget(subset_tuple):
            nonlocal lookups_made_by_this_call
            # Always return from cache if available, without penalty to budget.
            if subset_tuple in cache:
                return cache[subset_tuple]
                
            # If not in cache, check if we have budget to compute it.
            if lookups_made_by_this_call >= max_unique_lookups:
                return -float('inf') # Budget exceeded, return failure.
            
            # Compute, cache, increment budget counter, and return.
            utility = self._compute_response_metric(context_str=self._get_ablated_context_from_vector(np.array(subset_tuple)), mode=utility_mode)
            cache[subset_tuple] = utility
            lookups_made_by_this_call += 1
            return utility

        v_empty_util = get_utility_with_budget(tuple([0] * self.n_items))
        v_full_util = get_utility_with_budget(tuple([1] * self.n_items))
        
        truncation_possible = v_empty_util > -float('inf') and v_full_util > -float('inf')

        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations_max), desc="TMC Iterations (Corrected)", disable=not self.verbose)
        
        for t in pbar:
            if lookups_made_by_this_call >= max_unique_lookups:
                if self.verbose: print(f"TMC: Budget of {max_unique_lookups} lookups reached.")
                pbar.close()
                break 
                
            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int) 

            for item_idx_to_add in permutation:
                # If the chain has already failed (e.g., v_prev_util is -inf), no point continuing.
                if v_prev_util == -float('inf'):
                    marginal_contribution = 0.0 # Cannot compute a valid marginal.
                    v_curr_util = -float('inf') # The chain remains broken.
                else:
                    can_truncate = t > 0 and truncation_possible and (abs(v_full_util - v_prev_util) < performance_tolerance)
                    if can_truncate:
                        v_curr_util = v_prev_util
                    else:
                        v_curr_np = current_subset_np.copy(); v_curr_np[item_idx_to_add] = 1
                        v_curr_util = get_utility_with_budget(tuple(v_curr_np))
                    
                    marginal_contribution = v_curr_util - v_prev_util if v_curr_util > -float('inf') else 0.0
                
                k_count = marginal_counts[item_idx_to_add] + 1
                shapley_values[item_idx_to_add] = ((k_count - 1) / k_count) * shapley_values[item_idx_to_add] + (1 / k_count) * marginal_contribution
                marginal_counts[item_idx_to_add] = k_count
                
                # CRITICAL: Update state for the next step in the permutation.
                v_prev_util = v_curr_util
                current_subset_np[item_idx_to_add] = 1
        
        return shapley_values

    def compute_beta_shap(self, num_iterations_max: int, beta_a: float, beta_b: float,  
                        max_unique_lookups: int, seed: int = None,
                        shared_cache: dict = None, utility_mode="logit-prob"):
        """
        Computes Shapley values using BetaShap sampling. This version uses a
        provided shared cache and manages its own lookup budget. It runs on the main process.
        """
        if not self.accelerator.is_main_process:
            return np.zeros(self.n_items)
            
        if beta_dist is None: raise ImportError("BetaShap requires scipy.")
        if seed is not None: random.seed(seed); np.random.seed(seed)
            
        cache = shared_cache if shared_cache is not None else self.utility_cache
        lookups_made_by_this_call = 0

        weighted_marginal_sums = np.zeros(self.n_items)
        total_weights_for_item = np.zeros(self.n_items)
        
        def get_utility_with_budget(subset_tuple):
            nonlocal lookups_made_by_this_call
            if subset_tuple in cache: return cache[subset_tuple]
            if lookups_made_by_this_call >= max_unique_lookups: return -float('inf')
            
            utility = self._compute_response_metric(context_str=self._get_ablated_context_from_vector(np.array(subset_tuple)), mode=utility_mode)
            cache[subset_tuple] = utility
            lookups_made_by_this_call += 1
            return utility

        v_empty_util = get_utility_with_budget(tuple([0] * self.n_items))
        if v_empty_util == -float('inf') and self.verbose:
            print("BetaShap Warning: Utility of empty set is -inf. This may lead to zero scores.")

        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations_max), desc="BetaShap Iterations (Corrected)", disable=not self.verbose)
        
        for t in pbar:
            if lookups_made_by_this_call >= max_unique_lookups:
                if self.verbose: print(f"BetaShap: Budget of {max_unique_lookups} lookups reached.")
                pbar.close()
                break

            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int)

            for k, item_idx_to_add in enumerate(permutation):
                # If the chain has already failed, break from this permutation.
                if v_prev_util == -float('inf'):
                    break

                v_curr_np = current_subset_np.copy(); v_curr_np[item_idx_to_add] = 1
                v_curr_util = get_utility_with_budget(tuple(v_curr_np))
                
                # Only proceed if the new utility is valid
                if v_curr_util > -float('inf'):
                    marginal_contribution = v_curr_util - v_prev_util
                    
                    # Calculate Beta weight
                    if self.n_items > 1: x_pos = k / (self.n_items - 1)
                    else: x_pos = 0.5
                    
                    try:
                        weight = beta_dist.pdf(x_pos, beta_a, beta_b)
                        if not np.isfinite(weight): weight = 1e6 # Use large stable weight if PDF is infinite
                    except Exception: weight = 1.0 # Fallback
                    
                    weighted_marginal_sums[item_idx_to_add] += weight * marginal_contribution
                    total_weights_for_item[item_idx_to_add] += weight
                
                v_prev_util = v_curr_util
                current_subset_np[item_idx_to_add] = 1
                
        pbar.close()
        
        shapley_values = np.zeros(self.n_items)
        non_zero_mask = total_weights_for_item > 1e-9
        shapley_values[non_zero_mask] = weighted_marginal_sums[non_zero_mask] / total_weights_for_item[non_zero_mask]

        return shapley_values

    def compute_loo(self, utility_mode= "logit-prob"):
        """
        Computes general Leave-One-Out (LOO) scores for each item
        using the specified utility function mode.
        """
        loo_scores = [0.0] * self.n_items
        
        # V(FullSet) for the given utility mode
        utility_full_context = self.get_utility(
            tuple([1] * self.n_items), 
            mode=utility_mode
        )

        pbar = tqdm(range(self.n_items), desc=f"LOO Calls ({utility_mode})", disable=not self.verbose)
        for i in pbar:
            v_loo_tuple = tuple(1 if j != i else 0 for j in range(self.n_items))
            utility_ablated = self.get_utility(v_loo_tuple, mode=utility_mode)
            
            if utility_full_context > -float('inf') and utility_ablated > -float('inf'):
                loo_scores[i] = utility_full_context - utility_ablated
            else:
                loo_scores[i] = 0.0 # Or some other indicator of failure

        return loo_scores

    def compute_arc_jsd(self):
        """
        Computes ARC-JSD scores. This is a specific application of Leave-One-Out
        using a divergence-based utility.
        """
        # This is now just a convenient wrapper around the general LOO method.
        return self.compute_loo(utility_mode="divergence_utility")

# Faithfull methods {Faith-Shap, Faith-Banzhaf, Spex}

    def _make_value_function(self, utility_mode: str):
        """
        Build a value function for SPEX given a utility mode.
        Returns a callable that maps binary subset vectors -> utility.
        """
        def raw_value_function(context_str: str) -> float:
            ablated_items = context_str.split("\n\n") if context_str else []
            subset_vector = tuple(1 if item in ablated_items else 0 for item in self.items)
            return self.get_utility(subset_vector, mode=utility_mode)

        valuef_counter = CallCounter(raw_value_function)

        def value_function(subsets: list) -> list:
            results = []
            for subset in subsets:
                selected_idxs = np.where(np.array(subset) == 1)[0]
                ablated_context = [self.items[i] for i in selected_idxs]
                ablated_context_str = "\n\n".join(ablated_context)
                results.append(valuef_counter(ablated_context_str))
            return results

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
        # explainer = spex.Explainer(
        #     value_function=value_function,
        #     features=self.items,
        #     sample_budget=sample_budget,
        #     max_order=max_order,
        # )

        # method_interactions = explainer.interactions(index=method)
        # converted = method_interactions.convert_fourier_interactions()
        moebius_interactions = approximator.approximate(budget=sample_budget, game=value_function)
        attribution = np.zeros(self.n_items)
        interaction_terms = {}

        for pattern, coef in moebius_interactions.dict_values.items():
            order = len(pattern)
            if order == 1:
                attribution[pattern] = coef
            elif order == 2:
                interaction_terms[pattern] = coef

        return attribution, interaction_terms, moebius_interactions

    # def compute_spex(self, sample_budget: int, max_order: int = 2, utility_mode: str = "logit-prob"):
    #     """Compute attribution scores using SPEX (Fourier method)."""
    #     return self._run_spex("fourier", sample_budget, max_order, utility_mode)

    def compute_fsii(self, sample_budget: int, max_order: int, utility_mode: str = "logit-prob"):
        """Compute attribution scores using SPEX (FSII method)."""
        return self._run_spex("FSII", sample_budget, max_order, utility_mode)

    def compute_fbii(self, sample_budget: int, max_order: int, utility_mode: str = "logit-prob"):
        """Compute attribution scores using SPEX (FBII method)."""
        return self._run_spex("FBII", sample_budget, max_order, utility_mode)
    # --------------------------------------------------------------------------
    # Helper & Internal Methods
    # --------------------------------------------------------------------------
   
    def compute_jsd_for_ablated_indices(self, ablated_indices):
        """
        Computes total JSD when removing specific documents
        
        Args:
            ablated_indices: List of document indices to remove
            
        Returns:
            Total JSD between full context and context without specified documents
        """
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
    
    def lds(self, results_dict, n_eval_util,mode, models):
        eval_subsets = self._generate_sampled_ablations(n_eval_util, seed=2)
        X_all = np.array(eval_subsets)
        exact_utilities = [self.get_utility(v_tuple, mode=mode) for v_tuple in eval_subsets]
        X_all_sparse = csr_matrix(X_all)
        lds={}
        # Predict effects for all subsets using surrogates
        for method_name, scores in results_dict.items():
            if "FM" in method_name:
                predicted_effect=models[method_name].predict(X_all_sparse)
            elif "II" in method_name:
                predicted_effect = np.zeros(len(X_all))
                for i, x in enumerate(X_all):
                    for loc, coef in models[method_name].dict_values.items():
                        # print(loc)
                        if all(x[l] == 1 for l in loc):
                            predicted_effect[i] += coef
            else:
                predicted_effect=[np.dot(scores, i) for i in X_all]

            # Calculate Spearman correlation
            lds[method_name], _ = spearmanr(exact_utilities, predicted_effect)
        return lds
    
    def r2(self, results_dict, n_eval_util, mode, models):
        eval_subsets = self._generate_sampled_ablations(n_eval_util, seed=2)
        X_all = np.array(eval_subsets)
        exact_utilities = [self.get_utility(v_tuple, mode=mode) for v_tuple in eval_subsets]
        # exact_utilities_perplexity = [self.get_utility(v_tuple, mode="log-perplexity") for v_tuple in eval_subsets]
        X_all_sparse = csr_matrix(X_all)
        r2={}
        for method_name, scores in results_dict.items():
            if "FM" in method_name:
                predicted_effect=models[method_name].predict(X_all_sparse)
            elif "ContextCite" in method_name:
                predicted_effect=models[method_name].predict(X_all)
            elif "II" in method_name:
                predicted_effect = np.zeros(len(X_all))
                for i, x in enumerate(X_all):
                    for loc, coef in models[method_name].dict_values.items():
                        # print(loc)
                        if all(x[l] == 1 for l in loc):
                            predicted_effect[i] += coef
            else:
                predicted_effect=[np.dot(scores, i) for i in X_all]
            try:
                # if "FSI" in method_name or "FB" in method_name or "Spex" in method_name:
                #     r2[method_name]=r2_score(exact_utilities_perplexity, predicted_effect)
                # else:
                r2[method_name]=r2_score(exact_utilities, predicted_effect)
            except Exception: pass
        return r2

    def compute_exhaustive_top_k(self, k: int, search_list, model=None):
        n = self.n_items
        best_k_indices_to_remove = None
        min_utility_after_removal = float('inf') # We want to minimize V(N - S_removed)

        possible_indices_to_remove = list(itertools.combinations(search_list, k))
        
        pbar_desc = f"Exhaustive Top-{k} Search"
        pbar_iter = tqdm(possible_indices_to_remove, desc=pbar_desc, disable=not self.verbose)

        for k_indices_tuple in pbar_iter:
            ablated_set_np = np.ones(n, dtype=int)
            ablated_set_np[list(k_indices_tuple)] = 0
            # ablated_set_tuple = tuple(ablated_set_np)
            if model:
                utility_of_ablated_set = model.predict(csr_matrix(ablated_set_np))
            else:
                utility_of_ablated_set = self.get_utility(tuple(ablated_set_np), mode="logit-prob")
            if utility_of_ablated_set < min_utility_after_removal:
                min_utility_after_removal = utility_of_ablated_set
                best_k_indices_to_remove = k_indices_tuple

        return best_k_indices_to_remove

    def top_k_response_probability(self, results_dict, k_values=[1, 3, 5]):

        n_docs = self.n_items
        evaluation_results = {}

        for method_name, scores in results_dict.items():
            method_probs = {}
            for k in k_values:
                if k > n_docs:
                    continue

                # Get top-k indices (largest scores)
                topk_indices = np.argsort(scores)[-k:]
                if not isinstance(topk_indices, np.ndarray):
                    topk_indices = np.array([topk_indices])
                elif topk_indices.ndim > 1:
                    topk_indices = topk_indices.flatten()

                # Create vector: 1 for selected docs, 0 for the rest
                selection_vector = np.zeros(n_docs, dtype=int)
                selection_vector[topk_indices] = 1

                # Get utility using only the selected docs
                prob_util = self.get_utility(tuple(selection_vector), mode="raw-prob")

                method_probs[k] = prob_util

            evaluation_results[method_name] = method_probs

        return evaluation_results

    def recall_at_k(self, gtset_k, results_dict, k_val ):
        recall={}
        for method_name, scores in results_dict.items():
            rec=[]
            for i in k_val:
                topk= np.array(scores).argsort()[-i:]
                rec.append(len(set(gtset_k).intersection(topk))/len(gtset_k))
            recall[method_name]=rec
        return recall
    
    
    def evaluate_topk_performance(self, results_dict, models, k_values:list):
    
        n_docs = self.n_items
        evaluation_results = {}
        # Get full context utility based on type
        full_utility = self.get_utility(tuple([1] * n_docs), mode="logit-prob")

        for method_name, scores in results_dict.items():
            # Skip non-attribution results
            method_drops = {}
            for k in k_values:
                if k > n_docs:
                    continue
                # Get indices of top k documents
                if "FM_Weights" in method_name:
                    topk_indices=self.compute_exhaustive_top_k(k, np.argsort(scores)[-10:], model=models[method_name])
                elif "Exact" in method_name:
                    topk_indices=self.compute_exhaustive_top_k(k, np.argsort(scores)[-10:])
                else:
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
                util_without_topk = self.get_utility(tuple(ablation_vector), mode="logit-prob")
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

class CallCounter:
    """Wrapper to count function calls."""
    def __init__(self, func):
        functools.update_wrapper(self, func)
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class FMDataset(Dataset):
    def __init__(self, X, y, sample_weights=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        if sample_weights is None:
            self.w = torch.ones(len(y), dtype=torch.float32)
        else:
            self.w = torch.tensor(sample_weights, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.w[idx]

class FactorizationMachine(nn.Module):
    def __init__(self, n_features, rank=8):
        super().__init__()
        self.n = n_features
        self.rank = rank
        # linear term
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(n_features))
        # latent factors (n_features x rank)
        self.V = nn.Parameter(torch.randn(n_features, rank) * 0.01)
    def forward(self, x):
        # x: batch x n
        linear = self.w0 + (x * self.w).sum(dim=1)
        # pairwise interactions: 0.5 * sum((xV)^2 - (x^2)(V^2))
        xv = torch.matmul(x, self.V)  # batch x rank
        xv2 = xv * xv
        x2 = x * x
        v2 = self.V * self.V
        x2v2 = torch.matmul(x2, v2)  # batch x rank
        interactions = 0.5 * torch.sum(xv2 - x2v2, dim=1)
        return linear + interactions

def train_fm_torch(X, y, sample_weights=None, rank=8, n_iter=2000, lr=1e-3, batch_size=128, l2_reg=1e-4, seed=42, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ds = FMDataset(X, y, sample_weights=sample_weights)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = FactorizationMachine(X.shape[1], rank=rank).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
    total_loss = 0.0
    for epoch in range(n_iter):
        for xb, yb, wb in loader:
            # ...
            pred = model(xb)
            loss = (wb * ((pred - yb) ** 2)).mean() # Much cleaner!
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.shape[0]
    # extract weights
    w0 = model.w0.detach().cpu().numpy().reshape(-1)
    w = model.w.detach().cpu().numpy()
    V = model.V.detach().cpu().numpy()  # n x rank
    # compute interaction matrix F = V @ V.T
    F = V @ V.T
    np.fill_diagonal(F, 0.0)
    attr = w + 0.5 * F.sum(axis=1)
    return model, attr, F