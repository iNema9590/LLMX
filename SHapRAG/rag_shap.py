import itertools
# import json
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
# import xgboost
# from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from fastFM import als
from scipy.sparse import csr_matrix
from scipy.stats import beta as beta_dist
# from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures
from tqdm.auto import tqdm

# import warnings
# from collections import defaultdict

# from transformers import AutoModelForCausalLM, AutoTokenizer


class ContextAttribution:

    def __init__(self, items, query, 
                 prepared_model_for_harness, tokenizer_for_harness, accelerator_for_harness, 
                 verbose=False, utility_cache_path=None):
        self.accelerator = accelerator_for_harness
        self.items = items
        self.query = query
        self.model = prepared_model_for_harness 
        self.tokenizer = tokenizer_for_harness
        self.verbose = verbose
        self.n_items = len(items)
        self.device = self.accelerator.device

        # The on-demand utility cache.
        self.utility_cache = {}

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._factorials = {k: math.factorial(k) for k in range(self.n_items + 1)}

        # Load existing cache if provided to warm-start the harness.
        if utility_cache_path and os.path.exists(utility_cache_path):
            if self.accelerator.is_main_process:
                print(f"Loading existing utility cache from {utility_cache_path}...")
                try:
                    with open(utility_cache_path, "rb") as f:
                        self.utility_cache = pickle.load(f)
                    print(f"Successfully loaded {len(self.utility_cache)} cached utilities.")
                except Exception as e:
                    print(f"Warning: Failed to load cache from {utility_cache_path}: {e}")
            # Ensure all processes are synchronized after main process loads the cache
            self.accelerator.wait_for_everyone()
            # Broadcast the loaded cache to all processes
            object_to_broadcast = [self.utility_cache]
            broadcast_object_list(object_to_broadcast, from_process=0)
            self.utility_cache = object_to_broadcast[0]
            if self.verbose and not self.accelerator.is_main_process:
                print(f"Rank {self.accelerator.process_index} received {len(self.utility_cache)} items in broadcasted cache.")


        # Generate the target response once, as it's the reference for all utility calculations.
        if self.verbose and self.accelerator.is_main_process:
            print("Generating target response based on full context...")
        self.target_response = self._llm_generate_response(context_str="\n".join(self.items))
        if self.verbose and self.accelerator.is_main_process:
            print(f"Target response: '{self.target_response}'")

    # --------------------------------------------------------------------------
    # Core Utility Management (On-Demand)
    # --------------------------------------------------------------------------

    def get_utility(self, subset_tuple: tuple) -> float:
        """
        Gatekeeper for utility values. Returns from cache or computes if not present.
        This is the primary method for on-demand computation.
        """
        if subset_tuple in self.utility_cache:
            return self.utility_cache[subset_tuple]
        
        # If not in cache, compute it, cache it, and return it.
        if self.verbose and self.accelerator.is_main_process:
            # This log can be noisy, enable if debugging cache misses.
            # print(f"Cache miss for subset {subset_tuple}. Computing utility...")
            pass

        v_np = np.array(subset_tuple)
        context_str = self._get_ablated_context_from_vector(v_np)
        utility = self._llm_compute_logprob(context_str=context_str)
        
        self.utility_cache[subset_tuple] = utility
        return utility

    def save_utility_cache(self, file_path: str):
        """Saves the current state of the utility cache to a file."""
        if self.accelerator.is_main_process:
            if self.verbose:
                print(f"Main process: Saving {len(self.utility_cache)} utilities to {file_path}...")
            with open(file_path, "wb") as f:
                pickle.dump(self.utility_cache, f)
            if self.verbose:
                print("Save complete.")
    
    # --------------------------------------------------------------------------
    # LLM Interaction Methods (The "Engine")
    # --------------------------------------------------------------------------
    
    @staticmethod
    def _logit(p, eps=1e-7):
        p = torch.clamp(p, eps, 1 - eps)
        return torch.log(p / (1 - p))

    def _llm_generate_response(self, context_str: str, max_new_tokens: int = 100) -> str:
        messages = [{"role": "system", "content": """You are a helpful assistant. You use the provided context to answer
                    questions in few words. Avoid using your own knowledge or make assumptions.
                    """}]
        if context_str:
            messages.append({"role": "user", "content": f"###context: {context_str}. ###question: {self.query}."})
        else:
            messages.append({"role": "user", "content": self.query})

        chat_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        tokenized = self.tokenizer(chat_text, return_tensors="pt", padding=True)

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        with torch.no_grad():
            outputs_gen = unwrapped_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else unwrapped_model.config.eos_token_id,
                top_p=1.0,
            )
        
        if isinstance(outputs_gen, torch.Tensor): generated_ids = outputs_gen
        elif hasattr(outputs_gen, 'sequences'): generated_ids = outputs_gen.sequences
        else:
            print(f"Warning: Unexpected output type from model.generate: {type(outputs_gen)}")
            del input_ids, attention_mask, unwrapped_model, outputs_gen
            torch.cuda.empty_cache()
            return ""

        response_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_text = response_text.lstrip().removeprefix("assistant").lstrip(": \n").strip()
        del input_ids, attention_mask, generated_ids, unwrapped_model, outputs_gen
        torch.cuda.empty_cache()
        return cleaned_text

    def _llm_compute_logprob(self, context_str: str, response=None) -> float:
        if response is None: response = self.target_response
        answer_ids = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        L = answer_ids.shape[1]
        if L == 0: return 0.0

        sys_msg = {"role": "system", "content": """You are a helpful assistant. You use the provided context to answer
                    questions in few words. Avoid using your own knowledge or make assumptions."""}
        
        # --- Calculate for prompt with context ---
        prompt_with_context_str = self.tokenizer.apply_chat_template(
            [sys_msg, {"role": "user", "content": f"###context: {context_str} ###question: {self.query}" if context_str else self.query}],
            add_generation_prompt=True, tokenize=False
        )
        prompt_with_context_ids = self.tokenizer(prompt_with_context_str, return_tensors="pt").input_ids.to(self.device)
        input_ids_with_context = torch.cat([prompt_with_context_ids, answer_ids], dim=1)
        with torch.no_grad():
            logits_with = self.model(input_ids=input_ids_with_context).logits
        
        prompt_len_with = input_ids_with_context.shape[1] - L
        shift_logits_with = logits_with[..., prompt_len_with-1:-1, :].contiguous()
        answer_log_probs_with = torch.gather(F.log_softmax(shift_logits_with, dim=-1), 2, answer_ids.unsqueeze(-1)).squeeze(-1)
        total_log_prob_with = answer_log_probs_with.sum()

        # --- Calculate for prompt with empty context ---
        prompt_empty_context_str = self.tokenizer.apply_chat_template(
            [sys_msg, {"role": "user", "content": self.query}], add_generation_prompt=True, tokenize=False
        )
        prompt_empty_context_ids = self.tokenizer(prompt_empty_context_str, return_tensors="pt").input_ids.to(self.device)
        input_ids_empty_context = torch.cat([prompt_empty_context_ids, answer_ids], dim=1)
        with torch.no_grad():
            logits_empty = self.model(input_ids=input_ids_empty_context).logits

        prompt_len_empty = input_ids_empty_context.shape[1] - L
        shift_logits_empty = logits_empty[..., prompt_len_empty-1:-1, :].contiguous()
        answer_log_probs_empty = torch.gather(F.log_softmax(shift_logits_empty, dim=-1), 2, answer_ids.unsqueeze(-1)).squeeze(-1)
        total_log_prob_empty = answer_log_probs_empty.sum()

        # --- Calculate logit gain ---
        logit_gain_total = self._logit(torch.exp(total_log_prob_with)) - self._logit(torch.exp(total_log_prob_empty))
        logit_gain_per_token = logit_gain_total / L

        del logits_with, logits_empty, shift_logits_with, shift_logits_empty, answer_log_probs_with, answer_log_probs_empty
        torch.cuda.empty_cache()
        return logit_gain_per_token.item()

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    # --------------------------------------------------------------------------
    # Exhaustive Attribution Methods (Use with Extreme Caution)
    # --------------------------------------------------------------------------
    
    def _calculate_exact_shapley(self) -> np.ndarray:
        """
        Computes exact Shapley values by evaluating all 2^n subsets.
        WARNING: Incredibly slow. Triggers n * 2^(n-1) on-demand utility calls.
        Only use for very small n (e.g., n < 12).
        """
        if not (self.verbose and self.accelerator.is_main_process):
            # This calculation should likely only be run on the main process
            # to avoid redundant (and slow) computations.
            return np.zeros(self.n_items)

        shapley_values = np.zeros(self.n_items)
        n = self.n_items
        all_subsets = list(itertools.product([0, 1], repeat=n))
        
        pbar_desc = "Calculating Exact Shapley (SLOW)"
        pbar = tqdm(all_subsets, desc=pbar_desc, disable=not self.verbose)

        for s_tuple in pbar:
            s_size = sum(s_tuple)
            s_util = self.get_utility(s_tuple)
            
            if s_util == -float('inf'): continue
            
            for i in range(n):
                if s_tuple[i] == 1: # Marginal contribution of i to S\{i}
                    s_without_i_list = list(s_tuple); s_without_i_list[i] = 0
                    s_without_i_tuple = tuple(s_without_i_list)
                    s_without_i_util = self.get_utility(s_without_i_tuple)
                    
                    marginal_contrib = s_util - s_without_i_util
                    weight = (self._factorials[s_size - 1] * self._factorials[n - s_size]) / self._factorials[n]
                    shapley_values[i] += weight * marginal_contrib
                    
        return shapley_values

    def compute_shapley_interaction_index_pairs_matrix(self):
        """
        Computes the full matrix of pairwise Shapley interaction indices.
        WARNING: Very slow. This will trigger on-demand utility calls for all
        subsets relevant to each pair.
        """
        if not self.accelerator.is_main_process: return np.zeros((self.n_items, self.n_items))

        n = self.n_items
        interaction_matrix = np.zeros((n, n), dtype=float)
        item_indices = list(range(n))
        pbar_pairs = tqdm(list(itertools.combinations(item_indices, 2)), desc="Pairwise Interactions (SLOW)", disable=not self.verbose)

        for i, j in pbar_pairs:
            interaction_sum = 0.0
            remaining_indices = [idx for idx in item_indices if idx != i and idx != j]
            
            for s_size in range(len(remaining_indices) + 1):
                for s_members_indices in itertools.combinations(remaining_indices, s_size):
                    v_S_np = np.zeros(n, dtype=int)
                    if s_members_indices: v_S_np[list(s_members_indices)] = 1
                    
                    v_S_union_i_np = v_S_np.copy(); v_S_union_i_np[i] = 1
                    v_S_union_j_np = v_S_np.copy(); v_S_union_j_np[j] = 1
                    v_S_union_ij_np = v_S_np.copy(); v_S_union_ij_np[i] = 1; v_S_union_ij_np[j] = 1

                    util_S = self.get_utility(tuple(v_S_np))
                    util_S_i = self.get_utility(tuple(v_S_union_i_np))
                    util_S_j = self.get_utility(tuple(v_S_union_j_np))
                    util_S_ij = self.get_utility(tuple(v_S_union_ij_np))

                    delta_ij_S = util_S_ij - util_S_i - util_S_j + util_S
                    
                    if n > 2:
                        weight = (self._factorials[s_size] * self._factorials[n - s_size - 2]) / self._factorials[n - 1]
                    else: # n=2
                        weight = 1.0

                    interaction_sum += weight * delta_ij_S

            interaction_matrix[i, j] = interaction_matrix[j, i] = interaction_sum
        return interaction_matrix

    # --------------------------------------------------------------------------
    # Sampling and Approximation Methods (Efficient)
    # --------------------------------------------------------------------------

    def _generate_sampled_ablations(self, num_samples: int, sampling_method: str = "uniform", seed: int = None) -> list[tuple]:
        """Generates a list of subset tuples based on a sampling strategy."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        n = self.n_items
        sampled_tuples_set = set()

        # Always include the empty and full sets for better surrogate modeling
        if num_samples >= 1: sampled_tuples_set.add(tuple([0] * n))
        if num_samples >= 2: sampled_tuples_set.add(tuple([1] * n))

        remaining_to_sample = num_samples - len(sampled_tuples_set)
        if remaining_to_sample <= 0: return list(sampled_tuples_set)

        if sampling_method == "uniform":
            while len(sampled_tuples_set) < num_samples:
                v_tuple = tuple(np.random.randint(0, 2, n))
                sampled_tuples_set.add(v_tuple)
        
        elif sampling_method == "kernelshap":
            # Sample sizes based on KernelSHAP weights, then sample subsets of that size
            weights = []
            for z in range(1, n): # sizes 1 to n-1
                denominator = z * (n - z)
                weights.append((n - 1) / denominator)
            
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            
            # Sample `remaining_to_sample` sizes
            sampled_sizes = np.random.choice(list(range(1, n)), size=remaining_to_sample, p=probabilities, replace=True)

            for z in sampled_sizes:
                # Sample a random subset of size z
                v_np = np.zeros(n, dtype=int)
                indices_to_set = random.sample(range(n), z)
                v_np[indices_to_set] = 1
                sampled_tuples_set.add(tuple(v_np))

        return list(sampled_tuples_set)

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
        pbar = tqdm(sampled_tuples, desc="Computing utilities for ContextCite", disable=not self.verbose)
        utilities_for_samples = [self.get_utility(v_tuple) for v_tuple in pbar]

        # Filter out any samples where utility computation failed
        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
        if len(valid_indices) < len(sampled_tuples):
            print(f"Warning: {len(sampled_tuples) - len(valid_indices)} utility computations failed. Training surrogate on {len(valid_indices)} samples.")

        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]

        if not utilities_for_train:
            print("Warning: No valid utilities could be computed for ContextCite. Returning empty weights.")
            return np.array([])

        _, weights, _, mse = self._train_surrogate(
            sampled_tuples_for_train, 
            utilities_for_train,
            sur_type="linear"
        )
        return weights

    def compute_wss(self, num_samples: int, seed: int = None, sampling="kernelshap", sur_type="fm"):
        """Computes Weighted Subset Sampling (WSS) attributions using a surrogate model."""
        if not self.accelerator.is_main_process: return None, None, None
        
        sampled_tuples = self._generate_sampled_ablations(num_samples, sampling_method=sampling, seed=seed)
        
        pbar = tqdm(sampled_tuples, desc=f"Computing utilities for WSS ({sampling})", disable=not self.verbose)
        utilities_for_samples = [self.get_utility(v_tuple) for v_tuple in pbar]

        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]

        _, attr, F, mse = self._train_surrogate(sampled_tuples_for_train, utilities_for_train, sur_type=sur_type)
        return attr, F, mse
    
    def compute_tmc_shap(self, num_iterations_max: int, performance_tolerance: float, 
                        max_unique_lookups: int, seed: int = None):

            
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        shapley_values = np.zeros(self.n_items)
        marginal_counts = np.zeros(self.n_items, dtype=int)
        
        # On-demand call for empty and full sets, which are crucial for truncation.
        # This also populates the cache for the first lookups.
        v_empty_util = self.get_utility(tuple([0] * self.n_items))
        v_full_util = self.get_utility(tuple([1] * self.n_items))
        
        # Check for fatal errors in endpoint utilities needed for truncation.
        truncation_possible = v_empty_util != -float('inf') and v_full_util != -float('inf')
        if not truncation_possible and self.verbose:
            print("  Warning: Truncation disabled because utility for empty or full set failed.")

        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations_max), desc="TMC Iterations (Corrected)", disable=not self.verbose)
        
        for t in pbar:
            if len(self.utility_cache) >= max_unique_lookups:
                if self.verbose:
                    print(f"Stopping TMC at iteration {t+1} due to lookup budget ({max_unique_lookups}).")
                break 

            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int) 

            for item_idx_to_add in permutation:
                # Check if we can truncate the rest of this permutation
                can_truncate = t > 0 and truncation_possible and (abs(v_full_util - v_prev_util) < performance_tolerance)
                
                if can_truncate:
                    # If we truncate, the utility is assumed not to change from the previous step.
                    # The marginal contribution is therefore 0.
                    v_curr_util = v_prev_util
                else:
                    # If not truncating, compute the utility for the new subset on-demand.
                    v_curr_np = current_subset_np.copy() 
                    v_curr_np[item_idx_to_add] = 1
                    v_curr_util = self.get_utility(tuple(v_curr_np))
                
                # Calculate marginal contribution if utilities are valid
                marginal_contribution = 0.0
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'):
                    marginal_contribution = v_curr_util - v_prev_util
                
                # Update the running average for the current item's Shapley value
                k_count = marginal_counts[item_idx_to_add] + 1
                shapley_values[item_idx_to_add] = ((k_count - 1) / k_count) * shapley_values[item_idx_to_add] + \
                                                (1 / k_count) * marginal_contribution
                marginal_counts[item_idx_to_add] = k_count
                
                # CRITICAL: Update state for the next step in the permutation
                v_prev_util = v_curr_util 
                current_subset_np[item_idx_to_add] = 1
        
        pbar.close()
        return shapley_values

    def compute_beta_shap(self, num_iterations_max: int, beta_a: float, beta_b: float, 
                        max_unique_lookups: int, seed: int = None):
        if not self.accelerator.is_main_process:
            return np.zeros(self.n_items)
            
        if beta_dist is None:
            raise ImportError("BetaShap requires scipy. `pip install scipy`")
            
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        weighted_marginal_sums = np.zeros(self.n_items)
        total_weights_for_item = np.zeros(self.n_items)
        
        # On-demand call for the empty set utility. This is the crucial first step.
        v_empty_util = self.get_utility(tuple([0] * self.n_items))
        if v_empty_util == -float('inf') and self.verbose:
            print("  FATAL WARNING for BetaShap: Utility of the empty set failed. All marginals will be invalid.")
            
        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations_max), desc="BetaShap Iterations (Fortified)", disable=not self.verbose)
        
        for t in pbar:
            if len(self.utility_cache) >= max_unique_lookups:
                if self.verbose:
                    print(f"Stopping BetaShap at iteration {t+1} due to lookup budget ({max_unique_lookups}).")
                break

            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int)

            for k, item_idx_to_add in enumerate(permutation):
                # Form the new subset by adding the item
                v_curr_np = current_subset_np.copy()
                v_curr_np[item_idx_to_add] = 1
                
                # On-demand call to get the utility for the new subset
                v_curr_util = self.get_utility(tuple(v_curr_np))
                
                # Calculate the Beta distribution weight
                if self.n_items > 1:
                    x_pos = k / (self.n_items - 1)
                else: # Avoid division by zero for n=1
                    x_pos = 0.5
                
                try:
                    weight = beta_dist.pdf(x_pos, beta_a, beta_b)
                    if not np.isfinite(weight):
                        weight = 1e6 # Use a large, stable weight if PDF is infinite (e.g., at boundaries)
                except Exception:
                    weight = 1.0 # Fallback weight
                
                # Only update sums if the marginal contribution is valid
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'):
                    marginal_contribution = v_curr_util - v_prev_util
                    weighted_marginal_sums[item_idx_to_add] += weight * marginal_contribution
                    total_weights_for_item[item_idx_to_add] += weight
                    if self.verbose and t < 2 and k < 5: # Log first few steps of first two iterations
                        print(f"  [Iter {t+1}, Step {k+1}] item {item_idx_to_add}: "
                            f"V(S U i)={v_curr_util:.3f}, V(S)={v_prev_util:.3f}, "
                            f"Marginal={marginal_contribution:.3f}, Weight={weight:.3f}")
                
                # Update state for the next step in the permutation
                v_prev_util = v_curr_util
                current_subset_np = v_curr_np
                
        pbar.close()
        
        # Calculate the final Shapley values
        shapley_values = np.zeros(self.n_items)
        # Avoid division by zero
        non_zero_weights_mask = total_weights_for_item > 1e-9
        shapley_values[non_zero_weights_mask] = weighted_marginal_sums[non_zero_weights_mask] / total_weights_for_item[non_zero_weights_mask]

        return shapley_values

    def compute_loo(self):
        """Computes Leave-One-Out (LOO) scores for each item."""
        if not self.accelerator.is_main_process: return np.zeros(self.n_items)

        loo_scores = np.zeros(self.n_items)
        util_all = self.get_utility(tuple([1] * self.n_items))

        pbar = tqdm(range(self.n_items), desc="Computing LOO scores", disable=not self.verbose)
        for i in pbar:
            v_loo_list = [1] * self.n_items
            v_loo_list[i] = 0
            util_loo = self.get_utility(tuple(v_loo_list))
            
            if util_all != -float('inf') and util_loo != -float('inf'):
                loo_scores[i] = util_all - util_loo
            else:
                loo_scores[i] = -float('inf') # Indicate failure
        return loo_scores

    # --------------------------------------------------------------------------
    # Helper & Internal Methods
    # --------------------------------------------------------------------------
    
    def _train_surrogate(self, ablations: list[tuple], utilities: list[float], sur_type="linear", alpha=0.01):
        """Internal method to train a surrogate model on utility data."""
        X_train = np.array(ablations)
        y_train = np.array(utilities)

        if sur_type == "linear":
            model = Lasso(alpha=alpha, fit_intercept=True, random_state=42, max_iter=2000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            return model, model.coef_, None, mean_squared_error(y_train, y_pred)
        
        elif sur_type == "fm":
            X_train_fm = csr_matrix(X_train)
            model = als.FMRegression(n_iter=100, rank=4, l2_reg_w=0.1, l2_reg_V=0.1, random_state=42)
            model.fit(X_train_fm, y_train)
            y_pred = model.predict(X_train_fm)
            w, V = model.w_, model.V_.T
            F = V @ V.T
            np.fill_diagonal(F, 0.0)
            attr = w + 0.5 * F.sum(axis=1)
            return model, attr, F, mean_squared_error(y_train, y_pred)
        
        elif sur_type == "full_poly2":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X_train)
            model = Ridge(alpha=0.01, fit_intercept=True) # Small alpha for stability
            model.fit(X_poly, y_train)
            y_pred = model.predict(X_poly)
            
            n = X_train.shape[1]
            linear, pairs = model.coef_[:n], np.zeros((n, n))
            idx = n
            for i in range(n):
                for j in range(i + 1, n):
                    pairs[i, j] = pairs[j, i] = model.coef_[idx]
                    idx += 1
            importance = linear + 0.5 * pairs.sum(axis=1)
            return model, importance, pairs, mean_squared_error(y_train, y_pred)
        
    def _get_response_token_distributions(self, context_str: str, response: str) -> torch.Tensor:
        """
        Computes the probability distribution for each token in the given response.

        Args:
            context_str (str): The context to condition the generation on.
            response (str): The response for which to calculate token distributions.

        Returns:
            torch.Tensor: A tensor of shape (num_response_tokens, vocab_size) containing
                          the probability distribution for each token in the response.
        """
        answer_ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        
        L = answer_ids.shape[1]
        if L == 0:
            return torch.tensor([], device=self.device)

        # Prepare the prompt
        messages = [
            {"role": "system", "content": """You are a helpful assistant. You use the provided context to answer questions in few words. Avoid using your own knowledge or make assumptions."""},
            {"role": "user", "content": f"###context: {context_str}. ###question: {self.query}." if context_str else self.query}
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = self.tokenizer(prompt_str, return_tensors="pt").input_ids.to(self.device)

        # Concatenate prompt and answer to get full input
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits

        # Get logits corresponding to the positions of the answer tokens
        # The logit for the k-th answer token is at index (prompt_len + k - 1)
        shifted_logits = logits[..., prompt_len - 1:-1, :].contiguous()
        
        # Calculate the probability distributions
        distributions = F.softmax(shifted_logits, dim=-1).squeeze(0) # Shape: (L, vocab_size)

        # Cleanup
        del outputs, logits, shifted_logits, prompt_ids, answer_ids, input_ids
        torch.cuda.empty_cache()

        return distributions

    @staticmethod
    def _jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> float:

        # Add epsilon to avoid log(0) issues
        p = p + epsilon
        q = q + epsilon

        # Normalize to ensure they sum to 1 after adding epsilon
        p /= p.sum()
        q /= q.sum()

        m = 0.5 * (p + q)
        
        # PyTorch's F.kl_div expects (input, target) where input is log-probabilities
        # It calculates sum(target * (log(target) - input))
        # So D_KL(p || m) is equivalent to F.kl_div(m.log(), p, reduction='sum')
        kl_p_m = F.kl_div(m.log(), p, reduction='sum')
        kl_q_m = F.kl_div(m.log(), q, reduction='sum')
        
        jsd = 0.5 * (kl_p_m + kl_q_m)
        return jsd.item()

    def compute_arc_jsd(self) -> list[float]:

        # Step 1: Get the baseline distributions for the target response with full context
        full_context_str = "\n\n".join(self.items)
        baseline_distributions = self._get_response_token_distributions(
            context_str=full_context_str,
            response=self.target_response
        )

        if baseline_distributions.nelement() == 0:
            if self.accelerator.is_main_process:
                print("Warning: Target response is empty. Cannot run ARC-JSD.")
            return [0.0] * self.n_items

        jsd_scores = [0.0] * self.n_items
        
        pbar = None
        if self.accelerator.is_main_process:
            pbar = tqdm(total=self.n_items)
        
        # Step 2 & 3: Iterate, ablate each document, and compute JSD
        for i in range(self.n_items):
            # Create ablated context by removing the i-th document
            ablated_items = [item for j, item in enumerate(self.items) if i != j]
            ablated_context_str = "\n\n".join(ablated_items)
            
            # Get distributions for the ablated context
            ablated_distributions = self._get_response_token_distributions(
                context_str=ablated_context_str,
                response=self.target_response
            )

            total_jsd_for_item = 0.0
            # Sum the JSD scores over all tokens in the response
            if ablated_distributions.nelement() > 0:
                for token_idx in range(len(baseline_distributions)):
                    p = baseline_distributions[token_idx]
                    q = ablated_distributions[token_idx]
                    token_jsd = self._jensen_shannon_divergence(p, q)
                    total_jsd_for_item += token_jsd
            
            jsd_scores[i] = total_jsd_for_item
            
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        
        return jsd_scores

    def llm_evaluation(self, gold_answer,embedder=None, metric="cosine", model=None, tokenizer=None):
        if metric=="cosine":
            return cosine_similarity(embedder.encode([gold_answer], convert_to_numpy=True), embedder.encode([self.target_response], convert_to_numpy=True))
        elif metric=="logprob":
            return self._llm_compute_logprob(context_str="\n\n".join(self.items), response=gold_answer)
        elif metric=="llm_judge":
            prompt = """
                You are an impartial evaluator. Your task is to compare two responses: one is a ground truth answer (ideal answer), 
                and the other is a generated answer produced by another language model.Your goal is to determine if 
                the generated answer matches the ground truth in meaning and factual accuracy. Respond in the following strict JSON format:
                {{
                "evaluation": "Yes" or "No",
                "explanation": "Short, clear explanation of your judgment"
                }}
                Evaluation Criteria:
                - Respond "Yes" if the generated answer expresses the same meaning as the ground truth and has no critical factual errors, omissions, or contradictions.
                - Respond "No" if the generated answer changes the meaning, omits important details, introduces factual inaccuracies, or contradicts the ground truth.
                Be concise in your explanation. Do not add any content outside the JSON structure.
                ---
                Example 1
                Question: What is the capital of France?  
                Ground Truth Answer: Paris is the capital of France.  
                Generated Answer: The capital of France is Paris.  
                Response:
                {{
                "evaluation": "Yes",
                "explanation": "The generated answer is semantically identical and factually correct."
                }}
                
                Example 2
                Question: What is the capital of France?  
                Ground Truth Answer: Paris is the capital of France.  
                Generated Answer: The capital of France is Marseille.  
                Response:
                {{
                "evaluation": "No",
                "explanation": "Marseille is not the capital of France; this is a factual error."
                }}
                
                Example 3
                Question: Who wrote *Pride and Prejudice*?  
                Ground Truth Answer: Jane Austen  
                Generated Answer: Charlotte Bronte wrote *Pride and Prejudice*.  
                Response:
                {{
                "evaluation": "No",
                "explanation": "The generated answer misattributes the author, which is factually incorrect."
                }}
                
                Example 4
                Question: What is the boiling point of water at sea level in Celsius?  
                Ground Truth Answer: 100C  
                Generated Answer: Around 100 degrees Celsius.  
                Response:
                {{
                "evaluation": "Yes",
                "explanation": "The generated answer approximates the correct value and preserves the meaning."
                }}
                
                Be strict and unbiased. Always follow the exact JSON format.
                """
            messages = [{"role": "system", "content": """
                You are an impartial evaluator. Your task is to compare two responses: one is a ground truth answer (ideal answer), 
                and the other is a generated answer produced by another language model.Your goal is to determine if 
                the generated answer matches the ground truth in meaning and factual accuracy. Respond in the following strict JSON format:
                {
                "evaluation": "Yes" or "No",
                "explanation": "Short, clear explanation of your judgment"
                }
                Evaluation Criteria:
                - Respond "Yes" if the generated answer expresses the same meaning as the ground truth and has no critical factual errors, omissions, or contradictions.
                - Respond "No" if the generated answer changes the meaning, omits important details, introduces factual inaccuracies, or contradicts the ground truth.
                Be concise in your explanation. Do not add any content outside the JSON structure.
                ---
                Example 1
                Question: What is the capital of France?  
                Ground Truth Answer: Paris is the capital of France.  
                Generated Answer: The capital of France is Paris.  
                Response:
                {
                "evaluation": "Yes",
                "explanation": "The generated answer is semantically identical and factually correct."
                }
                
                Example 2
                Question: What is the capital of France?  
                Ground Truth Answer: Paris is the capital of France.  
                Generated Answer: The capital of France is Marseille.  
                Response:
                {
                "evaluation": "No",
                "explanation": "Marseille is not the capital of France; this is a factual error."
                }
                
                Example 3
                Question: Who wrote *Pride and Prejudice*?  
                Ground Truth Answer: Jane Austen  
                Generated Answer: Charlotte Bronte wrote *Pride and Prejudice*.  
                Response:
                {
                "evaluation": "No",
                "explanation": "The generated answer misattributes the author, which is factually incorrect."
                }
                
                Example 4
                Question: What is the boiling point of water at sea level in Celsius?  
                Ground Truth Answer: 100C  
                Generated Answer: Around 100 degrees Celsius.  
                Response:
                {
                "evaluation": "Yes",
                "explanation": "The generated answer approximates the correct value and preserves the meaning."
                }
                
                Be strict and unbiased. Always follow the exact JSON format.
                """
                }]
                
            prompt2 = f"""You are a strict evaluator comparing two responses. 
            Question: {self.query}
            Ground Truth Answer: {gold_answer}
            Generated Answer: {self.target_response}

            Now generate response in the defined format."""
            
            messages.append({"role": "user", "content": f"You are a strict evaluator comparing two responses. Question: {self.query} \
            Ground Truth Answer: {gold_answer} Generated Answer: {self.target_response} Now generate response in the defined format."})
            
            chat_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
            tokenized = tokenizer(chat_text, return_tensors="pt", padding=True)
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
            unwrapped_model = self.accelerator.unwrap_model(model)
            generated_ids = None # Initialize
            outputs_dict = None # Initialize for potential outputs if model returns dict

            with torch.no_grad():
                outputs = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
                             else unwrapped_model.config.eos_token_id,
                top_p=1.0,
                )
                
                # Tokenize and run generation
                #inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                #with torch.no_grad():
                #    outputs = model.generate(
                #        **inputs,
                #        max_new_tokens=50,
                #        do_sample=False,
                #        pad_token_id=tokenizer.eos_token_id
                #    )
                len_prompt = len(prompt)+len(prompt2)+5
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)[len_prompt:]
                cleaned_text = decoded_output.lstrip().removeprefix("assistant").lstrip(": \n").strip().lower()
                if '"evaluation": "yes"' in cleaned_text:
                    return True
                else:
                    return False
                #evaluation = json.loads(cleaned_text)["evaluation"]
                
                #return evaluation
                
                
                # Decode and extract model reply
                #decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                #model_reply = decoded_output[len(prompt):].strip().lower()
                #model_reply = decoded_output.strip().lower()

                #return model_reply