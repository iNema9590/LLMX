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
from accelerate.utils import broadcast_object_list
from scipy.sparse import csr_matrix
from scipy.stats import beta as beta_dist
from scipy.stats import spearmanr
# from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.metrics.pairwise import cosine_similarity
# from fastFM import als
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
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else unwrapped_model.config.eos_token_id,
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

    def get_utility(self, subset_tuple: tuple) -> float:
        if subset_tuple in self.utility_cache:
            return self.utility_cache[subset_tuple]
        
        v_np = np.array(subset_tuple)
        context_str = self._get_ablated_context_from_vector(v_np)
        utility = self._llm_compute_logprob(context_str=context_str)
        
        self.utility_cache[subset_tuple] = utility
        return utility


    def _llm_compute_logprob(self, context_str: str, response=None) -> float:
        """
        Compute the average log-prob *gain* per answer token compared to an empty context.
        This reduces length bias and centers the utility on the *value of the context*.
        """
        # Use target response if none provided
        if response is None:
            response = self.target_response

        answer_ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        L = answer_ids.shape[1]
        if L == 0:
            return 0.0

        sys_msg = {
            "role": "system",
            "content": """You are a helpful assistant. You use the provided context to answer
                    questions in few words. Avoid using your own knowledge or make assumptions."""
        }
        if context_str:
            user_msg = {"role": "user", "content": f"###context: {context_str} ###question: {self.query}"}
        else:
            user_msg = {"role": "user", "content": self.query}


        # --- Calculate for prompt with context ---
        prompt_with_context_str = self.tokenizer.apply_chat_template(
            [sys_msg, {"role": "user", "content": f"###context: {context_str} ###question: {self.query}" if context_str else self.query}], # Handle both cases
            add_generation_prompt=True,
            tokenize=False
        )
        input_ids_with_context = torch.cat(
            [self.tokenizer(prompt_with_context_str, return_tensors="pt").input_ids.to(self.device),
            answer_ids],
            dim=1
        )
        with torch.no_grad():
            logits_model_output_with = self.model(input_ids=input_ids_with_context).logits

        prompt_len_with = input_ids_with_context.shape[1] - L
        # Extract logits corresponding to the answer tokens
        shift_logits_with = logits_model_output_with[..., prompt_len_with-1:-1, :].contiguous()
        log_probs_tokens_with = F.log_softmax(shift_logits_with, dim=-1)
        # Gather log_probs of actual answer tokens
        answer_log_probs_with = torch.gather(log_probs_tokens_with, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
        total_log_prob_with = answer_log_probs_with.sum()
        # --- Calculate for prompt with empty context ---
        prompt_empty_context_str = self.tokenizer.apply_chat_template(
            [sys_msg, {"role": "user", "content": self.query}], # Query only
            add_generation_prompt=True,
            tokenize=False
        )
        input_ids_empty_context = torch.cat(
            [self.tokenizer(prompt_empty_context_str, return_tensors="pt").input_ids.to(self.device),
            answer_ids],
            dim=1
        )
        with torch.no_grad():
            logits_model_output_empty = self.model(input_ids=input_ids_empty_context).logits

        prompt_len_empty = input_ids_empty_context.shape[1] - L
        shift_logits_empty = logits_model_output_empty[..., prompt_len_empty-1:-1, :].contiguous()
        log_probs_tokens_empty = F.log_softmax(shift_logits_empty, dim=-1)
        answer_log_probs_empty = torch.gather(log_probs_tokens_empty, 2, answer_ids.unsqueeze(-1)).squeeze(-1)
        total_log_prob_empty = answer_log_probs_empty.sum() # This is log(P_empty)

        # --- Calculate probabilities and logits ---
        # Probability of the answer given the context
        prob_with = torch.exp(total_log_prob_with)
        # Probability of the answer given empty context
        prob_empty = torch.exp(total_log_prob_empty)

        # Logit of the probability of the answer given the context
        logit_with = logit(prob_with)
        # Logit of the probability of the answer given empty context
        logit_empty = logit(prob_empty)

        # --- Calculate logit gain ---
        # The gain on the logit scale
        logit_gain_total = logit_with - logit_empty

        # Clean up (important for long loops in CONTEXTCITE)
        del logits_model_output_with, logits_model_output_empty, shift_logits_with, shift_logits_empty
        del log_probs_tokens_with, log_probs_tokens_empty, answer_log_probs_with, answer_log_probs_empty
        del total_log_prob_with, total_log_prob_empty, prob_with, prob_empty, logit_with, logit_empty
        torch.cuda.empty_cache()
        return logit_gain_total.item()
        

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

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

        model, weights, _ = self._train_surrogate(
            sampled_tuples_for_train, 
            utilities_for_train,
            sur_type="linear"
        )
        return weights, model

    def compute_wss(self, num_samples: int, seed: int = None, sampling="kernelshap", sur_type="fm"):
        """
        Computes Weighted Subset Sampling (WSS) attributions using a surrogate model.
        Returns: (shapley_values, attr, F, model, mse)
        """
        if not self.accelerator.is_main_process: 
            return (np.zeros(self.n_items), np.zeros(self.n_items), None, None, 0.0)
        
        # Generate subsets and compute utilities
        sampled_tuples = self._generate_sampled_ablations(num_samples, sampling_method=sampling, seed=seed)
        pbar = tqdm(sampled_tuples, desc=f"Computing utilities for WSS ({sampling})", disable=not self.verbose)
        utilities_for_samples = [self.get_utility(v_tuple) for v_tuple in pbar]

        # Filter invalid utilities
        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]

        # Train surrogate model
        model, attr, F = self._train_surrogate(
            sampled_tuples_for_train, 
            utilities_for_train, 
            sur_type=sur_type
        )
        
        # Generate all possible subsets (2^n)
        all_subsets = list(itertools.product([0, 1], repeat=self.n_items))
        X_all = np.array(all_subsets)
        
        # Predict utilities for all subsets using surrogate
        if sur_type == "fm":
            X_all_sparse = csr_matrix(X_all)
            predicted_utilities = model.predict(X_all_sparse)
        else:  # linear or full_poly2
            predicted_utilities = model.predict(X_all)
        
        # Compute exact Shapley values using predicted utilities
        utility_dict = dict(zip(all_subsets, predicted_utilities))
        shapley_values = np.zeros(self.n_items)
        
        # Iterate through all subsets
        for s_tuple in all_subsets:
            s_size = sum(s_tuple)
            s_util = utility_dict[s_tuple]
            
            for i in range(self.n_items):
                if s_tuple[i] == 1:
                    # Create subset without item i
                    s_without_i_list = list(s_tuple)
                    s_without_i_list[i] = 0
                    s_without_i_tuple = tuple(s_without_i_list)
                    
                    s_without_i_util = utility_dict.get(s_without_i_tuple, 0.0)
                    marginal_contrib = s_util - s_without_i_util
                    
                    # Calculate Shapley weight
                    weight = (self._factorials[s_size - 1] * self._factorials[self.n_items - s_size]) / self._factorials[self.n_items]
                    shapley_values[i] += weight * marginal_contrib
        
        return shapley_values, attr, F, model

    def compute_tmc_shap(self, num_iterations_max: int, performance_tolerance: float, 
                        max_unique_lookups: int, seed: int = None, 
                        shared_cache: dict = None):
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
            utility = self._llm_compute_logprob(context_str=self._get_ablated_context_from_vector(np.array(subset_tuple)))
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
                        shared_cache: dict = None):
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
            
            utility = self._llm_compute_logprob(context_str=self._get_ablated_context_from_vector(np.array(subset_tuple)))
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
            return model, model.coef_, None
        
        # elif sur_type == "fm":
        #     X_train_fm = csr_matrix(X_train)
        #     model = als.FMRegression(n_iter=100, rank=4, l2_reg_w=0.1, l2_reg_V=0.1, random_state=42)
        #     model.fit(X_train_fm, y_train)
        #     w, V = model.w_, model.V_.T
        #     F = V @ V.T
        #     np.fill_diagonal(F, 0.0)
        #     attr = w + 0.5 * F.sum(axis=1)
        #     return model, attr, F
        
        elif sur_type == "full_poly2":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = poly.fit_transform(X_train)
            model = Ridge(alpha=0.01, fit_intercept=True) # Small alpha for stability
            model.fit(X_poly, y_train)
            
            n = X_train.shape[1]
            linear, pairs = model.coef_[:n], np.zeros((n, n))
            idx = n
            for i in range(n):
                for j in range(i + 1, n):
                    pairs[i, j] = pairs[j, i] = model.coef_[idx]
                    idx += 1
            importance = linear + 0.5 * pairs.sum(axis=1)
            return model, importance, pairs
        
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

        # Get top k (100) logits (top k most probable tokens)
        topk_values, topk_indices = torch.topk(shifted_logits, k=100, dim=-1) # shape: (1, L, vocab_size)
        
        # Calculate the probability distributions
        # distributions = F.softmax(shifted_logits, dim=-1).squeeze(0) # Shape: (L, vocab_size)
        topk_probs = F.softmax(topk_values, dim=-1).squeeze(0) # Shape: (L, 100)

        distributions = torch.zeros((L, logits.shape[-1]), device = self.device) # Full vocab size
        distributions.scatter_(dim=1, index = topk_indices.squeeze(0), src=topk_probs)

        # Cleanup
        del outputs, logits, shifted_logits, prompt_ids, answer_ids, input_ids, topk_values, topk_indices
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
    
    def lds(self, model_FM, model_cc, n_eval_util):
        eval_subsets = self._generate_sampled_ablations(n_eval_util, sampling_method='uniform', seed=2)
        X_all = np.array(eval_subsets)
        exact_utilities = [self.get_utility(v_tuple) for v_tuple in eval_subsets]
        # Predict utilities for all subsets using surrogate
        X_all_sparse = csr_matrix(X_all)
        predicted_utilities_fm = model_FM.predict(X_all_sparse)
        predicted_utilities_cc = model_cc.predict(X_all)
        spearman_cc, _ = spearmanr(exact_utilities, predicted_utilities_cc)
        spearman_fm, _ = spearmanr(exact_utilities, predicted_utilities_fm)
        
        return spearman_cc, spearman_fm
def logit(p, eps=1e-7):
    """Safe logit calculation with clamping to avoid numerical instability"""
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))