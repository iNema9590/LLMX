import numpy as np
import random
import math
import itertools
from collections import defaultdict
from sklearn.linear_model import Lasso, LinearRegression
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
# from accelerate import PartialState # Not used directly in this refactor
from accelerate.utils import gather_object, broadcast_object_list 
from tqdm.auto import tqdm
import time
# import pandas as pd # Not used in the provided snippet
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from scipy.stats import beta as beta_dist
from accelerate import Accelerator
# import torch.multiprocessing as mp # Not explicitly needed with accelerate launch

class ShapleyExperimentHarness:
    def __init__(self, items, query, model, tokenizer, verbose=False):
        self.accelerator = Accelerator(mixed_precision='fp16') # Or your desired precision
        self.items = items
        self.query = query
        # Model is passed in, assumed to be loaded on CPU or with device_map="auto"
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.n_items = len(items)
        self.device = self.accelerator.device # Device for the current process

        # Prepare model for distributed execution (moves to self.device)
        # This is where self.model might become a DDP-wrapped model
        self.model = self.accelerator.prepare(model)
        self.model.eval() # Call eval on the (potentially wrapped) model

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # If tokenizer's pad_token_id changed, update model config's pad_token_id
            # Access the original model's config using .module if DDP is used
            
            # Get the actual underlying model
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            if unwrapped_model.config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                unwrapped_model.config.pad_token_id = self.tokenizer.pad_token_id
            # Also, ensure the generation config is updated if it exists and is separate
            if hasattr(unwrapped_model, 'generation_config') and unwrapped_model.generation_config.pad_token_id is None and self.tokenizer.pad_token_id is not None:
                 unwrapped_model.generation_config.pad_token_id = self.tokenizer.pad_token_id


        self._factorials = {k: math.factorial(k) for k in range(self.n_items + 1)}

        if self.verbose and self.accelerator.is_main_process:
            print("Generating target response based on full context...")
        self.target_response = self._llm_generate_response(context_str="\n\n".join(self.items))
        if self.verbose and self.accelerator.is_main_process:
            print(f"Target response: '{self.target_response}'")

        self.all_true_utilities = self._precompute_all_utilities()

    def _precompute_all_utilities(self) -> dict[tuple, float]:
        all_ablations_tuples = list(itertools.product([0, 1], repeat=self.n_items))
        num_total_subsets = len(all_ablations_tuples)

        if self.verbose and self.accelerator.is_main_process:
            print(f"Starting pre-computation of utilities for {num_total_subsets} subsets "
                  f"using {self.accelerator.num_processes} processes...")

        local_results_for_this_process = {}

        # Distribute subsets among processes and compute utilities locally
        with self.accelerator.split_between_processes(all_ablations_tuples) as process_specific_subsets:
            if process_specific_subsets: # Ensure the list is not empty for this process
                progress_bar = tqdm(
                    process_specific_subsets,
                    desc=f"Process {self.accelerator.process_index} Computing Utilities",
                    disable=not (self.verbose and self.accelerator.is_local_main_process), # Show on local main process
                    position=self.accelerator.local_process_index, # For neat stacking of bars
                    leave=False
                )
                for v_tuple in progress_bar:
                    v_np = np.array(v_tuple)
                    context_str = self._get_ablated_context_from_vector(v_np)
                    try:
                        utility = self._llm_compute_logprob(context_str=context_str)
                        local_results_for_this_process[v_tuple] = utility
                    except Exception as e:
                        if self.verbose: # Basic error logging per process
                            print(f"Error on process {self.accelerator.process_index} for subset {v_tuple}: {e}")
                        local_results_for_this_process[v_tuple] = -float('inf') # Mark as failed

        # Each process wraps its dictionary in a list for gather_object.
        # In this environment, gather_object([local_dict]) results in [dict_rank0, dict_rank1, ...]
        object_payload_for_gather = [local_results_for_this_process]
        gathered_list_of_dicts = gather_object(object_payload_for_gather)

        # Initialize variables for broadcasting
        final_utilities_on_main = {}
        object_to_broadcast = [None] # Placeholder for all processes

        # Aggregation on the main process
        if self.accelerator.is_main_process:
            if isinstance(gathered_list_of_dicts, list):
                for i, single_process_dict in enumerate(gathered_list_of_dicts):
                    if isinstance(single_process_dict, dict):
                        final_utilities_on_main.update(single_process_dict)
                    elif self.verbose and single_process_dict is not None:
                        # This case should ideally not happen if gather_object works as expected now
                        print(f"Warning (Main Proc): Item from rank {i} in gathered data is not a dict "
                              f"(type: {type(single_process_dict)}). Skipping.")
            elif self.verbose:
                print(f"Warning (Main Proc): gathered_list_of_dicts is not a list "
                      f"(type: {type(gathered_list_of_dicts)}). Cannot aggregate.")

            if self.verbose:
                computed_count = len(final_utilities_on_main)
                failed_count = sum(1 for u in final_utilities_on_main.values() if u == -float('inf'))
                print(f"Pre-computation aggregation complete on main process.")
                print(f"Total utilities aggregated: {computed_count}/{num_total_subsets}")
                if failed_count > 0:
                    print(f"Warning (Main Proc): {failed_count} utility computations resulted in errors (-inf).")
                # Consider if this warning is too noisy if num_total_subsets is large and some minor discrepancies occur
                # or if some processes legitimately had no work for very small n_items.
                if computed_count != num_total_subsets and computed_count > 0 and num_total_subsets > 0 :
                     # print(f"Warning (Main Proc): Aggregated {computed_count} utilities, but expected {num_total_subsets}.")
                     pass # Potentially re-enable if strict count matching is critical

            object_to_broadcast = [final_utilities_on_main]

        # Broadcast the aggregated dictionary from the main process to all other processes
        broadcast_object_list(object_to_broadcast)

        # All processes now have the complete, aggregated utilities
        complete_utilities = object_to_broadcast[0]

        # Fallback if broadcast results in None (e.g., main process had issues)
        if complete_utilities is None:
            if self.verbose:
                print(f"Rank {self.accelerator.process_index}: Received None after broadcast. "
                      "Main process might have had an issue or broadcasted empty. Falling back to empty dict.")
            complete_utilities = {}

        # Synchronize all processes before proceeding
        self.accelerator.wait_for_everyone()

        if self.verbose and self.accelerator.is_main_process:
            final_size = len(complete_utilities if complete_utilities else {})
            print(f"All processes synchronized. Final size of utility map on main process: {final_size}")

        return complete_utilities if complete_utilities is not None else {}

    def _llm_generate_response(self, context_str: str, max_new_tokens: int = 50) -> str:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context_str:
            messages.append({"role": "user", "content": f"Given the context: {context_str}. Briefly answer the query: {self.query}"})
        else:
            messages.append({"role": "user", "content": self.query})

        chat_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        tokenized = self.tokenizer(chat_text, return_tensors="pt", padding=True)

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # --- KEY CHANGE FOR .generate() ---
        # Always call .generate() on the UNWRAPPED model when using DDP
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        with torch.no_grad():
            generated_ids = unwrapped_model.generate( # Call on unwrapped_model
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
                             else unwrapped_model.config.eos_token_id, # Use unwrapped_model.config here too
                temperature=1.0,
                top_p=1.0
            )
        response_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_text = response_text.lstrip().removeprefix("assistant").lstrip(": \n").strip()
        return cleaned_text

    def _llm_compute_logprob(self, context_str: str) -> float:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context_str:
            messages.append({"role": "user", "content": f"Given the context: {context_str}. Briefly answer the query: {self.query}"})
        else:
            messages.append({"role": "user", "content": self.query})

        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        answer_ids = self.tokenizer(self.target_response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        if answer_ids.shape[1] == 0:
            return 0.0 if not self.target_response else -float('inf')

        total_log_prob = 0.0
        current_input_ids = prompt_ids.clone()

        unwrapped_model = self.accelerator.unwrap_model(self.model) # Get unwrapped for config
        max_len = getattr(unwrapped_model.config, 'max_position_embeddings', 512)

        for i in range(answer_ids.shape[1]):
            with torch.no_grad():
                # For the standard forward pass model(...), using self.model (DDP wrapped) is fine
                # DDP handles forwarding the __call__ which maps to the forward method.
                outputs = self.model(input_ids=current_input_ids)
                logits = outputs.logits[:, -1, :]

            next_token_id = answer_ids[0, i].item()
            log_prob_next_token = F.log_softmax(logits, dim=-1)[0, next_token_id].item()
            total_log_prob += log_prob_next_token

            current_input_ids = torch.cat([current_input_ids, answer_ids[:, i:i+1]], dim=1)

            if current_input_ids.shape[1] >= max_len:
                if self.verbose and self.accelerator.is_local_main_process:
                    print(f"Warning (Proc {self.accelerator.process_index}): Input length {current_input_ids.shape[1]} exceeded max model length ({max_len}) during logprob. Prob may be truncated.")
                break
        return total_log_prob
    

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    def _calculate_shapley_from_cached_dict(self, utility_dict_to_use: dict[tuple, float]) -> np.ndarray:
        shapley_values = np.zeros(self.n_items)
        n = self.n_items
        factorials_local = self._factorials # Already initialized in __init__
        
        pbar_desc = "Calculating Shapley (from cache)"
        # Show tqdm only on main process for this CPU-bound calculation
        pbar_enabled = self.verbose and self.accelerator.is_main_process and len(utility_dict_to_use) > 1000 # Adjusted threshold
        
        # The iterator for the loop
        item_indices_iterator = range(n)
        if pbar_enabled:
            item_indices_iterator = tqdm(item_indices_iterator, desc=pbar_desc, leave=False)

        for i in item_indices_iterator:
            shap_i = 0.0
            for s_tuple, s_util in utility_dict_to_use.items():
                if len(s_tuple) != n : continue
                if s_util == -float('inf'): continue # Skip failed utilities
                if s_tuple[i] == 0: # If item i is not in subset S
                    s_size = sum(s_tuple)
                    s_union_i_list = list(s_tuple)
                    s_union_i_list[i] = 1
                    s_union_i_tuple = tuple(s_union_i_list)
                    if s_union_i_tuple in utility_dict_to_use:
                        s_union_i_util = utility_dict_to_use[s_union_i_tuple]
                        if s_union_i_util == -float('inf'): continue # Skip if union utility failed
                        marginal_contribution = s_union_i_util - s_util
                        weight = (factorials_local[s_size] * factorials_local[n - s_size - 1]) / factorials_local[n]
                        shap_i += weight * marginal_contribution
            shapley_values[i] = shap_i
        return shapley_values

    def _sample_ablations_from_all(self, num_samples: int) -> list[tuple]:
        max_possible = 2**self.n_items
        if num_samples > max_possible: num_samples = max_possible
        
        # Ensure all_true_utilities is not None (it shouldn't be after __init__)
        if self.all_true_utilities is None:
            if self.verbose and self.accelerator.is_main_process:
                print("Error: self.all_true_utilities is None during sampling. Pre-computation might have failed to broadcast.")
            return []
            
        all_tuples = list(self.all_true_utilities.keys())
        if not all_tuples and num_samples > 0: # No utilities precomputed, but samples requested
            if self.verbose and self.accelerator.is_main_process:
                print("Warning: No utilities available in self.all_true_utilities to sample from.")
            return []

        sampled_tuples_set = set()
        empty_v_tuple = tuple(np.zeros(self.n_items, dtype=int))
        full_v_tuple = tuple(np.ones(self.n_items, dtype=int))

        # Ensure grand coalition and empty set are included if requested and available
        if num_samples >= 1 and empty_v_tuple in self.all_true_utilities:
            sampled_tuples_set.add(empty_v_tuple)
        if num_samples >= 2 and full_v_tuple in self.all_true_utilities and empty_v_tuple != full_v_tuple:
            sampled_tuples_set.add(full_v_tuple)
        
        remaining_to_sample = num_samples - len(sampled_tuples_set)
        if remaining_to_sample > 0:
            # Exclude already added tuples from the pool for random sampling
            other_tuples = [t for t in all_tuples if t not in sampled_tuples_set]
            if len(other_tuples) >= remaining_to_sample:
                sampled_tuples_set.update(random.sample(other_tuples, remaining_to_sample))
            else: # Sample all remaining unique tuples if not enough for requested 'remaining_to_sample'
                sampled_tuples_set.update(other_tuples)
        return list(sampled_tuples_set)

    def _train_surrogate(self, ablations: list[tuple], utilities: list[float], lasso_alpha: float) -> tuple:
        X = np.array(ablations)
        y = np.array(utilities)
        if X.shape[0] == 0:
            if self.verbose and self.accelerator.is_main_process:
                 print("Error: Cannot train surrogate with zero samples.")
            # Return a model-like object that predicts zeros or mean, and zero coefficients
            dummy_coeffs = np.zeros(X.shape[1] if X.ndim > 1 and X.shape[1] > 0 else self.n_items)
            class DummyModel:
                def __init__(self, intercept, coef): self.intercept_ = intercept; self.coef_ = coef
                def predict(self, X_pred): return np.full(X_pred.shape[0], self.intercept_) + X_pred @ self.coef_
            return DummyModel(0.0, dummy_coeffs), dummy_coeffs, 0.0

        if lasso_alpha == 0: model = LinearRegression(fit_intercept=True)
        else: model = Lasso(alpha=lasso_alpha, fit_intercept=True, random_state=42, max_iter=2000)
        model.fit(X, y)
        return model, model.coef_, model.intercept_

    # --- Public Methods: These use self.all_true_utilities, which is now synced ---
    # Calculations are CPU-bound. They will run on all processes if called by all.
    # Print statements and tqdm are guarded to run primarily on the main process.

    def compute_exact_shap(self):
        if self.verbose and self.accelerator.is_main_process:
            print("Computing Exact Shapley (using pre-computed utilities)...")
        # self.all_true_utilities is available and identical on all processes
        return self._calculate_shapley_from_cached_dict(self.all_true_utilities)

    def compute_contextcite_weights(self, num_samples: int, lasso_alpha: float, seed: int = None):
        if self.verbose and self.accelerator.is_main_process:
            print(f"Computing ContextCite Weights (m={num_samples}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed) # Seed for all processes for consistent sampling

        sampled_tuples = self._sample_ablations_from_all(num_samples)
        if not sampled_tuples:
            if self.verbose and self.accelerator.is_main_process: print("Error: No tuples sampled for ContextCite.");
            return np.zeros(self.n_items) # Return zero array or handle as error

        utilities_for_samples = [self.all_true_utilities.get(v_tuple, -float('inf')) for v_tuple in sampled_tuples]

        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
        if not valid_indices:
            if self.verbose and self.accelerator.is_main_process: print("Error: All sampled precomputed utilities failed. Cannot train surrogate for ContextCite.");
            return np.zeros(self.n_items)

        if len(valid_indices) < len(sampled_tuples) and self.verbose and self.accelerator.is_main_process:
            print(f"Warning: {len(sampled_tuples) - len(valid_indices)} precomputed utilities were -inf, using {len(valid_indices)} for ContextCite surrogate.")

        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]

        _, weights, _ = self._train_surrogate(sampled_tuples_for_train, utilities_for_train, lasso_alpha)
        return weights

    # ... (Apply similar guards for verbose printing and tqdm in other public methods like compute_wss, compute_tmc_shap, etc.) ...
    # For brevity, I'll just show compute_wss as an example of further tqdm guarding. Other methods would follow suit.

    def compute_wss(self, num_samples: int, lasso_alpha: float, seed: int = None, return_weights: bool = False):
        if self.verbose and self.accelerator.is_main_process:
            print(f"Computing Weakly Supervised Shapley (m={num_samples}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed)

        sampled_tuples = self._sample_ablations_from_all(num_samples)
        if not sampled_tuples:
            if self.verbose and self.accelerator.is_main_process: print("Error: No tuples sampled for WSS.");
            return (np.zeros(self.n_items), np.zeros(self.n_items)) if return_weights else np.zeros(self.n_items)

        utilities_for_samples = [self.all_true_utilities.get(v_tuple, -float('inf')) for v_tuple in sampled_tuples]
        
        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
        if not valid_indices:
            if self.verbose and self.accelerator.is_main_process: print("Error: All sampled precomputed utilities failed for WSS. Cannot train surrogate.");
            return (np.zeros(self.n_items), np.zeros(self.n_items)) if return_weights else np.zeros(self.n_items)

        if len(valid_indices) < len(sampled_tuples) and self.verbose and self.accelerator.is_main_process:
            print(f"Warning: {len(sampled_tuples) - len(valid_indices)} precomputed utilities were -inf, using {len(valid_indices)} for WSS surrogate.")
            
        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]
        known_utilities_for_hybrid = {v_tuple: self.all_true_utilities[v_tuple] for v_tuple in sampled_tuples_for_train}
        
        surrogate_model, weights, _ = self._train_surrogate(sampled_tuples_for_train, utilities_for_train, lasso_alpha)

        hybrid_utilities = {}
        all_ablations_X = np.array(list(itertools.product([0, 1], repeat=self.n_items)), dtype=int)
        
        # Prediction can be done by all, it's fast.
        all_predictions = surrogate_model.predict(all_ablations_X)
        
        for i_pred in range(all_ablations_X.shape[0]): # Corrected loop variable name
            v_tuple = tuple(all_ablations_X[i_pred])
            hybrid_utilities[v_tuple] = known_utilities_for_hybrid.get(v_tuple, all_predictions[i_pred])
        
        shapley_values_wss = self._calculate_shapley_from_cached_dict(hybrid_utilities) # _calculate_shapley has tqdm guard
        
        return (shapley_values_wss, weights) if return_weights else shapley_values_wss

    def compute_tmc_shap(self, num_iterations: int, performance_tolerance: float, seed: int = None):
        if self.verbose and self.accelerator.is_main_process:
            print(f"Computing TMC-Shapley (T={num_iterations}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed)

        shapley_values = np.zeros(self.n_items)
        marginal_counts = np.zeros(self.n_items, dtype=int)

        v_empty_tuple = tuple(np.zeros(self.n_items, dtype=int))
        v_full_tuple = tuple(np.ones(self.n_items, dtype=int))
        v_empty_util = self.all_true_utilities.get(v_empty_tuple, -float('inf'))
        v_full_util = self.all_true_utilities.get(v_full_tuple, -float('inf'))

        if v_empty_util == -float('inf') or v_full_util == -float('inf'):
             if self.verbose and self.accelerator.is_main_process:
                 print("Warning: Truncation disabled in TMC due to endpoint utility failure in pre-computation.")
             performance_tolerance = float('inf') # Disable truncation

        indices = list(range(self.n_items))
        
        pbar_iter = range(num_iterations)
        if self.verbose and self.accelerator.is_main_process:
            pbar_iter = tqdm(pbar_iter, desc="TMC Iterations (from cache)", leave=False)
        
        for t in pbar_iter:
            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_indices_list = [] # Store indices of items in current subset

            for item_idx_to_add in permutation:
                current_subset_indices_list.append(item_idx_to_add)
                # No need to sort if constructing v_curr_np directly from indices
                
                v_curr_np = np.zeros(self.n_items, dtype=int)
                if current_subset_indices_list: # Ensure list is not empty before indexing
                    v_curr_np[current_subset_indices_list] = 1
                v_curr_tuple = tuple(v_curr_np)
                
                can_truncate = False
                if v_prev_util != -float('inf') and v_full_util != -float('inf'): # Check if valid utilities for truncation
                    is_near_full_set_performance = abs(v_full_util - v_prev_util) < performance_tolerance
                    can_truncate = t > 0 and is_near_full_set_performance # Original paper: t > min_iter
                
                if can_truncate: 
                    v_curr_util = v_prev_util 
                else:
                    v_curr_util = self.all_true_utilities.get(v_curr_tuple, -float('inf'))

                marginal_contribution = 0.0
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'):
                    marginal_contribution = v_curr_util - v_prev_util
                
                k_count = marginal_counts[item_idx_to_add] + 1
                shapley_values[item_idx_to_add] = ( (k_count - 1) / k_count ) * shapley_values[item_idx_to_add] + \
                                                  ( 1 / k_count ) * marginal_contribution
                marginal_counts[item_idx_to_add] = k_count
                
                v_prev_util = v_curr_util # This should be v_curr_util (utility of current set)
        return shapley_values

    def compute_beta_shap(self, num_iterations: int, beta_a: float, beta_b: float, seed: int = None):
        if beta_dist is None: raise ImportError("BetaShap requires scipy.")
        if self.verbose and self.accelerator.is_main_process:
            print(f"Computing Beta-Shapley (T={num_iterations}, α={beta_a}, β={beta_b}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed)

        weighted_marginal_sums = np.zeros(self.n_items)
        total_weights_for_item = np.zeros(self.n_items)

        v_empty_tuple = tuple(np.zeros(self.n_items, dtype=int))
        v_empty_util = self.all_true_utilities.get(v_empty_tuple, -float('inf'))
        
        indices = list(range(self.n_items))
        
        pbar_iter = range(num_iterations)
        if self.verbose and self.accelerator.is_main_process:
            pbar_iter = tqdm(pbar_iter, desc="BetaShap Iterations (from cache)", leave=False)

        for _ in pbar_iter:
            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_indices_list = [] 

            for k_minus_1, item_idx_to_add in enumerate(permutation): # k_minus_1 is the size of the set *before* adding item_idx_to_add
                current_subset_indices_list.append(item_idx_to_add)
                # No need to sort for v_curr_np construction
                
                v_curr_np = np.zeros(self.n_items, dtype=int)
                if current_subset_indices_list:
                    v_curr_np[current_subset_indices_list] = 1
                v_curr_tuple = tuple(v_curr_np)
                
                v_curr_util = self.all_true_utilities.get(v_curr_tuple, -float('inf'))
                
                marginal_contribution = 0.0
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'):
                     marginal_contribution = v_curr_util - v_prev_util
                
                if self.n_items > 1:
                    x_pos = k_minus_1 / (self.n_items - 1) # Position based on size of S (set before adding i)
                elif self.n_items == 1: # Only one item
                    x_pos = 0.0 # Or 0.5, depends on interpretation for single item; 0 is |S|=0
                else: # No items
                    x_pos = 0.5 # Default, though loop won't run

                weight = 1.0 # Default weight
                try:
                    if self.n_items > 0: # Avoid issues with n_items=0 or pdf definition at edges for certain beta params
                        if self.n_items == 1 and (beta_a <= 1 or beta_b <= 1): # For single item, pdf can be inf at 0 or 1
                             # Standard BetaShap often implies uniform weighting if beta params would cause issues
                             weight = 1.0
                        else:
                             weight = beta_dist.pdf(x_pos, beta_a, beta_b)
                    if not np.isfinite(weight): # Catch inf/nan from pdf
                        # A large finite number if inf, or 1.0 as a fallback
                        weight = 1e6 if np.isinf(weight) else 1.0
                except Exception: # Catch any other errors from pdf calculation
                    weight = 1.0
                
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'): # Only add if MC is valid
                    weighted_marginal_sums[item_idx_to_add] += weight * marginal_contribution
                    total_weights_for_item[item_idx_to_add] += weight
                
                v_prev_util = v_curr_util
                
        shapley_values = np.zeros(self.n_items)
        non_zero_weights_mask = total_weights_for_item > 1e-9 # Avoid division by zero
        shapley_values[non_zero_weights_mask] = weighted_marginal_sums[non_zero_weights_mask] / total_weights_for_item[non_zero_weights_mask]
        return shapley_values

    def compute_loo(self):
        if self.verbose and self.accelerator.is_main_process:
            print(f"Computing LOO (n={self.n_items}, using pre-computed utilities)...")
        loo_scores = np.zeros(self.n_items)
        v_all_tuple = tuple(np.ones(self.n_items, dtype=int))
        util_all = self.all_true_utilities.get(v_all_tuple, -float('inf'))

        for i in range(self.n_items):
            v_loo_list = list(v_all_tuple) # Start with all items
            v_loo_list[i] = 0 # Remove item i
            v_loo_tuple = tuple(v_loo_list)
            util_loo = self.all_true_utilities.get(v_loo_tuple, -float('inf'))
            
            if util_all == -float('inf') and util_loo == -float('inf'): loo_scores[i] = 0.0 # Both failed, no diff
            elif util_loo == -float('inf'): loo_scores[i] = np.inf # Removing item made it "infinitely" worse (or revealed full set was bad)
            elif util_all == -float('inf'): loo_scores[i] = -np.inf # Full set failed, but LOO set succeeded
            else: loo_scores[i] = util_all - util_loo
        return loo_scores