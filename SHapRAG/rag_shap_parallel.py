import numpy as np
import os
import random
import math
import pickle
import itertools
import xgboost
from sklearn.linear_model import Lasso
from pygam import LinearGAM, s, te, f, l
from interpret.glassbox import ExplainableBoostingRegressor
from fastFM import als
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import gather_object, broadcast_object_list 
from tqdm.auto import tqdm
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import beta as beta_dist
from scipy.sparse import csr_matrix
class ShapleyExperimentHarness:
    def __init__(self, items, query, 
                 prepared_model_for_harness, tokenizer_for_harness, accelerator_for_harness, 
                 verbose=False, utility_path=None):
        self.accelerator = accelerator_for_harness
        self.items = items
        self.query = query
        self.model = prepared_model_for_harness 
        self.tokenizer = tokenizer_for_harness
        self.verbose = verbose
        self.n_items = len(items)
        self.device = self.accelerator.device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._factorials = {k: math.factorial(k) for k in range(self.n_items + 1)}

        if self.verbose and self.accelerator.is_main_process:
            print("Generating target response based on full context...")
        self.target_response = self._llm_generate_response(context_str="\n\n".join(self.items))
        if self.verbose and self.accelerator.is_main_process:
            print(f"Target response: '{self.target_response}'")
        loaded_utilities = False
        if utility_path and self.accelerator.is_main_process: # Only main process tries to load initially
            if os.path.exists(utility_path):
                try:
                    self.all_true_utilities = self.load_utilities(utility_path) # Main process loads
                    if self.verbose:
                        print(f"Successfully loaded utilities from {utility_path}. "
                              f"Found {len(self.all_true_utilities)} entries.")
                    if len(self.all_true_utilities) != 2**self.n_items:
                        print(f"Warning: Loaded utilities count ({len(self.all_true_utilities)}) "
                              f"does not match expected ({2**self.n_items}) for n_items={self.n_items}. "
                              "This might be for a different 'items' set or an incomplete computation.")
                    loaded_utilities = True
                except Exception as e:
                    print(f"Warning: Failed to load utilities from {utility_path}: {e}. Will recompute.")
            else:
                if self.verbose:
                    print(f"Utility file {utility_path} not found. Will precompute utilities.")

        load_status_broadcast = [loaded_utilities]
        if self.accelerator.is_main_process:
            utilities_to_broadcast = [self.all_true_utilities if loaded_utilities else None]
        else:
            utilities_to_broadcast = [None]

        broadcast_object_list(load_status_broadcast) # Broadcast if utilities were loaded
        loaded_utilities_on_all_procs = load_status_broadcast[0]

        if loaded_utilities_on_all_procs:
            broadcast_object_list(utilities_to_broadcast) # Broadcast the actual utilities
            self.all_true_utilities = utilities_to_broadcast[0]
            if self.all_true_utilities is None and self.verbose and not self.accelerator.is_main_process:
                print(f"Rank {self.accelerator.process_index}: Received None for utilities after broadcast, expected loaded utilities.")
                self.all_true_utilities = {} # Fallback
            if self.verbose and self.accelerator.is_main_process:
                print("Broadcasted loaded utilities to all processes.")
        else:
            if self.verbose and self.accelerator.is_main_process:
                print("Pre-computing utilities as they were not loaded.")
            self.all_true_utilities = self._precompute_all_utilities()
            # Optionally save after computing if utility_path was given but file didn't exist
            if utility_path and self.accelerator.is_main_process and not os.path.exists(utility_path):
                if self.verbose:
                    print(f"Saving newly computed utilities to {utility_path}")
                self.save_utilities(utility_path)

    def save_utilities(self, file_path: str, method: str = "pickle"):

        if not self.accelerator.is_main_process:
            return

        if self.verbose:
            print(f"Main process: Saving utilities to {file_path} using {method}...")
        with open(file_path, "wb") as f:
            pickle.dump(self.all_true_utilities, f)
        if self.verbose:
            print(f"Main process: Successfully saved {len(self.all_true_utilities)} utilities to {file_path}.")

    @classmethod 
    def load_utilities_static(cls, file_path: str, method: str = "pickle") -> dict:

        loaded_utilities = {}
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Utility file {file_path} not found for loading.")


        with open(file_path, "rb") as f:
            loaded_utilities = pickle.load(f)
            
        return loaded_utilities

    def load_utilities(self, file_path: str, method: str = "pickle") -> dict:
    
        if not self.accelerator.is_main_process:
             raise RuntimeError("load_utilities instance method should ideally be called by main process, then broadcast results.")
        
        self.all_true_utilities = ShapleyExperimentHarness.load_utilities_static(file_path, method)
        return self.all_true_utilities
    
    def _precompute_all_utilities(self) -> dict[tuple, float]:
        all_ablations_tuples = list(itertools.product([0, 1], repeat=self.n_items))
        num_total_subsets = len(all_ablations_tuples)

        if self.verbose and self.accelerator.is_main_process:
            print(f"Starting pre-computation of utilities for {num_total_subsets} subsets "
                  f"using {self.accelerator.num_processes} processes...")

        local_results_for_this_process = {}

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
                    utility = self._llm_compute_logprob(context_str=context_str)
                    local_results_for_this_process[v_tuple] = utility

        object_payload_for_gather = [local_results_for_this_process]
        gathered_list_of_dicts = gather_object(object_payload_for_gather)

        final_utilities_on_main = {}
        object_to_broadcast = [None] # Placeholder for all processes

        if self.accelerator.is_main_process:
            if isinstance(gathered_list_of_dicts, list):
                for i, single_process_dict in enumerate(gathered_list_of_dicts):
                    if isinstance(single_process_dict, dict):
                        final_utilities_on_main.update(single_process_dict)

            elif self.verbose:
                print(f"Warning (Main Proc): gathered_list_of_dicts is not a list "
                      f"(type: {type(gathered_list_of_dicts)}). Cannot aggregate.")

            if self.verbose:
                computed_count = len(final_utilities_on_main)
                failed_count = sum(1 for u in final_utilities_on_main.values() if u == -float('inf'))
                print(f"Total utilities aggregated: {computed_count}/{num_total_subsets}")
                if failed_count > 0:
                    print(f"Warning (Main Proc): {failed_count} utility computations resulted in errors (-inf).")
           
                if computed_count != num_total_subsets and computed_count > 0 and num_total_subsets > 0 :
                     pass 

            object_to_broadcast = [final_utilities_on_main]

        broadcast_object_list(object_to_broadcast)

        complete_utilities = object_to_broadcast[0]

        if complete_utilities is None:
            if self.verbose:
                print(f"Rank {self.accelerator.process_index}: Received None after broadcast. "
                      "Main process might have had an issue or broadcasted empty. Falling back to empty dict.")
            complete_utilities = {}

        self.accelerator.wait_for_everyone()

        if self.verbose and self.accelerator.is_main_process:
            final_size = len(complete_utilities if complete_utilities else {})

        return complete_utilities if complete_utilities is not None else {}

    def _llm_generate_response(self, context_str: str, max_new_tokens: int = 200) -> str:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context_str:
            messages.append({"role": "user", "content": f"Use only the context: {context_str}. Briefly answer the query: {self.query}."})
        else:
            messages.append({"role": "user", "content": self.query})

        chat_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        tokenized = self.tokenizer(chat_text, return_tensors="pt", padding=True)

        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        generated_ids = None # Initialize
        outputs_dict = None # Initialize for potential outputs if model returns dict

        with torch.no_grad():
     
            outputs_gen = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None \
                             else unwrapped_model.config.eos_token_id,
                top_p=1.0,
            )
            # Handle different return types of generate
            if isinstance(outputs_gen, torch.Tensor):
                generated_ids = outputs_gen
            elif isinstance(outputs_gen, dict) and "sequences" in outputs_gen: # Common for GenerateOutput
                generated_ids = outputs_gen["sequences"]
            elif hasattr(outputs_gen, 'sequences'): # For GenerateOutput like objects
                generated_ids = outputs_gen.sequences
            else:
                print(f"Warning: Unexpected output type from model.generate: {type(outputs_gen)}")
                # Try to find a tensor that looks like token IDs
                if isinstance(outputs_gen, (list, tuple)) and len(outputs_gen) > 0 and isinstance(outputs_gen[0], torch.Tensor):
                    generated_ids = outputs_gen[0] # Guessing the first tensor is it
                else: # Cannot determine, return empty or raise error
                    del input_ids, attention_mask, unwrapped_model, outputs_gen
                    torch.cuda.empty_cache()
                    return ""


        response_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        cleaned_text = response_text.lstrip().removeprefix("assistant").lstrip(": \n").strip()

        # Explicitly delete tensors
        del input_ids, attention_mask, generated_ids, unwrapped_model, outputs_gen
        torch.cuda.empty_cache()
        return cleaned_text

    def _llm_compute_logprob(self, context_str: str, response=None) -> float:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if context_str:
            messages.append({"role": "user", "content": f"Use only the context: {context_str}. Briefly answer the query: {self.query}."})
        else:
            messages.append({"role": "user", "content": self.query})
        if not response:
            response=self.target_response
        prompt_templated = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        answer_ids = self.tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        if answer_ids.shape[1] == 0:
            return 0.0 if not response else -float('inf')

        prompt_tokens = self.tokenizer(prompt_templated, return_tensors="pt").to(self.device)

        input_ids = torch.cat([prompt_tokens.input_ids, answer_ids], dim=1)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        max_model_len = getattr(unwrapped_model.config, 'max_position_embeddings', 512)
        
        if input_ids.shape[1] > max_model_len:
            if self.verbose and self.accelerator.is_main_process:
                print(f"Warning (logprob): Combined input length {input_ids.shape[1]} exceeds model max length {max_model_len}. Result may be inaccurate or OOM.")
        
            del prompt_tokens, answer_ids, input_ids, unwrapped_model
            torch.cuda.empty_cache()
            return -float('inf')

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids) # Pass the full sequence
            logits = outputs.logits


        shift_logits = logits[..., prompt_tokens.input_ids.shape[1]-1:-1, :].contiguous()
        shift_labels = answer_ids.contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
 
        answer_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        total_log_prob = answer_log_probs.sum().item()

        del prompt_tokens, answer_ids, input_ids, outputs, logits, shift_logits, shift_labels, answer_log_probs, unwrapped_model
        torch.cuda.empty_cache()
        return total_log_prob
    
    def llm_evaluation(self, gold_answer,embedder, metric="cosine"):
        if metric=="'cosine":
            return cosine_similarity(embedder.encode([gold_answer], convert_to_numpy=True), embedder.encode([self.target_response], convert_to_numpy=True))
        elif metric=="logprob":
            return self._llm_compute_logprob(context_str="\n\n".join(self.items), response=gold_answer)
        

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    def _calculate_shapley_from_cached_dict(self, utility_dict_to_use: dict[tuple, float], second_util={}) -> np.ndarray:
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
                    if s_union_i_tuple in second_util and s_tuple in second_util:
                        s_union_i_util = second_util[s_union_i_tuple]
                        s_util = second_util[s_tuple]
                    elif s_union_i_tuple in utility_dict_to_use:
                        s_union_i_util = utility_dict_to_use[s_union_i_tuple]
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
    
    def _sample_ablations_kernelshap_weighted(self, num_samples: int, seed: int = None) -> list[tuple]:

        if seed is not None:
            random.seed(seed)
            # np.random.seed(seed) # Not strictly needed if only using random.choices

        max_possible = 2**self.n_items
        if num_samples > max_possible:
            num_samples = max_possible
        if num_samples == 0:
            return []

        all_available_tuples = list(self.all_true_utilities.keys())
        if not all_available_tuples:
             if self.verbose: print("Warning: all_true_utilities is empty, cannot sample.")
             return []

        sampled_tuples_set = set()
        
        # 1. Explicitly include empty and full sets if num_samples allows
        empty_v_tuple = tuple(np.zeros(self.n_items, dtype=int))
        full_v_tuple = tuple(np.ones(self.n_items, dtype=int))

        if num_samples >= 1 and empty_v_tuple in self.all_true_utilities:
            sampled_tuples_set.add(empty_v_tuple)
        if num_samples >= 2 and full_v_tuple in self.all_true_utilities and empty_v_tuple != full_v_tuple:
            sampled_tuples_set.add(full_v_tuple)

        remaining_to_sample = num_samples - len(sampled_tuples_set)

        if remaining_to_sample > 0:
            # Filter out already selected tuples for weighted sampling
            candidate_tuples = [t for t in all_available_tuples if t not in sampled_tuples_set]
            if not candidate_tuples and remaining_to_sample > 0:
                 return list(sampled_tuples_set)

            # Calculate KernelSHAP weights only for the candidates
            n_kernel = self.n_items
            candidate_weights = []
            for v_tuple in candidate_tuples:
                z = sum(v_tuple)
                if z == 0 or z == n_kernel:

                    weight = 0.00001 
                else:
                    denominator = z * (n_kernel - z)
                    if denominator == 0: weight = 1.0 # Should not happen for n_kernel > 1
                    else: weight = (n_kernel - 1) / denominator
                candidate_weights.append(weight)
            
            if not candidate_weights or sum(candidate_weights) == 0:
                # Fallback to uniform sampling if all weights are zero (unlikely) or no candidates
                if self.verbose and candidate_tuples: print("Warning: All candidate weights are zero for KernelSHAP sampling. Falling back to uniform.")
                if len(candidate_tuples) >= remaining_to_sample:
                    sampled_from_candidates = random.sample(candidate_tuples, remaining_to_sample)
                    sampled_tuples_set.update(sampled_from_candidates)
                else:
                    sampled_tuples_set.update(candidate_tuples) # Add all remaining if less than needed
            else:
  
                num_to_actually_sample = min(remaining_to_sample, len(candidate_tuples))
                
                # Normalize weights for probability distribution
                total_weight = sum(candidate_weights)
                probabilities = [w / total_weight for w in candidate_weights]
                
          
                if seed is not None: np.random.seed(seed) # Ensure numpy's RNG is also seeded
                
                chosen_indices = np.random.choice(
                    len(candidate_tuples), 
                    size=num_to_actually_sample, 
                    replace=False, 
                    p=probabilities
                )
                for idx in chosen_indices:
                    sampled_tuples_set.add(candidate_tuples[idx])
        
        if len(sampled_tuples_set) < num_samples and self.verbose:
            print(f"Warning: KernelSHAP sampling resulted in {len(sampled_tuples_set)} unique samples, "
                  f"less than requested {num_samples}. This might happen if num_samples is close to 2^n.")

        return list(sampled_tuples_set)
    
    def compute_exhaustive_top_k(self, k: int):
        n = self.n_items
        best_k_indices_to_remove = None
        min_utility_after_removal = float('inf') # We want to minimize V(N - S_removed)

        possible_indices_to_remove = list(itertools.combinations(range(n), k))
        
        pbar_desc = f"Exhaustive Top-{k} Search"
        pbar_iter = tqdm(possible_indices_to_remove, desc=pbar_desc, disable=not self.verbose)

        for k_indices_tuple in pbar_iter:
            ablated_set_np = np.ones(n, dtype=int)
            ablated_set_np[list(k_indices_tuple)] = 0
            ablated_set_tuple = tuple(ablated_set_np)

            utility_of_ablated_set = self.all_true_utilities.get(ablated_set_tuple, -float('inf'))

            if utility_of_ablated_set < min_utility_after_removal:
                min_utility_after_removal = utility_of_ablated_set
                best_k_indices_to_remove = k_indices_tuple

        return best_k_indices_to_remove  

    def _train_surrogate(self, ablations: list[tuple], utilities: list[float], eval=True, sur_type="linear") -> tuple:
        X_train = np.array(ablations)
        y_train = np.array(utilities)

        # X_test=np.array(list(self.all_true_utilities.keys()))
        # y_test=np.array(list(self.all_true_utilities.values()))

        # Train the model
        if sur_type == "linear":

            model = Lasso(alpha=0.01, fit_intercept=True, random_state=42, max_iter=2000)
            model.fit(X_train, y_train)
            coefficients = model.coef_
            intercept = model.intercept_
            return model, coefficients, intercept

        elif sur_type=="xgboost":                              
            model = xgboost.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            reg_alpha=0.01,
            reg_lambda=0,
            objective='reg:squarederror' # Common for regression
            )
            model.fit(X_train, y_train)
            return model

        elif sur_type == "fm":
            print(X_train.shape)
            X_train_fm = csr_matrix(X_train)  # Convert to sparse
            print(X_train_fm.shape)
            model = als.FMRegression(
                n_iter=100,
                init_stdev=0.1,
                rank=4,
                l2_reg_w=0.1,
                l2_reg_V=0.1,
                random_state=42
            )
            model.fit(X_train_fm, y_train)
            return model

        elif sur_type == "gam":
            gam_lam = 0.6
            n_features = X_train.shape[1]

            # Start with univariate smooth terms
            terms = s(0, lam=gam_lam)
            for i in range(1, n_features):
                terms += s(i, lam=gam_lam)

            # Add pairwise interaction terms
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    terms += te(i, j, lam=gam_lam)

            # Fit the model
            model = LinearGAM(terms).fit(X_train, y_train)
            return model
        
        elif sur_type == "boosted_gam":
            model = ExplainableBoostingRegressor(
                max_bins=256,
                max_interaction_bins=64,
                interactions=5,
                learning_rate=0.01,
                max_leaves=3,
                max_rounds=100,
                early_stopping_rounds=5,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model
        else:
            raise ValueError("Unknown type")




        # if eval:
        #     # Predict on the same training data for evaluation
        #     y_pred = model.predict(X_test)

        #     # Overall MSE
        #     overall_mse = mean_squared_error(y_test, y_pred)
        #     if hasattr(self, 'verbose') and self.verbose: # Check if self.verbose exists
        #         print(f"Overall Mean Squared Error (MSE): {overall_mse:.6f}")

        #     # MSE by subset size
        #     mse_by_size = defaultdict(list)
        #     predictions_by_size = defaultdict(list)
        #     true_values_by_size = defaultdict(list)

        #     for i, ablation_vector in enumerate(X_test):
        #         subset_size = int(np.sum(ablation_vector))
        #         true_values_by_size[subset_size].append(y_test[i])
        #         predictions_by_size[subset_size].append(y_pred[i])

        #     if hasattr(self, 'verbose') and self.verbose:
        #         print("\nMSE by Subset Size (Number of Active Items):")
        #         print("------------------------------------------")
        #         print("| Size | Num Subsets | MSE        |")
        #         print("|------|-------------|------------|")

        #     sorted_sizes = sorted(true_values_by_size.keys())
        #     for size in sorted_sizes:
        #         true_vals_s = np.array(true_values_by_size[size])
        #         pred_vals_s = np.array(predictions_by_size[size])
        #         num_subsets_at_size = len(true_vals_s)

        #         if num_subsets_at_size > 0:
        #             mse_s = mean_squared_error(true_vals_s, pred_vals_s)
        #             if hasattr(self, 'verbose') and self.verbose:
        #                 print(f"| {size:<4} | {num_subsets_at_size:<11} | {mse_s:<10.6f} |")
        #         # else: # Should not happen if size is a key from X_train
        #         #     if hasattr(self, 'verbose') and self.verbose:
        #         #         print(f"| {size:<4} | {num_subsets_at_size:<11} | {'N/A':<10} |")
            
        #     if hasattr(self, 'verbose') and self.verbose:
        #         print("------------------------------------------\n")
                

    def compute_exact_shap(self):
        if self.verbose and self.accelerator.is_main_process:
        # self.all_true_utilities is available and identical on all processes
            return self._calculate_shapley_from_cached_dict(self.all_true_utilities)

    def compute_contextcite_weights(self, num_samples: int, seed: int = None,  sampling="uniform"):
        if self.accelerator.is_main_process:

            if seed is not None: random.seed(seed); np.random.seed(seed) # Seed for all processes for consistent sampling

            sampled_tuples = self._sample_ablations_from_all(num_samples) if sampling=="uniform" else self._sample_ablations_kernelshap_weighted(num_samples, seed=seed)

            utilities_for_samples = [self.all_true_utilities.get(v_tuple, -float('inf')) for v_tuple in sampled_tuples]

            valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
            if not valid_indices:
                if self.verbose and self.accelerator.is_main_process: print("Error: All sampled precomputed utilities failed. Cannot train surrogate for ContextCite.");
                return np.zeros(self.n_items)

            if len(valid_indices) < len(sampled_tuples) and self.verbose and self.accelerator.is_main_process:
                print(f"Warning: {len(sampled_tuples) - len(valid_indices)} precomputed utilities were -inf, using {len(valid_indices)} for ContextCite surrogate.")

            sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
            utilities_for_train = [utilities_for_samples[i] for i in valid_indices]
            _, weights, _ = self._train_surrogate(sampled_tuples_for_train, utilities_for_train)
        return weights

    def compute_wss(self, num_samples: int, seed: int = None, distil: list=None,  sampling="kernelshap",sur_type="xgboost", util="pure-surrogate", pairchecking=False):
        if self.accelerator.is_main_process:
            if seed is not None: random.seed(seed); np.random.seed(seed)

            sampled_tuples = self._sample_ablations_from_all(num_samples) if sampling=="uniform" else self._sample_ablations_kernelshap_weighted(num_samples, seed=seed)

            utilities_for_samples = [self.all_true_utilities.get(v_tuple, -float('inf')) for v_tuple in sampled_tuples]
            
            valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]

            if len(valid_indices) < len(sampled_tuples) and self.verbose and self.accelerator.is_main_process:
                print(f"Warning: {len(sampled_tuples) - len(valid_indices)} precomputed utilities were -inf, using {len(valid_indices)} for WSS surrogate.")
                
            sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
            utilities_for_train = [utilities_for_samples[i] for i in valid_indices]
            known_utilities_for_hybrid = {v_tuple: self.all_true_utilities[v_tuple] for v_tuple in sampled_tuples_for_train}
            if sur_type=="linear":
                surrogate_model, _, _ = self._train_surrogate(sampled_tuples_for_train, utilities_for_train)
            else:
                surrogate_model= self._train_surrogate(sampled_tuples_for_train, utilities_for_train, sur_type=sur_type)

            hybrid_utilities = {}
            all_ablations_X = np.array(list(itertools.product([0, 1], repeat=self.n_items)), dtype=int)
            
            # Prediction can be done by all, it's fast.
            if sur_type=="fm":
                all_ablations_X_sp=csr_matrix(all_ablations_X)
                all_predictions = surrogate_model.predict(all_ablations_X_sp)
            else:
                all_predictions = surrogate_model.predict(all_ablations_X)
            
            if util=='hybrid':
                for i_pred in range(all_ablations_X.shape[0]):
                    v_tuple = tuple(all_ablations_X[i_pred])
                    hybrid_utilities[v_tuple] = known_utilities_for_hybrid.get(v_tuple, all_predictions[i_pred])
            elif util=="pure-surrogate":
                for i_pred in range(all_ablations_X.shape[0]):
                    v_tuple = tuple(all_ablations_X[i_pred])
                    hybrid_utilities[v_tuple] = all_predictions[i_pred]
                if pairchecking:
                    shapley_values_wss = self._calculate_shapley_from_cached_dict(hybrid_utilities, known_utilities_for_hybrid)
                    return shapley_values_wss
            if distil:
                {k: v for k, v in hybrid_utilities.items() if sum(k) not in distil}

            shapley_values_wss = self._calculate_shapley_from_cached_dict(hybrid_utilities) 
        
            return shapley_values_wss

    def compute_tmc_shap(self, num_iterations_max: int, performance_tolerance: float, 
                          max_unique_lookups: int, # Budget for utility lookups
                          seed: int = None): # Removed return_actual_lookups
        
        if seed is not None: random.seed(seed); np.random.seed(seed)

        shapley_values = np.zeros(self.n_items)
        marginal_counts = np.zeros(self.n_items, dtype=int)
        unique_utility_lookups_tracker = set() # Still useful for internal logic and verbose print

        v_empty_tuple = tuple(np.zeros(self.n_items, dtype=int))
        v_full_tuple = tuple(np.ones(self.n_items, dtype=int))
        
        v_empty_util = self.all_true_utilities.get(v_empty_tuple, -float('inf'))
        v_full_util = self.all_true_utilities.get(v_full_tuple, -float('inf'))
        
        if v_empty_tuple in self.all_true_utilities: unique_utility_lookups_tracker.add(v_empty_tuple)
        if v_full_tuple in self.all_true_utilities: unique_utility_lookups_tracker.add(v_full_tuple)

        effective_tolerance = performance_tolerance
        if v_empty_util == -float('inf') or v_full_util == -float('inf'):
             if self.verbose: print("  Warning: Truncation disabled due to endpoint utility failure.")
             effective_tolerance = float('inf')

        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations_max), desc="TMC Iterations (budgeted)", disable=not self.verbose, leave=False)
        
        actual_iterations_run = 0
        for t in pbar:
            actual_iterations_run += 1
            if len(unique_utility_lookups_tracker) >= max_unique_lookups:
                break 

            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int) 

            for item_idx_to_add in permutation:
                if len(unique_utility_lookups_tracker) >= max_unique_lookups:
                    break 

                v_curr_np = current_subset_np.copy() 
                v_curr_np[item_idx_to_add] = 1
                v_curr_tuple = tuple(v_curr_np)
                
                can_truncate = False
                if v_prev_util != -float('inf') and v_full_util != -float('inf'):
                    is_near_full = abs(v_full_util - v_prev_util) < effective_tolerance
                    can_truncate = t > 0 and is_near_full
                
                if can_truncate: 
                    v_curr_util = v_prev_util 
                else:
                    if v_curr_tuple not in unique_utility_lookups_tracker and len(unique_utility_lookups_tracker) < max_unique_lookups:
                        if v_curr_tuple in self.all_true_utilities: # Check if key exists before adding
                             unique_utility_lookups_tracker.add(v_curr_tuple)
                    elif v_curr_tuple in self.all_true_utilities and v_curr_tuple not in unique_utility_lookups_tracker:

                        pass

                    v_curr_util = self.all_true_utilities.get(v_curr_tuple, -float('inf'))
                    
                marginal_contribution = 0.0
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'):
                    marginal_contribution = v_curr_util - v_prev_util
                
                k_count = marginal_counts[item_idx_to_add] + 1
                shapley_values[item_idx_to_add] = ( (k_count - 1) / k_count ) * shapley_values[item_idx_to_add] + \
                                                  ( 1 / k_count ) * marginal_contribution
                marginal_counts[item_idx_to_add] = k_count
                
                v_prev_util = v_curr_util 
                current_subset_np = v_curr_np
        
        pbar.close()
        final_lookups_count = len(unique_utility_lookups_tracker)

        return shapley_values 

    def compute_beta_shap(self, num_iterations_max: int, beta_a: float, beta_b: float, 
                           max_unique_lookups: int, # Budget for utility lookups
                           seed: int = None): # Removed return_actual_lookups
        if beta_dist is None: raise ImportError("BetaShap requires scipy.")
        if seed is not None: random.seed(seed); np.random.seed(seed)

        weighted_marginal_sums = np.zeros(self.n_items)
        total_weights_for_item = np.zeros(self.n_items)
        unique_utility_lookups_tracker = set()

        v_empty_tuple = tuple(np.zeros(self.n_items, dtype=int))
        v_empty_util = self.all_true_utilities.get(v_empty_tuple, -float('inf'))
        if v_empty_tuple in self.all_true_utilities: unique_utility_lookups_tracker.add(v_empty_tuple)
        
        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations_max), desc="BetaShap Iterations (budgeted)", disable=not self.verbose, leave=False)
        
        actual_iterations_run = 0
        for _ in pbar:
            actual_iterations_run +=1
            if len(unique_utility_lookups_tracker) >= max_unique_lookups:
                break

            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int)

            for k_minus_1, item_idx_to_add in enumerate(permutation):
                if len(unique_utility_lookups_tracker) >= max_unique_lookups:
                    break

                v_curr_np = current_subset_np.copy()
                v_curr_np[item_idx_to_add] = 1
                v_curr_tuple = tuple(v_curr_np)
                
                if v_curr_tuple not in unique_utility_lookups_tracker and len(unique_utility_lookups_tracker) < max_unique_lookups :
                    if v_curr_tuple in self.all_true_utilities: # Check if key exists before adding
                        unique_utility_lookups_tracker.add(v_curr_tuple)
                
                v_curr_util = self.all_true_utilities.get(v_curr_tuple, -float('inf'))
                
                marginal_contribution = 0.0
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'):
                     marginal_contribution = v_curr_util - v_prev_util
                
                # ... (Beta weight calculation) ...
                if self.n_items > 1: x_pos = k_minus_1 / (self.n_items - 1)
                else: x_pos = 0.5 
                try:
                    if self.n_items == 1 and (beta_a < 1 or beta_b < 1): weight = 1.0
                    elif self.n_items > 0: weight = beta_dist.pdf(x_pos, beta_a, beta_b)
                    else: weight = 1.0
                    if not np.isfinite(weight): weight = 1.0 if beta_a == 1 and beta_b == 1 else 1e6
                except Exception: weight = 1.0
                
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'):
                    weighted_marginal_sums[item_idx_to_add] += weight * marginal_contribution
                    total_weights_for_item[item_idx_to_add] += weight
                
                v_prev_util = v_curr_util
                current_subset_np = v_curr_np
                
        pbar.close()
        shapley_values = np.zeros(self.n_items)
        non_zero_weights_mask = total_weights_for_item > 1e-9
        shapley_values[non_zero_weights_mask] = weighted_marginal_sums[non_zero_weights_mask] / total_weights_for_item[non_zero_weights_mask]
        
        final_lookups_count = len(unique_utility_lookups_tracker)

        return shapley_values

    def compute_loo(self):
        if self.accelerator.is_main_process:
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