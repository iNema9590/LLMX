import numpy as np
import random
import math
import itertools
from collections import defaultdict
from sklearn.linear_model import Lasso, LinearRegression
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM 
from tqdm.auto import tqdm
import time
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Helper for Beta Distribution (scipy is a common dependency)
try:
    from scipy.stats import beta as beta_dist
except ImportError:
    print("Warning: scipy.stats.beta not found. BetaShap will not be available.")
    print("Please install scipy: pip install scipy")
    beta_dist = None

def compute_logprob(
    query: str,
    ground_truth_answer: str = None,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    context: str = None,
    max_new_tokens: int = 30,
    response: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Construct chat messages using Hugging Face chat template
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if context:
        messages.append({"role": "user", "content": f"Given the context: {context}. Briefly answer the query: {query}"})
    else:
        messages.append({"role": "user", "content": query})

    # Tokenize prompt using chat template
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=not ground_truth_answer,
        return_tensors="pt"
    ).to(device)

    log_probs = []

    if response:
        with torch.no_grad():
            generated = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(generated[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        cleaned_text = output_text.lstrip().removeprefix("assistant").lstrip(": \n")
        return cleaned_text
    else:
        answer_ids = tokenizer(ground_truth_answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        for i in range(answer_ids.shape[1]):
            input_ids = torch.cat([prompt_ids, answer_ids[:, :i]], dim=1)

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]

            next_token_id = answer_ids[0, i].item()
            log_prob = F.log_softmax(logits, dim=-1)[0, next_token_id].item()
            log_probs.append(log_prob)

        total_log_prob = sum(log_probs)
        return total_log_prob
# ---------------------------------------------------

class ShapleyExperimentHarness:
    """
    A harness for experimenting with different Shapley-based attribution methods.
    It pre-computes utilities for ALL 2^n subsets once, and all methods
    then use this cached data, avoiding further LLM calls.
    """
    def __init__(self, items: list[str], query: str, target_response: str, llm_caller: callable, verbose: bool = True):
        if not isinstance(items, list): raise ValueError("items must be a list of strings")
        if not callable(llm_caller): raise ValueError("llm_caller must be a callable function")

        self.items = items
        self.query = query
        self.target_response = target_response
        self.llm_caller = llm_caller
        self.n_items = len(items)
        self.verbose = verbose
        if self.n_items == 0: raise ValueError("items list cannot be empty")
        if self.n_items > 16 and self.verbose: # Feasibility warning for pre-computation
            print(f"Warning: n_items={self.n_items} means {2**self.n_items} subsets. Pre-computation will be very slow.")

        self._factorials = [math.factorial(i) for i in range(self.n_items + 1)]
        self.all_true_utilities = self._precompute_all_utilities()

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    def _precompute_all_utilities(self) -> dict[tuple, float]:
        """Computes and caches utilities for all 2^n subsets using the LLM caller."""
        num_subsets = 2**self.n_items
        if self.verbose: print(f"Pre-computing utilities for all {num_subsets} subsets (n={self.n_items})...")
        
        utilities = {}
        all_ablations_tuples = list(itertools.product([0, 1], repeat=self.n_items))
        
        llm_call_count = 0
        for v_tuple in tqdm(all_ablations_tuples, desc="Pre-computing Utilities", disable=not self.verbose):
            v_np = np.array(v_tuple)
            context_str = self._get_ablated_context_from_vector(v_np)
            try:
                # Pass query, target_response first as expected by compute_logprob_placeholder
                utility = self.llm_caller(query=self.query, ground_truth_answer=self.target_response, context=context_str)
                utilities[v_tuple] = utility
                llm_call_count +=1
            except Exception as e:
                 print(f"Error calling llm_caller for subset {v_tuple}: {e}")
                 utilities[v_tuple] = -float('inf') # Mark as failed

        if self.verbose: print(f"Pre-computation complete. Made {llm_call_count} LLM calls.")
        if len(utilities) != num_subsets:
             print(f"Error: Only {len(utilities)} of {num_subsets} utilities were pre-computed. Check LLM caller.")
        return utilities

    def _calculate_shapley_from_cached_dict(self, utility_dict_to_use: dict[tuple, float]) -> np.ndarray:
        """Calculates exact Shapley values given a complete utility dictionary (from cache)."""
        shapley_values = np.zeros(self.n_items)
        n = self.n_items
        factorials_local = self._factorials

        # No need to warn about dict size here, as it's controlled by the harness
        pbar_desc = "Calculating Shapley (from cache)"
        pbar = tqdm(range(n), desc=pbar_desc, leave=False) if self.verbose and len(utility_dict_to_use) > 5000 else range(n)

        for i in pbar:
            shap_i = 0.0
            for s_tuple, s_util in utility_dict_to_use.items():
                if len(s_tuple) != n : continue 
                if s_util == -float('inf'): continue # Skip if base utility failed

                if s_tuple[i] == 0: # Only consider subsets S where i is NOT present
                    s_size = sum(s_tuple)
                    s_union_i_list = list(s_tuple)
                    s_union_i_list[i] = 1
                    s_union_i_tuple = tuple(s_union_i_list)

                    if s_union_i_tuple in utility_dict_to_use:
                        s_union_i_util = utility_dict_to_use[s_union_i_tuple]
                        if s_union_i_util == -float('inf'): continue # Skip if paired utility failed
                        
                        marginal_contribution = s_union_i_util - s_util
                        weight = (factorials_local[s_size] * factorials_local[n - s_size - 1]) / factorials_local[n]
                        shap_i += weight * marginal_contribution
            shapley_values[i] = shap_i
        return shapley_values

    def _sample_ablations_from_all(self, num_samples: int) -> list[tuple]:
        """Samples unique binary ablation vector tuples from the precomputed set."""
        max_possible = 2**self.n_items
        if num_samples > max_possible: num_samples = max_possible
        
        all_tuples = list(self.all_true_utilities.keys())
        
        # Ensure empty and full sets are included if possible and num_samples allows
        sampled_tuples_set = set()
        empty_v_tuple = tuple(np.zeros(self.n_items, dtype=int))
        full_v_tuple = tuple(np.ones(self.n_items, dtype=int))

        if num_samples >= 1 and empty_v_tuple in self.all_true_utilities:
            sampled_tuples_set.add(empty_v_tuple)
        if num_samples >= 2 and full_v_tuple in self.all_true_utilities and empty_v_tuple != full_v_tuple:
            sampled_tuples_set.add(full_v_tuple)
        
        # Sample remaining from the rest, ensuring uniqueness
        remaining_to_sample = num_samples - len(sampled_tuples_set)
        if remaining_to_sample > 0:
            other_tuples = [t for t in all_tuples if t not in sampled_tuples_set]
            if len(other_tuples) >= remaining_to_sample:
                sampled_from_other = random.sample(other_tuples, remaining_to_sample)
                sampled_tuples_set.update(sampled_from_other)
            else: # Not enough other unique tuples, add all of them
                sampled_tuples_set.update(other_tuples)
                if self.verbose: print(f"Warning: Requested {num_samples} samples, but only {len(sampled_tuples_set)} unique tuples exist/sampled after adding empty/full.")
        
        return list(sampled_tuples_set)

    def _train_surrogate(self, ablations: list[tuple], utilities: list[float], lasso_alpha: float) -> tuple:
        X = np.array(ablations)
        y = np.array(utilities)
        if X.shape[0] == 0: raise ValueError("Cannot train surrogate with zero samples.")
        if lasso_alpha == 0: model = LinearRegression(fit_intercept=True)
        else: model = Lasso(alpha=lasso_alpha, fit_intercept=True, random_state=42, max_iter=2000)
        model.fit(X, y)
        return model, model.coef_, model.intercept_

    # --- Public Methods for Experiments ---

    def compute_exact_shap(self):
        if self.verbose: print("Computing Exact Shapley (using pre-computed utilities)...")
        return self._calculate_shapley_from_cached_dict(self.all_true_utilities)

    def compute_contextcite_weights(self, num_samples: int, lasso_alpha: float, seed: int = None):
        if self.verbose: print(f"Computing ContextCite Weights (m={num_samples}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed)
            
        sampled_tuples = self._sample_ablations_from_all(num_samples)
        if not sampled_tuples: print("Error: No tuples sampled for ContextCite."); return None
            
        utilities_for_samples = [self.all_true_utilities[v_tuple] for v_tuple in sampled_tuples]
        
        _, weights, _ = self._train_surrogate(sampled_tuples, utilities_for_samples, lasso_alpha)
        return weights

    def compute_wss(self, num_samples: int, lasso_alpha: float, seed: int = None, return_weights: bool = False):
        if self.verbose: print(f"Computing Weakly Supervised Shapley (m={num_samples}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed)

        sampled_tuples = self._sample_ablations_from_all(num_samples)
        if not sampled_tuples: print("Error: No tuples sampled for WSS."); return (None,None) if return_weights else None

        utilities_for_samples = [self.all_true_utilities[v_tuple] for v_tuple in sampled_tuples]
        known_utilities_for_hybrid = {v_tuple: self.all_true_utilities[v_tuple] for v_tuple in sampled_tuples}
        
        surrogate_model, weights, _ = self._train_surrogate(sampled_tuples, utilities_for_samples, lasso_alpha)

        hybrid_utilities = {}
        all_ablations_X = np.array(list(itertools.product([0, 1], repeat=self.n_items)), dtype=int)
        all_predictions = surrogate_model.predict(all_ablations_X)
        
        for i in range(2**self.n_items):
            v_tuple = tuple(all_ablations_X[i])
            hybrid_utilities[v_tuple] = known_utilities_for_hybrid.get(v_tuple, all_predictions[i])
        
        shapley_values_wss = self._calculate_shapley_from_cached_dict(hybrid_utilities)
        
        return (shapley_values_wss, weights) if return_weights else shapley_values_wss

    def compute_tmc_shap(self, num_iterations: int, performance_tolerance: float, seed: int = None):
        if self.verbose: print(f"Computing TMC-Shapley (T={num_iterations}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed)

        shapley_values = np.zeros(self.n_items)
        marginal_counts = np.zeros(self.n_items, dtype=int)

        v_empty_tuple = tuple(np.zeros(self.n_items, dtype=int))
        v_full_tuple = tuple(np.ones(self.n_items, dtype=int))
        v_empty_util = self.all_true_utilities.get(v_empty_tuple, -float('inf'))
        v_full_util = self.all_true_utilities.get(v_full_tuple, -float('inf'))

        if v_empty_util == -float('inf') or v_full_util == -float('inf'):
             print("Warning: Truncation disabled in TMC due to endpoint utility failure in pre-computation.")
             performance_tolerance = float('inf')

        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations), desc="TMC Iterations (from cache)", disable=not self.verbose, leave=False)
        
        for t in pbar:
            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_list = [] # Store indices

            for item_idx_to_add in permutation:
                current_subset_list.append(item_idx_to_add)
                current_subset_list.sort() # Keep tuple representation canonical if needed, or use np.array
                
                v_curr_np = np.zeros(self.n_items, dtype=int)
                v_curr_np[current_subset_list] = 1
                v_curr_tuple = tuple(v_curr_np)
                
                can_truncate = False
                if v_prev_util != -float('inf') and v_full_util != -float('inf'):
                    is_near_full_set_performance = abs(v_full_util - v_prev_util) < performance_tolerance
                    can_truncate = t > 0 and is_near_full_set_performance
                
                if can_truncate: 
                    v_curr_util = v_prev_util 
                else:
                    v_curr_util = self.all_true_utilities.get(v_curr_tuple, -float('inf'))

                if v_curr_util != -float('inf') and v_prev_util != -float('inf'): marginal_contribution = v_curr_util - v_prev_util
                else: marginal_contribution = 0 
                
                k_count = marginal_counts[item_idx_to_add] + 1
                shapley_values[item_idx_to_add] = ( (k_count - 1) / k_count ) * shapley_values[item_idx_to_add] + \
                                                  ( 1 / k_count ) * marginal_contribution
                marginal_counts[item_idx_to_add] = k_count
                
                v_prev_util = v_curr_util
        return shapley_values

    def compute_beta_shap(self, num_iterations: int, beta_a: float, beta_b: float, seed: int = None):
        if beta_dist is None: raise ImportError("BetaShap requires scipy.")
        if self.verbose: print(f"Computing Beta-Shapley (T={num_iterations}, α={beta_a}, β={beta_b}, using pre-computed utilities)...")
        if seed is not None: random.seed(seed); np.random.seed(seed)

        weighted_marginal_sums = np.zeros(self.n_items)
        total_weights_for_item = np.zeros(self.n_items)

        v_empty_tuple = tuple(np.zeros(self.n_items, dtype=int))
        v_empty_util = self.all_true_utilities.get(v_empty_tuple, -float('inf'))
        
        indices = list(range(self.n_items))
        pbar = tqdm(range(num_iterations), desc="BetaShap Iterations (from cache)", disable=not self.verbose, leave=False)

        for _ in pbar:
            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_list = []

            for k_minus_1, item_idx_to_add in enumerate(permutation):
                current_subset_list.append(item_idx_to_add)
                current_subset_list.sort() # For canonical tuple
                
                v_curr_np = np.zeros(self.n_items, dtype=int)
                v_curr_np[current_subset_list] = 1
                v_curr_tuple = tuple(v_curr_np)
                
                v_curr_util = self.all_true_utilities.get(v_curr_tuple, -float('inf'))
                
                if v_curr_util != -float('inf') and v_prev_util != -float('inf'): marginal_contribution = v_curr_util - v_prev_util
                else: marginal_contribution = 0 
                
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
                
        shapley_values = np.zeros(self.n_items)
        non_zero_weights_mask = total_weights_for_item > 1e-9
        shapley_values[non_zero_weights_mask] = weighted_marginal_sums[non_zero_weights_mask] / total_weights_for_item[non_zero_weights_mask]
        return shapley_values

    def compute_loo(self):
        if self.verbose: print(f"Computing LOO (n={self.n_items}, using pre-computed utilities)...")
        loo_scores = np.zeros(self.n_items)
        v_all_tuple = tuple(np.ones(self.n_items, dtype=int))
        util_all = self.all_true_utilities.get(v_all_tuple, -float('inf'))

        for i in range(self.n_items):
            v_loo_list = list(v_all_tuple)
            v_loo_list[i] = 0
            v_loo_tuple = tuple(v_loo_list)
            util_loo = self.all_true_utilities.get(v_loo_tuple, -float('inf'))
            
            if util_all == -float('inf') and util_loo == -float('inf'): loo_scores[i] = 0.0
            elif util_loo == -float('inf'): loo_scores[i] = np.inf
            elif util_all == -float('inf'): loo_scores[i] = -np.inf
            else: loo_scores[i] = util_all - util_loo
        return loo_scores