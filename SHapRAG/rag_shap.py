import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM 
import math
import itertools
from collections import defaultdict
from sklearn.linear_model import Lasso, LinearRegression
from scipy.stats import beta as beta_dist
from tqdm.auto import tqdm
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def compute_logprob(
    query: str,
    ground_truth_answer: None,
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    context=None,
    max_new_tokens: int = 30,
    response: bool = False
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TinyLlama Chat Prompt Format
    if context:
        prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n Given the context: {context}. answer the query:{query}\n<|assistant|>\n"
    else:
        prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{query}\n<|assistant|>\n"

    # Tokenize the prompt and answer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    log_probs = []

    if response:   
        with torch.no_grad():
            generated = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            output_text = tokenizer.decode(generated[0][prompt_ids.shape[1]:], skip_special_tokens=True)
        return output_text
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
        # prob = math.exp(total_log_prob)

        return total_log_prob

class ShapleyAttributor:
    """
    Computes document/item attribution using various Shapley-based methods.
    Each method call is independent for a 'deployable' version.
    """
    def __init__(self, items: list[str], query: str, target_response: str, llm_caller: callable):
        if not isinstance(items, list):
            raise ValueError("items must be a list of strings")
        if not callable(llm_caller):
            raise ValueError("llm_caller must be a callable function")

        self.items = items # Renamed from docs for generality
        self.query = query
        self.target_response = target_response
        self.llm_caller = llm_caller
        self.n_items = len(items)
        if self.n_items == 0:
            raise ValueError("items list cannot be empty")

        self._factorials = [math.factorial(i) for i in range(self.n_items + 1)]
        # No persistent cross-method cache in the deployable version.
        # Each method might use an internal, run-specific cache.

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        """Constructs the context string based on the binary ablation vector."""
        if len(v_np) != self.n_items:
            raise ValueError(f"Ablation vector length {len(v_np)} must match {self.n_items}")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    def _get_ablated_context_from_tuple(self, v_tuple: tuple) -> str:
        return self._get_ablated_context_from_vector(np.array(v_tuple))

    def _calculate_true_utility_for_vector(self, v_np: np.ndarray, run_cache: dict) -> float:
        """Calculates utility for a vector, using run-specific cache."""
        v_tuple = tuple(v_np)
        if v_tuple in run_cache:
            return run_cache[v_tuple]
        
        context_str = self._get_ablated_context_from_vector(v_np)
        utility = self.llm_caller(self.query, self.target_response, context=context_str)
        run_cache[v_tuple] = utility
        return utility

    def _calculate_shapley_from_utility_dict(self, utility_dict: dict[tuple, float]) -> np.ndarray:
        """Calculates exact Shapley values given a complete utility dictionary."""
        shapley_values = np.zeros(self.n_items)
        n = self.n_items
        factorials_local = self._factorials

        if len(utility_dict) != 2**n:
            print(f"Warning: Utility dict size {len(utility_dict)} != 2**{n}. Results might be partial.")

        for i in tqdm(range(n), desc="Calculating Shapley from Dict", leave=False):
            shap_i = 0.0
            for s_tuple, s_util in utility_dict.items():
                 if len(s_tuple) != n: continue
                 if s_tuple[i] == 0:
                     s_size = sum(s_tuple)
                     s_union_i_list = list(s_tuple)
                     s_union_i_list[i] = 1
                     s_union_i_tuple = tuple(s_union_i_list)
                     if s_union_i_tuple in utility_dict:
                         marginal_contribution = utility_dict[s_union_i_tuple] - s_util
                         weight = (factorials_local[s_size] * factorials_local[n - s_size - 1]) / factorials_local[n]
                         shap_i += weight * marginal_contribution
            shapley_values[i] = shap_i
        return shapley_values

    # --- Method Implementations ---

    def _compute_exact_shap(self, verbose=True, **kwargs):
        """Computes Exact Shapley values by evaluating all 2^n subsets."""
        num_subsets = 2**self.n_items
        if verbose: print(f"Starting Exact Shapley (n={self.n_items}, {num_subsets} LLM calls)...")
        
        max_n_warn = kwargs.get('max_n_for_exact_no_warn', 12)
        if self.n_items > max_n_warn:
             print(f"Warning: Exact Shapley for n={self.n_items} is very expensive!")
             if kwargs.get('exact_confirm', True): # Allow override for testing
                confirm = input("Proceed? (yes/no): ").lower()
                if confirm != 'yes': return None

        run_cache = {} # Fresh cache for this run
        all_utilities = {}
        
        all_ablations_tuples = list(itertools.product([0, 1], repeat=self.n_items))
        for v_tuple in tqdm(all_ablations_tuples, desc="Exact LLM Calls", disable=not verbose):
            utility = self._calculate_true_utility_for_vector(np.array(v_tuple), run_cache)
            all_utilities[v_tuple] = utility
            
        if len(all_utilities) != num_subsets:
            print("Error: Not all utilities computed for Exact Shapley.")
            return None
            
        return self._calculate_shapley_from_utility_dict(all_utilities)

    def _compute_contextcite_or_wss(self, method_name:str, num_samples: int, lasso_alpha: float, return_weights: bool, verbose: bool):
        """Implements ContextCite (returns weights) and Weakly Supervised Shapley."""
        if verbose: print(f"Starting {method_name} (n={self.n_items}, m={num_samples})...")
        
        run_cache = {} # Fresh cache for this run
        sampled_ablation_vectors = self._sample_ablations(num_samples) # Returns list of np.arrays
        
        if verbose: print(f"Computing true utilities for {len(sampled_ablation_vectors)} samples...")
        true_utilities_list = [
            self._calculate_true_utility_for_vector(v_np, run_cache)
            for v_np in tqdm(sampled_ablation_vectors, desc=f"{method_name} LLM Calls", disable=not verbose)
        ]
        
        # For WSS, the "known_utilities" are those computed in this run
        known_utilities_for_hybrid = run_cache.copy()

        if verbose: print("Training surrogate model...")
        X = np.array(sampled_ablation_vectors)
        y = np.array(true_utilities_list)
        
        if lasso_alpha == 0: model = LinearRegression(fit_intercept=True)
        else: model = Lasso(alpha=lasso_alpha, fit_intercept=True, random_state=42, max_iter=2000)
        
        try:
            model.fit(X, y)
        except ValueError as e: # e.g. if X is empty due to num_samples=0
            print(f"Error training surrogate: {e}")
            return (None, None) if return_weights else None

        weights = model.coef_
        intercept = model.intercept_
        if verbose:
            print(f"  Surrogate Weights (w): {np.round(weights, 4)}")
            print(f"  Surrogate Intercept (b): {intercept:.4f}")

        if method_name == "contextcite":
            # Return weights as the primary result, Shapley values are None
            return None, weights 

        # --- For Weakly Supervised Shapley (WSS) ---
        if verbose: print("Building hybrid utility set for WSS...")
        hybrid_utilities = {}
        all_ablations_X = np.array(list(itertools.product([0, 1], repeat=self.n_items)), dtype=int)
        all_predictions = model.predict(all_ablations_X)
        
        for i in range(2**self.n_items):
            v_tuple = tuple(all_ablations_X[i])
            hybrid_utilities[v_tuple] = known_utilities_for_hybrid.get(v_tuple, all_predictions[i])
        
        if verbose: print("Calculating Shapley values from hybrid utilities for WSS...")
        shapley_values_wss = self._calculate_shapley_from_utility_dict(hybrid_utilities)
        
        if return_weights:
            return shapley_values_wss, weights
        else:
            # Still return two values, but the second is None if weights not needed
            return shapley_values_wss, None


    def _sample_ablations(self, num_samples: int) -> list[np.ndarray]: # Returns list of np.arrays
        """Samples unique binary ablation vectors (np.array)."""
        max_possible = 2**self.n_items
        if num_samples > max_possible: num_samples = max_possible
        
        sampled_tuples_set = set() # Use set of tuples for uniqueness
        # Ensure empty and full sets are included
        empty_v_tuple = tuple(np.zeros(self.n_items, dtype=int))
        full_v_tuple = tuple(np.ones(self.n_items, dtype=int))

        if num_samples >= 1: sampled_tuples_set.add(empty_v_tuple)
        if num_samples >= 2 and empty_v_tuple != full_v_tuple: sampled_tuples_set.add(full_v_tuple)
        
        # pbar = tqdm(total=num_samples, desc="Sampling Ablations", leave=False)
        # pbar.update(len(sampled_tuples_set))
        attempts, max_attempts = 0, num_samples * 10 
        while len(sampled_tuples_set) < num_samples and attempts < max_attempts:
            v_tuple = tuple(np.random.randint(0, 2, size=self.n_items, dtype=int))
            sampled_tuples_set.add(v_tuple)
            # pbar.update(len(sampled_tuples_set) - pbar.n) # Update progress
            attempts += 1
        # pbar.close()
        return [np.array(v_tuple) for v_tuple in sampled_tuples_set]


    def _compute_tmc_shap(self, num_iterations: int, performance_tolerance: float, verbose: bool):
        """Implements Truncated Monte Carlo Shapley."""
        if verbose: print(f"Starting TMC-Shapley (n={self.n_items}, T={num_iterations})...")
        
        run_cache = {} # Fresh cache for this run
        shapley_values = np.zeros(self.n_items)
        marginal_counts = np.zeros(self.n_items) # For running average

        # V({}) and V(D)
        v_empty_np = np.zeros(self.n_items, dtype=int)
        v_full_np = np.ones(self.n_items, dtype=int)
        
        v_empty_util = self._calculate_true_utility_for_vector(v_empty_np, run_cache)
        v_full_util = self._calculate_true_utility_for_vector(v_full_np, run_cache)

        indices = list(range(self.n_items))

        for t in tqdm(range(num_iterations), desc="TMC Iterations", disable=not verbose):
            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int)

            for j_idx, item_idx_to_add in enumerate(permutation):
                v_curr_np = current_subset_np.copy()
                v_curr_np[item_idx_to_add] = 1
                
                # Truncation logic
                # Truncate if adding more to S is unlikely to change performance much from V(S) towards V(D)
                # Or if S is already close to D in terms of items and performance
                is_near_full_set_performance = abs(v_full_util - v_prev_util) < performance_tolerance
                
                if t > 0 and is_near_full_set_performance: # Avoid truncation on first pass, or if not close
                    v_curr_util = v_prev_util # Marginal contribution is 0
                else:
                    v_curr_util = self._calculate_true_utility_for_vector(v_curr_np, run_cache)

                marginal_contribution = v_curr_util - v_prev_util
                
                # Update Shapley value estimate (running average)
                k_count = marginal_counts[item_idx_to_add] + 1
                shapley_values[item_idx_to_add] = ( (k_count - 1) / k_count ) * shapley_values[item_idx_to_add] + \
                                                  ( 1 / k_count ) * marginal_contribution
                marginal_counts[item_idx_to_add] = k_count
                
                v_prev_util = v_curr_util
                current_subset_np = v_curr_np # Update for next step in permutation
        
        if verbose: print(f"TMC-Shapley made {len(run_cache)} unique LLM calls.")
        return shapley_values

    def _compute_loo(self, verbose: bool, **kwargs):
        """Computes Leave-One-Out (LOO) attribution scores."""
        n = self.n_items
        if verbose: print(f"Starting Leave-One-Out (LOO) computation (n={n})...")
        
        run_cache = {} # Fresh cache for this run
        loo_scores = np.zeros(n)

        # 1. Calculate V(N) - Utility of the full set
        v_all_np = np.ones(n, dtype=int)
        if verbose: print("  Calculating utility of full set V(N)...")
        util_all = self._calculate_true_utility_for_vector(v_all_np, run_cache)
        
        if util_all == -float('inf'):
            print("Warning: Utility of full set is -infinity. LOO scores might not be meaningful.")
            # Depending on desired behavior, could return zeros or NaNs
            # return np.full(n, np.nan) 

        # 2. Calculate V(N - {i}) for each item i
        if verbose: print(f"  Calculating utility V(N-i) for {n} items...")
        pbar = tqdm(range(n), desc="LOO Calls", disable=not verbose, leave=False)
        for i in pbar:
            v_loo_np = np.ones(n, dtype=int)
            v_loo_np[i] = 0 # Leave out item i
            
            util_loo = self._calculate_true_utility_for_vector(v_loo_np, run_cache)
            
            if util_loo == -float('inf'):
                 print(f"Warning: Utility of set without item {i} is -infinity.")
                 # How to define marginal contribution? If V(N) is finite, the drop is infinite.
                 # If V(N) is also -inf, the drop is NaN or 0? Let's use NaN for undefined.
                 if util_all == -float('inf'):
                      loo_scores[i] = 0.0 # Or np.nan
                 else:
                      loo_scores[i] = np.inf # Or np.nan or a large number? Let's use inf for now.
            elif util_all == -float('inf'):
                 # V(N) is -inf, V(N-i) is finite -> removing i caused infinite gain?
                 loo_scores[i] = -np.inf # Or np.nan
            else:
                 # Standard case: V(N) - V(N-i)
                 loo_scores[i] = util_all - util_loo

        if verbose: print(f"LOO computation made {len(run_cache)} unique LLM calls.")
        return loo_scores

    def _compute_beta_shap(self, num_iterations: int, beta_a: float, beta_b: float, verbose: bool):
        """Implements Beta Shapley."""
        if beta_dist is None:
            raise ImportError("BetaShap requires scipy.stats.beta. Please install scipy.")
        if verbose: print(f"Starting Beta-Shapley (n={self.n_items}, T={num_iterations}, α={beta_a}, β={beta_b})...")

        run_cache = {}
        weighted_marginal_sums = np.zeros(self.n_items)
        total_weights_for_item = np.zeros(self.n_items)

        v_empty_util = self._calculate_true_utility_for_vector(np.zeros(self.n_items, dtype=int), run_cache)
        indices = list(range(self.n_items))

        for _ in tqdm(range(num_iterations), desc="BetaShap Iterations", disable=not verbose):
            permutation = random.sample(indices, self.n_items)
            v_prev_util = v_empty_util
            current_subset_np = np.zeros(self.n_items, dtype=int)

            for k_minus_1, item_idx_to_add in enumerate(permutation): # k_minus_1 is |S_before_add|
                v_curr_np = current_subset_np.copy()
                v_curr_np[item_idx_to_add] = 1
                
                v_curr_util = self._calculate_true_utility_for_vector(v_curr_np, run_cache)
                marginal_contribution = v_curr_util - v_prev_util
                
                # Calculate Beta weight
                # x is normalized position: k / (n-1) where k is num items BEFORE adding current one
                # For k_minus_1 = 0 (first item added), x = 0. For k_minus_1 = n-1 (last item added), x = 1
                if self.n_items > 1:
                    x_pos = k_minus_1 / (self.n_items - 1)
                else: # single item, effectively k_minus_1 = 0, n-1 = 0
                    x_pos = 0.5 # Or 0, depends on convention, Beta PDF can be inf at 0/1 for a,b<1
                                # Let's use a central point if n=1
                
                try:
                    # Beta PDF can be infinite at 0 or 1 if alpha or beta < 1.
                    # Clip x_pos to avoid issues, or handle these cases.
                    # For simplicity, we rely on scipy's handling.
                    # If beta_a, beta_b = 1,1 (uniform), pdf is 1.
                    if self.n_items == 1 and (beta_a < 1 or beta_b < 1): # Special handling for n=1, single point.
                        # The idea of "position" is less meaningful. The average is just the marginal.
                        # Beta(1,1) leads to weight 1. For other Beta, might need specific interpretation.
                        # For now, let's assume Beta(1,1) if n=1 for simplicity, or handle carefully.
                        # Here, we'll let scipy compute, but be aware.
                        weight = beta_dist.pdf(x_pos, beta_a, beta_b) if (self.n_items > 1 or (beta_a >=1 and beta_b >=1)) else 1.0

                    elif self.n_items > 1: # General case
                         weight = beta_dist.pdf(x_pos, beta_a, beta_b)
                    else: # n_items == 0, should not happen
                         weight = 1.0


                    # Guard against potential inf/nan from beta_dist.pdf if a,b < 1 and x is exactly 0 or 1
                    if not np.isfinite(weight):
                        # Heuristic: If Beta(a,b) with a,b < 1, pdf is large at ends.
                        # A simple large constant might be more stable than inf.
                        # Or, if uniform (a=b=1), weight is 1.
                        if beta_a == 1 and beta_b == 1: weight = 1.0
                        else: weight = 1e6 # Large finite weight as a heuristic
                        # print(f"Warning: Beta PDF non-finite for x={x_pos}, a={beta_a}, b={beta_b}. Using weight={weight}")


                except Exception as e:
                    print(f"Error calculating Beta PDF: x={x_pos}, a={beta_a}, b={beta_b}, error: {e}. Using weight 1.")
                    weight = 1.0
                
                weighted_marginal_sums[item_idx_to_add] += weight * marginal_contribution
                total_weights_for_item[item_idx_to_add] += weight
                
                v_prev_util = v_curr_util
                current_subset_np = v_curr_np

        # Calculate final Beta Shapley values
        shapley_values = np.zeros(self.n_items)
        for i in range(self.n_items):
            if total_weights_for_item[i] > 1e-9: # Avoid division by zero
                shapley_values[i] = weighted_marginal_sums[i] / total_weights_for_item[i]
            # else: value remains 0 if it was never weighted/contributed

        if verbose: print(f"Beta-Shapley made {len(run_cache)} unique LLM calls.")
        return shapley_values


    # --- Main Public Compute Method ---
    def compute(self, method_name: str, verbose: bool = True, **kwargs):
        """
        Computes attribution scores using the specified Shapley-based method.

        Args:
            method_name (str): One of 'contextcite', 'wss', 'exact', 'tmc', 'betashap'.
            verbose (bool): If True, prints progress and details.
            **kwargs: Method-specific parameters:
                - For 'contextcite', 'wss':
                    - num_samples (int): Number of samples for surrogate. Default 32.
                    - lasso_alpha (float): Alpha for LASSO. Default 0.01.
                    - return_weights (bool): For WSS, also return surrogate weights. Default False.
                - For 'exact':
                    - max_n_for_exact_no_warn (int): Max n for no warning. Default 12.
                    - exact_confirm (bool): Whether to ask for confirmation. Default True.
                - For 'tmc':
                    - num_iterations (int): Number of MC iterations. Default n_items * 20.
                    - performance_tolerance (float): For truncation. Default 0.001.
                - For 'betashap':
                    - num_iterations (int): Number of MC iterations. Default n_items * 20.
                    - beta_a (float): Alpha for Beta distribution. Default 1.0.
                    - beta_b (float): Beta for Beta distribution. Default 1.0.

        Returns:
            np.ndarray or tuple: Attribution scores. For 'wss' with return_weights=True,
                                 returns (shapley_values, surrogate_weights).
                                 Returns None if computation fails or is aborted.
        """
        start_time = time.time()
        result = None

        if method_name.lower() == "contextcite":
            num_samples = kwargs.get('num_samples', 32)
            lasso_alpha = kwargs.get('lasso_alpha', 0.01)
            # ContextCite returns the surrogate weights directly
            # The first return value (shapley) will be None
            _, weights = self._compute_contextcite_or_wss("contextcite", num_samples, lasso_alpha, True, verbose) # Request weights
            result = weights # Assign the weights to result

        elif method_name.lower() == "wss": # Weakly Supervised Shapley
            num_samples = kwargs.get('num_samples', 32)
            lasso_alpha = kwargs.get('lasso_alpha', 0.01)
            return_weights = kwargs.get('return_weights', False)
            # This call now correctly expects two return values
            shapley_values_wss, weights_wss = self._compute_contextcite_or_wss("wss", num_samples, lasso_alpha, True, verbose) # Always get weights internally
            
            # Decide what to return based on user request
            if return_weights:
                result = (shapley_values_wss, weights_wss)
            else:
                result = shapley_values_wss

        elif method_name.lower() == "exact":
            result = self._compute_exact_shap(verbose=verbose, **kwargs)

        elif method_name.lower() == "loo":
             result = self._compute_loo(verbose=verbose, **kwargs)

        elif method_name.lower() == "tmc":
            num_iterations = kwargs.get('num_iterations', self.n_items ) # Default iterations
            performance_tolerance = kwargs.get('performance_tolerance', 0.001)
            result = self._compute_tmc_shap(num_iterations, performance_tolerance, verbose=verbose)
            
        elif method_name.lower() == "betashap":
            if beta_dist is None:
                print("Error: BetaShap method requires scipy to be installed.")
                return None
            num_iterations = kwargs.get('num_iterations', self.n_items)
            beta_a = kwargs.get('beta_a', 1.0) # Default to Uniform (like standard MC)
            beta_b = kwargs.get('beta_b', 1.0) # Default to Uniform
            result = self._compute_beta_shap(num_iterations, beta_a, beta_b, verbose=verbose)
            
        else:
            raise ValueError(f"Unknown method_name: {method_name}. "
                             f"Choose from 'contextcite', 'wss', 'exact', 'tmc', 'betashap'.")
        
        end_time = time.time()
        if verbose and result is not None:
            print(f"Method '{method_name}' finished in {end_time - start_time:.2f}s.")
        return result