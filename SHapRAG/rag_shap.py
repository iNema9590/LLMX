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
import numpy as np
from scipy.special import comb
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, gather_object
from fastFM import als
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from scipy.stats import beta as beta_dist
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, ndcg_score
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

    def get_utility(self, subset_tuple: tuple, mode: str) -> float:
        """Gatekeeper for utility values. Returns from cache or computes if not present."""
        if mode in self.utility_cache.get(subset_tuple, {}):
            return self.utility_cache[subset_tuple][mode]
        
        # Compute the utility if not found in cache
        print(f"Computing utility for subset {subset_tuple} in mode '{mode}'...")
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


    def _calculate_shapley(self, mode: str = "log-perplexity") -> np.ndarray:
        explainer = shapiq.game_theory.exact.ExactComputer(
            n_players=self.n_items,
            game=self._make_value_function(mode, scale=True)
        )
        shapley_values = explainer('SV')

        return shapley_values.values[1:]


    def compute_shapley_interaction_index_pairs_matrix(self, mode: str = "log-perplexity") -> np.ndarray:
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
    

    def _get_ablated_context_from_vector(self, v_np: np.ndarray) -> str:
        if len(v_np) != self.n_items: raise ValueError("Ablation vector length mismatch")
        included_items = [self.items[i] for i, include in enumerate(v_np) if include == 1]
        return "\n\n".join(included_items)

    def _generate_sampled_ablations(self, num_samples: int, sampling_method: str = "uniform", seed: int = None):

        # Set random seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n = self.n_items
        sampled_tuples_set = set()

        # Always include empty and full set (good for surrogate stability)
        if num_samples >= 1:
            sampled_tuples_set.add(tuple([0] * n))
        if num_samples >= 2:
            sampled_tuples_set.add(tuple([1] * n))

        remaining_to_sample = num_samples - len(sampled_tuples_set)
        if remaining_to_sample <= 0:
            return list(sampled_tuples_set)

        if sampling_method == "uniform":
            while len(sampled_tuples_set) < num_samples:
                v_tuple = tuple(np.random.randint(0, 2, n))
                sampled_tuples_set.add(v_tuple)

        elif sampling_method == "kernelshap":
            # KernelSHAP distribution over coalition sizes
            sizes = np.arange(1, n)  # exclude 0 and n (already added)
            weights = (n - 1) / (sizes * (n - sizes))
            probabilities = weights / weights.sum()

            # Sample subset sizes directly from KernelSHAP distribution
            sampled_sizes = np.random.choice(sizes, size=remaining_to_sample, p=probabilities)

            for z in sampled_sizes:
                indices_to_set = np.random.choice(n, size=z, replace=False)
                v_np = np.zeros(n, dtype=int)
                v_np[indices_to_set] = 1
                sampled_tuples_set.add(tuple(v_np))

            # If duplicates caused fewer samples, fill up by retrying
            while len(sampled_tuples_set) < num_samples:
                z = np.random.choice(sizes, p=probabilities)
                indices_to_set = np.random.choice(n, size=z, replace=False)
                v_np = np.zeros(n, dtype=int)
                v_np[indices_to_set] = 1
                sampled_tuples_set.add(tuple(v_np))

        else:
            raise ValueError("Please input a valid sampling method: 'uniform' or 'kernelshap'")

        return list(sampled_tuples_set)


    def _train_surrogate(self, ablations: list[tuple], utilities: list[float], sur_type="linear",rank=None):
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
                rank=2,
                l2_reg_w=0.01,
                l2_reg_V=0.001,
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
            if rank is None:
                candidate_ranks = [1, 2, 3, 4, 5, 8]
                n_splits = 5
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                results = {}
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                for r in candidate_ranks:
                    fold_mse = []
                    for train_idx, val_idx in kf.split(X_train_fm):
                        X_tr, X_val = X_train_fm[train_idx], X_train_fm[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]

                        model = als.FMRegression(
                            n_iter=200,
                            rank=r,
                            l2_reg_w=0.01,
                            l2_reg_V=0.001,
                            random_state=42
                        )

                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_val)
                        mse = np.mean((preds - y_val) ** 2)
                        fold_mse.append(mse)

                    results[r] = np.mean(fold_mse)
                best_rank = min(results, key=results.get)
                print(f"[fastFM] Selected rank={best_rank}")

            # --- Train final model with best rank ---
            model = als.FMRegression(
                n_iter=2000,
                rank=best_rank,
                l2_reg_w=0.01,
                l2_reg_V=0.01,
                random_state=42
            )
            model.fit(X_train_fm, y_train)

            w, V = model.w_, model.V_.T
            F = V @ V.T
            np.fill_diagonal(F, 0.0)
            attr = w + 0.5 * F.sum(axis=1)

            return model, attr, F

        elif sur_type == "fmsgd":
            weights = shapley_kernel_weight(X_train)
            # ablations: list[tuple] -> X, utilities -> y
            X_train = np.array(ablations, dtype=np.int32)
            y_train = np.array(utilities, dtype=np.float32)
            best_rank=tune_fm_rank_cv(
                            X_train,
                            y_train,
                            shapley_kernel_weights=weights,
                            ranks=(1,2,4,8),
                            n_splits=5,
                            n_epochs=30,
                            lambda_l2_w=1e-3,
                            lambda_l2_V=1e-3,
                            device='cuda',
                            verbose=False,
                            warm_start=True,
                            random_state=42,
                        )
            print(f"Best rank selected: {best_rank}")
            wrapper_model, attr, F = train_torch_fm_surrogate_shapley_als(
                                                X_train,
                                                y_train,
                                                shapley_kernel_weights=weights,
                                                rank= best_rank,
                                                lambda_l2_w= 1e-3,
                                                lambda_l2_V= 1e-3,
                                                n_epochs= 200,
                                                device="cuda" ,
                                                verbose= False)
            return wrapper_model, attr, F

        
        elif sur_type == "full_poly2":
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=True)
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
        
        else:
            print("Please insert a valid surrogate type")

    def compute_contextcite(self, num_samples: int, seed: int = None, utility_mode="log-perplexity"):

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
        utilities_for_samples = [self.get_utility(v_tuple, mode=utility_mode) for v_tuple in sampled_tuples]

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

    
    def compute_wss_dynamic_pruning_reuse_utility(self, num_samples: int, seed: int = None,
                                                initial_rank: int = 1, final_rank: int = 2,
                                                pruning_strategy: str = 'top_k', # New parameter: 'elbow' or 'top_k'
                                                utility_mode="log-perplexity", sur_type="fm", sampling_method="kernelshap"):
        """
        Computes attributions using a two-stage process with a choice of pruning strategies.
        - 'elbow': Dynamically finds the drop-off point in importance scores.
        - 'top_k': Keeps the top n_items / 2 most important documents.
        """
        
        # === STAGE 1: Initial Low-Rank Model to Identify Important Documents ===
        
        # 1a. Generate initial samples and utilities (no changes here)
        sampled_tuples = self._generate_sampled_ablations(num_samples, sampling_method=sampling_method, seed=seed)
        utilities_for_samples = [self.get_utility(v_tuple, mode=utility_mode) for v_tuple in sampled_tuples]
        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u is not None and u != -float('inf')]
        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]

        # 1b. Train initial low-rank surrogate model (no changes here)
        _,initial_attr , _ = self._train_surrogate(
           sampled_tuples_for_train, 
            utilities_for_train,
            rank=initial_rank,
            sur_type="fm"
        )
        # initial_attr=np.sum(in_weight+initial_F, axis=1)
        if initial_attr is None:
            raise ValueError("Initial surrogate model failed to train. Cannot proceed with pruning.")
        
        # 1c. === DYNAMIC PRUNING LOGIC ===
        
        abs_attr = np.abs(initial_attr)
        print(f'Initial scores: {abs_attr}')
        # print(f'Initial scores no abs: {initial_attr}')
        sorted_indices = np.argsort(initial_attr)[::-1] # Sort indices from most to least important
        sorted_attr = initial_attr[sorted_indices]
        if pruning_strategy == 'elbow':
            gaps = sorted_attr[:-1] / sorted_attr[1:]  
            if len(gaps) > 0:
                elbow_point = np.argmax(gaps) + 1
            else:
                elbow_point = len(sorted_indices)
            
            docs_to_keep_indices = sorted_indices[:elbow_point]

        elif pruning_strategy == 'top_k':
            # Keep the top half of the documents
            num_to_keep = 7
            # Ensure we keep at least one document
            if num_to_keep == 0 and self.n_items > 0:
                num_to_keep = 1
                
            docs_to_keep_indices = sorted_indices[:num_to_keep]
        elif pruning_strategy== "hybrid":
            Q1, Q3 = np.percentile(sorted_attr, [25, 75])
            IQR = Q3 - Q1
            upper_cutoff = Q3 + 1.5 * IQR
            lower_cutoff = Q1 - 1.5 * IQR

            # Also ensure you always keep top 10%
            upper_quant = np.quantile(sorted_attr, 0.9)
            lower_quant = np.quantile(sorted_attr, 0.1)

            upper_final = max(upper_cutoff, upper_quant)
            lower_final = min(lower_cutoff, lower_quant)

            docs_to_keep_indices = np.where((sorted_attr >= upper_final) | (sorted_attr <= lower_final))[0]
        else:
            raise ValueError(f"Unknown pruning_strategy: '{pruning_strategy}'. Choose 'elbow' or 'top_k'.")

        # The rest of the logic remains the same...
        docs_to_prune_indices = np.setdiff1d(np.arange(self.n_items), docs_to_keep_indices)

        if len(docs_to_keep_indices) == self.n_items:
            return self.compute_wss(num_samples, seed, rank=final_rank, utility_mode=utility_mode, sur_type=sur_type)
        if len(docs_to_keep_indices) == 0:
            return np.zeros(self.n_items), np.zeros((self.n_items, self.n_items)), None
        print(f'We are keeping {len(docs_to_keep_indices)} documents')
        # === STAGE 2: High-Rank Model on Pruned Document Set with Reused Utilities ===
        
        # 2a. Create the new, smaller training set
        pruned_sampled_tuples = [tuple(np.array(t)[docs_to_keep_indices]) for t in sampled_tuples_for_train]

        # 2b. Reuse the original utilities
        reused_utilities = utilities_for_train

        # 2c. Train the final, higher-rank model
        final_model_pruned, attr_pruned, F_pruned = self._train_surrogate(
            pruned_sampled_tuples,
            reused_utilities,
            sur_type=sur_type,
            rank=final_rank if final_rank<len(docs_to_keep_indices) else len(docs_to_keep_indices)
        )
        if attr_pruned is None:
            raise ValueError("Final surrogate model failed to train.")

        # 2d. Re-integrate the results back into the original M-dimensional space
        final_attr = np.zeros(self.n_items)
        final_F = np.zeros((self.n_items, self.n_items))
        
        final_attr[docs_to_keep_indices] = attr_pruned
        
        # Reconstruct the interaction matrix
        for i_idx, i_original in enumerate(docs_to_keep_indices):
            for j_idx, j_original in enumerate(docs_to_keep_indices):
                if i_idx < F_pruned.shape[0] and j_idx < F_pruned.shape[1]:
                    final_F[i_original, j_original] = F_pruned[i_idx, j_idx]

        # Don't forget to also return the kept indices for the r2 function!
        model_info = (final_model_pruned, docs_to_keep_indices)
        return final_attr, final_F, model_info

    def compute_wss(self, num_samples: int, seed: int = None, sampling=None, sur_type="fmsgd",rank=None, utility_mode="log-perplexity"):
  
        # Generate subsets and compute utilities
        sampled_tuples = self._generate_sampled_ablations(num_samples, sampling_method=sampling, seed=seed)
        utilities_for_samples = [self.get_utility(v_tuple, mode=utility_mode) for v_tuple in sampled_tuples]

        # Filter invalid utilities
        valid_indices = [i for i, u in enumerate(utilities_for_samples) if u != -float('inf')]
        sampled_tuples_for_train = [sampled_tuples[i] for i in valid_indices]
        utilities_for_train = [utilities_for_samples[i] for i in valid_indices]

        # Train surrogate model
        model, attr, F = self._train_surrogate(
            sampled_tuples_for_train, 
            utilities_for_train, 
            sur_type=sur_type, rank=rank
        )
        return attr, F, model

    def compute_tmc_shap(self, num_iterations_max: int, performance_tolerance: float, 
                        max_unique_lookups: int, seed: int = None, 
                        shared_cache: dict = None, utility_mode="log-perplexity"):
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
                        shared_cache: dict = None, utility_mode="log-perplexity"):
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

    def compute_loo(self, utility_mode= "log-perplexity"):
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

        for i in range(self.n_items):
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

    def compute_exact_fsii(self, max_order: int, utility_mode: str = "log-perplexity", aggregate: bool = True):
  
        explainer = shapiq.game_theory.exact.ExactComputer(
            n_players=self.n_items,
            game=self._make_value_function(utility_mode)
        )
        interaction_values = explainer.compute_fii('FSII', max_order)

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


    def compute_shapiq_fsii(self, budget, utility_mode: str = "log-perplexity"):
  
        explainer = shapiq.explainer.agnostic.AgnosticExplainer(game=self._make_value_function(utility_mode), n_players=self.n_items, index='FSII', max_order=1, approximator='montecarlo', random_state=42)
        main_effects=explainer.explain_function(budget).dict_values

        explainer2 = shapiq.explainer.agnostic.AgnosticExplainer(game=self._make_value_function(utility_mode), n_players=self.n_items, index='FSII', max_order=2, approximator='montecarlo', random_state=42)
        interaction_terms=explainer2.explain_function(budget).dict_values
        interaction_values = interaction_terms|main_effects
        return np.array(list(main_effects.values())), interaction_terms, interaction_values

    def compute_fsii(self, sample_budget: int, max_order: int, utility_mode: str = "log-perplexity"):
        """Compute attribution scores using SPEX (FSII method)."""
        return self._run_spex("FSII", sample_budget, max_order, utility_mode)

    def compute_fbii(self, sample_budget: int, max_order: int, utility_mode: str = "log-perplexity"):
        """Compute attribution scores using SPEX (FBII method)."""
        return self._run_spex("FBII", sample_budget, max_order, utility_mode)
    # --------------------------------------------------------------------------
    # Helper & Internal Methods
    # --------------------------------------------------------------------------
    
    def compute_jsd_for_ablated_indices(self, ablated_indices):

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
    
    def lds(self, results_dict, n_eval_util, mode, models):
        eval_subsets = self._generate_sampled_ablations(n_eval_util, sampling_method='kernelshap', seed=2)
        X_all = np.array(eval_subsets)
        exact_utilities = [self.get_utility(v_tuple, mode=mode) for v_tuple in eval_subsets]
        X_all_sparse = csr_matrix(X_all)
        lds = {}
        # Predict effects for all subsets using surrogates
        for method_name, scores in results_dict.items():
            if "FM" in method_name:
                model_or_info = models[method_name]
                # Check if this is a pruned model (i.e., we stored a tuple)
                if isinstance(model_or_info, tuple):
                    model, kept_indices = model_or_info
                    # Slice the full evaluation data to get the pruned feature space
                    X_pruned = X_all[:, kept_indices]
                    X_pruned_sparse = csr_matrix(X_pruned)
                    predicted_effect = model.predict(X_pruned_sparse)
                else:
                    # This is a standard, non-pruned model
                    model = model_or_info
                    predicted_effect = model.predict(X_all_sparse)
            elif "II" in method_name:
                predicted_effect = np.zeros(len(X_all))
                for i, x in enumerate(X_all):
                    for loc, coef in models[method_name].items():
                        if all(x[l] == 1 for l in loc):
                            predicted_effect[i] += coef
            else:
                predicted_effect = [np.dot(scores, i) for i in X_all]

            try:
                spearman,_ = spearmanr(exact_utilities, predicted_effect)
            except Exception:
                spearman = float('nan')
            lds[method_name] = spearman
        return lds
    
    def r2(self, results_dict, n_eval_util, mode, models):
        eval_subsets = self._generate_sampled_ablations(n_eval_util, sampling_method='kernelshap', seed=2)
        X_all = np.array(eval_subsets)
        exact_utilities = [self.get_utility(v_tuple, mode=mode) for v_tuple in eval_subsets]
        # exact_utilities_perplexity = [self.get_utility(v_tuple, mode="log-perplexity") for v_tuple in eval_subsets]
        X_all_sparse = csr_matrix(X_all)
        r2_scores={}
        for method_name, scores in results_dict.items():
            if "FM" in method_name:
                # --- THIS IS THE NEW LOGIC ---
                model_or_info = models[method_name]
                # print(f"for {method_name} type is {isinstance(model_or_info, tuple)}")
                # Check if this is a pruned model (i.e., we stored a tuple)
                if isinstance(model_or_info, tuple):
                    model, kept_indices = model_or_info
                    
                    # Slice the full evaluation data to get the pruned feature space
                    X_pruned = X_all[:, kept_indices]
                    X_pruned_sparse = csr_matrix(X_pruned)
                    
                    predicted_effect = model.predict(X_pruned_sparse)
                else:
                    # This is a standard, non-pruned model
                    model = model_or_info
                    predicted_effect = model.predict(X_all_sparse)
            elif "ContextCite" in method_name:
                predicted_effect=models[method_name].predict(X_all)
            elif "II" in method_name:
                predicted_effect = np.zeros(len(X_all))
                for i, x in enumerate(X_all):
                    for loc, coef in models[method_name].items():
                        # print(loc)
                        if all(x[l] == 1 for l in loc):
                            predicted_effect[i] += coef
            else:
                predicted_effect=[np.dot(scores, i) for i in X_all]
            try:
                # if "FSI" in method_name or "FB" in method_name or "Spex" in method_name:
                #     r2[method_name]=r2_score(exact_utilities_perplexity, predicted_effect)
                # else:
                r2_scores[method_name]=r2_score(exact_utilities, predicted_effect)
            except Exception: pass
        return r2_scores

    def compute_exhaustive_top_k(self, k: int, search_list, model=None):
        n = self.n_items
        best_k_indices_to_remove = None
        min_utility_after_removal = float('inf') # We want to minimize V(N - S_removed)

        possible_indices_to_remove = list(itertools.combinations(search_list, k))
    
        for k_indices_tuple in possible_indices_to_remove:
            ablated_set_np = np.ones(n, dtype=int)
            ablated_set_np[list(k_indices_tuple)] = 0
            # ablated_set_tuple = tuple(ablated_set_np)
            if model:
                utility_of_ablated_set = model.predict(csr_matrix(ablated_set_np))
            else:
                utility_of_ablated_set = self.get_utility(tuple(ablated_set_np), mode="log-perplexity")
            if utility_of_ablated_set < min_utility_after_removal:
                min_utility_after_removal = utility_of_ablated_set
                best_k_indices_to_remove = k_indices_tuple

        return best_k_indices_to_remove



    def recall_at_k(self, gtset_k, results_dict, k_val ):
        recall={}
        for method_name, scores in results_dict.items():
            rec=[]
            for i in k_val:
                topk= np.array(scores).argsort()[-i:]
                rec.append(len(set(gtset_k).intersection(topk))/len(gtset_k))
            recall[method_name]=rec
        return recall
        
    def delta_r2(self, results_dict, num_samples=100, mode='log-perplexity', models=None):

        # Generate evaluation samples using KernelSHAP distribution
        eval_subsets = self._generate_sampled_ablations(num_samples, sampling_method='kernelshap', seed=2)
        
        # Compute actual deltas between S and S\{i}
        actual_deltas = []
        subset_player_pairs = []  # Track which (S,i) pairs we computed
        
        # For progress tracking
        total_pairs = sum(sum(s) for s in eval_subsets)  # Number of 1s across all subsets
        if self.verbose:
            print(f"Computing actual utility deltas for {total_pairs} (subset, player) pairs...")
        
        for S in eval_subsets:
            S_array = np.array(S)
            S_utility = self.get_utility(tuple(S), mode=mode)
            
            # Only consider removing present players (where S[i] = 1)
            for i in np.where(S_array == 1)[0]:
                # Create S\{i} by flipping bit i
                S_without_i = S_array.copy()
                S_without_i[i] = 0
                
                # Only compute delta if both utilities are valid
                if S_utility is not None and S_utility != -float('inf'):
                    utility_without_i = self.get_utility(tuple(S_without_i), mode=mode)
                    if utility_without_i is not None and utility_without_i != -float('inf'):
                        actual_deltas.append(S_utility - utility_without_i)
                        subset_player_pairs.append((S, i))
        
        if not actual_deltas:  # If no valid deltas found
            return {method: 0.0 for method in results_dict.keys()}
            
        actual_deltas = np.array(actual_deltas)
        delta_r2_scores = {}
        
        # For each method, compute predicted deltas and calculate R²
        for method_name, scores in results_dict.items():
            if self.verbose:
                print(f"Computing delta R² for {method_name}...")
                
            predicted_deltas = []
            
            # Handle factorization machine models
            if "FM" in method_name and models is not None:
                # Prepare all S vectors and S\{i} vectors in batch
                all_S = []
                all_S_without_i = []
                all_kept_indices = []
                
                model_or_info = models[method_name]
                if isinstance(model_or_info, tuple):
                    model, kept_indices = model_or_info
                    # For pruned models, need to track which indices to use for each prediction
                    for S, i in subset_player_pairs:
                        S_pruned = np.array(S)[kept_indices]
                        S_without_i_pruned = S_pruned.copy()
                        if i in kept_indices:
                            idx = np.where(kept_indices == i)[0][0]
                            S_without_i_pruned[idx] = 0
                        all_S.append(S_pruned)
                        all_S_without_i.append(S_without_i_pruned)
                        all_kept_indices.append(kept_indices)
                else:
                    model = model_or_info
                    # For standard models, use full feature space
                    for S, i in subset_player_pairs:
                        S_array = np.array(S)
                        S_without_i = S_array.copy()
                        S_without_i[i] = 0
                        all_S.append(S_array)
                        all_S_without_i.append(S_without_i)
                        all_kept_indices.append(None)
                
                # Batch predict for all S and S\{i}
                all_S = np.stack(all_S)
                all_S_without_i = np.stack(all_S_without_i)
                
                # Convert to sparse matrices
                S_sparse = csr_matrix(all_S)
                S_without_i_sparse = csr_matrix(all_S_without_i)
                
                # Get predictions
                pred_S = model.predict(S_sparse)
                pred_S_without_i = model.predict(S_without_i_sparse)
                predicted_deltas = pred_S - pred_S_without_i
            
            # Handle interaction-based models
            elif "II" in method_name and models is not None:
                for S, i in subset_player_pairs:
                    S_array = np.array(S)
                    S_without_i = S_array.copy()
                    S_without_i[i] = 0
                    
                    # Accumulate predictions from interaction terms
                    pred_S = 0.0
                    pred_S_without_i = 0.0
                    
                    for loc, coef in models[method_name].items():
                        # Add coefficient if all players in interaction are present
                        if all(S_array[l] == 1 for l in loc):
                            pred_S += coef
                        if all(S_without_i[l] == 1 for l in loc):
                            pred_S_without_i += coef
                            
                    predicted_deltas.append(pred_S - pred_S_without_i)
            
            # Handle linear attribution models
            else:
                # Direct prediction using attribution scores (linear model)
                for S, i in subset_player_pairs:
                    pred_S = np.dot(scores, S)
                    S_without_i = list(S)
                    S_without_i[i] = 0
                    pred_S_without_i = np.dot(scores, S_without_i)
                    predicted_deltas.append(pred_S - pred_S_without_i)
            
            predicted_deltas = np.array(predicted_deltas)
            
            # Calculate R² between actual and predicted deltas
            if len(predicted_deltas) > 0:
                delta_r2_scores[method_name] = r2_score(actual_deltas, predicted_deltas)
                if self.verbose:
                    print(f"{method_name} delta R² score: {delta_r2_scores[method_name]:.4f}")
            else:
                delta_r2_scores[method_name] = 0.0
                if self.verbose:
                    print(f"Warning: No valid predictions for {method_name}")
                
        return delta_r2_scores
    
    
    def evaluate_topk_performance(self, results_dict, models, k_values:list):
    
        n_docs = self.n_items
        evaluation_results = {}
        # Get full context utility based on type
        full_utility = self.get_utility(tuple([1] * n_docs), mode="log-perplexity")

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
                util_without_topk = self.get_utility(tuple(ablation_vector), mode="log-perplexity")
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