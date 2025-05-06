# Shapley Attributor for LLM Context/Feature Importance

This repository provides a Python class, `ShapleyAttributor`, designed to calculate attribution scores for input items (e.g., retrieved documents in RAG, features) based on their contribution to a Language Model's (LLM) output probability. It implements several methods derived from Shapley values, offering different trade-offs between computational cost (number of LLM calls) and accuracy/fidelity.

The core idea is to define a **utility function**, typically the log-probability of generating a specific `target_response` given a `query` and a subset of the input `items`. The attribution methods then determine how much each individual item contributes to this utility on average.

This library implements the following methods:

### 1. Exact Shapley (`exact`)

*   **Core Idea:** Directly calculates the Shapley value using its definition.
*   **Process:** Computes the utility `V(S)` for *all* `2^n` possible subsets `S` of the `n` input items by calling the LLM. It then uses the formula:
    `φᵢ = Σ_{S ⊆ N\{i}} [ |S|! * (n - |S| - 1)! / n! ] * [ V(S ∪ {i}) - V(S) ]`
    where `N` is the set of all items.
*   **Pros:** Provides the theoretically exact, fair attribution according to Shapley axioms based on the defined utility function. Serves as the ground truth.
*   **Cons:** Computationally infeasible for `n > ~15-20` due to the exponential number of LLM calls (`2^n`).

### 2. Truncated Monte Carlo Shapley (`tmc`)

*   **Core Idea:** Estimates Shapley values by averaging marginal contributions over random permutations of items, adding an efficiency heuristic. Based on Data Shapley (Ghorbani & Zou, 2019).
*   **Process:**
    1.  Repeatedly samples random orderings (permutations) of the `n` items (`T` times).
    2.  For each permutation, iterates through the items, adding them one by one to the current subset `S`.
    3.  Calculates the marginal contribution `V(S ∪ {i}) - V(S)` by calling the LLM for `V(S ∪ {i})`.
    4.  **Truncation:** If the utility `V(S)` is already very close to the utility of the full set `V(N)`, it assumes the marginal contribution of subsequent items in this permutation is negligible (zero) and skips the LLM call.
    5.  Averages the calculated marginal contributions for each item `i` across all permutations using a running average.
*   **Pros:** Much faster than Exact Shapley. Often provides good approximations.
*   **Cons:** Still requires many LLM calls (`<= T*n`, often `T ~ k*n`). Truncation is a heuristic and might underestimate the contribution (especially negative) of items appearing late in permutations. Results are stochastic.

### 3. Beta Shapley (`betashap`)

*   **Core Idea:** Refines Monte Carlo estimation by weighting marginal contributions based on *when* an item appears in the permutation, using the Beta distribution. Based on Beta Shapley (Kwon & Zou, 2021). Requires `scipy`.
*   **Process:**
    1.  Similar to TMC, samples `T` random permutations.
    2.  Calculates the marginal contribution `Δᵢ = V(S ∪ {i}) - V(S)` for item `i` added at position `k` (i.e., `|S|=k`).
    3.  Calculates a weight `w` based on the Beta distribution's probability density function `BetaPDF(k/(n-1); α, β)`.
    4.  Computes the final Shapley value as the *weighted* average of `Δᵢ` across all permutations, using `w` as the weight.
*   **Parameters:** `α` (beta_a) and `β` (beta_b) control the weighting shape.
    *   `α=1, β=1`: Uniform weighting (equivalent to standard Monte Carlo).
    *   `α<1, β<1`: U-shaped, emphasizes start and end contributions (often reduces variance).
*   **Pros:** More principled weighting than TMC truncation. Can achieve faster convergence (lower variance) than standard MC/TMC with fewer iterations `T` by tuning `α, β`. Unified framework.
*   **Cons:** Requires `scipy`. Still stochastic and requires `~T*n` LLM calls. Choosing optimal `α, β` might require experimentation. Beta PDF calculation needs careful handling near 0 and 1.

### 4. Weakly Supervised Shapley (`wss`)

*   **Core Idea:** Drastically reduces LLM calls by training a fast surrogate model on a small sample of true utilities, then uses this surrogate to approximate the full utility landscape for Shapley calculation. Inspired by ContextCite (Cohen-Wang et al., 2024) and Datamodels (Ilyas et al., 2022).
*   **Process:**
    1.  Randomly samples `m` (e.g., 32) distinct subsets.
    2.  Computes the *true* utility `V(S)` using the LLM for only these `m` subsets.
    3.  Trains a simple (e.g., LASSO or Linear Regression) surrogate model `g(v)` to predict `V(S)` based on the binary subset vector `v` using the `m` samples.
    4.  Constructs a "hybrid" utility dictionary for all `2^n` subsets: uses the true `V(S)` for the `m` sampled subsets and the predicted utility `g(v)` for the remaining `2^n - m` subsets.
    5.  Calculates the *exact* Shapley formula using this *hybrid* utility dictionary.
*   **Pros:** Extremely fast in terms of LLM calls (only `m` calls needed). Feasible even when Exact/TMC/Beta are too slow.
*   **Cons:** The final Shapley values are *approximations* whose accuracy depends heavily on how well the simple linear surrogate can capture the true (potentially non-linear) utility function based on only `m` samples.

### 5. ContextCite (Weights) (`contextcite`)

*   **Core Idea:** Uses the same surrogate modeling approach as WSS but returns the learned weights of the linear surrogate *directly* as attribution scores.
*   **Process:**
    1.  Samples `m` subsets.
    2.  Computes true utility `V(S)` for these `m` subsets.
    3.  Trains a linear surrogate model `g(v) = w⋅v + b`.
    4.  Returns the weight vector `w` as the attribution scores. `wᵢ` represents the model's linear approximation of item `i`'s contribution.
*   **Pros:** Same speed as WSS (`m` LLM calls). Very simple interpretation (the learned linear importance).
*   **Cons:** Accuracy depends entirely on the validity of the linear approximation. Doesn't satisfy all Shapley axioms formally like the other methods attempt to (even approximately).

### 6. Leave-One-Out (`loo`)

*   **Core Idea:** Attributes importance based on the drop in utility when a single item is removed.
*   **Process:** Calculates `V(N) - V(N - {i})` for each item `i`, where `N` is the set of all items. Requires `n+1` LLM calls.
*   **Pros:** Very simple and fast to compute. Intuitive.
*   **Cons:** Fails to capture interactions between items (synergy, redundancy). Often a poor approximation of Shapley values.

## How `ShapleyAttributor` Works

This class provides a unified interface to these methods.

**Initialization:**

```python
attributor = ShapleyAttributor(
    items=list_of_document_strings,
    query: "User query string",
    target_response: "The specific response string to explain",
    llm_caller=your_log_probability_function 
)
```

* items: A list of strings, where each string is an item (document, feature description) to be evaluated.

* query: The query provided to the LLM along with the context.

* target_response: The specific output sequence generated by the LLM (often using the full set of items) whose log-probability serves as the utility function.

llm_caller: A Python function you provide. It must accept (query, target_response, context) and return a single float representing log P(target_response | context, query). This function encapsulates the interaction with your chosen LLM.

### Computing Attributions:

Use the compute() method:
```python
scores = attributor.compute(
    method_name="wss",      # Choose the method ('exact', 'tmc', 'betashap', 'wss', 'contextcite', 'loo')
    num_samples=32,       # Kwarg for 'wss' or 'contextcite'
    lasso_alpha=0.01,     # Kwarg for 'wss' or 'contextcite' (0 for LinReg)
    num_iterations=100,   # Kwarg for 'tmc' or 'betashap'
    beta_a=0.5,           # Kwarg for 'betashap'
    beta_b=0.5,           # Kwarg for 'betashap'
    performance_tolerance=0.001, # Kwarg for 'tmc' truncation
    seed=42,              # Optional seed for stochastic methods
    verbose=True          # Optional flag for progress prints
    # return_weights=True # Optional for 'wss' to get surrogate weights too
    # exact_confirm=False # Optional for 'exact' to skip confirmation prompt
)
```

* method_name: String specifying the method.

* verbose: Controls progress bars and status messages.

* seed: For reproducibility of methods involving random sampling ('tmc', 'betashap', 'wss', 'contextcite').

**kwargs: Pass method-specific parameters as needed (see method descriptions and code docstrings).

The method returns a NumPy array of attribution scores, one for each input item. For wss with return_weights=True, it returns a tuple (shapley_values, surrogate_weights). Returns None if the computation fails or is aborted (e.g., exact Shapley confirmation).

Important Note on Caching: This deployable version of the class does not maintain a persistent cache of LLM utilities between different calls to compute(). Each call performs the required LLM evaluations independently, ensuring predictable behavior in deployment.

## Method Complexity Comparison

The primary computational cost lies in the number of calls to the potentially expensive llm_caller function.

| Method Name             | Approx. Number of LLM Calls       | Key Characteristic / Assumption                        |
|-------------------------|-----------------------------------|--------------------------------------------------------|
| Exact Shapley (`exact`) | 2^n                               | Theoretical Ground Truth                               |
| Leave-One-Out (`loo`)   | n + 1                             | Ignores interactions                                   |
| TMC-Shapley (`tmc`)     | ≤ T * n (often T ≈ k * n)         | Monte Carlo estimation with heuristic truncation       |
| Beta Shapley (`betashap`)| T * n (often T ≈ k * n)          | Monte Carlo estimation with Beta PDF weighting         |
| WSS (`wss`)             | m (e.g., 32, 64)                  | Surrogate model + Hybrid utilities                     |
| ContextCite (`contextcite`)| m (e.g., 32, 64)               | Uses surrogate model weights directly                  |


* n: Number of input items.

* T: Number of iterations (permutations) for TMC/BetaShap. Often needs to be O(n) or O(n*log n) for convergence.

* m: Number of samples used to train the surrogate model in WSS/ContextCite. Typically small relative to 2^n.

Trade-offs: There is a clear trade-off between computational cost (LLM calls) and the fidelity of the attribution scores to the theoretical Shapley values. WSS and ContextCite are significantly faster but rely on the quality of the surrogate model approximation. TMC and BetaShap offer better approximations than LOO but require more calls than WSS/ContextCite. Exact Shapley is the gold standard but typically infeasible.

