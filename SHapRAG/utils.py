# attribution_utils.py

import itertools
import numpy as np
import torch
import torch.nn.functional as F
from fastFM import als
from scipy.sparse import csr_matrix
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from tqdm.auto import tqdm

def logit(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes the logit function."""
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p / (1 - p))

def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> float:
    """Computes the Jensen-Shannon Divergence between two probability distributions."""
    p_norm = p + epsilon
    q_norm = q + epsilon
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    m = 0.5 * (p_norm + q_norm)
    
    # F.kl_div expects (input, target) where input is log-probabilities
    kl_p_m = F.kl_div(m.log(), p_norm, reduction='sum')
    kl_q_m = F.kl_div(m.log(), q_norm, reduction='sum')
    
    return 0.5 * (kl_p_m + kl_q_m).item()

def train_surrogate(
    ablations: list[tuple], 
    utilities: list[float], 
    sur_type: str = "linear", 
    alpha: float = 0.01
):
    """
    Trains a surrogate model on utility data.

    Returns:
        A tuple of (model, attributions, interactions, mse).
    """
    X_train = np.array(ablations)
    y_train = np.array(utilities)
    n = X_train.shape[1]

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
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_poly, y_train)
        y_pred = model.predict(X_poly)
        
        linear, pairs = model.coef_[:n], np.zeros((n, n))
        idx = n
        for i, j in itertools.combinations(range(n), 2):
            pairs[i, j] = pairs[j, i] = model.coef_[idx]
            idx += 1
        importance = linear + 0.5 * pairs.sum(axis=1)
        return model, importance, pairs, mean_squared_error(y_train, y_pred)

def calculate_shapley_for_surrogate(
    surrogate_model, 
    n_items: int,
    factorials: dict,
    sur_type: str = "fm", 
    verbose: bool = False
) -> np.ndarray:
    """Calculates exact Shapley values for a trained surrogate model."""
    all_subsets_np = np.array(list(itertools.product([0, 1], repeat=n_items)), dtype=int)
    
    if sur_type == "fm":
        all_subsets_sparse = csr_matrix(all_subsets_np)
        predicted_utilities = surrogate_model.predict(all_subsets_sparse)
    else:
        predicted_utilities = surrogate_model.predict(all_subsets_np)

    utility_dict = {tuple(subset): util for subset, util in zip(all_subsets_np, predicted_utilities)}
    
    shapley_values = np.zeros(n_items)
    pbar = tqdm(utility_dict.items(), desc="Surrogate Shapley", total=len(utility_dict), disable=not verbose)

    for s_tuple, s_util in pbar:
        s_size = sum(s_tuple)
        for i in range(n_items):
            if s_tuple[i] == 1:
                s_without_i_list = list(s_tuple); s_without_i_list[i] = 0
                s_without_i_tuple = tuple(s_without_i_list)
                marginal_contrib = s_util - utility_dict.get(s_without_i_tuple, 0.0)
                
                weight = (factorials[s_size - 1] * factorials[n_items - s_size]) / factorials[n_items]
                shapley_values[i] += weight * marginal_contrib
    
    return shapley_values