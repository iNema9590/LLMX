import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix

# ---------------------------
# Torch FM model and wrapper
# ---------------------------
class _TorchFMImpl(nn.Module):
    """
    Minimal Factorization Machine implementation:
      - n_features: number of original features (players/documents)
      - rank: latent dimension for interactions
    Predicts scalar value per row of X.
    """
    def __init__(self, n_features, rank, device=None):
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.n_features = int(n_features)
        self.rank = int(rank)
        # bias and linear weights
        self.w0 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.w = nn.Parameter(torch.zeros(self.n_features, dtype=torch.float32, device=self.device))
        # latent factors: shape (n_features, rank)
        self.V = nn.Parameter(torch.randn(self.n_features, self.rank, dtype=torch.float32, device=self.device) * 0.01)

    def forward(self, X):
        # X: dense torch tensor float32 shape (batch, n_features)
        # linear
        linear = torch.matmul(X, self.w) + self.w0
        # pairwise term via factorization trick
        XV = X @ self.V  # (batch, rank)
        XV_sq = XV * XV
        X_sq = X * X
        V_sq = self.V * self.V
        pairwise = 0.5 * torch.sum(XV_sq - (X_sq @ V_sq), dim=1)
        return linear + pairwise


class TorchFMSurrogate:
    """
    Wrapper to provide a sklearn-like .predict interface and to extract attributions.
    Holds the trained torch model and helper numpy copies of parameters.
    """
    def __init__(self, torch_model: _TorchFMImpl, device=None):
        self.device = device or torch.device("cpu")
        self.model = torch_model
        # After training, we'll copy params to numpy for quick CPU prediction on csr matrices
        self._synced = False
        self._sync_to_numpy()

    def _sync_to_numpy(self):
        # Sync parameters to numpy arrays for efficient non-grad predictions (csr-based)
        with torch.no_grad():
            self.w0_ = float(self.model.w0.detach().cpu().numpy())
            self.w_ = self.model.w.detach().cpu().numpy().astype(np.float64)
            self.V_ = self.model.V.detach().cpu().numpy().astype(np.float64)  # shape (n_features, rank)
            self.n_features_, self.rank_ = self.V_.shape
            # compute F = V V^T (pairwise interactions)
            self.F_ = self.V_.dot(self.V_.T)
            np.fill_diagonal(self.F_, 0.0)
            # marginal attribution: w + 0.5 * sum_j F_ij
            self.attr_ = self.w_.copy() + 0.5 * np.sum(self.F_, axis=1)
            self._synced = True

    def predict(self, X):
        """
        Accepts:
          - scipy csr_matrix of shape (n_samples, n_features)
          - numpy ndarray (dense)
        Returns numpy array of length n_samples
        """
        if isinstance(X, csr_matrix):
            # use numpy operations for speed and memory (no torch)
            # linear term
            linear = X.dot(self.w_) + self.w0_
            # pairwise term: 0.5 * sum( (X*V)^2 - (X^2) * (V^2) )
            XV = X.dot(self.V_)              # (n_samples, rank)
            XV_sq = XV * XV
            X_sq = X.multiply(X)             # still sparse
            V_sq = self.V_ * self.V_
            second = X_sq.dot(V_sq)          # (n_samples, rank)
            pairwise = 0.5 * np.sum(XV_sq - second, axis=1)
            return np.asarray(linear).ravel() + pairwise
        else:
            # assume dense numpy
            X_np = np.asarray(X, dtype=np.float64)
            linear = X_np.dot(self.w_) + self.w0_
            XV = X_np.dot(self.V_)
            XV_sq = XV * XV
            X_sq = X_np * X_np
            second = X_sq.dot(self.V_ * self.V_)
            pairwise = 0.5 * np.sum(XV_sq - second, axis=1)
            return linear + pairwise

    def get_attr_and_F(self):
        if not self._synced:
            self._sync_to_numpy()
        return self.attr_, self.F_

# ---------------------------
# Training function (KernelSHAP weighted + Ridge)
# ---------------------------

def train_torch_fm_surrogate_shapley_als(
    ablations: list[tuple],
    utilities: list[float],
    shapley_kernel_weights: list[float] = None,
    rank= None,
    lambda_l2_w: float = 1e-3,
    lambda_l2_V: float = 1e-3,
    n_epochs: int = 50,
    V_init=None,
    device: torch.device = None,
    verbose: bool = False,
):
    """
    Train a Factorization Machine surrogate using Alternating Least Squares (ALS)
    with KernelSHAP-style weighted loss.
    """

    # ---- Data prep ----
    X_np = np.asarray(ablations, dtype=np.float64)
    y_np = np.asarray(utilities, dtype=np.float64)
    n_samples, n_features = X_np.shape

    # ---- Kernel weights ----
    if shapley_kernel_weights is None:
        kernel_w = np.ones(n_samples, dtype=np.float64)
    else:
        kernel_w = np.asarray(shapley_kernel_weights, dtype=np.float64)
        kernel_w = np.maximum(kernel_w, 0.0)
        if kernel_w.sum() == 0:
            kernel_w = np.ones_like(kernel_w)
    kernel_w = kernel_w / np.mean(kernel_w)

    # ---- Model init ----
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = _TorchFMImpl(n_features, rank, device=device).to(device)

    if V_init is not None:
        Vinit = np.asarray(V_init, dtype=np.float64)
        if Vinit.shape == (rank, n_features):
            Vinit = Vinit.T
        assert Vinit.shape == (n_features, rank)
        model.V.data = torch.tensor(Vinit, dtype=torch.float32, device=device)

    # ---- Convert to numpy ----
    X, y, w_kernel = X_np, y_np, kernel_w
    w0 = model.w0.detach().cpu().item()
    w = model.w.detach().cpu().numpy().astype(np.float64)
    V = model.V.detach().cpu().numpy().astype(np.float64)

    # ---- ALS loop ----
    for epoch in range(n_epochs):
        # Step 1: Update w (ridge regression)
        pairwise_term = 0.5 * np.sum((X @ V) ** 2 - (X ** 2) @ (V ** 2), axis=1)
        y_resid = y - pairwise_term - w0
        WX = X * w_kernel[:, None]
        A = WX.T @ X + lambda_l2_w * np.eye(n_features)
        b = WX.T @ y_resid
        w = np.linalg.solve(A, b)

        # Step 2: Update each factor column in V
        for f in range(rank):
            # Temporarily zero out column f
            V[:, f] = 0.0
            # Compute residual excluding factor f
            pred_no_f = (X @ w) + w0 + 0.5 * np.sum((X @ V) ** 2 - (X ** 2) @ (V ** 2), axis=1)
            r = y - pred_no_f  # residual signal to fit with factor f

            # Weighted regression for factor f
            # Define design matrix for factor f
            X_f = X
            WXf = X_f * w_kernel[:, None]
            A_f = WXf.T @ X_f + lambda_l2_V * np.eye(n_features)
            b_f = WXf.T @ r
            # Solve for the single vector of shape (n_features,)
            V[:, f] = np.linalg.solve(A_f, b_f)

        # Step 3: Update intercept
        preds = (X @ w) + 0.5 * np.sum((X @ V) ** 2 - (X ** 2) @ (V ** 2), axis=1)
        w0 = np.average(y - preds, weights=w_kernel)

        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            preds_all = w0 + (X @ w) + 0.5 * np.sum((X @ V) ** 2 - (X ** 2) @ (V ** 2), axis=1)
            mse_all = np.average((preds_all - y) ** 2, weights=w_kernel)
            print(f"[FM ALS] epoch {epoch+1}/{n_epochs}, weighted MSE={mse_all:.6f}")

    # ---- Copy back to Torch ----
    with torch.no_grad():
        model.w0.copy_(torch.tensor(w0, dtype=torch.float32, device=device))
        model.w.copy_(torch.tensor(w, dtype=torch.float32, device=device))
        model.V.copy_(torch.tensor(V, dtype=torch.float32, device=device))

    wrapper = TorchFMSurrogate(model, device=device)
    attr, F = wrapper.get_attr_and_F()
    return wrapper, attr, F

from sklearn.model_selection import KFold

def tune_fm_rank_cv(
    ablations,
    utilities,
    shapley_kernel_weights=None,
    ranks=(1,2,4,8),
    n_splits=5,
    n_epochs=30,
    lambda_l2_w=1e-3,
    lambda_l2_V=1e-3,
    device=None,
    verbose=False,
    warm_start=True,
    random_state=42,
):
    X_full = np.asarray(ablations, dtype=np.float64)
    y_full = np.asarray(utilities, dtype=np.float64)
    n_samples = X_full.shape[0]

    # default kernel weights
    if shapley_kernel_weights is None:
        kernel_w_full = np.ones(n_samples, dtype=np.float64)
    else:
        kernel_w_full = np.asarray(shapley_kernel_weights, dtype=np.float64)
        kernel_w_full = np.maximum(kernel_w_full, 0.0)
        if kernel_w_full.sum() == 0:
            kernel_w_full = np.ones_like(kernel_w_full)
        kernel_w_full = kernel_w_full / np.mean(kernel_w_full)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {}
    prev_V = None
    prev_rank = None

    for rank in ranks:
        fold_scores = []
        if verbose:
            print(f"\n=== Evaluating rank={rank} ===")

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full)):
            X_train = X_full[train_idx]
            y_train = y_full[train_idx]
            w_train = kernel_w_full[train_idx]

            X_val = X_full[val_idx]
            y_val = y_full[val_idx]
            w_val = kernel_w_full[val_idx]

            # Convert training ablations to list-of-tuples or dense np arrays as expected by your trainer
            # Your trainer expects ablations: list[tuple] or np.ndarray (n_samples, n_features)
            # It will convert internally, so pass numpy arrays
            V_init = None
            if warm_start and (prev_V is not None) and (prev_rank is not None):
                # If prev_rank <= rank, expand prev_V with small random cols; if prev_rank > rank, truncate
                if prev_rank <= rank:
                    # prev_V shape: (n_features, prev_rank)
                    n_features = X_train.shape[1]
                    pad_cols = rank - prev_rank
                    rnd = np.random.randn(n_features, pad_cols) * 0.01
                    V_init = np.hstack([prev_V, rnd])
                else:
                    # truncate
                    V_init = prev_V[:, :rank].copy()

            # Train on this fold
            wrapper, attr, F = train_torch_fm_surrogate_shapley_als(
                ablations=X_train,
                utilities=y_train,
                shapley_kernel_weights=w_train,
                rank=rank,
                lambda_l2_w=lambda_l2_w,
                lambda_l2_V=lambda_l2_V,
                n_epochs=n_epochs,
                V_init=V_init,
                device=device,
                verbose=False,
            )

            # Save V for warm start (use last fold's V; will be overwritten by next fold/rank)
            prev_V = wrapper.model.V.detach().cpu().numpy().astype(np.float64)
            prev_rank = rank

            # Predict on validation set (wrapper.predict accepts numpy or csr)
            # Convert val X to csr if it's sparse-like; wrapper.predict handles csr_matrix efficiently
            X_val_input = csr_matrix(X_val) if not isinstance(X_val, np.ndarray) else (csr_matrix(X_val) if (X_val.size > 0 and float(np.count_nonzero(X_val)) / X_val.size < 0.5) else X_val)
            preds = wrapper.predict(X_val_input)
            # weighted MSE
            mse = np.average((preds - y_val) ** 2, weights=w_val)
            fold_scores.append(mse)

            if verbose:
                print(f" rank={rank} fold={fold_idx+1}/{n_splits} weighted_mse={mse:.6f}")

        mean_mse = float(np.mean(fold_scores))
        std_mse = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
        results[rank] = {"mean_mse": mean_mse, "std_mse": std_mse, "folds": fold_scores}

    # choose best rank (lowest mean_mse)
    ranks_sorted = sorted(results.keys())
    means = [results[r]["mean_mse"] for r in ranks_sorted]
    best_idx = int(np.argmin(means))
    best_rank = ranks_sorted[best_idx]

    return best_rank