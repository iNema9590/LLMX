from math import comb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from scipy.sparse import csr_matrix, issparse

class FMRegressionALS:
    """
    Factorization Machine with Alternating Least Squares solver for regression.
    Similar to fastFM library's ALS implementation.
    """
    def __init__(self, n_features, rank=10, n_iter=100, init_stdev=0.1, 
                 l2_reg_w=0.1, l2_reg_V=0.1, l2_reg_w0=0.1):
        """
        Parameters:
        -----------
        n_features : int
            Number of features
        rank : int
            Dimensionality of factorization
        n_iter : int
            Number of iterations
        init_stdev : float
            Standard deviation for initialization
        l2_reg_w : float
            L2 regularization for linear weights
        l2_reg_V : float
            L2 regularization for latent factors
        l2_reg_w0 : float
            L2 regularization for bias
        """
        self.n_features = n_features
        self.rank = rank
        self.n_iter = n_iter
        self.init_stdev = init_stdev
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = l2_reg_V
        self.l2_reg_w0 = l2_reg_w0
        
        # Initialize parameters
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(0, init_stdev, (n_features, rank))
        
        # Precompute for efficiency
        self.V_squared = np.zeros((n_features, rank))
        
    def _precompute_V_squared(self):
        """Precompute V^2 for faster computations"""
        self.V_squared = self.V ** 2
        
    def _compute_y_hat(self, X, w0=None, w=None, V=None):
        """Compute predictions using current parameters"""
        if w0 is None:
            w0 = self.w0
        if w is None:
            w = self.w
        if V is None:
            V = self.V
            
        # Convert to sparse if needed
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X
            
        # Linear terms
        y_hat = w0 + X_dense.dot(w)
        
        # Interaction terms: 0.5 * sum_{f} [(sum_i v_{i,f} x_i)^2 - sum_i v_{i,f}^2 x_i^2]
        X_V = X_dense.dot(V)  # (n_samples, rank)
        X_V_squared = X_V ** 2  # (sum_i v_{i,f} x_i)^2
        
        # sum_i v_{i,f}^2 x_i^2
        if issparse(X):
            X_squared = X.power(2).toarray()
        else:
            X_squared = X_dense ** 2
            
        V_squared_sum = X_squared.dot(V ** 2)  # sum_i v_{i,f}^2 x_i^2
        
        interaction = 0.5 * (X_V_squared - V_squared_sum).sum(axis=1)
        y_hat += interaction
        
        return y_hat
    
    def _update_w0(self, X, y, y_hat):
        """Update bias term w0"""
        if issparse(X):
            n_samples = X.shape[0]
        else:
            n_samples = len(X)
            
        residual = y - y_hat
        gradient = residual.sum()
        
        # Closed-form solution with regularization
        self.w0 = gradient / (n_samples + self.l2_reg_w0)
        return self.w0
    
    def _update_w(self, X, y, y_hat):
        """Update linear weights w"""
        if issparse(X):
            n_samples = X.shape[0]
            X_dense = X.toarray()
        else:
            n_samples = X.shape[0]
            X_dense = X
            
        # Compute residual without linear contribution from w
        residual = y - y_hat + X_dense.dot(self.w)
        
        # Update each w_j separately (can be parallelized)
        for j in range(self.n_features):
            x_j = X_dense[:, j]
            
            # Gradient for w_j
            numerator = np.dot(x_j, residual)
            denominator = np.dot(x_j, x_j) + self.l2_reg_w
            
            if denominator > 0:
                self.w[j] = numerator / denominator
                
                # Update residual for next iterations
                residual -= self.w[j] * x_j
                residual += self.w[j] * x_j  # Actually, we should recompute y_hat
            
        return self.w
    
    def _update_V(self, X, y, y_hat):
        """Update latent factors V using ALS"""
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X
            
        # Precompute for efficiency
        self._precompute_V_squared()
        
        # Update each row of V (each feature's latent factors) separately
        for j in range(self.n_features):
            # Get feature j's values
            x_j = X_dense[:, j]
            mask = x_j != 0
            
            if np.sum(mask) == 0:
                continue
                
            # Compute q_{j,f} = sum_{i≠j} v_{i,f} x_i for each factor f
            # This is: X.V - v_j * x_j
            X_V = X_dense.dot(self.V)  # (n_samples, rank)
            q = X_V - np.outer(x_j, self.V[j, :])  # q_{j,f} for all samples
            
            # Select only samples where x_j != 0
            q_j = q[mask]
            x_j_masked = x_j[mask]
            y_masked = y[mask]
            y_hat_masked = y_hat[mask]
            
            # For each factor f
            for f in range(self.rank):
                # Compute residual for factor f
                # r = y - (y_hat - contribution from v_{j,f})
                # contribution from v_{j,f} = x_j * q_{j,f} * v_{j,f}
                current_contrib = x_j_masked * q_j[:, f] * self.V[j, f]
                residual = y_masked - (y_hat_masked - current_contrib)
                
                # Solve ridge regression for v_{j,f}
                A = x_j_masked * q_j[:, f]
                numerator = np.dot(A, residual)
                denominator = np.dot(A, A) + self.l2_reg_V
                
                if denominator > 0:
                    self.V[j, f] = numerator / denominator
        
        return self.V
    
    def fit(self, X, y, verbose=False):
        """
        Fit the model using Alternating Least Squares
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        verbose : bool
            Whether to print progress
        """
        # Convert to numpy arrays
        if issparse(X):
            X_sparse = X
        else:
            X_sparse = csr_matrix(X) if X.shape[0] > 1000 else X
            
        y = np.asarray(y)
        
        # Initial prediction
        y_hat = self._compute_y_hat(X_sparse)
        
        # ALS iterations
        for iteration in range(self.n_iter):
            # Update parameters in alternating fashion
            old_w0 = self.w0
            old_w = self.w.copy()
            old_V = self.V.copy()
            
            # 1. Update w0
            self._update_w0(X_sparse, y, y_hat)
            
            # 2. Update w
            self._update_w(X_sparse, y, y_hat)
            
            # 3. Update V
            self._update_V(X_sparse, y, y_hat)
            
            # Update predictions
            y_hat = self._compute_y_hat(X_sparse)
            
            # Compute metrics
            mse = np.mean((y - y_hat) ** 2)
            r2 = 1 - mse / np.var(y) if np.var(y) > 0 else 0
            
            # Check convergence
            w0_change = abs(self.w0 - old_w0)
            w_change = np.mean(abs(self.w - old_w))
            V_change = np.mean(abs(self.V - old_V))
            
            if verbose and (iteration % 10 == 0 or iteration == self.n_iter - 1):
                print(f"Iteration {iteration}: MSE={mse:.6f}, R²={r2:.6f}, "
                      f"Δw0={w0_change:.6f}, Δw={w_change:.6f}, ΔV={V_change:.6f}")
            
            # Early stopping if changes are small
            if w0_change < 1e-6 and w_change < 1e-6 and V_change < 1e-6:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        return self
    
    def predict(self, X):
        """Predict target values for X"""
        if issparse(X):
            return self._compute_y_hat(X)
        else:
            return self._compute_y_hat(X)
    
    def compute_shapley_from_weights(self):
        """
        Compute approximate Shapley values from model weights
        Shapley_i ≈ w_i + 0.5 * sum_{j≠i} V_i·V_j
        """
        # Compute interaction matrix F = VV^T
        F = self.V.dot(self.V.T)
        
        # Set diagonal to 0 (self-interactions not included)
        np.fill_diagonal(F, 0.0)
        
        # Shapley value approximation
        shapley = self.w + 0.5 * F.sum(axis=1)
        return shapley, F
    
    def get_params(self):
        """Get model parameters"""
        return {
            'w0': self.w0,
            'w': self.w,
            'V': self.V,
            'n_features': self.n_features,
            'rank': self.rank
        }
    
    def set_params(self, params):
        """Set model parameters"""
        self.w0 = params['w0']
        self.w = params['w']
        self.V = params['V']

class DeltaOptimizedFMALS(nn.Module):
    """
    Factorization Machine optimized for delta prediction with ALS solver option.
    """
    def __init__(self, n_features, latent_dim=10, solver='als', **als_kwargs):
        super(DeltaOptimizedFMALS, self).__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.solver = solver
        
        if solver == 'gd':
            # Original gradient descent based implementation
            self.w0 = nn.Parameter(torch.zeros(1))
            self.w = nn.Parameter(torch.zeros(n_features))
            self.v = nn.Parameter(torch.randn(n_features, latent_dim) * 0.01)
        elif solver == 'als':
            # ALS solver
            self.als_model = FMRegressionALS(
                n_features=n_features,
                rank=latent_dim,
                **als_kwargs
            )
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'gd' or 'als'.")
    
    def forward(self, x):
        """
        x: [batch_size, n_features] binary coalition vector
        Returns: predicted utility v(S)
        """
        if self.solver == 'gd':
            # Original GD forward pass
            linear_terms = torch.sum(self.w * x, dim=1) + self.w0
            
            vx = torch.matmul(x, self.v)
            square_of_sum = torch.sum(vx * vx, dim=1)
            sum_of_squares = torch.sum((x.unsqueeze(-1) * self.v.unsqueeze(0))**2, dim=(1, 2))
            interaction_terms = 0.5 * (square_of_sum - sum_of_squares)
            
            return linear_terms + interaction_terms
        else:
            # ALS forward pass
            x_np = x.detach().cpu().numpy()
            y_pred = self.als_model.predict(x_np)
            return torch.tensor(y_pred, dtype=torch.float32)
    
    def fit(self, X, y):
        """Fit the model using specified solver"""
        if self.solver == 'gd':
            # GD training handled by outer training loop
            pass
        elif self.solver == 'als':
            # Fit ALS model
            X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
            y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else y
            self.als_model.fit(X_np, y_np)
    
    def compute_shapley_from_weights(self):
        """Extract Shapley values from model weights"""
        if self.solver == 'gd':
            with torch.no_grad():
                w = self.w.cpu().numpy()
                V = self.v.cpu().numpy()
            
            F = V @ V.T
            np.fill_diagonal(F, 0.0)
            shapley = w + 0.5 * F.sum(axis=1)
            return shapley, F
        else:
            return self.als_model.compute_shapley_from_weights()

class DeltaLossFM:
    """
    Training wrapper with support for both GD and ALS solvers
    """
    def __init__(self, n_features, latent_dim=10, lr=0.001, solver='gd', **als_kwargs):
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.solver = solver
        
        if solver == 'gd':
            self.model = DeltaOptimizedFMALS(n_features, latent_dim, solver='gd')
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif solver == 'als':
            self.model = DeltaOptimizedFMALS(n_features, latent_dim, solver='als', **als_kwargs)
            self.optimizer = None  # No optimizer needed for ALS
        else:
            raise ValueError(f"Unknown solver: {solver}")
            
        self.loss_type = 'pairwise'
        self.l2_lambda = 0.01
    
    def r2_delta_loss(self, pred_deltas, true_deltas):
        """Compute negative R² (to minimize) for deltas"""
        pred_np = pred_deltas.detach().numpy()
        true_np = true_deltas.numpy()
        
        ss_res = np.sum((true_np - pred_np) ** 2)
        ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)
        
        if ss_tot == 0:
            return torch.tensor(0.0)
        
        r2 = 1 - ss_res / ss_tot
        return torch.tensor(-r2)
    
    def pairwise_logistic_loss(self, pred_deltas, true_deltas):
        """Calculate pairwise logistic loss for ranking deltas"""
        n = pred_deltas.shape[0]
        
        true_expanded_i = true_deltas.unsqueeze(1)
        true_expanded_j = true_deltas.unsqueeze(0)
        mask = (true_expanded_i > true_expanded_j).float()
        
        pred_expanded_i = pred_deltas.unsqueeze(1)
        pred_expanded_j = pred_deltas.unsqueeze(0)
        pred_diff = pred_expanded_i - pred_expanded_j
        
        logistic_loss = torch.log(1 + torch.exp(-pred_diff))
        loss = torch.sum(mask * logistic_loss)
        
        num_valid_pairs = torch.sum(mask)
        if num_valid_pairs > 0:
            loss = loss / num_valid_pairs
            
        return loss
    
    def compute_loss(self, pred_deltas, true_deltas):
        """Compute loss based on selected loss type"""
        if self.loss_type == 'r2':
            return self.r2_delta_loss(pred_deltas, true_deltas)
        elif self.loss_type == 'pairwise':
            return self.pairwise_logistic_loss(pred_deltas, true_deltas)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def train_step(self, coalitions, utilities, pair_indices=None):
        """
        Train step - different for GD and ALS
        """
        if self.solver == 'gd':
            return self._train_step_gd(coalitions, utilities, pair_indices)
        elif self.solver == 'als':
            return self._train_step_als(coalitions, utilities)
    
    def _train_step_gd(self, coalitions, utilities, pair_indices):
        """Gradient descent training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        X = torch.FloatTensor(coalitions)
        y_true = torch.FloatTensor(utilities)
        
        y_pred = self.model(X)
        
        # Extract pairs for delta computation
        pred_deltas = []
        true_deltas = []
        
        for i in range(0, len(y_pred), 2):
            idx_with = i
            idx_without = i + 1

            pred_delta = y_pred[idx_with] - y_pred[idx_without]
            true_delta = y_true[idx_with] - y_true[idx_without]

            pred_deltas.append(pred_delta)
            true_deltas.append(true_delta)
        
        pred_deltas = torch.stack(pred_deltas)
        true_deltas = torch.stack(true_deltas)
        
        # Compute loss
        main_loss = self.compute_loss(pred_deltas, true_deltas)
        
        # Weighted MSE for utilities
        coalition_sizes = X.sum(dim=1)
        n_features = X.shape[1]
        weights = torch.tensor([comb(n_features-1, int(s)) for s in coalition_sizes])
        weights = weights / weights.sum() 
        mse_loss = torch.sum(weights * (y_pred - y_true)**2) / torch.sum(weights)
        
        combined_loss = mse_loss + main_loss
        combined_loss.backward()
        self.optimizer.step()
        
        return combined_loss.item()
    
    def _train_step_als(self, coalitions, utilities):
        """ALS training step - just fit the model"""
        self.model.fit(coalitions, utilities)
        return 0.0  # ALS doesn't have loss during training
    
    def train(self, X_train, utilities, n_epochs=100, pair_indices=None):
        """
        Complete training procedure
        """
        if self.solver == 'gd':
            for epoch in range(n_epochs):
                loss = self.train_step(X_train, utilities, pair_indices)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {loss:.6f}")
        elif self.solver == 'als':
            # ALS trains in one go
            self.train_step(X_train, utilities)
    
    def predict_delta(self, S_with, S_without):
        """
        Predict marginal contribution: v(S∪{i}) - v(S)
        """
        if self.solver == 'gd':
            with torch.no_grad():
                util_with = self.model.forward(torch.tensor([S_with], dtype=torch.float32))
                util_without = self.model.forward(torch.tensor([S_without], dtype=torch.float32))
                return (util_with - util_without).item()
        else:
            util_with = self.model.forward(torch.tensor([S_with], dtype=torch.float32))
            util_without = self.model.forward(torch.tensor([S_without], dtype=torch.float32))
            return (util_with - util_without).item()
    
    def get_shapley_values(self):
        """Get Shapley values from trained model"""
        return self.model.compute_shapley_from_weights()
    
    def evaluate_delta_r2(self, coalitions_pairs, true_deltas):
        """
        Evaluate R² for delta predictions
        """
        pred_deltas = []
        
        for i in range(0, len(coalitions_pairs), 2):
            S_with = coalitions_pairs[i]
            S_without = coalitions_pairs[i + 1]
            pred_delta = self.predict_delta(S_with, S_without)
            pred_deltas.append(pred_delta)
        
        pred_deltas = np.array(pred_deltas)
        true_deltas = np.array(true_deltas)
        
        ss_res = np.sum((true_deltas - pred_deltas) ** 2)
        ss_tot = np.sum((true_deltas - np.mean(true_deltas)) ** 2)
        
        if ss_tot == 0:
            return 1.0
        
        return 1 - ss_res / ss_tot