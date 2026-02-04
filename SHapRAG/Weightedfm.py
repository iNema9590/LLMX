from math import comb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

class DeltaOptimizedFM(nn.Module):
    """
    Factorization Machine optimized for delta (marginal contribution) prediction.
    Predicts utilities, but optimizes R² of deltas.
    """
    def __init__(self, n_features, latent_dim=10):
        super(DeltaOptimizedFM, self).__init__()
        self.n_features = n_features
        
        # Linear terms (for Shapley value components)
        self.w0 = nn.Parameter(torch.zeros(1))  # global bias
        self.w = nn.Parameter(torch.zeros(n_features))  # linear weights
        
        # Factorization terms (for interactions)
        self.v = nn.Parameter(torch.randn(n_features, latent_dim) * 0.01)
        
    def forward(self, x):
        """
        x: [batch_size, n_features] binary coalition vector
        Returns: predicted utility v(S)
        """
        # Linear terms
        linear_terms = torch.sum(self.w * x, dim=1) + self.w0
        
        # Interaction terms
        vx = torch.matmul(x, self.v)  # [batch_size, latent_dim]
        
        # Sum of squares of row-wise dot products
        square_of_sum = torch.sum(vx * vx, dim=1)  # (Σ v_i*x_i)²
        sum_of_squares = torch.sum((x.unsqueeze(-1) * self.v.unsqueeze(0))**2, dim=(1, 2))  # Σ (v_i*x_i)²
        
        interaction_terms = 0.5 * (square_of_sum - sum_of_squares)
        
        return linear_terms + interaction_terms
    
    def predict(self, S_with_i):
        """
        Predict marginal contribution: v(S∪{i}) - v(S)
        """
        with torch.no_grad():
            util_with = self.forward(torch.tensor(S_with_i, dtype=torch.float32))
            return util_with.detach().numpy()
    
    def compute_shapley_from_weights(self):
        """
        Extract Shapley values from model weights.
        For FM, linear weights approximate Shapley values when interaction terms are small.
        """
        with torch.no_grad():
            w = self.w.cpu().numpy()
            V = self.v.cpu().numpy()
            
        F = V @ V.T
        
        # Set diagonal to 0 (self-interactions not included in Shapley)
        np.fill_diagonal(F, 0.0)
        shapley = w + 0.5 * F.sum(axis=1)
        return shapley, F

class DeltaLossFM:
    """
    Training wrapper that optimizes R² of deltas
    """
    def __init__(self, n_features, latent_dim=10, lr=0.001):
        self.model = DeltaOptimizedFM(n_features, latent_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_type = 'pairwise'  # or 'pairwise'
        self.n_features = n_features
        self.l2_lambda = 0.01
        
    def r2_delta_loss(self, pred_deltas, true_deltas):
        """
        Compute negative R² (to minimize) for deltas
        R² = 1 - SS_res / SS_tot
        We minimize -R² to maximize R²
        """
        # Convert to numpy for R² calculation
        pred_np = pred_deltas.detach().numpy()
        true_np = true_deltas.numpy()
        
        # Compute R²
        ss_res = np.sum((true_np - pred_np) ** 2)
        ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)
        
        if ss_tot == 0:
            return torch.tensor(0.0)
        
        r2 = 1 - ss_res / ss_tot
        return torch.tensor(-r2)  # Negative because we want to maximize R²
    
    def pairwise_logistic_loss(self, pred_deltas, true_deltas):
        """
        Calculate the pairwise logistic loss for ranking deltas.
        loss = sum_i sum_j I[y_true_i > y_true_j] * log(1 + exp(-(y_pred_i - y_pred_j)))
        
        Args:
            pred_deltas: Predicted deltas tensor [batch_size]
            true_deltas: True deltas tensor [batch_size]
        """
        n = pred_deltas.shape[0]
        
        # Create masks for pairs where true_deltas[i] > true_deltas[j]
        true_expanded_i = true_deltas.unsqueeze(1)  # [n, 1]
        true_expanded_j = true_deltas.unsqueeze(0)  # [1, n]
        mask = (true_expanded_i > true_expanded_j).float()  # [n, n]
        
        # Compute pairwise differences for predictions
        pred_expanded_i = pred_deltas.unsqueeze(1)  # [n, 1]
        pred_expanded_j = pred_deltas.unsqueeze(0)  # [1, n]
        pred_diff = pred_expanded_i - pred_expanded_j  # [n, n]
        
        # Compute logistic loss for all pairs
        logistic_loss = torch.log(1 + torch.exp(-pred_diff))  # [n, n]
        
        # Apply mask and sum
        loss = torch.sum(mask * logistic_loss)
        
        # Normalize by number of valid pairs (optional but recommended)
        num_valid_pairs = torch.sum(mask)
        if num_valid_pairs > 0:
            loss = loss / num_valid_pairs
            
        return loss    
    
    def compute_loss(self, pred_deltas, true_deltas):
        """
        Compute loss based on selected loss type
        """
        if self.loss_type == 'r2':
            return self.r2_delta_loss(pred_deltas, true_deltas)
        elif self.loss_type == 'pairwise':
            return self.pairwise_logistic_loss(pred_deltas, true_deltas)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Use 'r2' or 'pairwise'.")
    
    def train_step(self, coalitions, utilities, pair_indices):
        """
        Train on paired coalitions to optimize delta R²
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(coalitions))
        y_true = torch.FloatTensor(utilities)
        
        # Predict all utilities 
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
        
        # # Compute loss (negative R² of deltas)
        # main_loss = self.compute_loss(pred_deltas, true_deltas)
        
        # Alternative: weighted with true_deltas MSE of deltas (more stable gradient)
        weights = 1 + true_deltas.abs()/ true_deltas.abs().quantile(0.7)  # or true_deltas, depending on your intent

        mse = weights * (pred_deltas - true_deltas) ** 2
        weighted_mse = mse.sum() / len(mse)
        mseut = (y_pred - y_true) ** 2
        weighted_mseut = mseut.sum() / len(mseut)
        
        l2_reg = torch.tensor(0.)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        combined_loss = weighted_mse + weighted_mseut + self.l2_lambda * l2_reg
        combined_loss.backward()
        self.optimizer.step()
        
        return combined_loss.item()
    
    def train(self, X_train, utilities, n_epochs=100, pair_indices=None):
        """
        Complete training procedure with paired sampling
        """
        for epoch in range(n_epochs):
                        
            # Training step
            metrics = self.train_step(X_train, utilities, pair_indices)
       
   
    def get_shapley_values(self):
        """
        Get Shapley values from trained model weights
        """
        return self.model.compute_shapley_from_weights()
