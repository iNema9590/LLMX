import numpy as np
from tqdm import tqdm

class WeightedFM:
    def __init__(self, rank=8, n_epochs=100, learning_rate=0.01, reg=0.01, verbose=True):
        self.rank = rank
        self.n_epochs = n_epochs
        self.lr = learning_rate
        self.reg = reg
        self.verbose = verbose
        
    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(0, 0.01, (n_features, self.rank))
        
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
            
        # Normalize weights
        sample_weights = sample_weights / np.mean(sample_weights)
        
        # Training loop with progress bar
        pbar = tqdm(range(self.n_epochs), desc="Training FM", disable=not self.verbose)
        for epoch in pbar:
            total_loss = 0
            for i in range(n_samples):
                # Prediction
                linear = np.dot(X[i], self.w)
                
                # Interaction term
                interactions = 0
                for f in range(self.rank):
                    dot_product = np.dot(X[i], self.V[:, f])
                    squared_sum = np.dot(X[i]**2, self.V[:, f]**2)
                    interactions += dot_product**2 - squared_sum
                interactions *= 0.5
                
                prediction = self.w0 + linear + interactions
                
                # Error
                error = y[i] - prediction
                total_loss += sample_weights[i] * error**2
                
                # Update parameters with weight
                weight = sample_weights[i]
                
                # Update bias
                self.w0 += self.lr * weight * error
                
                # Update linear weights
                self.w += self.lr * weight * error * X[i] - self.lr * self.reg * self.w
                
                # Update factors
                for f in range(self.rank):
                    # Compute gradient for factors
                    grad_V = np.zeros(n_features)
                    for j in range(n_features):
                        if X[i, j] != 0:
                            grad_V[j] = error * (X[i, j] * np.dot(X[i], self.V[:, f]) - 
                                               self.V[j, f] * X[i, j]**2)
                    
                    # Update with weight and regularization
                    self.V[:, f] += self.lr * weight * grad_V - self.lr * self.reg * self.V[:, f]
            
            if self.verbose:
                pbar.set_postfix({"Loss": f"{total_loss/n_samples:.6f}"})
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Linear term
            linear = np.dot(X[i], self.w)
            
            # Interaction term
            interactions = 0
            for f in range(self.rank):
                dot_product = np.dot(X[i], self.V[:, f])
                squared_sum = np.dot(X[i]**2, self.V[:, f]**2)
                interactions += dot_product**2 - squared_sum
            interactions *= 0.5
            
            predictions[i] = self.w0 + linear + interactions
        
        return predictions
    
    def get_attributions(self):
        """Compute feature attributions (Shapley values) from the FM model"""
        # Linear contributions
        linear_attr = self.w.copy()
        
        # Interaction contributions (sum of interactions with other features)
        interaction_attr = np.zeros(len(self.w))
        for i in range(len(self.w)):
            for j in range(len(self.w)):
                if i != j:
                    # Interaction between i and j is 〈v_i, v_j〉
                    interaction_strength = np.dot(self.V[i], self.V[j])
                    interaction_attr[i] += 0.5 * interaction_strength
        
        # Total attribution
        attr = linear_attr + interaction_attr
        
        # Create interaction matrix
        F = self.V @ self.V.T
        np.fill_diagonal(F, 0.0)  # Set diagonal to zero (self-interactions not included)
        
        return attr, F