#!/usr/bin/env python3
"""
Nonlinear Factor Calculator
Implements ML-based features: autoencoders, nonlinear PCA
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Autoencoder features will be disabled.")

class Autoencoder(nn.Module):
    """Simple autoencoder for nonlinear factor extraction"""
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=10):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class NonlinearFactorCalculator:
    """Calculates nonlinear factors using ML techniques"""
    
    def __init__(self, linear_factors: pd.DataFrame, latent_dim: int = 10):
        """
        Initialize nonlinear factor calculator
        
        Args:
            linear_factors: DataFrame of linear factors
            latent_dim: Dimension of latent space for autoencoder
        """
        self.linear_factors = linear_factors
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()
        self.pca = None
        self.autoencoder = None
        
    def prepare_data(self) -> np.ndarray:
        """Prepare data for ML models with enhanced validation"""
        print("Preparing data for nonlinear factor extraction...")
        
        # Remove any remaining NaN values
        clean_factors = self.linear_factors.fillna(method='ffill').fillna(method='bfill')
        
        # Check for infinite values
        if np.any(np.isinf(clean_factors.values)):
            print("  Warning: Infinite values detected, replacing with NaN")
            clean_factors = clean_factors.replace([np.inf, -np.inf], np.nan)
            clean_factors = clean_factors.fillna(0)
        
        # Check for extreme values before scaling
        extreme_threshold = 10
        extreme_mask = np.abs(clean_factors.values) > extreme_threshold
        if np.any(extreme_mask):
            print(f"  Warning: {np.sum(extreme_mask)} extreme values detected, clipping to ±{extreme_threshold}")
            clean_factors = clean_factors.clip(-extreme_threshold, extreme_threshold)
        
        # Standardize the data
        scaled_data = self.scaler.fit_transform(clean_factors)
        
        # Final bounds check on scaled data
        scaled_data = np.clip(scaled_data, -3, 3)
        
        print(f"  Prepared data: {scaled_data.shape}")
        return scaled_data
    
    def extract_pca_factors(self, scaled_data: np.ndarray) -> pd.DataFrame:
        """Extract factors using PCA with enhanced bounds checking"""
        print("Extracting PCA factors...")
        
        # Clip input data to prevent extreme PCA components
        clipped_data = np.clip(scaled_data, -3, 3)
        
        # Fit PCA
        self.pca = PCA(n_components=min(self.latent_dim, clipped_data.shape[1]))
        pca_factors = self.pca.fit_transform(clipped_data)
        
        # Clip PCA factors to reasonable range
        pca_factors = np.clip(pca_factors, -5, 5)
        
        # Create DataFrame
        pca_df = pd.DataFrame(
            pca_factors,
            index=self.linear_factors.index,
            columns=[f'PCA_Factor_{i+1}' for i in range(pca_factors.shape[1])]
        )
        
        # Calculate explained variance
        explained_variance = self.pca.explained_variance_ratio_
        print(f"  PCA factors: {pca_factors.shape[1]} factors")
        print(f"  Explained variance: {explained_variance[:5].sum():.3f}")
        
        return pca_df
    
    def extract_autoencoder_factors(self, scaled_data: np.ndarray) -> pd.DataFrame:
        """Extract factors using autoencoder with enhanced bounds checking"""
        if not TORCH_AVAILABLE:
            print("  Autoencoder disabled - PyTorch not available")
            return pd.DataFrame()
        
        print("Extracting autoencoder factors...")
        
        # Clip input data to prevent extreme values
        clipped_data = np.clip(scaled_data, -3, 3)
        X = torch.FloatTensor(clipped_data)
        
        # Initialize autoencoder with output activation
        input_dim = clipped_data.shape[1]
        self.autoencoder = Autoencoder(input_dim, hidden_dim=64, latent_dim=self.latent_dim)
        
        # Training with gradient clipping
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        
        # Enhanced training loop
        self.autoencoder.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.autoencoder(X)
            loss = criterion(outputs, X)
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Extract encoded features with bounds checking
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_features = self.autoencoder.encode(X).numpy()
            
            # Clip encoded features to reasonable range
            encoded_features = np.clip(encoded_features, -5, 5)
        
        # Create DataFrame
        ae_df = pd.DataFrame(
            encoded_features,
            index=self.linear_factors.index,
            columns=[f'AE_Factor_{i+1}' for i in range(encoded_features.shape[1])]
        )
        
        print(f"  Autoencoder factors: {encoded_features.shape[1]} factors")
        return ae_df
    
    def extract_nonlinear_combinations(self, scaled_data: np.ndarray) -> pd.DataFrame:
        """Extract nonlinear combinations with aggressive bounds checking"""
        print("Extracting nonlinear combinations...")
        
        # Clip input data first to prevent explosion
        clipped_data = np.clip(scaled_data, -3, 3)
        
        combinations = {}
        
        # Square terms with bounds checking
        for i in range(min(10, clipped_data.shape[1])):
            square_values = clipped_data[:, i] ** 2
            # Clip square terms to reasonable range
            square_values = np.clip(square_values, 0, 9)  # Max 3^2 = 9
            combinations[f'Square_Factor_{i+1}'] = square_values
        
        # Cross terms with bounds checking
        count = 0
        for i in range(min(5, clipped_data.shape[1])):
            for j in range(i+1, min(6, clipped_data.shape[1])):
                if count < 10:
                    cross_values = clipped_data[:, i] * clipped_data[:, j]
                    # Clip cross terms to reasonable range
                    cross_values = np.clip(cross_values, -9, 9)  # Max ±3×3 = ±9
                    combinations[f'Cross_Factor_{i+1}_{j+1}'] = cross_values
                    count += 1
        
        # Absolute values with bounds
        for i in range(min(5, clipped_data.shape[1])):
            abs_values = np.abs(clipped_data[:, i])
            # Absolute values are already bounded by input clipping
            combinations[f'Abs_Factor_{i+1}'] = abs_values
        
        # Create DataFrame
        nl_df = pd.DataFrame(
            combinations,
            index=self.linear_factors.index
        )
        
        print(f"  Nonlinear combinations: {nl_df.shape[1]} factors")
        return nl_df
    
    def normalize_nonlinear_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize nonlinear factors with multiple bounds checking"""
        print("Normalizing nonlinear factors...")
        
        # Remove any remaining NaN values
        factors_df = factors_df.fillna(0)
        
        # First pass: clip extreme values before normalization
        factors_df = factors_df.clip(-10, 10)
        
        # Z-score normalization with robust statistics
        rolling_mean = factors_df.rolling(252, min_periods=50).mean()
        rolling_std = factors_df.rolling(252, min_periods=50).std()
        
        # Handle zero standard deviation
        rolling_std = rolling_std.replace(0, 1)
        
        normalized = (factors_df - rolling_mean) / rolling_std
        
        # Multiple clipping passes for safety
        # First clip to ±5 range
        normalized = normalized.clip(-5, 5)
        
        # Additional outlier removal using IQR method
        for col in normalized.columns:
            Q1 = normalized[col].quantile(0.25)
            Q3 = normalized[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2 * IQR  # More conservative than 1.5
            upper_bound = Q3 + 2 * IQR
            
            normalized[col] = normalized[col].clip(lower_bound, upper_bound)
        
        # Final clipping to ensure bounds
        normalized = normalized.clip(-5, 5)
        
        # Fill any remaining NaN with 0
        normalized = normalized.fillna(0)
        
        print(f"  Normalized factors: {normalized.shape[1]} factors")
        return normalized
    
    def extract_factors(self) -> pd.DataFrame:
        """Extract all nonlinear factors"""
        print("Extracting nonlinear factors...")
        
        # Prepare data
        scaled_data = self.prepare_data()
        
        # Extract different types of nonlinear factors
        pca_factors = self.extract_pca_factors(scaled_data)
        autoencoder_factors = self.extract_autoencoder_factors(scaled_data)
        nonlinear_combinations = self.extract_nonlinear_combinations(scaled_data)
        
        # Combine all nonlinear factors
        all_nonlinear = []
        if not pca_factors.empty:
            all_nonlinear.append(pca_factors)
        if not autoencoder_factors.empty:
            all_nonlinear.append(autoencoder_factors)
        if not nonlinear_combinations.empty:
            all_nonlinear.append(nonlinear_combinations)
        
        if not all_nonlinear:
            print("  No nonlinear factors extracted")
            return pd.DataFrame()
        
        # Combine and normalize
        combined_factors = pd.concat(all_nonlinear, axis=1)
        normalized_factors = self.normalize_nonlinear_factors(combined_factors)
        
        print(f"Nonlinear factor extraction completed: {normalized_factors.shape[1]} factors")
        return normalized_factors

def main():
    """Test the nonlinear factor calculator"""
    print("Nonlinear Factor Calculator - Test Mode")
    print("This module is designed to be called from the main pipeline")

if __name__ == "__main__":
    main()
