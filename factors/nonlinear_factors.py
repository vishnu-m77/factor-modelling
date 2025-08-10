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
        """Prepare data for ML models"""
        print("Preparing data for nonlinear factor extraction...")
        
        # Remove any remaining NaN values
        clean_factors = self.linear_factors.fillna(method='ffill').fillna(method='bfill')
        
        # Standardize the data
        scaled_data = self.scaler.fit_transform(clean_factors)
        
        print(f"  Prepared data: {scaled_data.shape}")
        return scaled_data
    
    def extract_pca_factors(self, scaled_data: np.ndarray) -> pd.DataFrame:
        """Extract factors using Principal Component Analysis"""
        print("Extracting PCA factors...")
        
        # Fit PCA
        self.pca = PCA(n_components=min(self.latent_dim, scaled_data.shape[1]))
        pca_factors = self.pca.fit_transform(scaled_data)
        
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
        """Extract factors using autoencoder"""
        if not TORCH_AVAILABLE:
            print("  Autoencoder disabled - PyTorch not available")
            return pd.DataFrame()
        
        print("Extracting autoencoder factors...")
        
        # Convert to PyTorch tensors
        X = torch.FloatTensor(scaled_data)
        
        # Initialize autoencoder
        input_dim = scaled_data.shape[1]
        self.autoencoder = Autoencoder(input_dim, hidden_dim=64, latent_dim=self.latent_dim)
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        
        # Training loop
        self.autoencoder.train()
        for epoch in range(100):  # Simple training for demonstration
            optimizer.zero_grad()
            outputs = self.autoencoder(X)
            loss = criterion(outputs, X)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"    Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Extract encoded features
        self.autoencoder.eval()
        with torch.no_grad():
            encoded_features = self.autoencoder.encode(X).numpy()
        
        # Create DataFrame
        ae_df = pd.DataFrame(
            encoded_features,
            index=self.linear_factors.index,
            columns=[f'AE_Factor_{i+1}' for i in range(encoded_features.shape[1])]
        )
        
        print(f"  Autoencoder factors: {encoded_features.shape[1]} factors")
        return ae_df
    
    def extract_nonlinear_combinations(self, scaled_data: np.ndarray) -> pd.DataFrame:
        """Extract nonlinear combinations of original factors"""
        print("Extracting nonlinear combinations...")
        
        # Create nonlinear combinations
        combinations = {}
        
        # Square terms
        for i in range(min(10, scaled_data.shape[1])):  # Limit to first 10 factors
            combinations[f'Square_Factor_{i+1}'] = scaled_data[:, i] ** 2
        
        # Cross terms (limited to avoid explosion)
        count = 0
        for i in range(min(5, scaled_data.shape[1])):
            for j in range(i+1, min(6, scaled_data.shape[1])):
                if count < 10:  # Limit cross terms
                    combinations[f'Cross_Factor_{i+1}_{j+1}'] = scaled_data[:, i] * scaled_data[:, j]
                    count += 1
        
        # Absolute values
        for i in range(min(5, scaled_data.shape[1])):
            combinations[f'Abs_Factor_{i+1}'] = np.abs(scaled_data[:, i])
        
        # Create DataFrame
        nl_df = pd.DataFrame(
            combinations,
            index=self.linear_factors.index
        )
        
        print(f"  Nonlinear combinations: {nl_df.shape[1]} factors")
        return nl_df
    
    def normalize_nonlinear_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize nonlinear factors"""
        print("Normalizing nonlinear factors...")
        
        # Remove any remaining NaN values
        factors_df = factors_df.fillna(0)
        
        # Z-score normalization
        normalized = (factors_df - factors_df.rolling(252, min_periods=50).mean()) / factors_df.rolling(252, min_periods=50).std()
        
        # Clip extreme values
        normalized = normalized.clip(-3, 3)
        
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
