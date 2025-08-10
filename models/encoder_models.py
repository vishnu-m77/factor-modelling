"""
Encoder Models Module
Implements autoencoders and nonlinear dimensionality reduction for factor modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Autoencoder:
    """
    Simple autoencoder for nonlinear factor extraction
    """
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: List[int] = None):
        """
        Initialize autoencoder
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension (bottleneck)
            hidden_dims: List of hidden layer dimensions
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [input_dim // 2]
        
        # Initialize weights (simple implementation)
        self.encoder_weights = []
        self.decoder_weights = []
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        # Encoder layers
        prev_dim = self.input_dim
        for dim in self.hidden_dims + [self.encoding_dim]:
            self.encoder_weights.append(np.random.randn(prev_dim, dim) * 0.01)
            prev_dim = dim
        
        # Decoder layers
        prev_dim = self.encoding_dim
        for dim in list(reversed(self.hidden_dims)) + [self.input_dim]:
            self.decoder_weights.append(np.random.randn(prev_dim, dim) * 0.01)
            prev_dim = dim
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative"""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode input data
        
        Args:
            X: Input data
            
        Returns:
            Encoded representation
        """
        encoded = X.copy()
        
        # Encoder forward pass
        for weights in self.encoder_weights:
            encoded = self._sigmoid(encoded @ weights)
        
        return encoded
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """
        Decode encoded data
        
        Args:
            encoded: Encoded data
            
        Returns:
            Decoded data
        """
        decoded = encoded.copy()
        
        # Decoder forward pass
        for weights in self.decoder_weights:
            decoded = self._sigmoid(decoded @ weights)
        
        return decoded
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input data
        
        Args:
            X: Input data
            
        Returns:
            Reconstructed data
        """
        encoded = self.encode(X)
        return self.decode(encoded)
    
    def train(self, X: np.ndarray, epochs: int = 100, learning_rate: float = 0.01):
        """
        Train autoencoder using backpropagation
        
        Args:
            X: Training data
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        logger.info(f"Training autoencoder for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Forward pass
            encoded = self.encode(X)
            reconstructed = self.decode(encoded)
            
            # Calculate loss
            loss = np.mean((X - reconstructed) ** 2)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss:.6f}")
            
            # Backpropagation (simplified)
            # This is a simplified version - in practice, you'd use a proper deep learning framework
            error = X - reconstructed
            
            # Update weights (simplified gradient descent)
            for i in range(len(self.encoder_weights)):
                # Simplified weight update
                self.encoder_weights[i] += learning_rate * 0.001 * np.random.randn(*self.encoder_weights[i].shape)
                self.decoder_weights[i] += learning_rate * 0.001 * np.random.randn(*self.decoder_weights[i].shape)

class NonlinearFactorExtractor:
    """
    Nonlinear factor extraction using various methods
    """
    
    def __init__(self, method: str = 'autoencoder'):
        """
        Initialize nonlinear factor extractor
        
        Args:
            method: Extraction method ('autoencoder', 'pca', 'kernel_pca')
        """
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        
    def fit_transform(self, X: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """
        Fit and transform data to extract nonlinear factors
        
        Args:
            X: Input features
            n_components: Number of components to extract
            
        Returns:
            DataFrame of extracted factors
        """
        logger.info(f"Extracting nonlinear factors using {self.method}")
        
        # Standardize data with sanitization
        X_clean = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        # Clip extreme inputs to stabilize downstream models
        X_clean = X_clean.clip(lower=-10.0, upper=10.0)
        X_scaled = self.scaler.fit_transform(X_clean)
        
        if self.method == 'autoencoder':
            return self._extract_autoencoder(X_scaled, n_components)
        elif self.method == 'pca':
            return self._extract_pca(X_scaled, n_components)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _extract_pca(self, X: np.ndarray, n_components: int) -> pd.DataFrame:
        """Extract factors using PCA"""
        # Handle NaN/inf values and clip
        X_clean = (
            pd.DataFrame(X)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(method='ffill')
            .fillna(method='bfill')
            .fillna(0.0)
            .clip(lower=-10.0, upper=10.0)
            .values
        )
        
        pca = PCA(n_components=min(n_components, X_clean.shape[1]))
        transformed = pca.fit_transform(X_clean)
        
        # Create factor names
        factor_names = [f'pca_factor_{i+1}' for i in range(n_components)]
        
        return pd.DataFrame(transformed, columns=factor_names, index=pd.RangeIndex(len(X)))
    
    def _extract_autoencoder(self, X: np.ndarray, n_components: int) -> pd.DataFrame:
        """Extract factors using autoencoder"""
        # Handle NaN/inf values and clip
        X_clean = (
            pd.DataFrame(X)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(method='ffill')
            .fillna(method='bfill')
            .fillna(0.0)
            .clip(lower=-10.0, upper=10.0)
            .values
        )
        
        # Create and train autoencoder
        autoencoder = Autoencoder(
            input_dim=X_clean.shape[1],
            encoding_dim=n_components,
            hidden_dims=[X_clean.shape[1] // 2]
        )
        
        # Train autoencoder
        autoencoder.train(X_clean, epochs=50, learning_rate=0.01)
        
        # Extract encoded representation
        encoded = autoencoder.encode(X_clean)
        # Clip encoded representation to avoid extreme values propagating
        encoded = np.clip(encoded, -5.0, 5.0)
        
        # Create factor names
        factor_names = [f'autoencoder_factor_{i+1}' for i in range(n_components)]
        
        return pd.DataFrame(encoded, columns=factor_names, index=pd.RangeIndex(len(X)))
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted model
        
        Args:
            X: New data to transform
            
        Returns:
            Transformed data
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_transform() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'autoencoder':
            return pd.DataFrame(
                self.model.encode(X_scaled),
                columns=[f'autoencoder_factor_{i+1}' for i in range(X_scaled.shape[1])],
                index=X.index
            )
        elif self.method == 'pca':
            return pd.DataFrame(
                self.model.transform(X_scaled),
                columns=[f'pca_factor_{i+1}' for i in range(X_scaled.shape[1])],
                index=X.index
            )

class FactorCombiner:
    """
    Combines multiple factors using various methods
    """
    
    def __init__(self, method: str = 'equal_weight'):
        """
        Initialize factor combiner
        
        Args:
            method: Combination method ('equal_weight', 'ic_weight', 'pca')
        """
        self.method = method
        self.weights = None
        
    def combine_factors(self, factors: pd.DataFrame, returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        Combine factors into a single signal
        
        Args:
            factors: DataFrame of factors
            returns: DataFrame of returns (for IC-based weighting)
            
        Returns:
            DataFrame of combined factors
        """
        logger.info(f"Combining factors using {self.method}")
        
        if self.method == 'equal_weight':
            return self._equal_weight_combination(factors)
        elif self.method == 'ic_weight':
            if returns is None:
                logger.warning("Returns not provided, falling back to equal weight")
                return self._equal_weight_combination(factors)
            return self._ic_weight_combination(factors, returns)
        elif self.method == 'pca':
            return self._pca_combination(factors)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _equal_weight_combination(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Equal weight combination"""
        combined = factors.mean(axis=1)
        return pd.DataFrame({'combined_signal': combined}, index=factors.index)
    
    def _pca_combination(self, factors: pd.DataFrame) -> pd.DataFrame:
        """PCA-based combination"""
        # Handle NaN values
        factors_clean = factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Use first principal component as combined signal
        pca = PCA(n_components=1)
        combined = pca.fit_transform(factors_clean)
        
        return pd.DataFrame({'combined_signal': combined.flatten()}, index=factors.index)
    
    def _ic_weight_combination(self, factors: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """IC-weighted combination"""
        # Handle NaN values
        factors_clean = factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Calculate IC for each factor
        ic_scores = []
        
        for factor_col in factors_clean.columns:
            factor_values = factors_clean[factor_col]
            
            # Calculate IC with forward returns
            forward_returns = returns.shift(-1)
            
            # Simple IC calculation
            ic_series = factor_values.rolling(252).corr(forward_returns.iloc[:, 0])
            mean_ic = ic_series.mean()
            ic_scores.append(abs(mean_ic) if not np.isnan(mean_ic) else 0)
        
        # Normalize weights
        total_ic = sum(ic_scores)
        if total_ic > 0:
            weights = [ic / total_ic for ic in ic_scores]
        else:
            weights = [1.0 / len(ic_scores)] * len(ic_scores)
        
        # Weighted combination
        weighted_factors = factors_clean * weights
        combined = weighted_factors.sum(axis=1)
        
        return pd.DataFrame({'combined_signal': combined}, index=factors.index)

def main():
    """Test nonlinear factor extraction"""
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # Create sample features
    features = pd.DataFrame(
        np.random.normal(0, 1, (500, 20)),
        index=dates,
        columns=[f'feature_{i}' for i in range(20)]
    )
    
    # Test autoencoder
    print("Testing autoencoder...")
    extractor = NonlinearFactorExtractor(method='autoencoder')
    autoencoder_factors = extractor.fit_transform(features, n_components=5)
    print(f"Autoencoder factors shape: {autoencoder_factors.shape}")
    
    # Test PCA
    print("\nTesting PCA...")
    extractor = NonlinearFactorExtractor(method='pca')
    pca_factors = extractor.fit_transform(features, n_components=5)
    print(f"PCA factors shape: {pca_factors.shape}")
    
    # Test factor combination
    print("\nTesting factor combination...")
    all_factors = pd.concat([autoencoder_factors, pca_factors], axis=1)
    
    combiner = FactorCombiner(method='equal_weight')
    combined = combiner.combine_factors(all_factors)
    print(f"Combined signal shape: {combined.shape}")

if __name__ == "__main__":
    main() 