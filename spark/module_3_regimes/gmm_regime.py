from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMRegimeDetector:
    """
    Gaussian Mixture Model for regime detection.
    Assumes data comes from mixture of Gaussian distributions (softer clustering).
    More flexible than K-Means, allows overlapping regimes.
    """
    def __init__(self, n_regimes=3, random_state=42):
        """
        Args:
            n_regimes: Number of mixture components (regimes)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        print(f"[INFO] GMM initialized with {n_regimes} regimes")
    
    def detect(self, df: DataFrame) -> DataFrame:
        """
        Detect regimes using Gaussian Mixture Model.
        Returns:
            DataFrame with 'regime' and 'regime_probability' columns
        """
        print("[GMM] Starting regime detection...")
        
        # Feature columns
        feature_cols = [
            'log_return',
            'vol_60m',
            'vol_240m',
            'vol_ratio_60_240',
            'cum_return_60m',
            'return_skew_60m',
            'volume_spike',
            'rsi',
            'bb_width',
            'atr_pct'
        ]
        
        # Check which features exist
        available_features = [c for c in feature_cols if c in df.columns]
        
        if len(available_features) < 3:
            raise ValueError(f"Need at least 3 features, only found {len(available_features)}")
        
        print(f"[GMM] Using {len(available_features)} features")
        
        # Convert to Pandas
        print("[GMM] Converting to Pandas for GMM training...")
        df_pandas = df.select(['timestamp'] + available_features).toPandas()
        df_pandas = df_pandas.sort_values('timestamp').reset_index(drop=True)
        
        # Prepare feature matrix
        X = df_pandas[available_features].values
        
        # Handle NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        print("[GMM] Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train GMM
        print(f"[GMM] Training Gaussian Mixture Model with {self.n_regimes} components...")
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            max_iter=100,
            n_init=10,
            random_state=self.random_state
        )
        
        try:
            self.model.fit(X_scaled)
            print("[GMM] Training complete")
        except Exception as e:
            print(f"[GMM] Training failed with full covariance: {e}")
            print("[GMM] Retrying with diagonal covariance...")
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='diag',
                max_iter=100,
                n_init=10,
                random_state=self.random_state
            )
            self.model.fit(X_scaled)
        
        # Predict regimes (hard assignment)
        print("[GMM] Predicting regimes...")
        regimes = self.model.predict(X_scaled)
        
        # Get probabilities (soft assignment)
        regime_probs = self.model.predict_proba(X_scaled)
        max_probs = regime_probs.max(axis=1)
        
        # Map regimes based on volatility
        regimes = self._map_regimes_by_volatility(regimes, df_pandas['vol_60m'].values)
        
        # Add to pandas DataFrame
        df_pandas['regime'] = regimes
        df_pandas['regime_probability'] = max_probs
        
        # Add regime name
        regime_names = {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        df_pandas['regime_name'] = df_pandas['regime'].map(regime_names)
        
        # Calculate regime uncertainty (entropy of probability distribution)
        regime_probs_sorted = np.sort(regime_probs, axis=1)[:, ::-1]
        df_pandas['regime_uncertainty'] = 1 - (regime_probs_sorted[:, 0] - regime_probs_sorted[:, 1])
        
        # Convert back to Spark DataFrame
        print("[GMM] Converting back to Spark DataFrame...")
        spark = df.sql_ctx.sparkSession
        df_result = spark.createDataFrame(df_pandas)
        
        # Join with original DataFrame
        df_result = df.join(
            df_result.select('timestamp', 'regime', 'regime_name', 
                           'regime_probability', 'regime_uncertainty'),
            on='timestamp',
            how='inner'
        )
        
        # Cache to avoid lazy evaluation issues
        df_result = df_result.cache()
        df_result.count()  # Force materialization
        
        print("[GMM] Detection complete")
        return df_result
    
    def _map_regimes_by_volatility(self, regimes, volatility):
        """
        Map regime labels to match volatility levels.
        Ensures 0=low vol, 1=medium vol, 2=high vol.
        """
        # Calculate mean volatility for each regime
        regime_vols = {}
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() > 0:
                regime_vols[r] = volatility[mask].mean()
            else:
                regime_vols[r] = 0
        
        # Sort regimes by volatility
        sorted_regimes = sorted(regime_vols.items(), key=lambda x: x[1])
        
        # Create mapping
        regime_map = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Apply mapping
        mapped_regimes = np.array([regime_map[r] for r in regimes])
        
        return mapped_regimes
    
    def get_bic(self):
        """Get Bayesian Information Criterion (lower is better)."""
        if self.model is None:
            return None
        return self.model.bic(self.scaler.transform(X))
    
    def get_aic(self):
        """Get Akaike Information Criterion (lower is better)."""
        if self.model is None:
            return None
        return self.model.aic(self.scaler.transform(X))
    
    def get_component_means(self):
        """Get mean vectors for each mixture component."""
        if self.model is None:
            return None
        return self.model.means_
    
    def get_component_weights(self):
        """Get mixture weights (prior probabilities of each regime)."""
        if self.model is None:
            return None
        return self.model.weights_