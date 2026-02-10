from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.
    Assumes market has hidden states (regimes) that generate observed features.
    """
    def __init__(self, n_regimes=3, random_state=42):
        """
        Args:
            n_regimes: Number of hidden states (regimes)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        print(f"[INFO] HMM initialized with {n_regimes} regimes")
    
    def detect(self, df: DataFrame) -> DataFrame:
        """
        Detect regimes using HMM.
        Returns:
            DataFrame with 'regime' column added
        """
        print("[HMM] Starting regime detection...")
        
        # Feature columns for HMM
        feature_cols = [
            'log_return',
            'vol_60m',
            'vol_240m',
            'cum_return_60m',
            'return_skew_60m',
            'volume_spike',
            'bb_width',
            'atr_pct'
        ]
        
        # Check which features exist
        available_features = [c for c in feature_cols if c in df.columns]
        
        if len(available_features) < 3:
            raise ValueError(f"Need at least 3 features, only found {len(available_features)}")
        
        print(f"[HMM] Using {len(available_features)} features")
        
        # Convert to Pandas (HMM requires sequential data in memory)
        print("[HMM] Converting to Pandas for HMM training...")
        df_pandas = df.select(['timestamp'] + available_features).toPandas()
        df_pandas = df_pandas.sort_values('timestamp').reset_index(drop=True)
        
        # Prepare feature matrix
        X = df_pandas[available_features].values
        
        # Handle any remaining NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        print("[HMM] Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train HMM
        print(f"[HMM] Training Gaussian HMM with {self.n_regimes} states...")
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
            verbose=False
        )
        
        try:
            self.model.fit(X_scaled)
            print("[HMM] Training complete")
        except Exception as e:
            print(f"[HMM] Training failed: {e}")
            # Fallback to diagonal covariance
            print("[HMM] Retrying with diagonal covariance...")
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                n_iter=100,
                random_state=self.random_state,
                verbose=False
            )
            self.model.fit(X_scaled)
        
        # Predict regimes
        print("[HMM] Predicting regimes...")
        regimes = self.model.predict(X_scaled)
        
        # Map regimes based on volatility (0=low, 1=medium, 2=high)
        regimes = self._map_regimes_by_volatility(regimes, df_pandas['vol_60m'].values)
        
        # Add to pandas DataFrame
        df_pandas['regime'] = regimes
        
        # Add regime name
        regime_names = {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        df_pandas['regime_name'] = df_pandas['regime'].map(regime_names)
        
        # Convert back to Spark DataFrame
        print("[HMM] Converting back to Spark DataFrame...")
        spark = df.sql_ctx.sparkSession
        df_result = spark.createDataFrame(df_pandas)
        
        # Join with original DataFrame to get all columns
        df_result = df.join(
            df_result.select('timestamp', 'regime', 'regime_name'),
            on='timestamp',
            how='inner'
        )
        
        # Cache to avoid lazy evaluation issues
        df_result = df_result.cache()
        df_result.count()  # Force materialization
        
        print("[HMM] Detection complete")
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
        
        # Create mapping: old_regime -> new_regime
        regime_map = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Apply mapping
        mapped_regimes = np.array([regime_map[r] for r in regimes])
        
        return mapped_regimes
    
    def get_transition_matrix(self):
        """Get regime transition probabilities."""
        if self.model is None:
            return None
        return self.model.transmat_
    
    def get_regime_characteristics(self, df_pandas, regime_col='regime'):
        """
        Compute summary statistics for each regime.
        
        Args:
            df_pandas: Pandas DataFrame with regime labels
            regime_col: Name of regime column
            
        Returns:
            DataFrame with regime characteristics
        """
        if regime_col not in df_pandas.columns:
            return None
        
        # Features to summarize
        summary_features = ['log_return', 'vol_60m', 'cum_return_60m', 
                           'volume_spike', 'atr_pct']
        available = [f for f in summary_features if f in df_pandas.columns]
        
        # Group by regime
        regime_stats = df_pandas.groupby(regime_col)[available].agg(['mean', 'std', 'min', 'max'])
        
        return regime_stats