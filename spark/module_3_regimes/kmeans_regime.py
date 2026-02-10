from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class KMeansRegimeDetector:
    """
    K-Means clustering for regime detection.
    Groups similar market conditions into regimes based on feature similarity.
    """
    def __init__(self, n_regimes=3, random_state=42):
        """
        Args:
            n_regimes: Number of clusters (regimes)
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        print(f"[INFO] K-Means initialized with {n_regimes} regimes")
    
    def detect(self, df: DataFrame) -> DataFrame:
        """
        Detect regimes using K-Means clustering.=
        Returns:
            DataFrame with 'regime' column added
        """
        print("[K-MEANS] Starting regime detection...")
        
        # Feature columns for clustering
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
        
        print(f"[K-MEANS] Using {len(available_features)} features")
        
        # Convert to Pandas
        print("[K-MEANS] Converting to Pandas for clustering...")
        df_pandas = df.select(['timestamp'] + available_features).toPandas()
        df_pandas = df_pandas.sort_values('timestamp').reset_index(drop=True)
        
        # Prepare feature matrix
        X = df_pandas[available_features].values
        
        # Handle any remaining NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize features
        print("[K-MEANS] Standardizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Train K-Means
        print(f"[K-MEANS] Training K-Means with {self.n_regimes} clusters...")
        self.model = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        regimes = self.model.fit_predict(X_scaled)
        
        print("[K-MEANS] Clustering complete")
        
        # Map regimes based on volatility (0=low, 1=medium, 2=high)
        regimes = self._map_regimes_by_volatility(regimes, df_pandas['vol_60m'].values)
        
        # Add to pandas DataFrame
        df_pandas['regime'] = regimes
        
        # Add regime name
        regime_names = {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        df_pandas['regime_name'] = df_pandas['regime'].map(regime_names)
        
        # Convert back to Spark DataFrame
        print("[K-MEANS] Converting back to Spark DataFrame...")
        spark = df.sql_ctx.sparkSession
        df_result = spark.createDataFrame(df_pandas)
        
        # Join with original DataFrame
        df_result = df.join(
            df_result.select('timestamp', 'regime', 'regime_name'),
            on='timestamp',
            how='inner'
        )
        
        # Cache to avoid lazy evaluation issues
        df_result = df_result.cache()
        df_result.count()  # Force materialization
        
        print("[K-MEANS] Detection complete")
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
    
    def get_cluster_centers(self):
        """Get cluster centers (regime characteristics)."""
        if self.model is None:
            return None
        return self.model.cluster_centers_
    
    def get_inertia(self):
        """Get within-cluster sum of squares."""
        if self.model is None:
            return None
        return self.model.inertia_
    
    def compute_elbow_curve(self, X, max_k=10):
        """
        Compute inertia for different k values (elbow method).
        Args:
            X: Feature matrix 
            max_k: Maximum number of clusters to test
        Returns:
            List of (k, inertia) tuples
        """
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            inertias.append((k, kmeans.inertia_))
        
        return inertias