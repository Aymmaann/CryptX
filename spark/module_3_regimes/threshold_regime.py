from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np


class ThresholdRegimeDetector:
    """
    Simple threshold-based regime detection (baseline method).
    Uses volatility quantiles to define regimes:
    - Low volatility: bottom 33%
    - Medium volatility: middle 34%
    - High volatility: top 33%
    """
    def __init__(self):
        self.thresholds = None
        print("[INFO] Threshold detector initialized")
    
    def detect(self, df: DataFrame) -> DataFrame:
        """
        Detect regimes using volatility thresholds.
        Returns:
            DataFrame with 'regime' column added
        """
        print("[THRESHOLD] Starting regime detection...")
        
        # Use 60-minute volatility as the primary indicator
        if 'vol_60m' not in df.columns:
            raise ValueError("vol_60m column required for threshold detection")
        
        # Calculate quantiles for regime boundaries
        print("[THRESHOLD] Computing volatility quantiles...")
        quantiles = df.select('vol_60m').dropna() \
            .selectExpr("percentile_approx(vol_60m, array(0.33, 0.67))") \
            .collect()[0][0]
        
        low_threshold, high_threshold = quantiles
        self.thresholds = {'low': low_threshold, 'high': high_threshold}
        
        print(f"[THRESHOLD] Thresholds: low={low_threshold:.6f}, high={high_threshold:.6f}")
        
        # Assign regimes based on thresholds
        df_result = df.withColumn(
            'regime',
            F.when(F.col('vol_60m') <= low_threshold, 0)
            .when(F.col('vol_60m') <= high_threshold, 1)
            .otherwise(2)
        )
        
        # Add regime name
        df_result = df_result.withColumn(
            'regime_name',
            F.when(F.col('regime') == 0, 'low_vol')
            .when(F.col('regime') == 1, 'medium_vol')
            .otherwise('high_vol')
        )
        
        # Cache to avoid lazy evaluation issues
        df_result = df_result.cache()
        df_result.count()  # Force materialization
        
        print("[THRESHOLD] Detection complete")
        return df_result
    
    def detect_with_multiple_indicators(self, df: DataFrame) -> DataFrame:
        """
        Enhanced threshold detection using multiple indicators.
        Combines volatility, volume, and price momentum.
        """
        print("[THRESHOLD] Starting multi-indicator detection...")
        
        required_cols = ['vol_60m', 'volume_spike', 'cum_return_60m']
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            print(f"[THRESHOLD] Missing columns: {missing}, falling back to simple detection")
            return self.detect(df)
        
        # Convert to Pandas for complex logic
        print("[THRESHOLD] Converting to Pandas...")
        df_pandas = df.select(['timestamp', 'vol_60m', 'volume_spike', 
                              'cum_return_60m']).toPandas()
        df_pandas = df_pandas.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate quantiles for each indicator
        vol_q33, vol_q67 = df_pandas['vol_60m'].quantile([0.33, 0.67])
        volume_q33, volume_q67 = df_pandas['volume_spike'].quantile([0.33, 0.67])
        
        # Score-based regime assignment
        def assign_regime(row):
            score = 0
            
            # Volatility score (0, 1, or 2)
            if row['vol_60m'] <= vol_q33:
                score += 0
            elif row['vol_60m'] <= vol_q67:
                score += 1
            else:
                score += 2
            
            # Volume score (0, 1, or 2)
            if row['volume_spike'] <= volume_q33:
                score += 0
            elif row['volume_spike'] <= volume_q67:
                score += 0.5
            else:
                score += 1
            
            # Strong trends add volatility
            if abs(row['cum_return_60m']) > 0.02:  # >2% cumulative return
                score += 0.5
            
            # Map score to regime
            if score < 1.0:
                return 0  # Low volatility
            elif score < 2.5:
                return 1  # Medium volatility
            else:
                return 2  # High volatility
        
        df_pandas['regime'] = df_pandas.apply(assign_regime, axis=1)
        
        # Add regime name
        regime_names = {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        df_pandas['regime_name'] = df_pandas['regime'].map(regime_names)
        
        # Convert back to Spark
        print("[THRESHOLD] Converting back to Spark DataFrame...")
        spark = df.sql_ctx.sparkSession
        df_result = spark.createDataFrame(df_pandas)
        
        # Join with original
        df_result = df.join(
            df_result.select('timestamp', 'regime', 'regime_name'),
            on='timestamp',
            how='inner'
        )
        
        # Cache to avoid lazy evaluation issues
        df_result = df_result.cache()
        df_result.count()  # Force materialization
        
        print("[THRESHOLD] Multi-indicator detection complete")
        return df_result
    
    def detect_volatility_breakouts(self, df: DataFrame, lookback=240) -> DataFrame:
        """
        Detect regime changes based on volatility breakouts.
        Identifies when volatility breaks out of its recent range.
        """
        print("[THRESHOLD] Detecting volatility breakouts...")
        
        if 'vol_60m' not in df.columns:
            raise ValueError("vol_60m required")
        
        # Convert to Pandas for rolling calculations
        df_pandas = df.select(['timestamp', 'vol_60m']).toPandas()
        df_pandas = df_pandas.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate rolling statistics
        df_pandas['vol_ma'] = df_pandas['vol_60m'].rolling(window=lookback, min_periods=1).mean()
        df_pandas['vol_std'] = df_pandas['vol_60m'].rolling(window=lookback, min_periods=1).std()
        
        # Define breakout thresholds (mean ± 1 std dev)
        df_pandas['vol_lower'] = df_pandas['vol_ma'] - df_pandas['vol_std']
        df_pandas['vol_upper'] = df_pandas['vol_ma'] + df_pandas['vol_std']
        
        # Assign regimes
        def assign_breakout_regime(row):
            if row['vol_60m'] < row['vol_lower']:
                return 0  # Low volatility regime
            elif row['vol_60m'] > row['vol_upper']:
                return 2  # High volatility regime (breakout)
            else:
                return 1  # Normal volatility regime
        
        df_pandas['regime'] = df_pandas.apply(assign_breakout_regime, axis=1)
        
        # Add regime name
        regime_names = {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'}
        df_pandas['regime_name'] = df_pandas['regime'].map(regime_names)
        
        # Convert back to Spark
        spark = df.sql_ctx.sparkSession
        df_result = spark.createDataFrame(df_pandas[['timestamp', 'regime', 'regime_name']])
        
        # Join with original
        df_result = df.join(df_result, on='timestamp', how='inner')
        
        # Cache to avoid lazy evaluation issues
        df_result = df_result.cache()
        df_result.count()  # Force materialization
        
        print("[THRESHOLD] Breakout detection complete")
        return df_result
    
    def get_thresholds(self):
        """Get computed thresholds."""
        return self.thresholds