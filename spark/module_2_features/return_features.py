"""
Return Features for Regime Detection
Computes return statistics that characterize market regimes.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class ReturnFeatures:
    """
    Compute return-based features for regime detection.
    
    Different regimes exhibit different return characteristics:
    - Bull regime: Positive skew, low kurtosis, positive autocorr
    - Bear regime: Negative skew, high kurtosis (crashes), negative autocorr
    - Volatile regime: High kurtosis, low autocorr
    - Calm regime: Low kurtosis, high autocorr
    """
    
    def __init__(self, windows=[60, 240, 1440]):
        """
        Initialize return feature calculator.
        
        Args:
            windows: List of lookback windows in minutes
                    Default: [60, 240, 1440] = [1h, 4h, 24h]
        """
        self.windows = windows
        print(f"[INFO] Initialized ReturnFeatures with windows: {windows}")
    
    def compute_cumulative_returns(self, df: DataFrame, periods=[5, 15, 30, 60]) -> DataFrame:
        """
        Compute cumulative returns over multiple periods.
        
        These capture momentum at different time scales.
        
        Formula: return_Nm = sum(log_return[t-N:t])
        
        Args:
            df: DataFrame with 'log_return' column
            periods: List of periods in minutes
            
        Returns:
            DataFrame with return_{period}m columns
        """
        print("[INFO] Computing cumulative returns...")
        
        for period in periods:
            col_name = f"return_{period}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
            
            # Sum of log returns = log(P_t / P_{t-N}) = cumulative return
            df = df.withColumn(
                col_name,
                F.sum("log_return").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_return_skewness(self, df: DataFrame) -> DataFrame:
        """
        Compute skewness of returns - measures asymmetry.
        
        Skewness indicates regime type:
        - Positive skew (>0): More extreme positive returns (rallies)
        - Negative skew (<0): More extreme negative returns (crashes)
        - Near zero: Symmetric distribution
        
        Formula: E[(X - μ)³] / σ³
        
        Args:
            df: DataFrame with 'log_return' column
            
        Returns:
            DataFrame with return_skew_{window}m columns
        """
        print("[INFO] Computing return skewness...")
        
        for window in self.windows:
            col_name = f"return_skew_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # PySpark has built-in skewness function
            df = df.withColumn(
                col_name,
                F.skewness("log_return").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_return_kurtosis(self, df: DataFrame) -> DataFrame:
        """
        Compute kurtosis of returns - measures tail heaviness.
        
        Kurtosis indicates regime volatility:
        - High kurtosis (>3): Fat tails, extreme moves (volatile regime)
        - Low kurtosis (<3): Thin tails, no extremes (calm regime)
        - Normal = 3
        
        Formula: E[(X - μ)⁴] / σ⁴
        
        Args:
            df: DataFrame with 'log_return' column
            
        Returns:
            DataFrame with return_kurt_{window}m columns
        """
        print("[INFO] Computing return kurtosis...")
        
        for window in self.windows:
            col_name = f"return_kurt_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # PySpark has built-in kurtosis function
            df = df.withColumn(
                col_name,
                F.kurtosis("log_return").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_autocorrelation(self, df: DataFrame, lags=[1, 5, 15, 60]) -> DataFrame:
        """
        Compute autocorrelation of returns - measures momentum persistence.
        
        Autocorrelation indicates trend vs mean-reversion:
        - Positive autocorr: Momentum (trends continue)
        - Negative autocorr: Mean reversion (reversals)
        - Near zero: Random walk
        
        Formula: Corr(return_t, return_{t-lag})
        
        Args:
            df: DataFrame with 'log_return' column
            lags: List of lag periods in minutes
            
        Returns:
            DataFrame with autocorr_returns_{lag} columns
        """
        print("[INFO] Computing return autocorrelation...")
        
        for lag in lags:
            col_name = f"autocorr_returns_{lag}"
            
            # Create lagged return column
            lag_col = f"log_return_lag{lag}"
            window_spec = Window.orderBy("timestamp")
            
            df = df.withColumn(
                lag_col,
                F.lag("log_return", lag).over(window_spec)
            )
            
            # Compute correlation over a rolling window (using 60 periods)
            corr_window = Window.orderBy("timestamp").rowsBetween(-60 + 1, 0)
            
            df = df.withColumn(
                col_name,
                F.corr("log_return", lag_col).over(corr_window)
            )
            
            # Drop temporary lagged column
            df = df.drop(lag_col)
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_return_range(self, df: DataFrame, windows=[60, 240, 1440]) -> DataFrame:
        """
        Compute range of returns - max return minus min return in window.
        
        High range indicates volatile regime.
        
        Args:
            df: DataFrame with 'log_return' column
            windows: List of windows in minutes
            
        Returns:
            DataFrame with return_range_{window}m columns
        """
        print("[INFO] Computing return range...")
        
        for window in windows:
            col_name = f"return_range_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # Range = max - min
            df = df.withColumn(
                col_name,
                F.max("log_return").over(window_spec) - F.min("log_return").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_positive_return_ratio(self, df: DataFrame, windows=[60, 240, 1440]) -> DataFrame:
        """
        Compute ratio of positive returns in window.
        
        High ratio (>0.5) indicates bullish regime.
        Low ratio (<0.5) indicates bearish regime.
        
        Args:
            df: DataFrame with 'log_return' column
            windows: List of windows in minutes
            
        Returns:
            DataFrame with pos_return_ratio_{window}m columns
        """
        print("[INFO] Computing positive return ratio...")
        
        # Create binary column: 1 if return > 0, else 0
        df = df.withColumn(
            "is_positive_return",
            F.when(F.col("log_return") > 0, 1).otherwise(0)
        )
        
        for window in windows:
            col_name = f"pos_return_ratio_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # Ratio = mean of binary indicator
            df = df.withColumn(
                col_name,
                F.mean("is_positive_return").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        # Drop temporary column
        df = df.drop("is_positive_return")
        
        return df
    
    def compute_all_return_features(self, df: DataFrame) -> DataFrame:
        """
        Compute all return features in one go.
        
        Args:
            df: DataFrame with 'log_return' column
            
        Returns:
            DataFrame with all return features added
        """
        print("\n" + "="*70)
        print("COMPUTING ALL RETURN FEATURES")
        print("="*70)
        
        # 1. Cumulative returns (momentum)
        df = self.compute_cumulative_returns(df)
        
        # 2. Return skewness (asymmetry)
        df = self.compute_return_skewness(df)
        
        # 3. Return kurtosis (tail risk)
        df = self.compute_return_kurtosis(df)
        
        # 4. Autocorrelation (persistence)
        df = self.compute_autocorrelation(df)
        
        # 5. Return range (volatility)
        df = self.compute_return_range(df)
        
        # 6. Positive return ratio (directional bias)
        df = self.compute_positive_return_ratio(df)
        
        # Count total features added
        return_features = [col for col in df.columns if 'return' in col.lower() and col != 'log_return' and col != 'simple_return']
        print(f"\n[INFO] Added {len(return_features)} return features")
        print(f"[INFO] Features: {', '.join(return_features[:5])}... (and {len(return_features)-5} more)")
        
        return df


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("ReturnFeatures-Test") \
        .getOrCreate()
    
    print("\n=== Testing Return Features ===\n")
    
    # Load processed data
    from pathlib import Path
    base_path = Path(__file__).parent.parent.parent
    data_path = str(base_path / "data" / "processed" / "BTCUSDT" / "1m" / "historical")
    
    print(f"Loading data from: {data_path}")
    df = spark.read.parquet(data_path)
    
    print(f"Loaded {df.count():,} records")
    
    # Initialize return features
    return_calc = ReturnFeatures(windows=[60, 240, 1440])
    
    # Compute all features
    df_with_returns = return_calc.compute_all_return_features(df)
    
    # Show sample
    print("\n=== Sample Data with Return Features ===")
    df_with_returns.select(
        "timestamp", "price_close", "log_return",
        "return_5m", "return_60m",
        "return_skew_60m", "return_kurt_60m",
        "autocorr_returns_1", "pos_return_ratio_60m"
    ).orderBy(F.desc("timestamp")).show(10, truncate=False)
    
    # Summary statistics
    print("\n=== Return Feature Statistics ===")
    df_with_returns.select(
        "return_skew_60m", "return_kurt_60m", 
        "autocorr_returns_5", "pos_return_ratio_60m"
    ).describe().show()
    
    spark.stop()