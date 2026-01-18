"""
Volatility Estimators for Regime Detection
Implements multiple volatility measures at different time scales.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np


class VolatilityEstimators:
    """
    Compute various volatility measures for regime detection.
    
    Different regimes exhibit different volatility characteristics:
    - Low volatility regime: All measures low and stable
    - High volatility regime: All measures elevated
    - Transition regime: Volatility of volatility increases
    """
    
    def __init__(self, windows=[5, 15, 30, 60, 240, 1440]):
        """
        Initialize volatility estimators.
        
        Args:
            windows: List of lookback windows in minutes
                    Default: [5, 15, 30, 60, 240, 1440]
                    = [5min, 15min, 30min, 1h, 4h, 24h]
        """
        self.windows = windows
        print(f"[INFO] Initialized VolatilityEstimators with windows: {windows}")
    
    def compute_realized_volatility(self, df: DataFrame) -> DataFrame:
        """
        Compute realized volatility (rolling std of log returns).
        
        Realized Vol = sqrt(sum(r_t^2)) where r_t = log returns
        
        This is the most common volatility measure and works well for
        detecting volatility regime changes.
        
        Args:
            df: DataFrame with 'log_return' column
            
        Returns:
            DataFrame with realized_vol_{window}m columns added
        """
        print("[INFO] Computing realized volatility...")
        
        for window in self.windows:
            col_name = f"realized_vol_{window}m"
            
            # Rolling standard deviation of log returns
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(
                col_name,
                F.stddev("log_return").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_parkinson_volatility(self, df: DataFrame, windows=[60, 240, 1440]) -> DataFrame:
        """
        Parkinson volatility estimator - uses high-low range.
        
        More efficient than close-to-close volatility (30% more efficient).
        Based on the assumption that high-low range contains more info.
        
        Formula: sqrt(1/(4*ln(2)) * (ln(High/Low))^2)
        
        Args:
            df: DataFrame with 'price_high' and 'price_low' columns
            windows: Windows to compute (default: [60, 240, 1440] for 1h, 4h, 24h)
            
        Returns:
            DataFrame with parkinson_vol_{window}m columns
        """
        print("[INFO] Computing Parkinson volatility estimator...")
        
        # Calculate log(High/Low) for each candle
        df = df.withColumn(
            "hl_ratio",
            F.log(F.col("price_high") / F.col("price_low"))
        )
        
        # Parkinson constant
        parkinson_factor = 1.0 / (4.0 * np.log(2))
        
        for window in windows:
            col_name = f"parkinson_vol_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # Parkinson estimator: sqrt(mean((ln(H/L))^2) / (4*ln(2)))
            df = df.withColumn(
                col_name,
                F.sqrt(
                    F.mean(F.pow(F.col("hl_ratio"), 2)).over(window_spec) * parkinson_factor
                )
            )
            
            print(f"  ✓ Created {col_name}")
        
        # Drop temporary column
        df = df.drop("hl_ratio")
        
        return df
    
    def compute_garman_klass_volatility(self, df: DataFrame, windows=[60, 240, 1440]) -> DataFrame:
        """
        Garman-Klass volatility estimator - uses OHLC.
        
        More efficient than Parkinson (50% more efficient than close-to-close).
        Uses all four price points: Open, High, Low, Close.
        
        Formula combines:
        - 0.5 * (ln(High/Low))^2
        - (2*ln(2) - 1) * (ln(Close/Open))^2
        
        Args:
            df: DataFrame with OHLC columns
            windows: Windows to compute
            
        Returns:
            DataFrame with gk_vol_{window}m columns
        """
        print("[INFO] Computing Garman-Klass volatility estimator...")
        
        # Calculate components
        df = df.withColumn(
            "hl_component",
            0.5 * F.pow(F.log(F.col("price_high") / F.col("price_low")), 2)
        )
        
        df = df.withColumn(
            "co_component",
            (2 * np.log(2) - 1) * F.pow(F.log(F.col("price_close") / F.col("price_open")), 2)
        )
        
        for window in windows:
            col_name = f"gk_vol_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # Garman-Klass estimator
            df = df.withColumn(
                col_name,
                F.sqrt(
                    F.mean(F.col("hl_component") - F.col("co_component")).over(window_spec)
                )
            )
            
            print(f"  ✓ Created {col_name}")
        
        # Drop temporary columns
        df = df.drop("hl_component", "co_component")
        
        return df
    
    def compute_volatility_of_volatility(self, df: DataFrame, base_window=60, vov_window=60) -> DataFrame:
        """
        Volatility of volatility - second-order volatility measure.
        
        Measures how much volatility itself is changing.
        High VoV indicates regime transitions.
        
        Args:
            df: DataFrame (should already have realized_vol columns)
            base_window: Which realized vol to use (default: 60m)
            vov_window: Window for VoV calculation (default: 60m)
            
        Returns:
            DataFrame with vol_of_vol_{base_window}m column
        """
        print(f"[INFO] Computing volatility of volatility (base={base_window}m, window={vov_window}m)...")
        
        base_col = f"realized_vol_{base_window}m"
        vov_col = f"vol_of_vol_{base_window}m"
        
        if base_col not in df.columns:
            print(f"[WARNING] {base_col} not found. Skipping VoV calculation.")
            return df
        
        window_spec = Window.orderBy("timestamp").rowsBetween(-vov_window + 1, 0)
        
        df = df.withColumn(
            vov_col,
            F.stddev(base_col).over(window_spec)
        )
        
        print(f"  ✓ Created {vov_col}")
        
        return df
    
    def compute_all_volatility_features(self, df: DataFrame) -> DataFrame:
        """
        Compute all volatility features in one go.
        
        Args:
            df: DataFrame with OHLCV and log_return columns
            
        Returns:
            DataFrame with all volatility features added
        """
        print("\n" + "="*70)
        print("COMPUTING ALL VOLATILITY FEATURES")
        print("="*70)
        
        # 1. Realized volatility (all windows)
        df = self.compute_realized_volatility(df)
        
        # 2. Parkinson estimator (hourly+)
        df = self.compute_parkinson_volatility(df)
        
        # 3. Garman-Klass estimator (hourly+)
        df = self.compute_garman_klass_volatility(df)
        
        # 4. Volatility of volatility
        df = self.compute_volatility_of_volatility(df)
        
        # Count total features added
        vol_features = [col for col in df.columns if 'vol' in col.lower()]
        print(f"\n[INFO] Added {len(vol_features)} volatility features")
        print(f"[INFO] Features: {', '.join(vol_features[:5])}... (and {len(vol_features)-5} more)")
        
        return df


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("VolatilityEstimators-Test") \
        .getOrCreate()
    
    print("\n=== Testing Volatility Estimators ===\n")
    
    # Load processed data (use absolute path from CryptX root)
    from pathlib import Path
    base_path = Path(__file__).parent.parent.parent  # Go up to CryptX/
    data_path = str(base_path / "data" / "processed" / "BTCUSDT" / "1m" / "historical")
    
    print(f"Loading data from: {data_path}")
    df = spark.read.parquet(data_path)
    
    print(f"Loaded {df.count():,} records")
    print(f"Columns: {df.columns}")
    
    # Initialize estimators
    vol_estimator = VolatilityEstimators(windows=[5, 15, 30, 60, 240, 1440])
    
    # Compute all features
    df_with_vol = vol_estimator.compute_all_volatility_features(df)
    
    # Show sample
    print("\n=== Sample Data with Volatility Features ===")
    df_with_vol.select(
        "timestamp", "price_close", "log_return",
        "realized_vol_5m", "realized_vol_60m", "realized_vol_1440m",
        "parkinson_vol_60m", "gk_vol_60m"
    ).orderBy(F.desc("timestamp")).show(10, truncate=False)
    
    # Summary statistics
    print("\n=== Volatility Feature Statistics ===")
    df_with_vol.select(
        "realized_vol_60m", "parkinson_vol_60m", "gk_vol_60m", "vol_of_vol_60m"
    ).describe().show()
    
    spark.stop()