"""
Volume Features for Regime Detection
Computes volume-based features that indicate market participation and liquidity regimes.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class VolumeFeatures:
    """
    Compute volume-based features for regime detection.
    
    Different regimes exhibit different volume characteristics:
    - High volatility regime: High volume, large spikes
    - Trending regime: Sustained high volume, strong OBV
    - Consolidation regime: Low volume, low spikes
    - Panic regime: Extreme volume spikes, negative volume-price correlation
    """
    
    def __init__(self, windows=[60, 240, 1440]):
        """
        Initialize volume feature calculator.
        
        Args:
            windows: List of lookback windows in minutes
                    Default: [60, 240, 1440] = [1h, 4h, 24h]
        """
        self.windows = windows
        print(f"[INFO] Initialized VolumeFeatures with windows: {windows}")
    
    def compute_volume_moving_averages(self, df: DataFrame) -> DataFrame:
        """
        Compute volume moving averages at multiple time scales.
        
        Used to detect unusual volume activity (spikes).
        
        Args:
            df: DataFrame with 'volume_base' column
            
        Returns:
            DataFrame with volume_ma_{window}m columns
        """
        print("[INFO] Computing volume moving averages...")
        
        for window in self.windows:
            col_name = f"volume_ma_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(
                col_name,
                F.mean("volume_base").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_volume_spikes(self, df: DataFrame) -> DataFrame:
        """
        Compute volume spike indicators (current volume / MA volume).
        
        Spike > 2.0 indicates 2x normal volume (high activity).
        
        Formula: volume_spike = volume_t / MA(volume)
        
        Args:
            df: DataFrame with volume_ma columns (must run volume_ma first)
            
        Returns:
            DataFrame with volume_spike_{window}m columns
        """
        print("[INFO] Computing volume spikes...")
        
        for window in self.windows:
            ma_col = f"volume_ma_{window}m"
            spike_col = f"volume_spike_{window}m"
            
            if ma_col not in df.columns:
                print(f"[WARNING] {ma_col} not found. Skipping volume spike for {window}m.")
                continue
            
            # Avoid division by zero
            df = df.withColumn(
                spike_col,
                F.when(F.col(ma_col) > 0, F.col("volume_base") / F.col(ma_col)).otherwise(1.0)
            )
            
            print(f"  ✓ Created {spike_col}")
        
        return df
    
    def compute_vwap(self, df: DataFrame, windows=[60, 240, 1440]) -> DataFrame:
        """
        Compute Volume-Weighted Average Price (VWAP).
        
        VWAP = sum(price * volume) / sum(volume)
        
        Used to detect if price is above/below average transaction price.
        
        Args:
            df: DataFrame with 'price_close' and 'volume_base' columns
            windows: Windows to compute VWAP
            
        Returns:
            DataFrame with vwap_{window}m and price_vs_vwap_{window}m columns
        """
        print("[INFO] Computing VWAP...")
        
        # Calculate price * volume for each candle
        df = df.withColumn(
            "pv",
            F.col("price_close") * F.col("volume_base")
        )
        
        for window in windows:
            vwap_col = f"vwap_{window}m"
            ratio_col = f"price_vs_vwap_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # VWAP = sum(P*V) / sum(V)
            df = df.withColumn(
                vwap_col,
                F.sum("pv").over(window_spec) / F.sum("volume_base").over(window_spec)
            )
            
            # Price vs VWAP ratio (>1 = price above VWAP, <1 = below)
            df = df.withColumn(
                ratio_col,
                F.col("price_close") / F.col(vwap_col)
            )
            
            print(f"  ✓ Created {vwap_col} and {ratio_col}")
        
        # Drop temporary column
        df = df.drop("pv")
        
        return df
    
    def compute_obv(self, df: DataFrame) -> DataFrame:
        """
        Compute On-Balance Volume (OBV).
        
        OBV tracks cumulative volume flow:
        - Add volume if price up
        - Subtract volume if price down
        
        Rising OBV = accumulation (bullish)
        Falling OBV = distribution (bearish)
        
        Args:
            df: DataFrame with 'volume_base' and 'log_return' columns
            
        Returns:
            DataFrame with obv and obv_ma_{window}m columns
        """
        print("[INFO] Computing On-Balance Volume (OBV)...")
        
        # Direction: +1 if price up, -1 if price down, 0 if unchanged
        df = df.withColumn(
            "direction",
            F.when(F.col("log_return") > 0, 1)
            .when(F.col("log_return") < 0, -1)
            .otherwise(0)
        )
        
        # Signed volume
        df = df.withColumn(
            "signed_volume",
            F.col("direction") * F.col("volume_base")
        )
        
        # Cumulative OBV
        window_spec = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0)
        
        df = df.withColumn(
            "obv",
            F.sum("signed_volume").over(window_spec)
        )
        
        print("  ✓ Created obv")
        
        # OBV moving averages (to smooth the signal)
        for window in self.windows:
            col_name = f"obv_ma_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(
                col_name,
                F.mean("obv").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        # Drop temporary columns
        df = df.drop("direction", "signed_volume")
        
        return df
    
    def compute_volume_price_correlation(self, df: DataFrame, windows=[60, 240, 1440]) -> DataFrame:
        """
        Compute correlation between volume and price changes.
        
        Positive correlation: Volume increases with price (healthy trend)
        Negative correlation: Volume increases as price falls (panic selling)
        
        Args:
            df: DataFrame with 'volume_base' and 'log_return' columns
            windows: Windows to compute correlation
            
        Returns:
            DataFrame with volume_price_corr_{window}m columns
        """
        print("[INFO] Computing volume-price correlation...")
        
        for window in windows:
            col_name = f"volume_price_corr_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # Correlation between volume and absolute price change
            df = df.withColumn(
                col_name,
                F.corr("volume_base", F.abs("log_return")).over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_volume_momentum(self, df: DataFrame, windows=[60, 240]) -> DataFrame:
        """
        Compute rate of change in volume.
        
        Positive momentum: Volume increasing (growing interest)
        Negative momentum: Volume decreasing (fading interest)
        
        Args:
            df: DataFrame with 'volume_base' column
            windows: Windows to compute momentum
            
        Returns:
            DataFrame with volume_momentum_{window}m columns
        """
        print("[INFO] Computing volume momentum...")
        
        for window in windows:
            col_name = f"volume_momentum_{window}m"
            
            window_spec = Window.orderBy("timestamp")
            
            # Volume N periods ago
            volume_lag = F.lag("volume_base", window).over(window_spec)
            
            # Rate of change: (current - past) / past
            df = df.withColumn(
                col_name,
                F.when(volume_lag > 0, 
                       (F.col("volume_base") - volume_lag) / volume_lag
                ).otherwise(0.0)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_volume_volatility(self, df: DataFrame, windows=[60, 240, 1440]) -> DataFrame:
        """
        Compute volatility of volume (standard deviation of volume).
        
        High volume volatility = unstable participation (regime transition?)
        
        Args:
            df: DataFrame with 'volume_base' column
            windows: Windows to compute
            
        Returns:
            DataFrame with volume_vol_{window}m columns
        """
        print("[INFO] Computing volume volatility...")
        
        for window in windows:
            col_name = f"volume_vol_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(
                col_name,
                F.stddev("volume_base").over(window_spec)
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_all_volume_features(self, df: DataFrame) -> DataFrame:
        """
        Compute all volume features in one go.
        
        Args:
            df: DataFrame with 'volume_base', 'price_close', 'log_return' columns
            
        Returns:
            DataFrame with all volume features added
        """
        print("\n" + "="*70)
        print("COMPUTING ALL VOLUME FEATURES")
        print("="*70)
        
        # 1. Volume moving averages (baseline)
        df = self.compute_volume_moving_averages(df)
        
        # 2. Volume spikes (unusual activity)
        df = self.compute_volume_spikes(df)
        
        # 3. VWAP (average transaction price)
        df = self.compute_vwap(df)
        
        # 4. OBV (accumulation/distribution)
        df = self.compute_obv(df)
        
        # 5. Volume-price correlation
        df = self.compute_volume_price_correlation(df)
        
        # 6. Volume momentum
        df = self.compute_volume_momentum(df)
        
        # 7. Volume volatility
        df = self.compute_volume_volatility(df)
        
        # Count total features added
        volume_features = [col for col in df.columns if 'volume' in col.lower() or 'vwap' in col.lower() or 'obv' in col.lower()]
        # Exclude original volume columns
        original_cols = ['volume_base', 'volume_quote', 'taker_buy_base_vol', 'taker_buy_quote_vol']
        new_features = [col for col in volume_features if col not in original_cols]
        
        print(f"\n[INFO] Added {len(new_features)} volume features")
        print(f"[INFO] Features: {', '.join(new_features[:5])}... (and {len(new_features)-5} more)")
        
        return df


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("VolumeFeatures-Test") \
        .getOrCreate()
    
    print("\n=== Testing Volume Features ===\n")
    
    # Load processed data
    from pathlib import Path
    base_path = Path(__file__).parent.parent.parent
    data_path = str(base_path / "data" / "processed" / "BTCUSDT" / "1m" / "historical")
    
    print(f"Loading data from: {data_path}")
    df = spark.read.parquet(data_path)
    
    print(f"Loaded {df.count():,} records")
    
    # Initialize volume features
    volume_calc = VolumeFeatures(windows=[60, 240, 1440])
    
    # Compute all features
    df_with_volume = volume_calc.compute_all_volume_features(df)
    
    # Show sample
    print("\n=== Sample Data with Volume Features ===")
    df_with_volume.select(
        "timestamp", "price_close", "volume_base",
        "volume_ma_60m", "volume_spike_60m",
        "vwap_60m", "price_vs_vwap_60m",
        "obv", "volume_price_corr_60m"
    ).orderBy(F.desc("timestamp")).show(10, truncate=False)
    
    # Summary statistics
    print("\n=== Volume Feature Statistics ===")
    df_with_volume.select(
        "volume_spike_60m", "price_vs_vwap_60m",
        "volume_price_corr_60m", "volume_momentum_60m"
    ).describe().show()
    
    spark.stop()