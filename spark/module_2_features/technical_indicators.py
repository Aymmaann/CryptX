"""
Technical Indicators for Regime Detection
Implements classic technical indicators adapted for regime identification.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import numpy as np


class TechnicalIndicators:
    """
    Compute technical indicators for regime detection.
    
    Different regimes exhibit different technical patterns:
    - Trending regime: MACD crossovers, RSI trending
    - Overbought regime: RSI > 70, price near upper BB
    - Oversold regime: RSI < 30, price near lower BB
    - High volatility regime: Wide Bollinger Bands, high ATR
    """
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        print("[INFO] Initialized TechnicalIndicators")
    
    def compute_rsi(self, df: DataFrame, period=14) -> DataFrame:
        """
        Compute Relative Strength Index (RSI).
        
        RSI measures momentum and identifies overbought/oversold conditions:
        - RSI > 70: Overbought (potential reversal down)
        - RSI < 30: Oversold (potential reversal up)
        - RSI 40-60: Neutral regime
        
        Formula:
        1. Calculate gains and losses
        2. RS = Average Gain / Average Loss (over period)
        3. RSI = 100 - (100 / (1 + RS))
        
        Args:
            df: DataFrame with 'price_close' column
            period: RSI period (default: 14)
            
        Returns:
            DataFrame with rsi_{period} column
        """
        print(f"[INFO] Computing RSI (period={period})...")
        
        col_name = f"rsi_{period}"
        
        # Calculate price changes
        window_spec = Window.orderBy("timestamp")
        df = df.withColumn(
            "price_change",
            F.col("price_close") - F.lag("price_close", 1).over(window_spec)
        )
        
        # Separate gains and losses
        df = df.withColumn(
            "gain",
            F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0)
        )
        df = df.withColumn(
            "loss",
            F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0)
        )
        
        # Calculate average gain and loss using EMA-style smoothing
        window_spec_period = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
        
        df = df.withColumn(
            "avg_gain",
            F.mean("gain").over(window_spec_period)
        )
        df = df.withColumn(
            "avg_loss",
            F.mean("loss").over(window_spec_period)
        )
        
        # Calculate RS and RSI
        df = df.withColumn(
            "rs",
            F.when(F.col("avg_loss") > 0, F.col("avg_gain") / F.col("avg_loss")).otherwise(100)
        )
        
        df = df.withColumn(
            col_name,
            100 - (100 / (1 + F.col("rs")))
        )
        
        # Drop temporary columns
        df = df.drop("price_change", "gain", "loss", "avg_gain", "avg_loss", "rs")
        
        print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_bollinger_bands(self, df: DataFrame, period=20, num_std=2) -> DataFrame:
        """
        Compute Bollinger Bands.
        
        Bollinger Bands measure volatility and identify price extremes:
        - Price at upper band: Overbought/high volatility
        - Price at lower band: Oversold/high volatility
        - Narrow bands: Low volatility (consolidation regime)
        - Wide bands: High volatility (volatile regime)
        
        Formula:
        - Middle Band = SMA(close, period)
        - Upper Band = Middle + (std * num_std)
        - Lower Band = Middle - (std * num_std)
        - %B = (Price - Lower) / (Upper - Lower)
        - Bandwidth = (Upper - Lower) / Middle
        
        Args:
            df: DataFrame with 'price_close' column
            period: Period for MA and std (default: 20)
            num_std: Number of standard deviations (default: 2)
            
        Returns:
            DataFrame with bb_upper, bb_lower, bb_middle, bb_width, bb_pct columns
        """
        print(f"[INFO] Computing Bollinger Bands (period={period}, std={num_std})...")
        
        window_spec = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
        
        # Middle band (SMA)
        df = df.withColumn(
            f"bb_middle_{period}",
            F.mean("price_close").over(window_spec)
        )
        
        # Standard deviation
        df = df.withColumn(
            "bb_std",
            F.stddev("price_close").over(window_spec)
        )
        
        # Upper and lower bands
        df = df.withColumn(
            f"bb_upper_{period}",
            F.col(f"bb_middle_{period}") + (F.col("bb_std") * num_std)
        )
        
        df = df.withColumn(
            f"bb_lower_{period}",
            F.col(f"bb_middle_{period}") - (F.col("bb_std") * num_std)
        )
        
        # Bandwidth (normalized width of bands - volatility indicator)
        df = df.withColumn(
            f"bb_width_{period}",
            (F.col(f"bb_upper_{period}") - F.col(f"bb_lower_{period}")) / F.col(f"bb_middle_{period}")
        )
        
        # %B (position within bands: 0 = lower band, 1 = upper band)
        df = df.withColumn(
            f"bb_pct_{period}",
            (F.col("price_close") - F.col(f"bb_lower_{period}")) / 
            (F.col(f"bb_upper_{period}") - F.col(f"bb_lower_{period}"))
        )
        
        # Drop temporary column
        df = df.drop("bb_std")
        
        print(f"  ✓ Created bb_upper_{period}, bb_lower_{period}, bb_middle_{period}, bb_width_{period}, bb_pct_{period}")
        
        return df
    
    def compute_macd(self, df: DataFrame, fast=12, slow=26, signal=9) -> DataFrame:
        """
        Compute MACD (Moving Average Convergence Divergence).
        
        MACD measures momentum and trend:
        - MACD > 0: Bullish momentum
        - MACD < 0: Bearish momentum
        - MACD crosses signal line: Trend change
        - Histogram expanding: Increasing momentum
        
        Formula:
        - MACD Line = EMA(close, fast) - EMA(close, slow)
        - Signal Line = EMA(MACD, signal)
        - Histogram = MACD - Signal
        
        Args:
            df: DataFrame with 'price_close' column
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            DataFrame with macd, macd_signal, macd_hist columns
        """
        print(f"[INFO] Computing MACD (fast={fast}, slow={slow}, signal={signal})...")
        
        # For simplicity, use SMA instead of EMA (EMA is complex in Spark)
        # This is a reasonable approximation for regime detection
        
        window_fast = Window.orderBy("timestamp").rowsBetween(-fast + 1, 0)
        window_slow = Window.orderBy("timestamp").rowsBetween(-slow + 1, 0)
        
        # Calculate fast and slow moving averages
        df = df.withColumn(
            "ema_fast",
            F.mean("price_close").over(window_fast)
        )
        
        df = df.withColumn(
            "ema_slow",
            F.mean("price_close").over(window_slow)
        )
        
        # MACD line
        df = df.withColumn(
            "macd",
            F.col("ema_fast") - F.col("ema_slow")
        )
        
        # Signal line (SMA of MACD)
        window_signal = Window.orderBy("timestamp").rowsBetween(-signal + 1, 0)
        
        df = df.withColumn(
            "macd_signal",
            F.mean("macd").over(window_signal)
        )
        
        # MACD histogram
        df = df.withColumn(
            "macd_hist",
            F.col("macd") - F.col("macd_signal")
        )
        
        # Drop temporary columns
        df = df.drop("ema_fast", "ema_slow")
        
        print(f"  ✓ Created macd, macd_signal, macd_hist")
        
        return df
    
    def compute_atr(self, df: DataFrame, period=14) -> DataFrame:
        """
        Compute Average True Range (ATR).
        
        ATR measures volatility using high-low range:
        - High ATR: High volatility regime
        - Low ATR: Low volatility regime
        - Rising ATR: Volatility increasing
        
        Formula:
        1. True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        2. ATR = SMA(True Range, period)
        
        Args:
            df: DataFrame with OHLC columns
            period: ATR period (default: 14)
            
        Returns:
            DataFrame with atr_{period} column
        """
        print(f"[INFO] Computing ATR (period={period})...")
        
        col_name = f"atr_{period}"
        
        # Get previous close
        window_spec = Window.orderBy("timestamp")
        df = df.withColumn(
            "prev_close",
            F.lag("price_close", 1).over(window_spec)
        )
        
        # Calculate True Range
        df = df.withColumn(
            "tr",
            F.greatest(
                F.col("price_high") - F.col("price_low"),  # High - Low
                F.abs(F.col("price_high") - F.col("prev_close")),  # |High - PrevClose|
                F.abs(F.col("price_low") - F.col("prev_close"))   # |Low - PrevClose|
            )
        )
        
        # ATR = Average of True Range
        window_spec_period = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
        
        df = df.withColumn(
            col_name,
            F.mean("tr").over(window_spec_period)
        )
        
        # Normalized ATR (as % of price)
        df = df.withColumn(
            f"{col_name}_pct",
            (F.col(col_name) / F.col("price_close")) * 100
        )
        
        # Drop temporary columns
        df = df.drop("prev_close", "tr")
        
        print(f"  ✓ Created {col_name} and {col_name}_pct")
        
        return df
    
    def compute_ema(self, df: DataFrame, periods=[10, 20, 50, 200]) -> DataFrame:
        """
        Compute Exponential Moving Averages (approximated as SMA).
        
        EMAs identify trend:
        - Price > EMA: Uptrend
        - Price < EMA: Downtrend
        - EMA crossovers: Trend changes
        
        Args:
            df: DataFrame with 'price_close' column
            periods: List of EMA periods
            
        Returns:
            DataFrame with ema_{period} and price_vs_ema_{period} columns
        """
        print(f"[INFO] Computing EMAs (periods={periods})...")
        
        for period in periods:
            ema_col = f"ema_{period}"
            ratio_col = f"price_vs_ema_{period}"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
            
            # EMA (approximated as SMA for simplicity)
            df = df.withColumn(
                ema_col,
                F.mean("price_close").over(window_spec)
            )
            
            # Price vs EMA ratio
            df = df.withColumn(
                ratio_col,
                F.col("price_close") / F.col(ema_col)
            )
            
            print(f"  ✓ Created {ema_col} and {ratio_col}")
        
        return df
    
    def compute_price_momentum(self, df: DataFrame, periods=[10, 20, 50]) -> DataFrame:
        """
        Compute price momentum (rate of change).
        
        Momentum = (Current Price / Price N periods ago) - 1
        
        Args:
            df: DataFrame with 'price_close' column
            periods: List of periods
            
        Returns:
            DataFrame with momentum_{period} columns
        """
        print(f"[INFO] Computing price momentum (periods={periods})...")
        
        window_spec = Window.orderBy("timestamp")
        
        for period in periods:
            col_name = f"momentum_{period}"
            
            # Price N periods ago
            price_lag = F.lag("price_close", period).over(window_spec)
            
            # Momentum = (current / past) - 1
            df = df.withColumn(
                col_name,
                (F.col("price_close") / price_lag) - 1
            )
            
            print(f"  ✓ Created {col_name}")
        
        return df
    
    def compute_all_technical_indicators(self, df: DataFrame) -> DataFrame:
        """
        Compute all technical indicators in one go.
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all technical indicators added
        """
        print("\n" + "="*70)
        print("COMPUTING ALL TECHNICAL INDICATORS")
        print("="*70)
        
        # 1. RSI (14-period)
        df = self.compute_rsi(df, period=14)
        
        # 2. Bollinger Bands (20-period)
        df = self.compute_bollinger_bands(df, period=20, num_std=2)
        
        # 3. MACD (12, 26, 9)
        df = self.compute_macd(df, fast=12, slow=26, signal=9)
        
        # 4. ATR (14-period)
        df = self.compute_atr(df, period=14)
        
        # 5. EMAs (10, 20, 50, 200)
        df = self.compute_ema(df, periods=[10, 20, 50, 200])
        
        # 6. Price momentum
        df = self.compute_price_momentum(df, periods=[10, 20, 50])
        
        # Count total features added
        technical_features = [
            col for col in df.columns 
            if any(indicator in col.lower() for indicator in 
                   ['rsi', 'bb_', 'macd', 'atr', 'ema', 'momentum'])
        ]
        
        print(f"\n[INFO] Added {len(technical_features)} technical indicators")
        print(f"[INFO] Features: {', '.join(technical_features[:5])}... (and {len(technical_features)-5} more)")
        
        return df


# Example usage and testing
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    
    # Create Spark session
    spark = SparkSession.builder \
        .appName("TechnicalIndicators-Test") \
        .getOrCreate()
    
    print("\n=== Testing Technical Indicators ===\n")
    
    # Load processed data
    from pathlib import Path
    base_path = Path(__file__).parent.parent.parent
    data_path = str(base_path / "data" / "processed" / "BTCUSDT" / "1m" / "historical")
    
    print(f"Loading data from: {data_path}")
    df = spark.read.parquet(data_path)
    
    print(f"Loaded {df.count():,} records")
    
    # Initialize technical indicators
    tech_calc = TechnicalIndicators()
    
    # Compute all features
    df_with_tech = tech_calc.compute_all_technical_indicators(df)
    
    # Show sample
    print("\n=== Sample Data with Technical Indicators ===")
    df_with_tech.select(
        "timestamp", "price_close",
        "rsi_14", "bb_width_20", "bb_pct_20",
        "macd", "macd_signal", "macd_hist",
        "atr_14", "atr_14_pct"
    ).orderBy(F.desc("timestamp")).show(10, truncate=False)
    
    # Summary statistics
    print("\n=== Technical Indicator Statistics ===")
    df_with_tech.select(
        "rsi_14", "bb_width_20", "macd_hist", "atr_14_pct"
    ).describe().show()
    
    spark.stop()