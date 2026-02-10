from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class TechnicalIndicators:
    """Calculate technical indicators."""
    def __init__(self):
        print("[INFO] Technical indicators initialized")
    
    def compute_rsi(self, df: DataFrame, period=14) -> DataFrame:
        """
        Compute RSI (Relative Strength Index).
        RSI ranges 0-100:
        - >70 = Overbought
        - <30 = Oversold
        - 40-60 = Neutral
        """
        print(f"[INFO] Computing RSI (period={period})...")
        
        window_spec = Window.orderBy("timestamp")
        
        # Price changes
        df = df.withColumn(
            "price_change",
            F.col("price_close") - F.lag("price_close", 1).over(window_spec)
        )
        
        # Gains and losses
        df = df.withColumn("gain", F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0))
        df = df.withColumn("loss", F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0))
        
        # Average gain and loss
        window_period = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
        df = df.withColumn("avg_gain", F.mean("gain").over(window_period))
        df = df.withColumn("avg_loss", F.mean("loss").over(window_period))
        
        # RSI calculation
        df = df.withColumn(
            "rs",
            F.when(F.col("avg_loss") > 0, F.col("avg_gain") / F.col("avg_loss")).otherwise(100)
        )
        df = df.withColumn("rsi", 100 - (100 / (1 + F.col("rs"))))
        
        # Cleanup
        df = df.drop("price_change", "gain", "loss", "avg_gain", "avg_loss", "rs")
        
        return df
    
    def compute_moving_averages(self, df: DataFrame, periods=[20, 50]) -> DataFrame:
        """
        Compute simple moving averages.
        Used to identify trends:
        - Price > MA = Uptrend
        - Price < MA = Downtrend
        """
        print(f"[INFO] Computing moving averages (periods={periods})...")
        
        for period in periods:
            ma_col = f"ma_{period}"
            ratio_col = f"price_vs_ma_{period}"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
            
            # Moving average
            df = df.withColumn(ma_col, F.mean("price_close").over(window_spec))
            
            # Price vs MA ratio
            df = df.withColumn(ratio_col, F.col("price_close") / F.col(ma_col))
        
        return df
    
    def compute_bollinger_bands(self, df: DataFrame, period=20, num_std=2) -> DataFrame:
        """
        Compute Bollinger Bands.
        Measures volatility and price extremes:
        - Wide bands = high volatility
        - Narrow bands = low volatility
        - Price at upper band = overbought
        - Price at lower band = oversold
        """
        print(f"[INFO] Computing Bollinger Bands (period={period})...")
        
        window_spec = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
        
        # Middle band (SMA)
        df = df.withColumn("bb_middle", F.mean("price_close").over(window_spec))
        
        # Standard deviation
        df = df.withColumn("bb_std", F.stddev("price_close").over(window_spec))
        
        # Upper and lower bands
        df = df.withColumn("bb_upper", F.col("bb_middle") + (F.col("bb_std") * num_std))
        df = df.withColumn("bb_lower", F.col("bb_middle") - (F.col("bb_std") * num_std))
        
        # Bandwidth (volatility indicator)
        df = df.withColumn(
            "bb_width",
            (F.col("bb_upper") - F.col("bb_lower")) / F.col("bb_middle")
        )
        
        # %B (position within bands: 0=lower, 1=upper)
        df = df.withColumn(
            "bb_position",
            (F.col("price_close") - F.col("bb_lower")) / 
            (F.col("bb_upper") - F.col("bb_lower"))
        )
        
        # Drop temporary columns (keep only width and position)
        df = df.drop("bb_middle", "bb_std", "bb_upper", "bb_lower")
        
        return df
    
    def compute_atr(self, df: DataFrame, period=14) -> DataFrame:
        """
        Compute Average True Range (ATR).
        Measures volatility using high-low range:
        - High ATR = high volatility
        - Low ATR = low volatility
        """
        print(f"[INFO] Computing ATR (period={period})...")
        
        window_spec = Window.orderBy("timestamp")
        
        # Previous close
        df = df.withColumn("prev_close", F.lag("price_close", 1).over(window_spec))
        
        # True Range = max(H-L, |H-PrevClose|, |L-PrevClose|)
        df = df.withColumn(
            "true_range",
            F.greatest(
                F.col("price_high") - F.col("price_low"),
                F.abs(F.col("price_high") - F.col("prev_close")),
                F.abs(F.col("price_low") - F.col("prev_close"))
            )
        )
        
        # ATR = average of true range
        window_period = Window.orderBy("timestamp").rowsBetween(-period + 1, 0)
        df = df.withColumn("atr", F.mean("true_range").over(window_period))
        
        # ATR as percentage of price (normalized)
        df = df.withColumn("atr_pct", (F.col("atr") / F.col("price_close")) * 100)
        
        # Cleanup
        df = df.drop("prev_close", "true_range")
        
        return df
    
    def compute_all(self, df: DataFrame) -> DataFrame:
        """Compute all technical indicators."""
        df = self.compute_rsi(df, period=14)
        df = self.compute_moving_averages(df, periods=[20, 50])
        df = self.compute_bollinger_bands(df, period=20)
        df = self.compute_atr(df, period=14)
        return df