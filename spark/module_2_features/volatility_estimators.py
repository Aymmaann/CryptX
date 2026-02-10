from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class VolatilityEstimators:
    """Calculate volatility features at different time scales."""
    
    def __init__(self, windows=[30, 60, 240]):
        self.windows = windows
        print(f"[INFO] Volatility windows: {windows}m")
    
    def compute_realized_volatility(self, df: DataFrame) -> DataFrame:
        """
        Compute rolling standard deviation of returns.
        - Low vol: Calm market
        - High vol: Volatile market
        """
        print("[INFO] Computing volatility...")
        
        for window in self.windows:
            col_name = f"vol_{window}m"
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(col_name, F.stddev("log_return").over(window_spec))
        
        return df
    
    def compute_volatility_change(self, df: DataFrame) -> DataFrame:
        # Compute change in volatility (vol momentum).
        
        window_spec = Window.orderBy("timestamp")
        # Compare current vol to vol 30 minutes ago
        df = df.withColumn(
            "vol_change_30m",
            F.col("vol_30m") - F.lag("vol_30m", 30).over(window_spec)
        )
        
        return df
    
    def compute_volatility_ratio(self, df: DataFrame) -> DataFrame:
        """
        Compute volatility ratios (short-term vs long-term).
        High ratio = short-term vol spike (regime transition)
        """
        print("[INFO] Computing volatility ratios...")
        # Short-term vol / long-term vol
        df = df.withColumn(
            "vol_ratio_30_240",
            F.when(F.col("vol_240m") > 0, F.col("vol_30m") / F.col("vol_240m")).otherwise(1.0)
        )
        
        df = df.withColumn(
            "vol_ratio_60_240",
            F.when(F.col("vol_240m") > 0, F.col("vol_60m") / F.col("vol_240m")).otherwise(1.0)
        )
        
        return df
    
    def compute_high_low_range(self, df: DataFrame, windows=[60, 240]) -> DataFrame:
        """
        Compute high-low price range (alternative volatility measure).
        """
        print("[INFO] Computing high-low range...")
        
        for window in windows:
            col_name = f"hl_range_{window}m"
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # Average of (high - low) / close for each period
            df = df.withColumn(
                "hl_ratio_temp",
                (F.col("price_high") - F.col("price_low")) / F.col("price_close")
            )
            
            df = df.withColumn(
                col_name,
                F.mean("hl_ratio_temp").over(window_spec)
            )
        
        df = df.drop("hl_ratio_temp")
        return df
    
    def compute_all(self, df: DataFrame) -> DataFrame:
        """Compute all volatility features."""
        df = self.compute_realized_volatility(df)
        df = self.compute_volatility_change(df)
        df = self.compute_volatility_ratio(df)
        df = self.compute_high_low_range(df)
        return df