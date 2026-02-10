from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class VolumeFeatures:
    """Calculate volume-based features."""
    def __init__(self, windows=[30, 60, 240]):
        self.windows = windows
        print(f"[INFO] Volume windows: {windows}m")
    
    def compute_volume_averages(self, df: DataFrame) -> DataFrame:
        """Compute rolling volume averages."""
        print("[INFO] Computing volume averages...")
        
        for window in self.windows:
            col_name = f"vol_avg_{window}m"
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(col_name, F.mean("volume_base").over(window_spec))
        
        return df
    
    def compute_volume_spikes(self, df: DataFrame) -> DataFrame:
        """
        Compute volume spikes (current vol / average vol).
        >2.0 = unusual high volume (2x normal)
        """
        print("[INFO] Computing volume spikes...")
        
        # Use 60m average as baseline
        df = df.withColumn(
            "volume_spike",
            F.when(F.col("vol_avg_60m") > 0, 
                   F.col("volume_base") / F.col("vol_avg_60m")
            ).otherwise(1.0)
        )
        
        return df
    
    def compute_vwap(self, df: DataFrame, windows=[60, 240]) -> DataFrame:
        """
        Compute Volume-Weighted Average Price (VWAP).
        VWAP = sum(price * volume) / sum(volume)
        Shows if current price is above/below average transaction price.
        """
        print("[INFO] Computing VWAP...")
        # Price * volume for each candle
        df = df.withColumn("pv", F.col("price_close") * F.col("volume_base"))
        
        for window in windows:
            vwap_col = f"vwap_{window}m"
            ratio_col = f"price_vs_vwap_{window}m"
            
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            # VWAP calculation
            df = df.withColumn(
                vwap_col,
                F.sum("pv").over(window_spec) / F.sum("volume_base").over(window_spec)
            )
            
            # Price vs VWAP ratio
            df = df.withColumn(
                ratio_col,
                F.col("price_close") / F.col(vwap_col)
            )
        
        df = df.drop("pv")
        return df
    
    def compute_volume_momentum(self, df: DataFrame, windows=[60, 240]) -> DataFrame:
        """
        Compute rate of change in volume.
        Positive = volume increasing (growing interest)
        Negative = volume decreasing (fading interest)
        """
        print("[INFO] Computing volume momentum...")
        
        window_spec = Window.orderBy("timestamp")
        
        for window in windows:
            col_name = f"volume_momentum_{window}m"
            volume_lag = F.lag("volume_base", window).over(window_spec)
            
            df = df.withColumn(
                col_name,
                F.when(volume_lag > 0, 
                       (F.col("volume_base") - volume_lag) / volume_lag
                ).otherwise(0.0)
            )
        
        return df
    
    def compute_volume_volatility(self, df: DataFrame, windows=[60, 240]) -> DataFrame:
        """
        Compute volatility of volume (standard deviation).
        High volume volatility = unstable participation
        """
        print("[INFO] Computing volume volatility...")
        
        for window in windows:
            col_name = f"volume_vol_{window}m"
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(col_name, F.stddev("volume_base").over(window_spec))
        
        return df
    
    def compute_all(self, df: DataFrame) -> DataFrame:
        """Compute all volume features."""
        df = self.compute_volume_averages(df)
        df = self.compute_volume_spikes(df)
        df = self.compute_vwap(df)
        df = self.compute_volume_momentum(df)
        df = self.compute_volume_volatility(df)
        return df