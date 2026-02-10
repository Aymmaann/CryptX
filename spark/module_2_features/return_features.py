from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


class ReturnFeatures:
    """Calculate return-based features."""
    
    def __init__(self, windows=[30, 60, 240]):
        # windows: Time windows in minutes [30m, 1h, 4h]
        self.windows = windows
        print(f"[INFO] Return windows: {windows}m")
    
    def compute_cumulative_returns(self, df: DataFrame) -> DataFrame:
        """
        Compute cumulative returns (momentum).
        Positive = uptrend, Negative = downtrend
        """
        print("[INFO] Computing cumulative returns...")
        
        for window in self.windows:
            col_name = f"cum_return_{window}m"
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            # Sum of log returns = total return
            df = df.withColumn(col_name, F.sum("log_return").over(window_spec))
        
        return df
    
    def compute_return_direction(self, df: DataFrame) -> DataFrame:
        """
        Compute percentage of positive returns.
        >0.5 = bullish, <0.5 = bearish
        """
        print("[INFO] Computing return direction...")
        # Binary: 1 if return positive, 0 otherwise
        df = df.withColumn(
            "is_positive",
            F.when(F.col("log_return") > 0, 1).otherwise(0)
        )
        
        for window in self.windows:
            col_name = f"bullish_ratio_{window}m"
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(col_name, F.mean("is_positive").over(window_spec))
        
        df = df.drop("is_positive")
        return df
    
    def compute_return_skewness(self, df: DataFrame) -> DataFrame:
        """
        Compute skewness of returns (asymmetry measure).
        Positive skew = more extreme positive returns (rallies)
        Negative skew = more extreme negative returns (crashes)
        """
        print("[INFO] Computing return skewness...")
        
        for window in self.windows:
            col_name = f"return_skew_{window}m"
            window_spec = Window.orderBy("timestamp").rowsBetween(-window + 1, 0)
            
            df = df.withColumn(col_name, F.skewness("log_return").over(window_spec))
        
        return df
    
    def compute_return_momentum(self, df: DataFrame, periods=[10, 30, 60]) -> DataFrame:
        """
        Compute return momentum (rate of change).
        Momentum = (Current Price / Price N periods ago) - 1
        """
        print("[INFO] Computing return momentum...")
        
        window_spec = Window.orderBy("timestamp")
        
        for period in periods:
            col_name = f"momentum_{period}m"
            price_lag = F.lag("price_close", period).over(window_spec)
            
            df = df.withColumn(
                col_name,
                F.when(price_lag > 0, (F.col("price_close") / price_lag) - 1).otherwise(0.0)
            )
        
        return df
    
    def compute_all(self, df: DataFrame) -> DataFrame:
        """Compute all return features."""
        df = self.compute_cumulative_returns(df)
        df = self.compute_return_direction(df)
        df = self.compute_return_skewness(df)
        df = self.compute_return_momentum(df)
        return df