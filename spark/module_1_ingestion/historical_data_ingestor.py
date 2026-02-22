from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType
from pyspark.sql.window import Window
from pathlib import Path


class HistoricalDataIngestor:
    """Simplified batch ingestion for historical cryptocurrency market data."""
    
    def __init__(self, base_path="./CryptX"):
        """Initialize the data ingestor."""
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "data" / "raw" / "binance"
        self.processed_data_path = self.base_path / "data" / "processed"
        
        # Create Spark session
        self.spark = (SparkSession.builder
                     .appName("CryptoDataIngestion")
                     .config("spark.driver.memory", "4g")
                     .getOrCreate())
        
        # Define Binance CSV schema
        self.schema = StructType([
            StructField("open_time", LongType(), False),
            StructField("open", DoubleType(), False),
            StructField("high", DoubleType(), False),
            StructField("low", DoubleType(), False),
            StructField("close", DoubleType(), False),
            StructField("volume", DoubleType(), False),
            StructField("close_time", LongType(), False),
            StructField("quote_volume", DoubleType(), True),
            StructField("trades", LongType(), True),
            StructField("taker_buy_base", DoubleType(), True),
            StructField("taker_buy_quote", DoubleType(), True),
            StructField("ignore", StringType(), True)
        ])
    
    def ingest_data(self, symbol="BTCUSDT", interval="1m"):
        """
        Ingest and preprocess historical data.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            interval: Time interval (e.g., 1m)
            
        Returns:
            Processed DataFrame
        """
        print(f"[INFO] Loading data for {symbol} ({interval})...")
        
        # Read CSV files
        data_dir = self.raw_data_path / symbol / interval
        csv_pattern = str(data_dir / f"{symbol}-{interval}-*.csv")
        
        df = (self.spark.read
              .schema(self.schema)
              .option("header", "false")
              .csv(csv_pattern))
        
        print(f"[INFO] Loaded {df.count():,} records")
        
        # Preprocess
        df_clean = self._preprocess(df)
        
        print(f"[INFO] After cleaning: {df_clean.count():,} records")
        return df_clean
    
    def _preprocess(self, df):
        """Clean and transform data."""
        
        # Convert millisecond timestamps to datetime
        df = df.withColumn(
            "timestamp",
            F.when(
                F.col("open_time") > 1_000_000_000_000_000,  # microseconds
                F.from_unixtime(F.col("open_time") / 1_000_000)
            ).when(
                F.col("open_time") > 1_000_000_000_000,      # milliseconds
                F.from_unixtime(F.col("open_time") / 1_000)
            ).otherwise(
                F.from_unixtime(F.col("open_time"))           # seconds
            ).cast("timestamp")
        )
        
        # Remove duplicates
        df = df.dropDuplicates(["open_time"])
        
        # Filter invalid data
        df = df.filter(
            (F.col("open") > 0) &
            (F.col("close") > 0) &
            (F.col("high") >= F.col("low")) &
            (F.col("high") >= F.col("close")) &
            (F.col("low") <= F.col("close"))
        )
        
        # Sort by time
        df = df.orderBy("timestamp")
        
        # Calculate log returns
        window = Window.orderBy("timestamp")
        df = df.withColumn("prev_close", F.lag("close", 1).over(window))
        df = df.withColumn("log_return", F.log(F.col("close") / F.col("prev_close")))
        
        # Add date columns for partitioning
        df = df.withColumn("year", F.year("timestamp"))
        df = df.withColumn("month", F.month("timestamp"))
        df = df.withColumn("day", F.dayofmonth("timestamp"))
        df = df.withColumn("hour", F.hour("timestamp"))
        
        # Select final columns
        df = df.select(
            "timestamp",
            F.col("open").alias("price_open"),
            F.col("high").alias("price_high"),
            F.col("low").alias("price_low"),
            F.col("close").alias("price_close"),
            F.col("volume").alias("volume_base"),
            F.col("quote_volume").alias("volume_quote"),
            F.col("trades").alias("num_trades"),
            "log_return",
            "year", "month", "day", "hour"
        )
        
        return df
    
    def save_data(self, df, symbol="BTCUSDT", interval="1m"):
        """
        Save processed data as Parquet.
        Uses OVERWRITE mode to create clean, consolidated files.
        """
        output_path = self.processed_data_path / symbol / interval
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Saving to {output_path}")
        
        # Use overwrite mode and coalesce to create fewer files
        df.coalesce(12).write \
            .mode("overwrite") \
            .partitionBy("year", "month") \
            .parquet(str(output_path))
        
        print("[INFO] Data saved successfully")
    
    def show_sample(self, df, n=10):
        """Display sample data."""
        print(f"\n[INFO] Sample data (first {n} rows):")
        df.select("timestamp", "price_close", "volume_base", "log_return").show(n)
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()


# Usage example
if __name__ == "__main__":
    ingestor = HistoricalDataIngestor(base_path=".")
    
    # Load and process data
    df = ingestor.ingest_data(symbol="BTCUSDT", interval="1m")
    
    # Show sample
    ingestor.show_sample(df)
    
    # Save
    ingestor.save_data(df, symbol="BTCUSDT", interval="1m")
    
    ingestor.stop()