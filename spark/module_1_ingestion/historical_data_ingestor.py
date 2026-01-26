"""
Module 1: Crypto Market Data Collection and Preprocessing - Historical Batch Ingestion
Handles scalable ingestion and preprocessing of historical OHLCV data from Binance CSV files.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType
from pyspark.sql.window import Window
import os
from datetime import datetime
from pathlib import Path


class HistoricalDataIngestor:
    """
    Scalable batch ingestion for historical cryptocurrency market data.
    Handles CSV files from Binance, performs validation, cleaning, and normalization.
    """
    
    def __init__(self, spark_session=None, base_path="./CryptX"):
        """
        Initialize the data ingestor.
        
        Args:
            spark_session: Optional existing Spark session
            base_path: Base directory path for the project
        """
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "data" / "raw" / "binance"
        self.processed_data_path = self.base_path / "data" / "processed"
        
        # Initialize or use existing Spark session
        self.spark = spark_session or self._create_spark_session()
        
        # Define Binance 1-minute candle schema
        self.binance_schema = StructType([
            StructField("open_time", LongType(), False),
            StructField("open", DoubleType(), False),
            StructField("high", DoubleType(), False),
            StructField("low", DoubleType(), False),
            StructField("close", DoubleType(), False),
            StructField("volume", DoubleType(), False),
            StructField("close_time", LongType(), False),
            StructField("quote_asset_volume", DoubleType(), True),
            StructField("number_of_trades", LongType(), True),
            StructField("taker_buy_base_volume", DoubleType(), True),
            StructField("taker_buy_quote_volume", DoubleType(), True),
            StructField("ignore", StringType(), True)
        ])
    
    def _create_spark_session(self):
        """Create optimized Spark session for crypto data processing."""
        return (SparkSession.builder
                .appName("CryptoVolatilityFramework-HistoricalIngestion")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.files.maxPartitionBytes", "128MB")
                .config("spark.sql.shuffle.partitions", "200")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .getOrCreate())
    
    def ingest_historical_data(self, symbol="BTCUSDT", interval="1m", 
                              start_month="2025-07", end_month="2025-12"):
        """
        Ingest historical data from multiple CSV files.
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            interval: Time interval (e.g., 1m, 5m, 1h)
            start_month: Start month in YYYY-MM format
            end_month: End month in YYYY-MM format
            
        Returns:
            Preprocessed PySpark DataFrame
        """
        print(f"[INFO] Starting historical data ingestion for {symbol} ({interval})")
        print(f"[INFO] Period: {start_month} to {end_month}")
        
        # Construct file pattern
        data_dir = self.raw_data_path / symbol / interval
        file_pattern = str(data_dir / f"{symbol}-{interval}-*.csv")
        
        print(f"[INFO] Reading CSV files from: {file_pattern}")
        
        # Read all CSV files with schema
        df_raw = (self.spark.read
                  .schema(self.binance_schema)
                  .option("header", "false")
                  .csv(file_pattern))
        
        initial_count = df_raw.count()
        print(f"[INFO] Loaded {initial_count:,} raw records")
        
        # Preprocess the data
        df_processed = self._preprocess_data(df_raw)
        
        final_count = df_processed.count()
        print(f"[INFO] After preprocessing: {final_count:,} valid records")
        print(f"[INFO] Removed {initial_count - final_count:,} invalid/duplicate records")
        
        return df_processed
    
    def _preprocess_data(self, df):
        """
        Comprehensive data preprocessing pipeline.
        
        Steps:
        1. Convert timestamps to proper datetime
        2. Remove duplicates
        3. Handle missing values
        4. Validate price/volume data
        5. Sort by timestamp
        6. Compute log returns
        7. Add metadata columns
        """
        print("[INFO] Starting data preprocessing...")
        
        # Detect timestamp format
        sample_timestamp = df.select("open_time").first()["open_time"]
        num_digits = len(str(abs(sample_timestamp)))
        
        print(f"[INFO] Sample timestamp: {sample_timestamp} ({num_digits} digits)")
        
        # Convert timestamps based on number of digits
        if num_digits >= 16:  # Microseconds (1,000,000 microseconds = 1 second)
            print("[INFO] Detected MICROSECOND timestamps (16+ digits)")
            print("[INFO] Converting from microseconds to datetime...")
            df = df.withColumn("timestamp", 
                              F.from_unixtime(F.col("open_time") / 1000000)
                              .cast(TimestampType()))
            df = df.withColumn("close_timestamp",
                              F.from_unixtime(F.col("close_time") / 1000000)
                              .cast(TimestampType()))
        elif num_digits >= 13:  # Milliseconds (standard Binance format)
            print("[INFO] Detected MILLISECOND timestamps (13-15 digits)")
            df = df.withColumn("timestamp", 
                              F.from_unixtime(F.col("open_time") / 1000)
                              .cast(TimestampType()))
            df = df.withColumn("close_timestamp",
                              F.from_unixtime(F.col("close_time") / 1000)
                              .cast(TimestampType()))
        elif num_digits >= 10:  # Seconds
            print("[INFO] Detected SECOND timestamps (10-12 digits)")
            df = df.withColumn("timestamp",
                              F.from_unixtime(F.col("open_time"))
                              .cast(TimestampType()))
            df = df.withColumn("close_timestamp",
                              F.from_unixtime(F.col("close_time"))
                              .cast(TimestampType()))
        else:
            raise ValueError(f"Unexpected timestamp format with {num_digits} digits: {sample_timestamp}")
        
        # VERIFY the conversion worked
        print("\n[VERIFICATION] Checking converted timestamps:")
        df.select(
            "open_time", 
            "timestamp", 
            F.year("timestamp").alias("year"),
            F.month("timestamp").alias("month"),
            F.dayofmonth("timestamp").alias("day")
        ).show(5, truncate=False)
        
        # Remove duplicates based on open_time
        df = df.dropDuplicates(["open_time"])
        
        # Validate OHLC relationships and prices
        df = df.filter(
            (F.col("open") > 0) &
            (F.col("high") > 0) &
            (F.col("low") > 0) &
            (F.col("close") > 0) &
            (F.col("volume") >= 0) &
            (F.col("high") >= F.col("low")) &
            (F.col("high") >= F.col("open")) &
            (F.col("high") >= F.col("close")) &
            (F.col("low") <= F.col("open")) &
            (F.col("low") <= F.col("close"))
        )
        
        # Sort by timestamp
        df = df.orderBy("timestamp")
        
        # Compute log returns
        window_spec = Window.orderBy("timestamp")
        df = df.withColumn("prev_close", F.lag("close", 1).over(window_spec))
        df = df.withColumn("log_return", 
                          F.log(F.col("close") / F.col("prev_close")))
        
        # Compute simple returns as well
        df = df.withColumn("simple_return",
                          (F.col("close") - F.col("prev_close")) / F.col("prev_close"))
        
        # Add temporal features for reference
        df = df.withColumn("year", F.year("timestamp"))
        df = df.withColumn("month", F.month("timestamp"))
        df = df.withColumn("day", F.dayofmonth("timestamp"))
        df = df.withColumn("hour", F.hour("timestamp"))
        df = df.withColumn("minute", F.minute("timestamp"))
        df = df.withColumn("day_of_week", F.dayofweek("timestamp"))
        
        # FINAL VERIFICATION: Check partition columns
        print("\n[VERIFICATION] Checking year/month values:")
        year_stats = df.select(
            F.min("year").alias("min_year"), 
            F.max("year").alias("max_year"),
            F.countDistinct("year").alias("unique_years")
        ).collect()[0]
        
        print(f"  Year range: {year_stats['min_year']} to {year_stats['max_year']}")
        print(f"  Unique years: {year_stats['unique_years']}")
        
        if year_stats['min_year'] < 2020 or year_stats['max_year'] > 2030:
            print(f"[WARNING] Unexpected year range! Expected 2025, got {year_stats['min_year']}-{year_stats['max_year']}")
        else:
            print("[SUCCESS] Year values look correct! ✓")
        
        # Select and rename columns for clarity
        df = df.select(
            F.col("timestamp"),
            F.col("open").alias("price_open"),
            F.col("high").alias("price_high"),
            F.col("low").alias("price_low"),
            F.col("close").alias("price_close"),
            F.col("volume").alias("volume_base"),
            F.col("quote_asset_volume").alias("volume_quote"),
            F.col("number_of_trades").alias("num_trades"),
            F.col("taker_buy_base_volume").alias("taker_buy_base_vol"),
            F.col("taker_buy_quote_volume").alias("taker_buy_quote_vol"),
            F.col("log_return"),
            F.col("simple_return"),
            F.col("year"),
            F.col("month"),
            F.col("day"),
            F.col("hour"),
            F.col("minute"),
            F.col("day_of_week")
        )
        
        print("[INFO] Preprocessing completed successfully")
        return df
    
    def save_processed_data(self, df, symbol="BTCUSDT", interval="1m", 
                       format="parquet", partition_by=None, mode="append"):
        """
        Save processed data in efficient Parquet format.
        
        Args:
            mode: "append" (default) or "overwrite"
                append = add to existing data (safe)
                overwrite = replace all data (destructive)
        """
        output_path = self.processed_data_path / symbol / interval / "historical"
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Saving processed data to: {output_path}")
        print(f"[INFO] Mode: {mode}")
        
        writer = df.write.mode(mode)
        
        if partition_by:
            print(f"[INFO] Partitioning by: {partition_by}")
            writer = writer.partitionBy(*partition_by)
        
        if format == "parquet":
            writer.parquet(str(output_path))
        elif format == "delta":
            writer.format("delta").save(str(output_path))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"[INFO] Data saved successfully in {format} format (mode: {mode})")
    

    def _save_metadata(self, df, symbol, interval, output_path):
        """Save ingestion metadata for tracking."""
        metadata = {
            "symbol": symbol,
            "interval": interval,
            "record_count": df.count(),
            "start_timestamp": df.agg(F.min("timestamp")).collect()[0][0],
            "end_timestamp": df.agg(F.max("timestamp")).collect()[0][0],
            "ingestion_timestamp": datetime.now().isoformat(),
            "columns": df.columns,
            "output_path": str(output_path)
        }
        
        # Save as JSON
        import json
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"[INFO] Metadata saved to: {metadata_path}")
    
    def validate_data_quality(self, df):
        """
        Comprehensive data quality validation.
        
        Returns:
            Dictionary with quality metrics
        """
        print("[INFO] Running data quality validation...")
        
        total_records = df.count()
        
        # Check for missing values in critical columns
        missing_counts = {}
        for col in ["price_close", "volume_base", "log_return"]:
            missing = df.filter(F.col(col).isNull()).count()
            missing_counts[col] = missing
        
        # Check for temporal gaps (should be continuous 1-minute data)
        time_diffs = df.select(
            (F.col("timestamp").cast("long") - 
             F.lag("timestamp", 1).over(Window.orderBy("timestamp")).cast("long"))
            .alias("time_diff_seconds")
        ).filter(F.col("time_diff_seconds").isNotNull())
        
        # Expected: 60 seconds for 1-minute data
        gaps = time_diffs.filter(F.col("time_diff_seconds") > 60).count()
        
        # Basic statistics
        stats = df.select(
            F.mean("log_return").alias("mean_return"),
            F.stddev("log_return").alias("std_return"),
            F.min("price_close").alias("min_price"),
            F.max("price_close").alias("max_price")
        ).collect()[0]
        
        quality_report = {
            "total_records": total_records,
            "missing_values": missing_counts,
            "temporal_gaps": gaps,
            "mean_log_return": float(stats["mean_return"]) if stats["mean_return"] else None,
            "std_log_return": float(stats["std_return"]) if stats["std_return"] else None,
            "min_price": float(stats["min_price"]),
            "max_price": float(stats["max_price"])
        }
        
        print(f"[INFO] Data Quality Report:")
        print(f"  Total Records: {quality_report['total_records']:,}")
        print(f"  Temporal Gaps: {quality_report['temporal_gaps']}")
        print(f"  Mean Log Return: {quality_report['mean_log_return']:.6f}" if quality_report['mean_log_return'] else "  Mean Log Return: N/A")
        print(f"  Std Log Return: {quality_report['std_log_return']:.6f}" if quality_report['std_log_return'] else "  Std Log Return: N/A")
        
        return quality_report
    
    def show_sample_data(self, df, n=10):
        """Display sample of processed data."""
        print(f"\n[INFO] Sample of processed data (first {n} rows):")
        df.select("timestamp", "price_close", "volume_base", "log_return").show(n, truncate=False)
    
    def stop(self):
        """Stop the Spark session."""
        self.spark.stop()
        print("[INFO] Spark session stopped")


# Main execution example
if __name__ == "__main__":
    # Initialize the ingestor
    ingestor = HistoricalDataIngestor(base_path="./CryptX")
    
    # Ingest historical data
    df_processed = ingestor.ingest_historical_data(
        symbol="BTCUSDT",
        interval="1m",
        start_month="2025-07",
        end_month="2025-12"
    )
    
    # Validate data quality
    quality_report = ingestor.validate_data_quality(df_processed)
    
    # Show sample
    ingestor.show_sample_data(df_processed, n=10)
    
    # Save processed data (partitioned by year and month for efficient access)
    ingestor.save_processed_data(
        df_processed,
        symbol="BTCUSDT",
        interval="1m",
        format="parquet",
        partition_by=["year", "month"]
    )
    
    # Clean up
    ingestor.stop()