from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import requests
from datetime import datetime
from pathlib import Path

class StreamingDataIngestor:
    """Simplified streaming data ingestion from Binance REST API."""
    def __init__(self, base_path="./CryptX"):
        """Initialize streaming ingestor."""
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / "data" / "processed"
        
        self.spark = (SparkSession.builder
                     .appName("CryptoStreamingIngestion")
                     .config("spark.driver.memory", "4g")
                     .getOrCreate())
        
        self.api_url = "https://api.binance.com/api/v3/klines"
    
    def fetch_recent_data(self, symbol="BTCUSDT", interval="1m", limit=100):
        """
        Fetch recent candles from Binance API.
        Returns:
            List of candle dictionaries
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            klines = response.json()
            
            # Parse into structured format
            data = []
            for k in klines:
                data.append({
                    "open_time": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": int(k[6]),
                    "quote_volume": float(k[7]),
                    "trades": int(k[8])
                })
            
            print(f"[INFO] Fetched {len(data)} candles")
            return data
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            return []
    
    def process_data(self, raw_data):
        """Convert raw data to processed DataFrame."""
        if not raw_data:
            return None
        
        # Create DataFrame
        df = self.spark.createDataFrame(raw_data)
        
        # Convert timestamps (Binance uses milliseconds)
        df = df.withColumn("timestamp",
                          F.from_unixtime(F.col("open_time") / 1000).cast("timestamp"))
        
        # Filter valid data
        df = df.filter(
            (F.col("open") > 0) &
            (F.col("close") > 0) &
            (F.col("high") >= F.col("low"))
        )
        
        # Add date columns
        df = df.withColumn("year", F.year("timestamp"))
        df = df.withColumn("month", F.month("timestamp"))
        df = df.withColumn("day", F.dayofmonth("timestamp"))
        df = df.withColumn("hour", F.hour("timestamp"))
        
        # Select columns (match historical format)
        df = df.select(
            "timestamp",
            F.col("open").alias("price_open"),
            F.col("high").alias("price_high"),
            F.col("low").alias("price_low"),
            F.col("close").alias("price_close"),
            F.col("volume").alias("volume_base"),
            F.col("quote_volume").alias("volume_quote"),
            F.col("trades").alias("num_trades"),
            F.lit(None).cast("double").alias("log_return"),  # Calculate later
            "year", "month", "day", "hour"
        )
        
        return df
    
    def append_data(self, df, symbol="BTCUSDT", interval="1m"):
        """
        Append streaming data to historical dataset.
        NOTE: This creates multiple small files - use consolidate_data() to fix.
        """
        if df is None or df.count() == 0:
            print("[INFO] No data to append")
            return
        
        output_path = self.processed_data_path / symbol / interval
        
        print(f"[INFO] Appending {df.count()} records to {output_path}")
        
        df.write \
            .mode("append") \
            .partitionBy("year", "month") \
            .parquet(str(output_path))
        
        print("[INFO] Data appended")
    
    def consolidate_data(self, symbol="BTCUSDT", interval="1m"):
        """
        Consolidate multiple small parquet files into fewer, larger files.
        This fixes the issue of having too many small files from streaming appends.
        """
        data_path = self.processed_data_path / symbol / interval
        
        print(f"[INFO] Consolidating data in {data_path}")
        
        # Read all existing data
        df = self.spark.read.parquet(str(data_path))
        
        # Remove duplicates
        df = df.dropDuplicates(["timestamp"])
        
        # Sort by timestamp
        df = df.orderBy("timestamp")
        
        # Recalculate log returns (since we may have new data)
        from pyspark.sql.window import Window
        window = Window.orderBy("timestamp")
        df = df.withColumn("prev_close", F.lag("price_close", 1).over(window))
        df = df.withColumn("log_return", 
                          F.log(F.col("price_close") / F.col("prev_close")))
        df = df.drop("prev_close")
        
        # Overwrite with consolidated files (fewer, larger files)
        print("[INFO] Rewriting with consolidated files...")
        df.coalesce(12).write \
            .mode("overwrite") \
            .partitionBy("year", "month") \
            .parquet(str(data_path))
        
        print("[INFO] Consolidation complete!")
    
    def backfill_missing_data(self, symbol="BTCUSDT", interval="1m",
                            start_date="2026-01-01", end_date="2026-01-17"):
        """
        Fetch historical data to fill gaps.
        
        Args:
            symbol: Trading pair
            interval: Time interval
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Processed DataFrame
        """
        import time
        
        print(f"[INFO] Backfilling {start_date} to {end_date}")
        
        # Convert to millisecond timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        
        while current_ts < end_ts:
            # Fetch 1000 candles at a time
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": 1000,
                "startTime": current_ts,
                "endTime": end_ts
            }
            
            try:
                response = requests.get(self.api_url, params=params, timeout=10)
                response.raise_for_status()
                batch = response.json()
                
                if not batch:
                    break
                
                # Parse batch
                for k in batch:
                    all_data.append({
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                        "close_time": int(k[6]),
                        "quote_volume": float(k[7]),
                        "trades": int(k[8])
                    })
                
                # Move to next batch
                current_ts = int(batch[-1][6]) + 1
                
                print(f"[INFO] Collected {len(all_data)} records so far...")
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"[ERROR] {e}")
                break
        
        print(f"[INFO] Backfill complete: {len(all_data)} records")
        
        # Process and return
        return self.process_data(all_data)
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()


# Usage example
if __name__ == "__main__":
    ingestor = StreamingDataIngestor(base_path="./CryptX")
    
    # Fetch recent data
    print("\n=== Fetching recent data ===")
    raw_data = ingestor.fetch_recent_data(symbol="BTCUSDT", interval="1m", limit=10)
    
    # Process and display
    if raw_data:
        df = ingestor.process_data(raw_data)
        df.show(5)
        
    ingestor.stop()