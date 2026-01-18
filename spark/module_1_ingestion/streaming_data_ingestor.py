"""
Module 1 (Extended): Crypto Market Data Collection - Streaming/Near Real-Time Ingestion
Handles near real-time data ingestion from Binance API/Websocket for continuous volatility analysis.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, TimestampType
import requests
import json
import websocket
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Queue


class StreamingDataIngestor:
    """
    Near real-time data ingestion for cryptocurrency market data.
    Supports both REST API polling and WebSocket streaming from Binance.
    """
    
    def __init__(self, spark_session=None, base_path="./CryptX"):
        """
        Initialize streaming data ingestor.
        
        Args:
            spark_session: Optional existing Spark session
            base_path: Base directory path for the project
        """
        self.base_path = Path(base_path)
        self.processed_data_path = self.base_path / "data" / "processed"
        self.checkpoint_path = self.base_path / "data" / "checkpoints"
        
        self.spark = spark_session or self._create_spark_session()
        
        # Binance API endpoints
        self.api_base_url = "https://api.binance.com"
        self.ws_base_url = "wss://stream.binance.com:9443"
        
        # Internal buffer for websocket data
        self.data_buffer = Queue(maxsize=10000)
        self.ws_thread = None
        self.ws_connection = None
        self.is_running = False
        
    def _create_spark_session(self):
        """Create Spark session optimized for streaming."""
        return (SparkSession.builder
                .appName("CryptoVolatilityFramework-StreamingIngestion")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
                .config("spark.sql.shuffle.partitions", "100")
                .getOrCreate())
    
    def fetch_latest_klines_rest(self, symbol="BTCUSDT", interval="1m", limit=100, start_time=None, end_time=None):
        """
        Fetch candles using Binance REST API.
        Useful for bootstrapping or filling gaps.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval (1m, 5m, etc.)
            limit: Number of candles to fetch (max 1000)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            
        Returns:
            List of candle data dictionaries
        """
        endpoint = f"{self.api_base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        # Add time range if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            klines = response.json()
            
            # Parse response into structured format
            parsed_data = []
            for kline in klines:
                parsed_data.append({
                    "open_time": int(kline[0]),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "close_time": int(kline[6]),
                    "quote_asset_volume": float(kline[7]),
                    "number_of_trades": int(kline[8]),
                    "taker_buy_base_volume": float(kline[9]),
                    "taker_buy_quote_volume": float(kline[10])
                })
            
            print(f"[INFO] Fetched {len(parsed_data)} candles via REST API")
            return parsed_data
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch klines: {e}")
            return []
    
    def start_websocket_stream(self, symbol="BTCUSDT", interval="1m"):
        """
        Start WebSocket connection for real-time candle updates.
        
        Args:
            symbol: Trading pair symbol (lowercase for websocket)
            interval: Candle interval
        """
        if self.is_running:
            print("[WARNING] WebSocket stream already running")
            return
        
        symbol_lower = symbol.lower()
        stream_name = f"{symbol_lower}@kline_{interval}"
        ws_url = f"{self.ws_base_url}/ws/{stream_name}"
        
        print(f"[INFO] Starting WebSocket stream: {ws_url}")
        
        self.is_running = True
        self.ws_thread = threading.Thread(
            target=self._websocket_worker,
            args=(ws_url,),
            daemon=True
        )
        self.ws_thread.start()
        
        print("[INFO] WebSocket stream started successfully")
    
    def _websocket_worker(self, ws_url):
        """Background worker for WebSocket connection."""
        import ssl
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'k' in data:  # Kline data
                    kline = data['k']
                    
                    # Only process closed candles for consistency
                    if kline['x']:  # is_closed
                        candle_data = {
                            "open_time": int(kline['t']),
                            "open": float(kline['o']),
                            "high": float(kline['h']),
                            "low": float(kline['l']),
                            "close": float(kline['c']),
                            "volume": float(kline['v']),
                            "close_time": int(kline['T']),
                            "quote_asset_volume": float(kline['q']),
                            "number_of_trades": int(kline['n']),
                            "taker_buy_base_volume": float(kline['V']),
                            "taker_buy_quote_volume": float(kline['Q']),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        if not self.data_buffer.full():
                            self.data_buffer.put(candle_data)
                        else:
                            print("[WARNING] Data buffer full, dropping candle")
                            
            except Exception as e:
                print(f"[ERROR] Error processing WebSocket message: {e}")
        
        def on_error(ws, error):
            print(f"[ERROR] WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print(f"[INFO] WebSocket connection closed")
        
        def on_open(ws):
            print(f"[INFO] WebSocket connection opened")
        
        # Create WebSocket connection
        self.ws_connection = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Run forever with SSL context to avoid certificate errors
        # This is safe for public Binance API
        self.ws_connection.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    
    def stop_websocket_stream(self):
        """Stop the WebSocket stream gracefully."""
        if not self.is_running:
            print("[WARNING] No active WebSocket stream to stop")
            return
        
        print("[INFO] Stopping WebSocket stream...")
        self.is_running = False
        
        if self.ws_connection:
            self.ws_connection.close()
        
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        
        print("[INFO] WebSocket stream stopped")
    
    def process_buffered_data(self, batch_size=100):
        """
        Process data from buffer and convert to Spark DataFrame.
        
        Args:
            batch_size: Number of records to process at once
            
        Returns:
            Processed Spark DataFrame
        """
        if self.data_buffer.empty():
            print("[INFO] No data in buffer to process")
            return None
        
        # Collect data from buffer
        batch_data = []
        count = 0
        while not self.data_buffer.empty() and count < batch_size:
            batch_data.append(self.data_buffer.get())
            count += 1
        
        if not batch_data:
            return None
        
        print(f"[INFO] Processing {len(batch_data)} candles from buffer")
        
        # Convert to Spark DataFrame
        df_raw = self.spark.createDataFrame(batch_data)
        
        # Apply preprocessing (similar to historical)
        df_processed = self._preprocess_streaming_data(df_raw)
        
        return df_processed
    
    def _preprocess_streaming_data(self, df):
        """
        Preprocess streaming data with same logic as historical.
        FIXED: Proper timestamp detection and conversion.
        """
        print("[INFO] Preprocessing streaming data...")
        
        # Detect timestamp format (Binance API returns milliseconds - 13 digits)
        sample_timestamp = df.select("open_time").first()["open_time"]
        num_digits = len(str(abs(sample_timestamp)))
        
        print(f"[INFO] Sample timestamp: {sample_timestamp} ({num_digits} digits)")
        
        # Convert timestamps based on number of digits
        if num_digits >= 16:  # Microseconds
            print("[INFO] Detected MICROSECOND timestamps")
            df = df.withColumn("candle_timestamp",
                              F.from_unixtime(F.col("open_time") / 1000000)
                              .cast(TimestampType()))
        elif num_digits >= 13:  # Milliseconds (standard Binance API format)
            print("[INFO] Detected MILLISECOND timestamps (Binance standard)")
            df = df.withColumn("candle_timestamp",
                              F.from_unixtime(F.col("open_time") / 1000)
                              .cast(TimestampType()))
        elif num_digits >= 10:  # Seconds
            print("[INFO] Detected SECOND timestamps")
            df = df.withColumn("candle_timestamp",
                              F.from_unixtime(F.col("open_time"))
                              .cast(TimestampType()))
        else:
            raise ValueError(f"Unexpected timestamp format: {sample_timestamp}")
        
        # Validate OHLCV
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
        
        # Add temporal features - MUST match historical data columns
        df = df.withColumn("year", F.year("candle_timestamp"))
        df = df.withColumn("month", F.month("candle_timestamp"))
        df = df.withColumn("day", F.dayofmonth("candle_timestamp"))
        df = df.withColumn("hour", F.hour("candle_timestamp"))
        df = df.withColumn("minute", F.minute("candle_timestamp"))
        df = df.withColumn("day_of_week", F.dayofweek("candle_timestamp"))
        
        # Rename columns for consistency with historical data
        df = df.select(
            F.col("candle_timestamp").alias("timestamp"),
            F.col("open").alias("price_open"),
            F.col("high").alias("price_high"),
            F.col("low").alias("price_low"),
            F.col("close").alias("price_close"),
            F.col("volume").alias("volume_base"),
            F.col("quote_asset_volume").alias("volume_quote"),
            F.col("number_of_trades").alias("num_trades"),
            F.col("taker_buy_base_volume").alias("taker_buy_base_vol"),
            F.col("taker_buy_quote_volume").alias("taker_buy_quote_vol"),
            # Add null columns for consistency (streaming doesn't calculate returns)
            F.lit(None).cast("double").alias("log_return"),
            F.lit(None).cast("double").alias("simple_return"),
            F.col("year"),
            F.col("month"),
            F.col("day"),
            F.col("hour"),
            F.col("minute"),
            F.col("day_of_week")
        )
        
        print(f"[INFO] Processed {df.count()} streaming records")
        return df
    
    def append_to_historical(self, df_streaming, symbol="BTCUSDT", interval="1m"):
        """
        Append streaming data to historical Parquet files.
        
        Args:
            df_streaming: Processed streaming DataFrame
            symbol: Trading pair symbol
            interval: Time interval
        """
        if df_streaming is None or df_streaming.count() == 0:
            print("[INFO] No data to append")
            return
        
        output_path = self.processed_data_path / symbol / interval / "historical"
        
        print(f"[INFO] Appending {df_streaming.count()} records to {output_path}")
        
        # Append mode - add to existing partitioned data
        df_streaming.write \
            .mode("append") \
            .partitionBy("year", "month") \
            .parquet(str(output_path))
        
        print("[INFO] Data appended successfully")
    
    def continuous_ingestion_loop(self, symbol="BTCUSDT", interval="1m", 
                                  process_interval=60, duration=3600):
        """
        Run continuous ingestion loop for specified duration.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            process_interval: Seconds between processing batches
            duration: Total duration to run (seconds)
        """
        print(f"[INFO] Starting continuous ingestion for {duration} seconds")
        print(f"[INFO] Processing batches every {process_interval} seconds")
        
        # Start WebSocket stream
        self.start_websocket_stream(symbol, interval)
        
        start_time = time.time()
        iteration = 0
        
        try:
            while (time.time() - start_time) < duration:
                time.sleep(process_interval)
                
                iteration += 1
                print(f"\n[INFO] Processing batch #{iteration}")
                
                # Process buffered data
                df_batch = self.process_buffered_data(batch_size=1000)
                
                if df_batch is not None:
                    # Append to historical data
                    self.append_to_historical(df_batch, symbol, interval)
                    
                    print(f"[INFO] Buffer size: {self.data_buffer.qsize()}")
                
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        finally:
            # Clean shutdown
            self.stop_websocket_stream()
            print(f"[INFO] Continuous ingestion completed after {iteration} iterations")
    

    def backfill_missing_data(self, symbol="BTCUSDT", interval="1m", 
                         start_date="2026-01-01", end_date="2026-01-17"):
        """
        Fetch historical data to fill gaps.
        Uses Binance REST API (max 1000 candles per request).
        
        Args:
            symbol: Trading pair
            interval: Candle interval
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            
        Returns:
            Processed Spark DataFrame ready to append
        """
        from datetime import datetime
        import time
        
        print(f"[INFO] Backfilling data from {start_date} to {end_date}")
        
        # Convert to millisecond timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        
        all_data = []
        current_ts = start_ts
        request_count = 0
        
        while current_ts < end_ts:
            print(f"[INFO] Fetching batch {request_count + 1} (from {datetime.fromtimestamp(current_ts/1000)})")
            
            # Fetch 1000 candles at a time (Binance limit)
            batch = self.fetch_latest_klines_rest(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_ts,
                end_time=end_ts
            )
            
            if not batch:
                print("[WARNING] No more data available")
                break
            
            all_data.extend(batch)
            request_count += 1
            
            # Move to next batch (use last candle's close_time + 1)
            current_ts = batch[-1]['close_time'] + 1
            
            # Rate limiting (Binance allows ~1200 requests/min)
            time.sleep(0.1)  # Small delay to avoid rate limits
            
            print(f"[INFO] Total records collected: {len(all_data)}")
        
        print(f"[INFO] Backfill complete! Collected {len(all_data)} candles in {request_count} requests")
        
        if not all_data:
            return None
        
        # Convert to Spark DataFrame
        df_raw = self.spark.createDataFrame(all_data)
        
        # Preprocess using same logic as streaming
        df_processed = self._preprocess_streaming_data(df_raw)
        
        return df_processed
    
    def stop(self):
        """Clean shutdown."""
        self.stop_websocket_stream()
        self.spark.stop()
        print("[INFO] Streaming ingestor stopped")


# Main execution example
if __name__ == "__main__":
    # Initialize streaming ingestor
    ingestor = StreamingDataIngestor(base_path="./CryptX")
    
    # Example 1: Fetch latest candles via REST API
    print("\n=== Fetching latest candles via REST API ===")
    latest_candles = ingestor.fetch_latest_klines_rest(
        symbol="BTCUSDT",
        interval="1m",
        limit=10
    )
    
    if latest_candles:
        df = ingestor.spark.createDataFrame(latest_candles)
        df_processed = ingestor._preprocess_streaming_data(df)
        df_processed.show(5, truncate=False)
    
    # Example 2: WebSocket streaming (run for 5 minutes)
    print("\n=== Starting WebSocket streaming (5 minutes) ===")
    ingestor.continuous_ingestion_loop(
        symbol="BTCUSDT",
        interval="1m",
        process_interval=60,  # Process every minute
        duration=300  # Run for 5 minutes
    )
    
    # Clean up
    ingestor.stop()