import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from pyspark.sql import functions as F


class DataIngestionOrchestrator:
    """
    Main orchestrator for coordinating data ingestion workflows.
    Handles both historical batch processing and real-time streaming.
    """
    
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        
        # Will be initialized when needed
        self.historical_ingestor = None
        self.streaming_ingestor = None
    
    def setup_logging(self):
        """Configure logging for the pipeline."""
        log_dir = self.config.paths.LOGS_PATH
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger('DataIngestionOrchestrator')
        self.logger.info(f"Logging initialized. Log file: {log_file}")
    
    def run_historical_ingestion(self, symbol=None, interval=None, 
                                 start_month=None, end_month=None, overwrite=False):
        """
        Execute historical data ingestion pipeline.
        
        Args:
            symbol: Trading pair symbol (defaults to config)
            interval: Candle interval (defaults to config)
            start_month: Start month YYYY-MM (defaults to config)
            end_month: End month YYYY-MM (defaults to config)
            overwrite: If True, replace all data. If False, append to existing.
        """
        self.logger.info("="*70)
        self.logger.info("STARTING HISTORICAL DATA INGESTION PIPELINE")
        self.logger.info("="*70)
        
        # Use config defaults if not provided
        symbol = symbol or self.config.ingestion.DEFAULT_SYMBOL
        interval = interval or self.config.ingestion.DEFAULT_INTERVAL
        start_month = start_month or self.config.ingestion.HISTORICAL_START_MONTH
        end_month = end_month or self.config.ingestion.HISTORICAL_END_MONTH
        
        try:
            # Import here to avoid circular dependencies
            from module_1_ingestion.historical_data_ingestor import HistoricalDataIngestor
            
            # Initialize ingestor
            self.logger.info("Initializing Historical Data Ingestor...")
            self.historical_ingestor = HistoricalDataIngestor(
                base_path=str(self.config.paths.BASE_PATH)
            )
            
            # Step 1: Ingest data
            self.logger.info(f"Step 1: Ingesting data for {symbol} ({interval})")
            df_processed = self.historical_ingestor.ingest_historical_data(
                symbol=symbol,
                interval=interval,
                start_month=start_month,
                end_month=end_month
            )
            
            if df_processed is None:
                self.logger.error("Failed to ingest data")
                return False
            
            # Step 2: Validate data quality
            self.logger.info("Step 2: Validating data quality...")
            quality_report = self.historical_ingestor.validate_data_quality(df_processed)
            
            # Check if quality meets thresholds
            if quality_report['temporal_gaps'] > self.config.data_quality.MAX_ALLOWED_GAPS:
                self.logger.warning(
                    f"Data has {quality_report['temporal_gaps']} temporal gaps "
                    f"(threshold: {self.config.data_quality.MAX_ALLOWED_GAPS})"
                )
            
            # Step 3: Display sample
            self.logger.info("Step 3: Sample of processed data:")
            self.historical_ingestor.show_sample_data(df_processed, n=10)
            
            # Step 4: Save processed data
            self.logger.info("Step 4: Saving processed data...")
            
            mode = "overwrite" if overwrite else "append"
            
            if overwrite:
                self.logger.warning("OVERWRITE mode enabled - existing data will be deleted!")
            
            self.historical_ingestor.save_processed_data(
                df_processed,
                symbol=symbol,
                interval=interval,
                format=self.config.spark.DEFAULT_OUTPUT_FORMAT,
                partition_by=self.config.spark.PARTITION_COLUMNS,
                mode=mode
            )
            
            self.logger.info("="*70)
            self.logger.info("HISTORICAL DATA INGESTION COMPLETED SUCCESSFULLY")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in historical ingestion: {e}", exc_info=True)
            return False
        
        finally:
            if self.historical_ingestor:
                self.historical_ingestor.stop()
    
    def run_streaming_ingestion(self, symbol=None, interval=None, 
                                duration=3600, process_interval=60):
        """
        Execute real-time streaming ingestion pipeline.
        
        Args:
            symbol: Trading pair symbol (defaults to config)
            interval: Candle interval (defaults to config)
            duration: How long to run streaming (seconds)
            process_interval: How often to process batches (seconds)
        """
        self.logger.info("="*70)
        self.logger.info("STARTING STREAMING DATA INGESTION PIPELINE")
        self.logger.info("="*70)
        
        # Use config defaults if not provided
        symbol = symbol or self.config.ingestion.DEFAULT_SYMBOL
        interval = interval or self.config.ingestion.DEFAULT_INTERVAL
        
        try:
            # Import here to avoid circular dependencies
            from module_1_ingestion.streaming_data_ingestor import StreamingDataIngestor
            
            # Initialize ingestor
            self.logger.info("Initializing Streaming Data Ingestor...")
            self.streaming_ingestor = StreamingDataIngestor(
                base_path=str(self.config.paths.BASE_PATH)
            )
            
            # Run continuous ingestion
            self.logger.info(f"Starting continuous ingestion for {duration} seconds")
            self.logger.info(f"Symbol: {symbol}, Interval: {interval}")
            
            self.streaming_ingestor.continuous_ingestion_loop(
                symbol=symbol,
                interval=interval,
                process_interval=process_interval,
                duration=duration
            )
            
            self.logger.info("="*70)
            self.logger.info("STREAMING DATA INGESTION COMPLETED")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in streaming ingestion: {e}", exc_info=True)
            return False
        
        finally:
            if self.streaming_ingestor:
                self.streaming_ingestor.stop()
    
    def run_hybrid_ingestion(self, symbol=None, interval=None, 
                            start_month=None, end_month=None,
                            streaming_duration=3600, overwrite=False):
        """
        Run both historical and streaming ingestion sequentially.
        First processes historical data, then starts real-time streaming.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start_month: Historical start month
            end_month: Historical end month
            streaming_duration: How long to run streaming (seconds)
            overwrite: Whether to overwrite existing historical data
        """
        self.logger.info("="*70)
        self.logger.info("STARTING HYBRID INGESTION PIPELINE")
        self.logger.info("="*70)
        
        # Phase 1: Historical ingestion
        self.logger.info("\n*** PHASE 1: HISTORICAL DATA INGESTION ***\n")
        historical_success = self.run_historical_ingestion(
            symbol=symbol,
            interval=interval,
            start_month=start_month,
            end_month=end_month,
            overwrite=overwrite
        )
        
        if not historical_success:
            self.logger.error("Historical ingestion failed. Aborting hybrid pipeline.")
            return False
        
        # Phase 2: Streaming ingestion
        self.logger.info("\n*** PHASE 2: STREAMING DATA INGESTION ***\n")
        streaming_success = self.run_streaming_ingestion(
            symbol=symbol,
            interval=interval,
            duration=streaming_duration
        )
        
        self.logger.info("="*70)
        self.logger.info("HYBRID INGESTION PIPELINE COMPLETED")
        self.logger.info("="*70)
        
        return historical_success and streaming_success
    
    def run_data_verification(self, symbol=None, interval=None):
        """
        Verify the integrity and completeness of ingested data.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
        """
        self.logger.info("="*70)
        self.logger.info("RUNNING DATA VERIFICATION")
        self.logger.info("="*70)
        
        symbol = symbol or self.config.ingestion.DEFAULT_SYMBOL
        interval = interval or self.config.ingestion.DEFAULT_INTERVAL
        
        try:
            from module_1_ingestion.historical_data_ingestor import HistoricalDataIngestor
            
            ingestor = HistoricalDataIngestor(
                base_path=str(self.config.paths.BASE_PATH)
            )
            
            # Load processed data
            processed_path = self.config.paths.PROCESSED_DATA_PATH / symbol / interval / "historical"
            
            if not processed_path.exists():
                self.logger.error(f"No processed data found at {processed_path}")
                return False
            
            self.logger.info(f"Loading data from: {processed_path}")
            
            # Read partitioned Parquet data - Spark will auto-detect partition columns
            df = ingestor.spark.read.parquet(str(processed_path))
            
            # Verify columns exist
            print(f"\n[INFO] Available columns: {df.columns}")
            print(f"\n[INFO] Schema:")
            df.printSchema()
            
            print(f"\n[INFO] Loaded {df.count():,} total records")
            
            # Show year/month distribution
            print("\n[INFO] Data distribution by year and month:")
            df.groupBy("year", "month").count().orderBy("year", "month").show(50)
            
            # Run validation
            quality_report = ingestor.validate_data_quality(df)
            
            # Display statistics
            self.logger.info("\nData Statistics:")
            df.select("price_close", "volume_base", "log_return").describe().show()
            
            # Show sample of data
            print("\n[INFO] Sample of recent data:")
            df.orderBy(F.desc("timestamp")).select(
                "timestamp", "price_close", "volume_base", "year", "month"
            ).show(10, truncate=False)
            
            ingestor.stop()
            
            self.logger.info("="*70)
            self.logger.info("DATA VERIFICATION COMPLETED")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in data verification: {e}", exc_info=True)
            return False
        
    def run_backfill_ingestion(self, symbol=None, interval=None, 
                           start_date=None, end_date=None):
        """
        Backfill missing data between two dates.
        
        Args:
            symbol: Trading pair symbol
            interval: Candle interval
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
        """
        self.logger.info("="*70)
        self.logger.info("STARTING BACKFILL DATA INGESTION")
        self.logger.info("="*70)
        
        symbol = symbol or self.config.ingestion.DEFAULT_SYMBOL
        interval = interval or self.config.ingestion.DEFAULT_INTERVAL
        
        if not start_date or not end_date:
            self.logger.error("Both --start-date and --end-date are required for backfill mode")
            return False
        
        try:
            from module_1_ingestion.streaming_data_ingestor import StreamingDataIngestor
            
            self.logger.info("Initializing Streaming Data Ingestor for backfill...")
            self.streaming_ingestor = StreamingDataIngestor(
                base_path=str(self.config.paths.BASE_PATH)
            )
            
            # Backfill missing data
            self.logger.info(f"Backfilling {symbol} from {start_date} to {end_date}")
            df_backfilled = self.streaming_ingestor.backfill_missing_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if df_backfilled is None:
                self.logger.error("Backfill failed - no data retrieved")
                return False
            
            # Show sample
            print("\n[INFO] Sample of backfilled data:")
            df_backfilled.select("timestamp", "price_close", "volume_base", "year", "month").show(10)
            
            # Append to historical data
            self.logger.info("Appending backfilled data to historical...")
            self.streaming_ingestor.append_to_historical(
                df_backfilled, 
                symbol=symbol, 
                interval=interval
            )
            
            self.logger.info("="*70)
            self.logger.info("BACKFILL COMPLETED SUCCESSFULLY")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in backfill: {e}", exc_info=True)
            return False
        
        finally:
            if self.streaming_ingestor:
                self.streaming_ingestor.stop()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Crypto Volatility Analysis Framework - Data Ingestion Layer'
    )

    parser.add_argument(
        '--overwrite', 
        action='store_true',
        help='Overwrite existing processed data (default: append)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['historical', 'streaming', 'hybrid', 'verify', 'backfill'],  
        default='historical',
        help='Ingestion mode to run'
    )
    
    parser.add_argument('--symbol', type=str, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, help='Candle interval (e.g., 1m, 5m)')
    parser.add_argument('--start-month', type=str, help='Start month (YYYY-MM)')
    parser.add_argument('--end-month', type=str, help='End month (YYYY-MM)')
    parser.add_argument('--duration', type=int, default=3600, 
                       help='Streaming duration in seconds (default: 3600)')
    parser.add_argument('--start-date', type=str, help='Start date for backfill (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backfill (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Import config
    from config import config
    
    # Setup environment
    config.setup_environment()
    config.summary()
    
    # Initialize orchestrator
    orchestrator = DataIngestionOrchestrator(config)
    
    # Run selected mode
    if args.mode == 'historical':
        success = orchestrator.run_historical_ingestion(
            symbol=args.symbol,
            interval=args.interval,
            start_month=args.start_month,
            end_month=args.end_month,
            overwrite=args.overwrite
        )
    elif args.mode == 'streaming':
        success = orchestrator.run_streaming_ingestion(
            symbol=args.symbol,
            interval=args.interval,
            duration=args.duration
        )
    elif args.mode == 'hybrid':
        success = orchestrator.run_hybrid_ingestion(
            symbol=args.symbol,
            interval=args.interval,
            start_month=args.start_month,
            end_month=args.end_month,
            streaming_duration=args.duration,
            overwrite=args.overwrite
        )
    elif args.mode == 'verify':
        success = orchestrator.run_data_verification(
            symbol=args.symbol,
            interval=args.interval
        )
    elif args.mode == 'backfill':
        success = orchestrator.run_backfill_ingestion(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )
    elif args.mode == 'features':
        print("\n=== Running Feature Engineering Mode ===")
        from module_2_features.feature_engineer import FeatureEngineer
        
        engineer = FeatureEngineer(base_path=base_path, spark=spark)
        df = engineer.run_full_pipeline(
            symbol=config.SYMBOL,
            interval=config.INTERVAL,
            save=True
        )
        print(f"\n[SUCCESS] Feature matrix created with {len(df.columns)} columns")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()