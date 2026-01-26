"""
Feature Engineer - Master Orchestrator for Module 2
Combines all feature engineering modules and creates the final feature matrix.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path
import sys

# Add parent directory to path to import feature modules
sys.path.append(str(Path(__file__).parent))

from volatility_estimators import VolatilityEstimators
from return_features import ReturnFeatures
from volume_features import VolumeFeatures
from technical_indicators import TechnicalIndicators


class FeatureEngineer:
    """
    Master class for feature engineering.
    
    Orchestrates all feature modules and creates the final feature matrix
    for regime detection models.
    """
    
    def __init__(self, base_path=".", spark=None):
        """
        Initialize Feature Engineer.
        
        Args:
            base_path: Base path to CryptX directory
            spark: Existing SparkSession (optional)
        """
        self.base_path = Path(base_path)
        
        # Create or get Spark session
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("CryptX-FeatureEngineering") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.ansi.enabled", "false") \
                .getOrCreate()
        else:
            self.spark = spark
        
        # Initialize all feature calculators
        print("\n" + "="*70)
        print("INITIALIZING FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        self.vol_estimator = VolatilityEstimators(windows=[5, 15, 30, 60, 240, 1440])
        self.return_calc = ReturnFeatures(windows=[60, 240, 1440])
        self.volume_calc = VolumeFeatures(windows=[60, 240, 1440])
        self.tech_calc = TechnicalIndicators()
        
        print("\n[INFO] All feature modules initialized successfully!")
    
    def load_processed_data(self, symbol="BTCUSDT", interval="1m"):
        """
        Load processed data from Module 1.
        
        Args:
            symbol: Trading symbol (default: BTCUSDT)
            interval: Time interval (default: 1m)
            
        Returns:
            DataFrame with processed OHLCV data
        """
        data_path = self.base_path / "data" / "processed" / symbol / interval / "historical"
        
        print("\n" + "="*70)
        print(f"LOADING PROCESSED DATA")
        print("="*70)
        print(f"Path: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        
        df = self.spark.read.parquet(str(data_path))
        
        print(f"[INFO] Loaded {df.count():,} records")
        print(f"[INFO] Columns: {', '.join(df.columns[:10])}...")
        print(f"[INFO] Date range: {df.agg(F.min('timestamp')).collect()[0][0]} to {df.agg(F.max('timestamp')).collect()[0][0]}")
        
        return df
    
    def engineer_all_features(self, df):
        """
        Apply all feature engineering steps.
        
        Args:
            df: DataFrame with processed OHLCV data
            
        Returns:
            DataFrame with all features added
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING PIPELINE - SEQUENTIAL EXECUTION")
        print("="*70)
        
        initial_cols = len(df.columns)
        print(f"[INFO] Starting with {initial_cols} columns")
        
        # Step 1: Volatility features
        print("\n[STEP 1/4] Computing Volatility Features...")
        df = self.vol_estimator.compute_all_volatility_features(df)
        print(f"[INFO] Now have {len(df.columns)} columns (+{len(df.columns) - initial_cols})")
        
        # Step 2: Return features
        print("\n[STEP 2/4] Computing Return Features...")
        step2_start = len(df.columns)
        df = self.return_calc.compute_all_return_features(df)
        print(f"[INFO] Now have {len(df.columns)} columns (+{len(df.columns) - step2_start})")
        
        # Step 3: Volume features
        print("\n[STEP 3/4] Computing Volume Features...")
        step3_start = len(df.columns)
        df = self.volume_calc.compute_all_volume_features(df)
        print(f"[INFO] Now have {len(df.columns)} columns (+{len(df.columns) - step3_start})")
        
        # Step 4: Technical indicators
        print("\n[STEP 4/4] Computing Technical Indicators...")
        step4_start = len(df.columns)
        df = self.tech_calc.compute_all_technical_indicators(df)
        print(f"[INFO] Now have {len(df.columns)} columns (+{len(df.columns) - step4_start})")
        
        total_features = len(df.columns) - initial_cols
        print("\n" + "="*70)
        print(f"FEATURE ENGINEERING COMPLETE!")
        print(f"Total features added: {total_features}")
        print(f"Final column count: {len(df.columns)}")
        print("="*70)
        
        return df
    
    def save_feature_matrix(self, df, symbol="BTCUSDT", interval="1m", mode="overwrite"):
        """
        Save feature matrix to parquet.
        
        Args:
            df: DataFrame with all features
            symbol: Trading symbol
            interval: Time interval
            mode: Write mode ('overwrite', 'append')
        """
        output_path = self.base_path / "data" / "features" / symbol / interval
        
        print("\n" + "="*70)
        print("SAVING FEATURE MATRIX")
        print("="*70)
        print(f"Output path: {output_path}")
        
        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet (partitioned by year and month for efficient querying)
        df.write \
            .mode(mode) \
            .partitionBy("year", "month") \
            .parquet(str(output_path / "feature_matrix.parquet"))
        
        print(f"[INFO] Feature matrix saved successfully!")
        print(f"[INFO] Mode: {mode}")
        print(f"[INFO] Partitions: year, month")
        
        # Verify the save
        saved_df = self.spark.read.parquet(str(output_path / "feature_matrix.parquet"))
        saved_count = saved_df.count()
        
        print(f"[INFO] Verification: {saved_count:,} records saved")
        
        return output_path
    
    def generate_feature_summary(self, df):
        """
        Generate summary statistics for all features.
        
        Args:
            df: DataFrame with all features
        """
        print("\n" + "="*70)
        print("FEATURE SUMMARY")
        print("="*70)
        
        # Categorize features
        feature_categories = {
            'Volatility': [col for col in df.columns if 'vol' in col.lower() and col not in ['volume_base', 'volume_quote']],
            'Returns': [col for col in df.columns if 'return' in col.lower() and col not in ['log_return', 'simple_return']],
            'Volume': [col for col in df.columns if any(x in col.lower() for x in ['volume_ma', 'volume_spike', 'vwap', 'obv', 'volume_price', 'volume_momentum'])],
            'Technical': [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'bb_', 'macd', 'atr', 'ema', 'momentum'])]
        }
        
        print("\n[FEATURE BREAKDOWN BY CATEGORY]")
        for category, features in feature_categories.items():
            print(f"\n{category} ({len(features)} features):")
            for feat in features[:5]:  # Show first 5
                print(f"  - {feat}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")
        
        # Sample of feature values
        print("\n[SAMPLE FEATURE VALUES]")
        sample_features = []
        for features in feature_categories.values():
            sample_features.extend(features[:2])  # 2 from each category
        
        df.select("timestamp", "price_close", *sample_features[:8]).orderBy(F.desc("timestamp")).show(5, truncate=False)
        
        # Statistics for key features
        print("\n[KEY FEATURE STATISTICS]")
        key_features = [
            'realized_vol_60m', 'return_skew_60m', 'volume_spike_60m', 
            'rsi_14', 'bb_width_20', 'atr_14_pct'
        ]
        existing_key_features = [f for f in key_features if f in df.columns]
        
        if existing_key_features:
            df.select(existing_key_features).describe().show()
        
        # Check for null values
        print("\n[NULL VALUE CHECK]")
        null_counts = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
        null_summary = null_counts.collect()[0].asDict()
        
        features_with_nulls = {k: v for k, v in null_summary.items() if v > 0}
        if features_with_nulls:
            print(f"[WARNING] {len(features_with_nulls)} features have null values:")
            for feat, count in sorted(features_with_nulls.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {feat}: {count:,} nulls ({count/df.count()*100:.2f}%)")
        else:
            print("[INFO] No null values found! ✓")
    
    def run_full_pipeline(self, symbol="BTCUSDT", interval="1m", save=True):
        """
        Run the complete feature engineering pipeline.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            save: Whether to save the feature matrix
            
        Returns:
            DataFrame with all features
        """
        print("\n" + "="*70)
        print("FEATURE ENGINEERING PIPELINE - FULL RUN")
        print("="*70)
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Save output: {save}")
        
        # Step 1: Load data
        df = self.load_processed_data(symbol=symbol, interval=interval)
        
        # Step 2: Engineer features
        df_with_features = self.engineer_all_features(df)
        
        # Step 3: Generate summary
        self.generate_feature_summary(df_with_features)
        
        # Step 4: Save (optional)
        if save:
            output_path = self.save_feature_matrix(df_with_features, symbol=symbol, interval=interval)
            print(f"\n[SUCCESS] Feature matrix saved to: {output_path}/feature_matrix.parquet")
        
        print("\n" + "="*70)
        print("FEATURE ENGINEERING PIPELINE COMPLETE! ✓")
        print("="*70)
        
        return df_with_features
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CryptX Feature Engineering")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="1m", help="Time interval")
    parser.add_argument("--base-path", type=str, default=".", help="Base path to CryptX")
    parser.add_argument("--no-save", action="store_true", help="Don't save output (test mode)")
    
    args = parser.parse_args()
    
    # Determine base path
    if args.base_path == ".":
        # Try to find CryptX directory
        current = Path(__file__).resolve()
        for parent in current.parents:
            if parent.name == "CryptX":
                args.base_path = parent
                break
    
    print("\n" + "="*70)
    print("CRYPTX - MODULE 2: FEATURE ENGINEERING")
    print("="*70)
    
    # Create feature engineer
    engineer = FeatureEngineer(base_path=args.base_path)
    
    try:
        # Run pipeline
        df = engineer.run_full_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            save=not args.no_save
        )
        
        print("\n[SUCCESS] Feature engineering completed successfully!")
        print(f"[INFO] Final dataset: {df.count():,} records with {len(df.columns)} columns")
        
    except Exception as e:
        print(f"\n[ERROR] Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engineer.stop()