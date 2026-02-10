from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path

from volatility_estimators import VolatilityEstimators
from return_features import ReturnFeatures
from volume_features import VolumeFeatures
from technical_indicators import TechnicalIndicators


class FeatureEngineer:
    def __init__(self, base_path=".", spark=None):
        self.base_path = Path(base_path)
        
        # Create or reuse Spark session
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("CryptX-FeatureEngineering") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = spark
        
        # Initialize feature modules
        print("\n" + "="*60)
        print("INITIALIZING FEATURE ENGINEERING")
        print("="*60)
        
        self.vol_calc = VolatilityEstimators(windows=[30, 60, 240])
        self.return_calc = ReturnFeatures(windows=[30, 60, 240])
        self.volume_calc = VolumeFeatures(windows=[30, 60, 240])
        self.tech_calc = TechnicalIndicators()
        
        print("[INFO] All modules initialized\n")
    
    def load_data(self, symbol="BTCUSDT", interval="1m"):
        # Load processed data from Module 1.
        data_path = self.base_path / "data" / "processed" / symbol / interval
        
        print("="*60)
        print(f"LOADING DATA: {symbol} {interval}")
        print("="*60)
        print(f"Path: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data not found at {data_path}")
        
        df = self.spark.read.parquet(str(data_path))
        
        count = df.count()
        print(f"Loaded: {count:,} records")
        print(f"Date range: {df.agg(F.min('timestamp')).collect()[0][0]} to "
              f"{df.agg(F.max('timestamp')).collect()[0][0]}")
        print()
        
        return df
    
    def engineer_features(self, df):
        """
        Apply all feature engineering.
        Returns DataFrame with features added
        """
        print("="*60)
        print("COMPUTING FEATURES")
        print("="*60)
        
        initial_cols = len(df.columns)
        
        # 1. Volatility features
        print("\n[1/4] Volatility features...")
        df = self.vol_calc.compute_all(df)
        print(f"      Added {len(df.columns) - initial_cols} features")
        
        # 2. Return features
        step2_start = len(df.columns)
        print("\n[2/4] Return features...")
        df = self.return_calc.compute_all(df)
        print(f"      Added {len(df.columns) - step2_start} features")
        
        # 3. Volume features
        step3_start = len(df.columns)
        print("\n[3/4] Volume features...")
        df = self.volume_calc.compute_all(df)
        print(f"      Added {len(df.columns) - step3_start} features")
        
        # 4. Technical indicators
        step4_start = len(df.columns)
        print("\n[4/4] Technical indicators...")
        df = self.tech_calc.compute_all(df)
        print(f"      Added {len(df.columns) - step4_start} features")
        
        total_features = len(df.columns) - initial_cols
        print(f"\n{'='*60}")
        print(f"TOTAL FEATURES ADDED: {total_features}")
        print(f"Final columns: {len(df.columns)}")
        print("="*60 + "\n")
        
        return df
    
    def save_features(self, df, symbol="BTCUSDT", interval="1m"):
        """
        Save feature matrix.
        """
        output_path = self.base_path / "data" / "features" / symbol / interval
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("SAVING FEATURES")
        print("="*60)
        print(f"Output: {output_path}")
        
        # Save as parquet (partitioned by year/month)
        df.coalesce(12).write \
            .mode("overwrite") \
            .partitionBy("year", "month") \
            .parquet(str(output_path))
        
        print(f"Saved: {df.count():,} records")
        print("="*60 + "\n")
    
    def show_summary(self, df):
        """Display feature summary."""
        print("="*60)
        print("FEATURE SUMMARY")
        print("="*60)
        
        # List all features (exclude base columns)
        base_cols = ['timestamp', 'price_open', 'price_high', 'price_low', 
                     'price_close', 'volume_base', 'volume_quote', 'num_trades',
                     'log_return', 'year', 'month', 'day', 'hour']
        
        features = [col for col in df.columns if col not in base_cols]
        
        print(f"\nTotal features: {len(features)}")
        print("\nFeature list:")
        for i, feat in enumerate(features, 1):
            print(f"  {i:2d}. {feat}")
        
        # Show sample
        print("\n" + "="*60)
        print("SAMPLE DATA")
        print("="*60)
        sample_cols = ['timestamp', 'price_close', 'vol_60m', 'cum_return_60m', 
                      'return_skew_60m', 'volume_spike', 'rsi', 'bb_width', 'atr_pct']
        existing_cols = [c for c in sample_cols if c in df.columns]
        
        df.select(existing_cols).orderBy(F.desc("timestamp")).show(10)
        
        # Check for nulls
        print("\n" + "="*60)
        print("NULL CHECK")
        print("="*60)
        
        null_counts = {}
        for col in features:
            nulls = df.filter(F.col(col).isNull()).count()
            if nulls > 0:
                null_counts[col] = nulls
        
        if null_counts:
            print(f"Found {len(null_counts)} features with nulls:")
            for feat, count in sorted(null_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / df.count() * 100
                print(f"  - {feat}: {count:,} ({pct:.1f}%)")
        else:
            print("✓ No null values!")
        
        print()
    
    def run_pipeline(self, symbol="BTCUSDT", interval="1m", save=True):
        """
        Run complete feature engineering pipeline.
        Returns:
            DataFrame with features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print(f"Save: {save}")
        print("="*60 + "\n")
        
        # Load data
        df = self.load_data(symbol=symbol, interval=interval)
        
        # Engineer features
        df_features = self.engineer_features(df)
        
        # Show summary
        self.show_summary(df_features)
        
        # Save
        if save:
            self.save_features(df_features, symbol=symbol, interval=interval)
        
        print("="*60)
        print("✓ PIPELINE COMPLETE")
        print("="*60 + "\n")
        
        return df_features
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()


# Main execution
if __name__ == "__main__":
    import os
    
    # Find CryptX directory
    current = Path(__file__).resolve()
    cryptx_root = None
    
    for parent in current.parents:
        if parent.name == "CryptX":
            cryptx_root = parent
            break
    
    if cryptx_root is None:
        # Try current directory
        if os.path.exists("./data/processed"):
            cryptx_root = Path(".")
        else:
            print("[ERROR] Cannot find CryptX directory!")
            exit(1)
    
    print(f"[INFO] CryptX root: {cryptx_root}\n")
    
    # Run pipeline
    engineer = FeatureEngineer(base_path=cryptx_root)
    
    try:
        df = engineer.run_pipeline(
            symbol="BTCUSDT",
            interval="1m",
            save=True
        )
        
        print(f"[SUCCESS] Feature engineering complete!")
        print(f"          {df.count():,} records with {len(df.columns)} columns")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        engineer.stop()