from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path
import pandas as pd
import numpy as np

from hmm_regime import HMMRegimeDetector
from kmeans_regime import KMeansRegimeDetector
from gmm_regime import GMMRegimeDetector
from threshold_regime import ThresholdRegimeDetector
from regime_evaluator import RegimeEvaluator
from regime_visualizer import RegimeVisualizer


class RegimeDetector:
    """
    Coordinate regime detection using multiple approaches.
    Tests HMM, K-Means, GMM, and threshold-based methods.
    """
    def __init__(self, base_path=".", spark=None):
        self.base_path = Path(base_path)
        
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("CryptX-RegimeDetection") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
        else:
            self.spark = spark
        
        print("\n" + "="*60)
        print("INITIALIZING REGIME DETECTION")
        print("="*60)
        
        # Initialize all detectors
        self.detectors = {
            'hmm': HMMRegimeDetector(n_regimes=3),
            'kmeans': KMeansRegimeDetector(n_regimes=3),
            'gmm': GMMRegimeDetector(n_regimes=3),
            'threshold': ThresholdRegimeDetector()
        }
        
        # Initialize evaluator and visualizer
        self.evaluator = RegimeEvaluator()
        self.visualizer = RegimeVisualizer()
        
        print("[INFO] All regime detectors initialized\n")
    
    def load_features(self, symbol="BTCUSDT", interval="1m"):
        """Load feature-engineered data from Module 2."""
        data_path = self.base_path / "data" / "features" / symbol / interval
        
        print("="*60)
        print(f"LOADING FEATURE DATA: {symbol} {interval}")
        print("="*60)
        print(f"Path: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Feature data not found at {data_path}")
        
        df = self.spark.read.parquet(str(data_path))
        
        count = df.count()
        print(f"Loaded: {count:,} records")
        print(f"Date range: {df.agg(F.min('timestamp')).collect()[0][0]} to "
              f"{df.agg(F.max('timestamp')).collect()[0][0]}")
        print()
        
        return df
    
    def prepare_regime_features(self, df):
        """
        Select and prepare features for regime detection.
        Focus on volatility and return characteristics.
        """
        print("="*60)
        print("PREPARING REGIME FEATURES")
        print("="*60)
        
        # Core features for regime detection
        regime_features = [
            'timestamp',
            'price_close',
            'log_return',
            'vol_60m',
            'vol_240m',
            'vol_ratio_60_240',
            'cum_return_60m',
            'return_skew_60m',
            'volume_spike',
            'rsi',
            'bb_width',
            'atr_pct'
        ]
        
        # Check which features exist
        existing_features = [f for f in regime_features if f in df.columns]
        missing = set(regime_features) - set(existing_features)
        
        if missing:
            print(f"[WARNING] Missing features: {missing}")
        
        print(f"Using {len(existing_features)} features for regime detection")
        
        # Select features and remove nulls
        df_clean = df.select(existing_features).dropna()
        
        print(f"Records after removing nulls: {df_clean.count():,}")
        print()
        
        return df_clean
    
    def detect_regimes_all(self, df):
        """
        Run all regime detection methods.
        Returns dict of DataFrames with regime labels.
        """
        print("="*60)
        print("DETECTING REGIMES - ALL METHODS")
        print("="*60)
        
        results = {}
        
        for name, detector in self.detectors.items():
            print(f"\n[{name.upper()}] Running detection...")
            
            try:
                # Each detector returns Spark DataFrame with regime column
                df_regime = detector.detect(df)
                results[name] = df_regime
                
                # Show regime distribution
                regime_counts = df_regime.groupBy('regime').count().orderBy('regime').collect()
                print(f"[{name.upper()}] Regime distribution:")
                for row in regime_counts:
                    pct = row['count'] / df_regime.count() * 100
                    print(f"  Regime {row['regime']}: {row['count']:,} ({pct:.1f}%)")
                
            except Exception as e:
                print(f"[{name.upper()}] ERROR: {e}")
                results[name] = None
        
        print("\n" + "="*60)
        print(f"✓ Completed {len([r for r in results.values() if r is not None])}/{len(self.detectors)} methods")
        print("="*60 + "\n")
        
        return results
    
    def evaluate_regimes(self, results, df_original):
        """
        Evaluate all regime detection methods.
        Returns comparison DataFrame.
        """
        print("="*60)
        print("EVALUATING REGIME QUALITY")
        print("="*60)
        
        evaluation_results = []
        
        for name, df_regime in results.items():
            if df_regime is None:
                continue
            
            print(f"\n[{name.upper()}] Evaluating...")
            
            # Compute metrics
            metrics = self.evaluator.evaluate(df_regime, df_original)
            metrics['method'] = name
            evaluation_results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(evaluation_results)
        
        # Reorder columns
        cols = ['method'] + [c for c in comparison_df.columns if c != 'method']
        comparison_df = comparison_df[cols]
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(comparison_df.to_string(index=False))
        print()
        
        return comparison_df
    
    def visualize_regimes(self, results, df_original, output_dir=None):
        """
        Create visualizations for all regime detection methods.
        """
        print("="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        if output_dir is None:
            output_dir = self.base_path / "results" / "regimes" / "visualizations"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df_regime in results.items():
            if df_regime is None:
                continue
            
            print(f"\n[{name.upper()}] Creating visualizations...")
            
            try:
                # Generate plots
                self.visualizer.plot_regime_timeseries(
                    df_regime, 
                    method_name=name,
                    output_path=output_dir / f"{name}_timeseries.png"
                )
                
                self.visualizer.plot_regime_characteristics(
                    df_regime,
                    method_name=name,
                    output_path=output_dir / f"{name}_characteristics.png"
                )
                
                print(f"[{name.upper()}] Saved visualizations to {output_dir}")
                
            except Exception as e:
                print(f"[{name.upper()}] Visualization error: {e}")
        
        print("\n" + "="*60)
        print(f"✓ Visualizations saved to {output_dir}")
        print("="*60 + "\n")
    
    def create_comparison_visualization(self, comparison_df, output_dir=None):
        """
        Create detailed comparison visualization from metrics DataFrame.
        """
        if output_dir is None:
            output_dir = self.base_path / "results" / "regimes" / "visualizations"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("CREATING DETAILED COMPARISON VISUALIZATION")
        print("="*60)
        
        output_path = output_dir / "detailed_comparison.png"
        
        try:
            summary = self.visualizer.plot_detailed_comparison(
                comparison_df,
                output_path=output_path
            )
            
            print(f"\n✓ Detailed comparison saved to {output_path}")
            print("\nMethod Rankings (Best to Worst):")
            for i, row in enumerate(summary, 1):
                print(f"  {i}. {row[0]} - Avg Rank: {row[5]}")
            
        except Exception as e:
            print(f"[ERROR] Failed to create comparison visualization: {e}")
        
        print("="*60 + "\n")
    
    def save_regimes(self, results, symbol="BTCUSDT", interval="1m"):
        """
        Save regime labels for all methods.
        """
        print("="*60)
        print("SAVING REGIME LABELS")
        print("="*60)
        
        for name, df_regime in results.items():
            if df_regime is None:
                continue
            
            output_path = self.base_path / "data" / "regimes" / symbol / interval / name
            output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"[{name.upper()}] Saving to {output_path}")
            
            # Add year and month columns for partitioning
            df_regime = df_regime.withColumn("year", F.year("timestamp"))
            df_regime = df_regime.withColumn("month", F.month("timestamp"))
            
            # Save as parquet
            df_regime.coalesce(12).write \
                .mode("overwrite") \
                .partitionBy("year", "month") \
                .parquet(str(output_path))
        
        print("\n" + "="*60)
        print("✓ All regimes saved")
        print("="*60 + "\n")
    
    def save_comparison(self, comparison_df, symbol="BTCUSDT", interval="1m"):
        """Save evaluation comparison results."""
        output_dir = self.base_path / "results" / "regimes"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"regime_comparison_{symbol}_{interval}.csv"
        comparison_df.to_csv(output_file, index=False)
        
        print(f"[INFO] Comparison saved to {output_file}")
    
    def run_pipeline(self, symbol="BTCUSDT", interval="1m", 
                     save_regimes=True, create_viz=True):
        """
        Run complete regime detection pipeline.
        """
        print("\n" + "="*60)
        print("REGIME DETECTION PIPELINE")
        print("="*60)
        print(f"Symbol: {symbol}")
        print(f"Interval: {interval}")
        print("="*60 + "\n")
        
        # Load features
        df = self.load_features(symbol=symbol, interval=interval)
        
        # Prepare features
        df_clean = self.prepare_regime_features(df)
        
        # Detect regimes with all methods
        results = self.detect_regimes_all(df_clean)
        
        # Evaluate
        comparison_df = self.evaluate_regimes(results, df_clean)
        
        # Save comparison
        self.save_comparison(comparison_df, symbol=symbol, interval=interval)
        
        # Visualize
        if create_viz:
            self.visualize_regimes(results, df_clean)
            # Create detailed comparison visualization
            self.create_comparison_visualization(comparison_df)
        
        # Save regimes
        if save_regimes:
            self.save_regimes(results, symbol=symbol, interval=interval)
        
        print("="*60)
        print("✓ REGIME DETECTION PIPELINE COMPLETE")
        print("="*60 + "\n")
        
        return results, comparison_df
    
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
        if os.path.exists("./data/features"):
            cryptx_root = Path(".")
        else:
            print("[ERROR] Cannot find CryptX directory!")
            exit(1)
    
    print(f"[INFO] CryptX root: {cryptx_root}\n")
    
    # Run pipeline
    detector = RegimeDetector(base_path=cryptx_root)
    
    try:
        results, comparison = detector.run_pipeline(
            symbol="BTCUSDT",
            interval="1m",
            save_regimes=True,
            create_viz=True
        )
        
        print("\n[SUCCESS] Regime detection complete!")
        print("\nBest performing method:")
        best_method = comparison.loc[comparison['silhouette_score'].idxmax()]
        print(f"  Method: {best_method['method']}")
        print(f"  Silhouette Score: {best_method['silhouette_score']:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        detector.stop()