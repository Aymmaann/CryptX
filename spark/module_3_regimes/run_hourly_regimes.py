"""
Hourly Regime Detection Pipeline

This script:
1. Loads your minute-level feature data
2. Aggregates to hourly bars
3. Computes hourly features
4. Runs all 6 regime detectors on hourly data
5. Compares results

Why this will work better:
- Regimes are clearer at hourly level
- Less noise
- DL models will find actual patterns
- More realistic (traders think in hours/days, not minutes)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pathlib import Path
import sys

# Add module path
sys.path.append(str(Path(__file__).parent))

from hmm_regime import HMMRegimeDetector
from kmeans_regime import KMeansRegimeDetector
from gmm_regime import GMMRegimeDetector
from threshold_regime import ThresholdRegimeDetector
from patchtst_regime import PatchTSTRegimeDetector
from informer_regime import InformerRegimeDetector
from regime_evaluator import RegimeEvaluator
from regime_visualizer import RegimeVisualizer


def aggregate_to_hourly(spark, minute_data_path):
    """
    Aggregate minute-level data to hourly bars.
    """
    print("\n" + "="*60)
    print("STEP 1: AGGREGATING MINUTE DATA TO HOURLY")
    print("="*60)
    
    print(f"[INFO] Loading minute data from {minute_data_path}")
    df_minute = spark.read.parquet(str(minute_data_path))
    
    minute_count = df_minute.count()
    print(f"[INFO] Loaded {minute_count:,} minute bars")
    
    # Add hour timestamp
    df_hour = df_minute.withColumn(
        "hour_ts",
        F.date_trunc("hour", F.col("timestamp"))
    )
    
    print("[INFO] Aggregating to hourly OHLCV...")
    
    # Aggregate OHLCV
    df_hourly = df_hour.groupBy("hour_ts").agg(
        F.first("price_open").alias("price_open"),
        F.max("price_high").alias("price_high"),
        F.min("price_low").alias("price_low"),
        F.last("price_close").alias("price_close"),
        F.sum("volume_base").alias("volume_base"),
        F.sum("volume_quote").alias("volume_quote"),
        F.sum("num_trades").alias("num_trades"),
        F.count("*").alias("n_bars")
    )
    
    df_hourly = df_hourly.withColumnRenamed("hour_ts", "timestamp")
    df_hourly = df_hourly.orderBy("timestamp")
    
    hourly_count = df_hourly.count()
    print(f"[INFO] Created {hourly_count:,} hourly bars")
    print(f"[INFO] Compression ratio: {minute_count/hourly_count:.1f}:1")
    
    return df_hourly


def compute_hourly_features(df_hourly):
    """
    Compute features on hourly data (similar to minute features but at hourly scale).
    """
    print("\n" + "="*60)
    print("STEP 2: COMPUTING HOURLY FEATURES")
    print("="*60)
    
    window = Window.orderBy("timestamp")
    
    # Log returns
    print("[INFO] Computing returns...")
    df = df_hourly.withColumn(
        "prev_close",
        F.lag("price_close", 1).over(window)
    )
    df = df.withColumn(
        "log_return",
        F.log(F.col("price_close") / F.col("prev_close"))
    )
    
    # Volatility (rolling std of returns)
    # Windows: 24h (1 day), 168h (1 week), 720h (30 days)
    print("[INFO] Computing volatility features...")
    
    window_24h = Window.orderBy("timestamp").rowsBetween(-23, 0)
    window_168h = Window.orderBy("timestamp").rowsBetween(-167, 0)
    window_720h = Window.orderBy("timestamp").rowsBetween(-719, 0)
    
    df = df.withColumn("vol_24h", F.stddev("log_return").over(window_24h))
    df = df.withColumn("vol_168h", F.stddev("log_return").over(window_168h))
    df = df.withColumn("vol_720h", F.stddev("log_return").over(window_720h))
    
    # Volatility ratios
    df = df.withColumn("vol_ratio_24_168",
                      F.when(F.col("vol_168h") > 0,
                            F.col("vol_24h") / F.col("vol_168h"))
                      .otherwise(1.0))
    
    df = df.withColumn("vol_ratio_168_720",
                      F.when(F.col("vol_720h") > 0,
                            F.col("vol_168h") / F.col("vol_720h"))
                      .otherwise(1.0))
    
    # Cumulative returns (momentum)
    print("[INFO] Computing return features...")
    df = df.withColumn("cum_return_24h", F.sum("log_return").over(window_24h))
    df = df.withColumn("cum_return_168h", F.sum("log_return").over(window_168h))
    df = df.withColumn("cum_return_720h", F.sum("log_return").over(window_720h))
    
    # Return skewness
    df = df.withColumn("return_skew_24h", F.skewness("log_return").over(window_24h))
    df = df.withColumn("return_skew_168h", F.skewness("log_return").over(window_168h))
    
    # Volume features
    print("[INFO] Computing volume features...")
    df = df.withColumn("vol_avg_24h", F.mean("volume_base").over(window_24h))
    df = df.withColumn("vol_avg_168h", F.mean("volume_base").over(window_168h))
    
    df = df.withColumn("volume_spike",
                      F.when(F.col("vol_avg_24h") > 0,
                            F.col("volume_base") / F.col("vol_avg_24h"))
                      .otherwise(1.0))
    
    # RSI (14-hour period)
    print("[INFO] Computing technical indicators...")
    df = df.withColumn("price_change",
                      F.col("price_close") - F.lag("price_close", 1).over(window))
    df = df.withColumn("gain",
                      F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0))
    df = df.withColumn("loss",
                      F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0))
    
    window_14h = Window.orderBy("timestamp").rowsBetween(-13, 0)
    df = df.withColumn("avg_gain", F.mean("gain").over(window_14h))
    df = df.withColumn("avg_loss", F.mean("loss").over(window_14h))
    df = df.withColumn("rs",
                      F.when(F.col("avg_loss") > 0,
                            F.col("avg_gain") / F.col("avg_loss"))
                      .otherwise(100))
    df = df.withColumn("rsi", 100 - (100 / (1 + F.col("rs"))))
    
    # ATR (14-hour)
    df = df.withColumn("true_range",
                      F.greatest(
                          F.col("price_high") - F.col("price_low"),
                          F.abs(F.col("price_high") - F.col("prev_close")),
                          F.abs(F.col("price_low") - F.col("prev_close"))
                      ))
    df = df.withColumn("atr", F.mean("true_range").over(window_14h))
    df = df.withColumn("atr_pct",
                      (F.col("atr") / F.col("price_close")) * 100)
    
    # Bollinger Bands (20-hour)
    window_20h = Window.orderBy("timestamp").rowsBetween(-19, 0)
    df = df.withColumn("bb_middle", F.mean("price_close").over(window_20h))
    df = df.withColumn("bb_std", F.stddev("price_close").over(window_20h))
    df = df.withColumn("bb_width",
                      (F.col("bb_std") * 2) / F.col("bb_middle"))
    
    # High-low range
    df = df.withColumn("hl_range_24h",
                      F.mean((F.col("price_high") - F.col("price_low")) / F.col("price_close"))
                      .over(window_24h))
    
    # Cleanup
    df = df.drop("price_change", "gain", "loss", "avg_gain", "avg_loss",
                "rs", "true_range", "bb_middle", "bb_std", "prev_close")
    
    # Add date columns
    df = df.withColumn("year", F.year("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    df = df.withColumn("day", F.dayofmonth("timestamp"))
    df = df.withColumn("hour", F.hour("timestamp"))
    
    # Remove nulls
    df = df.dropna()
    
    final_count = df.count()
    print(f"[INFO] Final hourly data: {final_count:,} bars with {len(df.columns)} columns")
    
    return df


def run_hourly_regime_detection(base_path):
    """
    Main pipeline: aggregate to hourly and run all detectors.
    """
    base_path = Path(base_path)
    
    spark = SparkSession.builder \
        .appName("Hourly-Regime-Detection") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Step 1 & 2: Aggregate and compute features
    minute_path = base_path / "data" / "features" / "BTCUSDT" / "1m"
    df_hourly = aggregate_to_hourly(spark, minute_path)
    df_hourly_features = compute_hourly_features(df_hourly)
    
    # Save hourly features
    hourly_output = base_path / "data" / "features" / "BTCUSDT" / "1h"
    hourly_output.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] Saving hourly features to {hourly_output}")
    df_hourly_features.coalesce(2).write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(str(hourly_output))
    print("[INFO] Hourly features saved")
    
    # Step 3: Initialize all detectors
    print("\n" + "="*60)
    print("STEP 3: INITIALIZING REGIME DETECTORS")
    print("="*60)
    
    detectors = {
        'threshold': ThresholdRegimeDetector(),
        'hmm':       HMMRegimeDetector(n_regimes=3),
        'kmeans':    KMeansRegimeDetector(n_regimes=3),
        'gmm':       GMMRegimeDetector(n_regimes=3),
        'patchtst':  PatchTSTRegimeDetector(n_regimes=3, seq_len=168),  # 1 week
        'informer':  InformerRegimeDetector(n_regimes=3, seq_len=168),  # 1 week
    }
    
    print(f"[INFO] Initialized {len(detectors)} detectors")
    
    # Step 4: Run all detectors
    print("\n" + "="*60)
    print("STEP 4: DETECTING REGIMES ON HOURLY DATA")
    print("="*60)
    
    results = {}
    for name, detector in detectors.items():
        print(f"\n[{name.upper()}] Running detection...")
        try:
            df_regime = detector.detect(df_hourly_features)
            results[name] = df_regime
            
            # Show distribution
            regime_counts = df_regime.groupBy('regime').count().orderBy('regime').collect()
            print(f"[{name.upper()}] Regime distribution:")
            for row in regime_counts:
                pct = row['count'] / df_regime.count() * 100
                print(f"  Regime {row['regime']}: {row['count']:,} ({pct:.1f}%)")
        except Exception as e:
            print(f"[{name.upper()}] ERROR: {e}")
            results[name] = None
    
    print(f"\n{'='*60}")
    print(f"✓ Completed {len([r for r in results.values() if r is not None])}/{len(detectors)} methods")
    print("="*60)
    
    # Step 5: Evaluate
    print("\n" + "="*60)
    print("STEP 5: EVALUATING REGIME QUALITY")
    print("="*60)
    
    evaluator = RegimeEvaluator()
    evaluation_results = []
    
    for name, df_regime in results.items():
        if df_regime is None:
            continue
        print(f"\n[{name.upper()}] Evaluating...")
        metrics = evaluator.evaluate(df_regime, df_hourly_features)
        metrics['method'] = name
        evaluation_results.append(metrics)
    
    import pandas as pd
    comparison_df = pd.DataFrame(evaluation_results)
    cols = ['method'] + [c for c in comparison_df.columns if c != 'method']
    comparison_df = comparison_df[cols]
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS - HOURLY DATA")
    print("="*60)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    results_dir = base_path / "results" / "regimes_hourly"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(results_dir / "regime_comparison_hourly.csv", index=False)
    print(f"\n[INFO] Comparison saved to {results_dir / 'regime_comparison_hourly.csv'}")
    
    # Step 6: Visualize
    print("\n" + "="*60)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*60)
    
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = RegimeVisualizer()
    
    # Individual plots - organized by model
    for name, df_regime in results.items():
        if df_regime is None:
            continue
        
        # Create model-specific directory
        model_dir = viz_dir / name
        model_dir.mkdir(exist_ok=True)
        
        print(f"\n[{name.upper()}] Creating visualizations in {model_dir.name}/...")
        
        visualizer.plot_regime_timeseries(
            df_regime, method_name=name,
            output_path=model_dir / "timeseries.png",
            sample_size=5000  # Less sampling needed for hourly
        )
        
        visualizer.plot_regime_characteristics(
            df_regime, method_name=name,
            output_path=model_dir / "characteristics.png"
        )
        
        print(f"[{name.upper()}] Saved to {model_dir.name}/")
    
    # Comparison plot - in root viz directory
    print("\n[INFO] Creating detailed comparison...")
    visualizer.plot_detailed_comparison(
        comparison_df,
        output_path=viz_dir / "detailed_comparison_hourly.png"
    )
    
    print(f"\n{'='*60}")
    print("✓ VISUALIZATIONS ORGANIZED")
    print(f"{'='*60}")
    print(f"Structure:")
    print(f"  {viz_dir}/")
    print(f"    detailed_comparison_hourly.png")
    for name in results.keys():
        if results[name] is not None:
            print(f"    {name}/")
            print(f"      timeseries.png")
            print(f"      characteristics.png")
    
    print(f"\n{'='*60}")
    print("✓ HOURLY REGIME DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nHourly features: {hourly_output}")
    print(f"Results: {results_dir}")
    print(f"Visualizations: {viz_dir}")
    
    # Show rankings
    print("\n" + "="*60)
    print("FINAL RANKINGS (HOURLY DATA)")
    print("="*60)
    
    # Calculate composite ranks
    ranks = []
    for idx, row in comparison_df.iterrows():
        method = row['method']
        silh_rank = comparison_df['silhouette_score'].rank(ascending=False)[idx]
        vol_sep_rank = comparison_df['volatility_separation'].rank(ascending=False)[idx]
        db_rank = comparison_df['davies_bouldin_score'].rank(ascending=True)[idx]
        avg_rank = (silh_rank + vol_sep_rank + db_rank) / 3
        ranks.append((method.upper(), avg_rank))
    
    ranks.sort(key=lambda x: x[1])
    for i, (method, rank) in enumerate(ranks, 1):
        print(f"  {i}. {method} - Avg Rank: {rank:.1f}")
    
    spark.stop()


if __name__ == "__main__":
    import os
    
    # Find CryptX root
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
    
    try:
        run_hourly_regime_detection(cryptx_root)
        print("\n[SUCCESS] Pipeline complete! Check your results.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()