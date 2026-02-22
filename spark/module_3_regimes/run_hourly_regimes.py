from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from hmm_regime import HMMRegimeDetector
from kmeans_regime import KMeansRegimeDetector
from gmm_regime import GMMRegimeDetector
from threshold_regime import ThresholdRegimeDetector
from patchtst_regime import PatchTSTRegimeDetector
from informer_regime import InformerRegimeDetector
from garch_regime import GARCHRegimeDetector
from regime_evaluator import RegimeEvaluator
from regime_visualizer import RegimeVisualizer


def aggregate_to_hourly(spark, minute_data_path):
    print("\n" + "="*60)
    print("STEP 1: AGGREGATING MINUTE DATA TO HOURLY")
    print("="*60)

    print(f"[INFO] Loading minute data from {minute_data_path}")
    df_minute = spark.read.parquet(str(minute_data_path))

    minute_count = df_minute.count()
    date_range = df_minute.agg(
        F.min('timestamp').alias('start'),
        F.max('timestamp').alias('end')
    ).collect()[0]
    print(f"[INFO] Loaded {minute_count:,} minute bars")
    print(f"[INFO] Date range: {date_range['start']} → {date_range['end']}")

    df_hour = df_minute.withColumn("hour_ts", F.date_trunc("hour", F.col("timestamp")))

    print("[INFO] Aggregating to hourly OHLCV...")
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
    df_hourly = df_hourly.withColumnRenamed("hour_ts", "timestamp").orderBy("timestamp")

    hourly_count = df_hourly.count()
    print(f"[INFO] Created {hourly_count:,} hourly bars")
    print(f"[INFO] Compression ratio: {minute_count/hourly_count:.1f}:1")

    return df_hourly, hourly_count


def compute_hourly_features(df_hourly):
    print("\n" + "="*60)
    print("STEP 2: COMPUTING HOURLY FEATURES")
    print("="*60)

    window = Window.orderBy("timestamp")

    print("[INFO] Computing returns...")
    df = df_hourly.withColumn("prev_close", F.lag("price_close", 1).over(window))
    df = df.withColumn("log_return", F.log(F.col("price_close") / F.col("prev_close")))

    print("[INFO] Computing volatility features...")
    window_24h  = Window.orderBy("timestamp").rowsBetween(-23, 0)
    window_168h = Window.orderBy("timestamp").rowsBetween(-167, 0)
    window_720h = Window.orderBy("timestamp").rowsBetween(-719, 0)

    df = df.withColumn("vol_24h",  F.stddev("log_return").over(window_24h))
    df = df.withColumn("vol_168h", F.stddev("log_return").over(window_168h))
    df = df.withColumn("vol_720h", F.stddev("log_return").over(window_720h))

    df = df.withColumn("vol_ratio_24_168",
                       F.when(F.col("vol_168h") > 0, F.col("vol_24h") / F.col("vol_168h"))
                        .otherwise(1.0))
    df = df.withColumn("vol_ratio_168_720",
                       F.when(F.col("vol_720h") > 0, F.col("vol_168h") / F.col("vol_720h"))
                        .otherwise(1.0))

    # --- Alias columns so detectors (written for minute data) work unchanged ---
    print("[INFO] Adding detector-compatibility alias columns...")
    df = df.withColumn("vol_60m",          F.col("vol_24h"))
    df = df.withColumn("vol_240m",         F.col("vol_168h"))
    df = df.withColumn("vol_ratio_60_240", F.col("vol_ratio_24_168"))

    print("[INFO] Computing return features...")
    df = df.withColumn("cum_return_24h",  F.sum("log_return").over(window_24h))
    df = df.withColumn("cum_return_168h", F.sum("log_return").over(window_168h))
    df = df.withColumn("cum_return_720h", F.sum("log_return").over(window_720h))
    df = df.withColumn("cum_return_60m",  F.col("cum_return_24h"))

    df = df.withColumn("return_skew_24h",  F.skewness("log_return").over(window_24h))
    df = df.withColumn("return_skew_168h", F.skewness("log_return").over(window_168h))
    df = df.withColumn("return_skew_60m",  F.col("return_skew_24h"))

    print("[INFO] Computing volume features...")
    df = df.withColumn("vol_avg_24h",  F.mean("volume_base").over(window_24h))
    df = df.withColumn("vol_avg_168h", F.mean("volume_base").over(window_168h))
    df = df.withColumn("volume_spike",
                       F.when(F.col("vol_avg_24h") > 0,
                              F.col("volume_base") / F.col("vol_avg_24h"))
                        .otherwise(1.0))

    print("[INFO] Computing technical indicators...")
    df = df.withColumn("price_change",
                       F.col("price_close") - F.lag("price_close", 1).over(window))
    df = df.withColumn("gain", F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0))
    df = df.withColumn("loss", F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0))

    window_14h = Window.orderBy("timestamp").rowsBetween(-13, 0)
    df = df.withColumn("avg_gain", F.mean("gain").over(window_14h))
    df = df.withColumn("avg_loss", F.mean("loss").over(window_14h))
    df = df.withColumn("rs",
                       F.when(F.col("avg_loss") > 0, F.col("avg_gain") / F.col("avg_loss"))
                        .otherwise(100))
    df = df.withColumn("rsi", 100 - (100 / (1 + F.col("rs"))))

    df = df.withColumn("true_range", F.greatest(
        F.col("price_high") - F.col("price_low"),
        F.abs(F.col("price_high") - F.col("prev_close")),
        F.abs(F.col("price_low") - F.col("prev_close"))
    ))
    df = df.withColumn("atr", F.mean("true_range").over(window_14h))
    df = df.withColumn("atr_pct", (F.col("atr") / F.col("price_close")) * 100)

    window_20h = Window.orderBy("timestamp").rowsBetween(-19, 0)
    df = df.withColumn("bb_middle", F.mean("price_close").over(window_20h))
    df = df.withColumn("bb_std",    F.stddev("price_close").over(window_20h))
    df = df.withColumn("bb_width",  (F.col("bb_std") * 2) / F.col("bb_middle"))

    df = df.withColumn("hl_range_24h",
                       F.mean((F.col("price_high") - F.col("price_low")) / F.col("price_close"))
                        .over(window_24h))

    df = df.drop("price_change", "gain", "loss", "avg_gain", "avg_loss",
                 "rs", "true_range", "bb_middle", "bb_std", "prev_close")

    df = df.withColumn("year",  F.year("timestamp"))
    df = df.withColumn("month", F.month("timestamp"))
    df = df.withColumn("day",   F.dayofmonth("timestamp"))
    df = df.withColumn("hour",  F.hour("timestamp"))

    df = df.dropna()
    final_count = df.count()
    print(f"[INFO] Final hourly data: {final_count:,} bars with {len(df.columns)} columns")

    return df, final_count


def pick_detector_settings(hourly_count: int) -> dict:
    """
    Auto-scale DL model settings to dataset size.

    < 6 months  (~4,400 bars)  → small settings (what you had before)
    6-18 months (~8,700 bars)  → medium settings
    18+ months  (~17,500 bars) → large settings (target after downloading 24 months)

    seq_len must be divisible by patch_len for PatchTST.
    All seq_lens below use patch_len=24 → integer patch counts.
    """
    print(f"\n[INFO] Auto-selecting model settings for {hourly_count:,} hourly bars...")

    if hourly_count < 6_000:
        label        = "small (< 6 months)"
        seq_len      = 168    # 1 week  → 7 patches
        epochs_patch = 20
        epochs_info  = 15
        batch_size   = 256

    elif hourly_count < 13_000:
        label        = "medium (6-18 months)"
        seq_len      = 336    # 2 weeks → 14 patches
        epochs_patch = 30
        epochs_info  = 25
        batch_size   = 512

    else:
        label        = "large (18+ months)"
        seq_len      = 504    # 3 weeks → 21 patches
        epochs_patch = 50
        epochs_info  = 35
        batch_size   = 512

    patch_len = 24   # always 24 so seq_len/patch_len is always an integer

    print(f"  Dataset size:  {label}")
    print(f"  seq_len:       {seq_len} hours ({seq_len//24} days)")
    print(f"  patch_len:     {patch_len}  →  {seq_len//patch_len} patches")
    print(f"  epochs (patch/informer): {epochs_patch} / {epochs_info}")
    print(f"  batch_size:    {batch_size}")

    return dict(seq_len=seq_len, patch_len=patch_len,
                epochs_patch=epochs_patch, epochs_info=epochs_info,
                batch_size=batch_size)


def run_hourly_regime_detection(base_path):
    base_path = Path(base_path)

    spark = SparkSession.builder \
        .appName("Hourly-Regime-Detection") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()

    minute_path = base_path / "data" / "features" / "BTCUSDT" / "1m"
    df_hourly, hourly_count         = aggregate_to_hourly(spark, minute_path)
    df_hourly_features, final_count = compute_hourly_features(df_hourly)

    hourly_output = base_path / "data" / "features" / "BTCUSDT" / "1h"
    hourly_output.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Saving hourly features to {hourly_output}")
    df_hourly_features.coalesce(4).write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(str(hourly_output))
    print("[INFO] Hourly features saved")

    settings = pick_detector_settings(final_count)

    print("\n" + "="*60)
    print("STEP 3: INITIALIZING REGIME DETECTORS")
    print("="*60)

    detectors = {
        'threshold': ThresholdRegimeDetector(),
        'hmm':       HMMRegimeDetector(n_regimes=3),
        'kmeans':    KMeansRegimeDetector(n_regimes=3),
        'gmm':       GMMRegimeDetector(n_regimes=3),
        'patchtst':  PatchTSTRegimeDetector(
                         n_regimes=3,
                         seq_len=settings['seq_len'],
                         patch_len=settings['patch_len'],
                         epochs=settings['epochs_patch'],
                         batch_size=settings['batch_size']
                     ),
        'garch':     GARCHRegimeDetector(n_regimes=3),
        'informer':  InformerRegimeDetector(
                         n_regimes=3,
                         seq_len=settings['seq_len'],
                         epochs=settings['epochs_info'],
                         batch_size=settings['batch_size']
                     ),
    }
    print(f"[INFO] Initialized {len(detectors)} detectors")

    print("\n" + "="*60)
    print("STEP 4: DETECTING REGIMES ON HOURLY DATA")
    print("="*60)

    results = {}
    for name, detector in detectors.items():
        print(f"\n[{name.upper()}] Running detection...")
        try:
            df_regime = detector.detect(df_hourly_features)
            results[name] = df_regime
            total = df_regime.count()
            for row in df_regime.groupBy('regime').count().orderBy('regime').collect():
                print(f"  Regime {row['regime']}: {row['count']:,} ({row['count']/total*100:.1f}%)")
        except Exception as e:
            print(f"[{name.upper()}] ERROR: {e}")
            import traceback; traceback.print_exc()
            results[name] = None

    succeeded = sum(1 for r in results.values() if r is not None)
    print(f"\n{'='*60}")
    print(f"✓ Completed {succeeded}/{len(detectors)} methods")
    print("="*60)

    print("\n" + "="*60)
    print("STEP 5: EVALUATING REGIME QUALITY")
    print("="*60)
    print("[INFO] Focus on these metrics for your research paper:")
    print("[INFO]   silhouette_score       → cluster separation quality")
    print("[INFO]   volatility_separation  → how distinct regimes are by vol")
    print("[INFO]   davies_bouldin_score   → compact, well-separated clusters")

    evaluator = RegimeEvaluator()
    evaluation_results = []
    for name, df_regime in results.items():
        if df_regime is None:
            continue
        print(f"\n[{name.upper()}] Evaluating...")
        try:
            metrics = evaluator.evaluate(df_regime, df_hourly_features)
            metrics['method'] = name
            evaluation_results.append(metrics)
        except Exception as e:
            print(f"[{name.upper()}] Evaluation ERROR: {e}")

    import pandas as pd
    comparison_df = pd.DataFrame(evaluation_results)
    if comparison_df.empty:
        print("[ERROR] No evaluation results.")
        spark.stop()
        return

    cols = ['method'] + [c for c in comparison_df.columns if c != 'method']
    comparison_df = comparison_df[cols]

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(comparison_df.to_string(index=False))

    results_dir = base_path / "results" / "regimes_hourly"
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(results_dir / "regime_comparison_hourly.csv", index=False)
    print(f"\n[INFO] Saved to {results_dir / 'regime_comparison_hourly.csv'}")

    print("\n" + "="*60)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("="*60)
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    visualizer = RegimeVisualizer()

    for name, df_regime in results.items():
        if df_regime is None:
            continue
        model_dir = viz_dir / name
        model_dir.mkdir(exist_ok=True)
        print(f"\n[{name.upper()}] Creating visualizations...")
        try:
            visualizer.plot_regime_timeseries(
                df_regime, method_name=name,
                output_path=model_dir / "timeseries.png", sample_size=5000)
            visualizer.plot_regime_characteristics(
                df_regime, method_name=name,
                output_path=model_dir / "characteristics.png")
        except Exception as e:
            print(f"[{name.upper()}] Visualization ERROR: {e}")

    try:
        visualizer.plot_detailed_comparison(
            comparison_df,
            output_path=viz_dir / "detailed_comparison_hourly.png")
    except Exception as e:
        print(f"[INFO] Comparison plot ERROR: {e}")

    print("\n" + "="*60)
    print("FINAL RANKINGS")
    print("="*60)
    ranks = []
    for idx, row in comparison_df.iterrows():
        avg_rank = (
            comparison_df['silhouette_score'].rank(ascending=False)[idx] +
            comparison_df['volatility_separation'].rank(ascending=False)[idx] +
            comparison_df['davies_bouldin_score'].rank(ascending=True)[idx]
        ) / 3
        ranks.append((row['method'].upper(), avg_rank))
    for i, (method, rank) in enumerate(sorted(ranks, key=lambda x: x[1]), 1):
        print(f"  {i}. {method} — Avg Rank: {rank:.1f}")

    spark.stop()


if __name__ == "__main__":
    import os
    current = Path(__file__).resolve()
    cryptx_root = None
    for parent in current.parents:
        if parent.name == "CryptX":
            cryptx_root = parent
            break
    if cryptx_root is None:
        cryptx_root = Path(".") if os.path.exists("./data/features") else None
    if cryptx_root is None:
        print("[ERROR] Cannot find CryptX directory!")
        exit(1)

    print(f"[INFO] CryptX root: {cryptx_root}\n")
    try:
        run_hourly_regime_detection(cryptx_root)
        print("\n[SUCCESS] Pipeline complete!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback; traceback.print_exc()