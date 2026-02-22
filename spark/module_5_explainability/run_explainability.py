"""
run_explainability.py  —  Module 5

Explainability & Regime Analysis Pipeline

Produces:
  1. Feature importance (permutation + SHAP) globally and per regime
  2. Regime-specific volatility characteristics and return distributions
  3. Transition probability matrix and persistence analysis
  4. Regime duration distributions
  5. Transition trigger analysis (what changes before a regime switch)

Results saved to:
  results/explainability/<SYMBOL>/
    feature_importance/
      global_importance.png
      per_regime_importance.png
      shap_summary.png          (if shap installed)
      global_importance.csv
      per_regime_importance.csv
    regime_characteristics/
      characteristics_heatmap.png
      return_distributions.png
      regime_stats.csv
    transitions/
      transition_matrix.png
      duration_distributions.png
      transition_triggers.png
      transition_matrix.csv
      regime_durations.csv
      transition_triggers.csv

Usage (standalone):
    python spark/module_5_explainability/run_explainability.py

Usage (importable):
    from run_explainability import run_explainability
    results = run_explainability(cryptx_root, symbol='BTCUSDT')
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
current     = Path(__file__).resolve()
cryptx_root = None
for parent in current.parents:
    if parent.name == "CryptX":
        cryptx_root = parent
        break
if cryptx_root is None:
    import os
    cryptx_root = Path(".") if os.path.exists("./data/features") else None

if cryptx_root:
    sys.path.insert(0, str(cryptx_root / "spark" / "module_3_regimes"))
    sys.path.insert(0, str(cryptx_root / "spark" / "module_5_explainability"))

from feature_importance import FeatureImportanceAnalyzer
from regime_analyzer import RegimeCharacteristicsAnalyzer


# ======================================================================
# Data loading — re-runs GARCH detector (same as Module 4)
# ======================================================================

def load_data_with_regimes(cryptx_root: Path,
                            symbol: str = 'BTCUSDT') -> pd.DataFrame:
    """
    Load hourly features and re-detect regimes using GARCH detector.
    Returns a pandas DataFrame with all features + regime column.
    """
    print(f"\n{'='*60}")
    print(f"LOADING DATA: {symbol}")
    print(f"{'='*60}")

    features_path = cryptx_root / "data" / "features" / symbol / "1h"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Hourly features not found at {features_path}\n"
            f"Run run_hourly_regimes.py first."
        )

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder \
        .appName(f"Module5-Explainability-{symbol}") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print(f"[INFO] Loading from {features_path}")
    df_spark = spark.read.parquet(str(features_path))

    count      = df_spark.count()
    date_range = df_spark.agg(
        F.min('timestamp').alias('s'),
        F.max('timestamp').alias('e')
    ).collect()[0]
    print(f"[INFO] {count:,} bars | {date_range['s']} → {date_range['e']}")

    print(f"[INFO] Re-running GARCH detector for regime labels...")
    from garch_regime import GARCHRegimeDetector
    detector        = GARCHRegimeDetector(n_regimes=3)
    df_with_regimes = detector.detect(df_spark)

    # Pull all useful columns to pandas
    keep_cols = [
        'timestamp', 'log_return', 'regime',
        'vol_60m', 'vol_240m', 'vol_ratio_60_240',
        'cum_return_60m', 'return_skew_60m',
        'volume_spike', 'rsi', 'bb_width', 'atr_pct',
        'price_close', 'volume_base'
    ]
    avail = [c for c in keep_cols if c in df_with_regimes.columns]

    df_pd = (df_with_regimes
             .select(avail)
             .toPandas()
             .sort_values('timestamp')
             .reset_index(drop=True))

    df_pd['regime'] = df_pd['regime'].fillna(0).astype(int)

    print(f"[INFO] Regime distribution:")
    for r, cnt in df_pd['regime'].value_counts().sort_index().items():
        print(f"  Regime {r}: {cnt:,} ({cnt/len(df_pd)*100:.1f}%)")

    spark.stop()
    return df_pd


# ======================================================================
# Main pipeline
# ======================================================================

def run_explainability(cryptx_root: Path,
                        symbol: str = 'BTCUSDT') -> dict:
    """
    Run full explainability and regime analysis pipeline.

    Args:
        cryptx_root: Path to CryptX project root
        symbol:      Trading pair (e.g. 'BTCUSDT', 'ETHUSDT')

    Returns:
        dict with keys: 'stats', 'transition_matrix', 'durations',
                        'triggers', 'global_importance'
    """
    print("\n" + "="*60)
    print(f"MODULE 5: EXPLAINABILITY & REGIME ANALYSIS")
    print("="*60)
    print(f"Symbol: {symbol}")
    print("="*60)

    # ── Output directories ─────────────────────────────────────────────
    results_dir = cryptx_root / "results" / "explainability" / symbol
    fi_dir      = results_dir / "feature_importance"
    rc_dir      = results_dir / "regime_characteristics"
    tr_dir      = results_dir / "transitions"

    for d in [fi_dir, rc_dir, tr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    df      = load_data_with_regimes(cryptx_root, symbol)
    regimes = df['regime'].values

    # ── Step 1: Feature Importance ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    fi_analyzer = FeatureImportanceAnalyzer(symbol=symbol, n_regimes=3)
    fi_analyzer.fit(df, regimes)

    # Global importance
    global_imp = fi_analyzer.global_importance()
    print(f"\n[FeatureImportance] Top 5 features (global):")
    print(global_imp[['rank', 'feature', 'perm_importance', 'gini_importance']]
          .head(5).to_string(index=False))

    fi_analyzer.plot_global_importance(fi_dir / "global_importance.png")
    fi_analyzer.plot_per_regime_importance(fi_dir / "per_regime_importance.png")

    if fi_analyzer.shap_available:
        fi_analyzer.plot_shap_summary(fi_dir / "shap_summary.png")
        shap_imp = fi_analyzer.shap_importance()
        for r, df_shap in shap_imp.items():
            df_shap.to_csv(fi_dir / f"shap_importance_regime_{r}.csv", index=False)

    # Save CSVs
    global_imp.to_csv(fi_dir / "global_importance.csv", index=False)
    regime_imp = fi_analyzer.per_regime_importance()
    for r, df_r in regime_imp.items():
        df_r.to_csv(fi_dir / f"per_regime_importance_{r}.csv", index=False)

    print(f"[INFO] Feature importance saved to {fi_dir}")

    # ── Step 2: Regime Characteristics ────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: REGIME CHARACTERISTICS")
    print("="*60)

    rc_analyzer = RegimeCharacteristicsAnalyzer(symbol=symbol, n_regimes=3)
    stats_df    = rc_analyzer.compute_regime_stats(df, regimes)

    print(f"\n[RegimeStats] Summary table:")
    display_cols = ['regime_name', 'n_bars', 'pct_of_total',
                    'return_mean', 'return_std', 'vol_mean',
                    'sharpe_hourly', 'rsi_mean', 'autocorr_lag1']
    avail_display = [c for c in display_cols if c in stats_df.columns]
    print(stats_df[avail_display].to_string(index=False))

    rc_analyzer.plot_regime_characteristics(stats_df,
                                             rc_dir / "characteristics_heatmap.png")
    rc_analyzer.plot_return_distributions(df, regimes,
                                          rc_dir / "return_distributions.png")

    stats_df.to_csv(rc_dir / "regime_stats.csv", index=False)
    print(f"[INFO] Regime characteristics saved to {rc_dir}")

    # ── Step 3: Transition Analysis ────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: TRANSITION ANALYSIS")
    print("="*60)

    # Transition matrix
    trans_matrix = rc_analyzer.compute_transition_matrix(regimes)
    print(f"\n[Transitions] Transition probability matrix:")
    print(trans_matrix.to_string())

    # Regime durations
    durations = rc_analyzer.compute_regime_durations(regimes)
    print(f"\n[Transitions] Regime duration statistics:")
    dur_rows = []
    for r, d in durations.items():
        if not d:
            continue
        d_arr = np.array(d)
        print(f"  Regime {r} ({fi_analyzer.shap_available and 'shap+perm' or 'perm'}): "
              f"mean={d_arr.mean():.1f}h  "
              f"median={np.median(d_arr):.1f}h  "
              f"max={d_arr.max()}h  "
              f"n_spells={len(d_arr)}")
        for spell_len in d_arr:
            dur_rows.append({'regime': r, 'duration_hours': spell_len})

    # Transition triggers
    triggers_df = rc_analyzer.compute_transition_triggers(df, regimes)
    if not triggers_df.empty:
        print(f"\n[Transitions] Most common transitions:")
        print(triggers_df[['from_name', 'to_name', 'n_transitions']]
              .to_string(index=False))

    # Plots
    rc_analyzer.plot_transition_matrix(trans_matrix,
                                        tr_dir / "transition_matrix.png")
    rc_analyzer.plot_duration_distributions(durations,
                                             tr_dir / "duration_distributions.png")
    if not triggers_df.empty:
        rc_analyzer.plot_transition_triggers(triggers_df,
                                              tr_dir / "transition_triggers.png")

    # Save CSVs
    trans_matrix.to_csv(tr_dir / "transition_matrix.csv")
    if dur_rows:
        pd.DataFrame(dur_rows).to_csv(tr_dir / "regime_durations.csv", index=False)
    if not triggers_df.empty:
        triggers_df.to_csv(tr_dir / "transition_triggers.csv", index=False)

    print(f"[INFO] Transition analysis saved to {tr_dir}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"MODULE 5 COMPLETE — {symbol}")
    print("="*60)
    print(f"\nKey findings:")

    if not stats_df.empty:
        highest_vol = stats_df.loc[stats_df['vol_mean'].idxmax(), 'regime_name']
        most_common = stats_df.loc[stats_df['n_bars'].idxmax(),   'regime_name']
        print(f"  Most volatile regime:  {highest_vol}")
        print(f"  Most common regime:    {most_common}")

    diagonal = np.diag(trans_matrix.values)
    most_sticky = trans_matrix.index[np.argmax(diagonal)]
    print(f"  Most persistent regime: {most_sticky} "
          f"(P(stay)={diagonal.max():.1%})")

    print(f"\n  All results saved to: {results_dir}")
    print("="*60)

    return {
        'stats':             stats_df,
        'transition_matrix': trans_matrix,
        'durations':         durations,
        'triggers':          triggers_df,
        'global_importance': global_imp,
    }


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    import os

    # Find CryptX root
    current     = Path(__file__).resolve()
    cryptx_root = None
    for parent in current.parents:
        if parent.name == "CryptX":
            cryptx_root = parent
            break
    if cryptx_root is None:
        cryptx_root = Path(".") if os.path.exists("./data/features") else None
    if cryptx_root is None:
        print("[ERROR] Cannot find CryptX root directory.")
        exit(1)

    print(f"[INFO] CryptX root: {cryptx_root}")

    try:
        results = run_explainability(
            cryptx_root=cryptx_root,
            symbol='BTCUSDT'    # change to 'ETHUSDT' etc. when you add more assets
        )
        print("\n[SUCCESS] Module 5 complete!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()