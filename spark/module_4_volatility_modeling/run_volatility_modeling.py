"""
run_volatility_modeling.py  —  Module 4

Volatility Modeling Pipeline

Compares two models:
  1. Baseline GARCH(1,1)        — single model, no regime info
  2. Regime-Switching GARCH(1,1) — separate model per detected regime

The comparison is the core empirical result of the research:
  Does knowing the current volatility regime improve forecast accuracy?

Forecasts produced:
  - Rolling 1-step ahead (OOS, 20% holdout)
  - Multi-step ahead: 24h, 48h, 168h from the last known bar

Results saved to:
  results/volatility_modeling/
    comparison_metrics.csv
    forecasts_combined.csv
    forecasts_ahead_24h.csv
    forecasts_ahead_48h.csv
    forecasts_ahead_168h.csv
    visualizations/
      forecast_comparison.png
      regime_forecast_breakdown.png
      metrics_comparison.png
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Allow imports from other modules
current = Path(__file__).resolve()
for parent in current.parents:
    if parent.name == "CryptX":
        cryptx_root = parent
        break
else:
    cryptx_root = Path(".")

sys.path.insert(0, str(cryptx_root / "spark" / "module_4_volatility_modeling"))

from baseline_garch import BaselineGARCH
from regime_switching_garch import RegimeSwitchingGARCH


# ======================================================================
# Data loading
# ======================================================================

def load_hourly_with_regimes(cryptx_root: Path, regime_method: str = 'garch'):
    """
    Load hourly features and assign regime labels by re-running the
    GARCH detector (Module 3 does not save regime parquets to disk).
    """
    print(f"\n{'='*60}")
    print(f"LOADING DATA + DETECTING REGIMES")
    print(f"{'='*60}")

    features_path = cryptx_root / "data" / "features" / "BTCUSDT" / "1h"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Hourly features not found at {features_path}\n"
            f"Run run_hourly_regimes.py first."
        )

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    spark = SparkSession.builder \
        .appName("Module4-VolatilityModeling") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    print(f"[INFO] Loading hourly features from {features_path}")
    df_spark = spark.read.parquet(str(features_path))

    count = df_spark.count()
    date_range = df_spark.agg(
        F.min('timestamp').alias('start'),
        F.max('timestamp').alias('end')
    ).collect()[0]
    print(f"[INFO] Loaded {count:,} hourly bars")
    print(f"[INFO] Date range: {date_range['start']} → {date_range['end']}")

    # Re-run GARCH detector to get regime labels
    # (Module 3 does not persist regime parquets — regimes live in memory only)
    print(f"\n[INFO] Re-running GARCH detector to assign regime labels...")
    garch_detector_path = cryptx_root / "spark" / "module_3_regimes"
    sys.path.insert(0, str(garch_detector_path))
    from garch_regime import GARCHRegimeDetector

    detector        = GARCHRegimeDetector(n_regimes=3)
    df_with_regimes = detector.detect(df_spark)

    df_pd = (df_with_regimes
             .select('timestamp', 'log_return', 'regime')
             .toPandas()
             .sort_values('timestamp')
             .reset_index(drop=True))

    df_pd['regime'] = df_pd['regime'].fillna(0).astype(int)

    print(f"\n[INFO] Regime distribution:")
    for r, cnt in df_pd['regime'].value_counts().sort_index().items():
        print(f"  Regime {r}: {cnt:,} ({cnt/len(df_pd)*100:.1f}%)")

    spark.stop()
    return df_pd


# ======================================================================
# Visualisation
# ======================================================================

def plot_forecast_comparison(baseline_df: pd.DataFrame,
                              rs_df: pd.DataFrame,
                              output_path: Path):
    """Side-by-side forecast vs realised variance."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Volatility Forecast Comparison\n'
                 'Baseline GARCH(1,1) vs Regime-Switching GARCH(1,1)',
                 fontsize=14, fontweight='bold')

    combined = baseline_df.merge(
        rs_df[['timestamp', 'rs_forecast', 'rs_vol', 'regime']],
        on='timestamp', how='inner')
    combined = combined.dropna()

    ts = combined['timestamp']
    rv = combined['realised_var']

    # Panel 1: Variance forecasts
    ax = axes[0]
    ax.plot(ts, rv,                          color='black',  alpha=0.4,
            lw=0.8, label='Realised Variance (r²)')
    ax.plot(ts, combined['baseline_forecast'], color='#e74c3c', alpha=0.8,
            lw=1.2, label='Baseline GARCH')
    ax.plot(ts, combined['rs_forecast'],       color='#2ecc71', alpha=0.8,
            lw=1.2, label='RS-GARCH')
    ax.set_ylabel('Conditional Variance')
    ax.set_title('Conditional Variance Forecasts')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Volatility (sqrt of variance)
    ax = axes[1]
    ax.plot(ts, combined['baseline_vol'], color='#e74c3c', alpha=0.8,
            lw=1.2, label='Baseline GARCH Vol')
    ax.plot(ts, combined['rs_vol'],       color='#2ecc71', alpha=0.8,
            lw=1.2, label='RS-GARCH Vol')
    ax.set_ylabel('Conditional Volatility (σ)')
    ax.set_title('Conditional Volatility Forecasts')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Regime overlay
    ax = axes[2]
    regime_colors = {0: '#3498db', 1: '#f39c12', 2: '#e74c3c'}
    regime_names  = {0: 'Low Vol', 1: 'Medium Vol', 2: 'High Vol'}
    for r, color in regime_colors.items():
        mask = combined['regime'] == r
        if mask.any():
            ax.fill_between(ts, 0, 1,
                            where=mask,
                            transform=ax.get_xaxis_transform(),
                            alpha=0.3, color=color,
                            label=regime_names[r])
    ax.plot(ts, np.sqrt(rv), color='black', alpha=0.5, lw=0.8,
            label='Realised Vol')
    ax.set_ylabel('Volatility')
    ax.set_title('Regimes (shaded) vs Realised Volatility')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {output_path.name}")


def plot_metrics_comparison(baseline_metrics: dict,
                             rs_metrics: dict,
                             output_path: Path):
    """Bar chart comparing key metrics."""
    metric_labels = {
        'rmse':     'RMSE ↓',
        'mae':      'MAE ↓',
        'qlike':    'QLIKE ↓',
        'corr':     'Correlation ↑',
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Forecast Accuracy: Baseline GARCH vs RS-GARCH',
                 fontsize=13, fontweight='bold')

    colors = ['#e74c3c', '#2ecc71']
    models = ['Baseline\nGARCH', 'RS-GARCH']

    for ax, (metric, label) in zip(axes, metric_labels.items()):
        b_val = baseline_metrics.get(metric, 0)
        r_val = rs_metrics.get(metric, 0)
        bars  = ax.bar(models, [b_val, r_val], color=colors, width=0.5,
                       edgecolor='black', linewidth=0.8)

        # Highlight winner
        if metric == 'corr':
            winner = 0 if b_val > r_val else 1
        else:
            winner = 0 if b_val < r_val else 1

        bars[winner].set_edgecolor('gold')
        bars[winner].set_linewidth(3)

        # Improvement %
        if b_val != 0:
            if metric == 'corr':
                pct = (r_val - b_val) / abs(b_val) * 100
            else:
                pct = (b_val - r_val) / abs(b_val) * 100
            ax.set_title(f'{label}\n{"RS-GARCH +" if pct > 0 else ""}'
                         f'{abs(pct):.1f}%'
                         f'{"↑" if pct > 0 else "↓"}',
                         fontsize=10)
        else:
            ax.set_title(label)

        ax.set_ylabel(metric.upper())
        for bar, val in zip(bars, [b_val, r_val]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {output_path.name}")


def plot_regime_forecast_breakdown(rs_df: pd.DataFrame,
                                   regime_params: dict,
                                   output_path: Path):
    """Per-regime GARCH parameter summary and forecast distribution."""
    n_regimes = len(regime_params)
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('RS-GARCH: Per-Regime Analysis', fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, n_regimes, figure=fig)

    regime_colors = ['#3498db', '#f39c12', '#e74c3c']
    regime_names  = ['Low Volatility', 'Medium Volatility', 'High Volatility']

    for r in range(n_regimes):
        color = regime_colors[r]
        name  = regime_names[r] if r < 3 else f'Regime {r}'

        # Top: parameter bars
        ax_top = fig.add_subplot(gs[0, r])
        params = regime_params.get(r, (0, 0, 0))
        omega, alpha, beta = params
        labels = ['ω (omega)', 'α (alpha)', 'β (beta)', 'α+β']
        values = [omega * 1e5, alpha, beta, alpha + beta]  # scale omega for vis
        bars   = ax_top.bar(labels, values, color=color, alpha=0.8,
                            edgecolor='black', linewidth=0.8)
        ax_top.set_title(f'{name}\nomega={omega:.2e}\n'
                         f'α={alpha:.3f}  β={beta:.3f}  α+β={alpha+beta:.3f}',
                         fontsize=9)
        ax_top.set_ylim(0, 1.1)
        ax_top.axhline(1.0, color='red', linestyle='--', alpha=0.5, lw=1)
        ax_top.grid(True, alpha=0.3, axis='y')
        if r == 0:
            ax_top.set_ylabel('Parameter Value\n(omega ×10⁻⁵)')

        # Bottom: forecast vol distribution
        ax_bot = fig.add_subplot(gs[1, r])
        mask   = rs_df['regime'] == r
        if mask.any():
            vols = rs_df.loc[mask, 'rs_vol'].dropna()
            ax_bot.hist(vols, bins=50, color=color, alpha=0.7,
                        edgecolor='black', linewidth=0.3)
            ax_bot.axvline(vols.mean(), color='black', linestyle='--',
                           lw=1.5, label=f'Mean: {vols.mean():.4f}')
            ax_bot.legend(fontsize=8)
        ax_bot.set_xlabel('Forecast Volatility (σ)')
        if r == 0:
            ax_bot.set_ylabel('Frequency')
        ax_bot.set_title(f'Forecast Vol Distribution\n(n={mask.sum():,})', fontsize=9)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {output_path.name}")


def plot_multi_step_forecast(forecasts_dict: dict, output_path: Path):
    """Plot 24h, 48h, 168h ahead forecasts."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('RS-GARCH Multi-Step Ahead Volatility Forecast',
                 fontsize=13, fontweight='bold')

    horizon_labels = {'24h': '24 Hours Ahead', '48h': '48 Hours Ahead',
                      '168h': '1 Week Ahead'}
    regime_colors  = {0: '#3498db', 1: '#f39c12', 2: '#e74c3c'}

    for ax, (key, label) in zip(axes, horizon_labels.items()):
        df = forecasts_dict.get(key)
        if df is None:
            continue
        color = regime_colors.get(int(df['active_regime'].iloc[0]), '#7f8c8d')
        ax.plot(df['step'], df['forecast_vol'], color=color, lw=2, marker='o',
                markersize=4, label=f'Regime {df["active_regime"].iloc[0]}')
        ax.fill_between(df['step'],
                        df['forecast_vol'] * 0.9,
                        df['forecast_vol'] * 1.1,
                        color=color, alpha=0.2, label='±10% band')
        ax.set_xlabel('Steps Ahead (hours)')
        ax.set_ylabel('Forecast Volatility (σ)')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved: {output_path.name}")


# ======================================================================
# Main pipeline
# ======================================================================

def run_volatility_modeling(cryptx_root: Path, regime_method: str = 'garch'):
    print("\n" + "="*60)
    print("MODULE 4: VOLATILITY MODELING PIPELINE")
    print("="*60)
    print(f"Regime source:  {regime_method.upper()} detector")
    print(f"Models:         Baseline GARCH(1,1) vs RS-GARCH(1,1)")
    print(f"Horizons:       1-step OOS + 24h / 48h / 168h ahead")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────────────
    df = load_hourly_with_regimes(cryptx_root, regime_method)
    returns = df['log_return'].fillna(0).values
    regimes = df['regime'].values
    timestamps = df['timestamp'].values

    # ── Baseline GARCH ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1: BASELINE GARCH(1,1)")
    print("="*60)
    baseline = BaselineGARCH(train_frac=0.8, refit_every=168)
    baseline.fit(returns)
    baseline_metrics = baseline.evaluate()
    baseline_df      = baseline.get_forecast_df(timestamps)

    # ── Regime-Switching GARCH ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2: REGIME-SWITCHING GARCH(1,1)")
    print("="*60)
    rs_garch = RegimeSwitchingGARCH(n_regimes=3, train_frac=0.8, refit_every=168)
    rs_garch.fit(returns, regimes)
    rs_metrics = rs_garch.evaluate()
    rs_df      = rs_garch.get_forecast_df(timestamps)

    # ── Save models to disk ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: SAVING MODELS")
    print("="*60)
    models_dir = cryptx_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(models_dir / "baseline_garch.pkl")
    rs_garch.save(models_dir / "regime_switching_garch.pkl")
    print(f"[INFO] Models saved to {models_dir}")

    # ── Multi-step forecasts ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 3: MULTI-STEP AHEAD FORECASTS")
    print("="*60)
    forecasts_ahead = {}
    for horizon in [24, 48, 168]:
        fc = rs_garch.forecast_ahead(steps=horizon)
        forecasts_ahead[f'{horizon}h'] = fc
        print(f"  {horizon}h ahead — "
              f"regime={fc['active_regime'].iloc[0]} | "
              f"vol_t+1={fc['forecast_vol'].iloc[0]:.6f} | "
              f"vol_t+{horizon}={fc['forecast_vol'].iloc[-1]:.6f}")

    # ── Print comparison ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS: BASELINE vs REGIME-SWITCHING GARCH")
    print("="*60)

    metrics_to_show = ['rmse', 'mae', 'mape_pct', 'qlike', 'corr']
    improvement_direction = {'rmse': -1, 'mae': -1, 'mape_pct': -1,
                              'qlike': -1, 'corr': +1}

    print(f"\n  {'Metric':<15} {'Baseline':>14} {'RS-GARCH':>14} {'Improvement':>14}")
    print(f"  {'-'*57}")

    for m in metrics_to_show:
        b = baseline_metrics.get(m, 0)
        r = rs_metrics.get(m, 0)
        direction = improvement_direction.get(m, -1)
        if b != 0:
            pct = (r - b) / abs(b) * 100 * direction
            arrow = '✓ better' if pct > 0 else '✗ worse'
            imp_str = f"{pct:+.1f}% {arrow}"
        else:
            imp_str = 'N/A'
        print(f"  {m:<15} {b:>14.6f} {r:>14.6f} {imp_str:>14}")

    # ── Save results ───────────────────────────────────────────────────
    results_dir = cryptx_root / "results" / "volatility_modeling"
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Metrics CSV
    comparison_df = pd.DataFrame([baseline_metrics, rs_metrics])
    comparison_df.to_csv(results_dir / "comparison_metrics.csv", index=False)
    print(f"\n[INFO] Saved comparison_metrics.csv")

    # Forecast CSVs
    combined_df = baseline_df.merge(
        rs_df[['timestamp', 'rs_forecast', 'rs_vol', 'regime']],
        on='timestamp', how='inner')
    combined_df.to_csv(results_dir / "forecasts_combined.csv", index=False)

    for key, fc_df in forecasts_ahead.items():
        fc_df.to_csv(results_dir / f"forecasts_ahead_{key}.csv", index=False)

    print(f"[INFO] Saved forecast CSVs to {results_dir}")

    # ── Visualisations ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4: VISUALIZATIONS")
    print("="*60)

    plot_forecast_comparison(
        baseline_df, rs_df,
        viz_dir / "forecast_comparison.png")

    plot_metrics_comparison(
        baseline_metrics, rs_metrics,
        viz_dir / "metrics_comparison.png")

    plot_regime_forecast_breakdown(
        rs_df, rs_garch.regime_params,
        viz_dir / "regime_forecast_breakdown.png")

    plot_multi_step_forecast(
        forecasts_ahead,
        viz_dir / "multi_step_forecast.png")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODULE 4 COMPLETE")
    print("="*60)

    rmse_imp = (baseline_metrics['rmse'] - rs_metrics['rmse']) / baseline_metrics['rmse'] * 100
    mae_imp  = (baseline_metrics['mae']  - rs_metrics['mae'])  / baseline_metrics['mae']  * 100
    qlike_imp = (baseline_metrics['qlike'] - rs_metrics['qlike']) / abs(baseline_metrics['qlike']) * 100

    if rmse_imp > 0:
        print(f"\n✓ RS-GARCH outperforms Baseline GARCH:")
        print(f"  RMSE improvement:  {rmse_imp:+.1f}%")
        print(f"  MAE  improvement:  {mae_imp:+.1f}%")
        print(f"  QLIKE improvement: {qlike_imp:+.1f}%")
    else:
        print(f"\n✗ Baseline GARCH outperforms RS-GARCH on this dataset.")
        print(f"  RMSE difference: {rmse_imp:+.1f}%")

    print(f"\n  Results saved to: {results_dir}")
    print(f"  Charts saved to:  {viz_dir}")
    print("="*60)

    return baseline_metrics, rs_metrics, forecasts_ahead


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    import os

    # Find project root
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
            print("[ERROR] Cannot find CryptX root directory.")
            exit(1)

    print(f"[INFO] CryptX root: {cryptx_root}")

    try:
        run_volatility_modeling(
            cryptx_root=cryptx_root,
            regime_method='garch'     # use GARCH detector regimes (ranked #1)
        )
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()