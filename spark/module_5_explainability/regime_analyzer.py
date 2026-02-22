"""
regime_analyzer.py

Regime-specific volatility characteristics and transition behavior analysis.

Covers:
  1. Per-regime distributional stats (vol, returns, skew, kurtosis, autocorr)
  2. Transition probability matrix
  3. Regime duration distributions
  4. What market conditions trigger regime transitions

Symbol-agnostic: pass any symbol's feature DataFrame + regime labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import product


REGIME_NAMES  = {0: 'Low Vol',    1: 'Medium Vol', 2: 'High Vol'}
REGIME_COLORS = {0: '#3498db',    1: '#f39c12',    2: '#e74c3c'}


class RegimeCharacteristicsAnalyzer:
    """
    Analyses the statistical properties of each regime and how
    the system transitions between them.

    Designed to work for any symbol — just pass different DataFrames.
    """

    def __init__(self, symbol: str = 'BTCUSDT', n_regimes: int = 3):
        self.symbol    = symbol
        self.n_regimes = n_regimes

    # ------------------------------------------------------------------
    # 1. Volatility characteristics per regime
    # ------------------------------------------------------------------

    def compute_regime_stats(self, df: pd.DataFrame,
                              regimes: np.ndarray) -> pd.DataFrame:
        """
        Compute distributional statistics for each regime.

        Returns a DataFrame with one row per regime covering:
          - Return stats: mean, std, skew, kurtosis, sharpe
          - Volatility stats: mean vol, vol-of-vol, percentiles
          - Volume: mean spike, std spike
          - Technical: mean RSI, mean ATR
          - Autocorrelation: return autocorr at lag 1, 2, 24
        """
        df = df.copy()
        df['regime'] = regimes

        rows = []
        for r in range(self.n_regimes):
            mask = df['regime'] == r
            sub  = df[mask]
            if len(sub) < 10:
                continue

            ret = sub['log_return'].dropna()
            vol = sub['vol_60m'].dropna()   if 'vol_60m'  in sub.columns else pd.Series()
            rsi = sub['rsi'].dropna()       if 'rsi'      in sub.columns else pd.Series()
            atr = sub['atr_pct'].dropna()   if 'atr_pct'  in sub.columns else pd.Series()
            spk = sub['volume_spike'].dropna() if 'volume_spike' in sub.columns else pd.Series()
            bb  = sub['bb_width'].dropna()  if 'bb_width' in sub.columns else pd.Series()

            # Autocorrelation of returns at various lags
            def autocorr(series, lag):
                if len(series) > lag + 1:
                    return float(pd.Series(series.values).autocorr(lag=lag))
                return np.nan

            row = {
                'regime':          r,
                'regime_name':     REGIME_NAMES.get(r, f'Regime {r}'),
                'n_bars':          int(mask.sum()),
                'pct_of_total':    float(mask.mean() * 100),

                # Return statistics
                'return_mean':     float(ret.mean()),
                'return_std':      float(ret.std()),
                'return_skew':     float(stats.skew(ret)),
                'return_kurtosis': float(stats.kurtosis(ret)),
                'sharpe_hourly':   float(ret.mean() / ret.std()) if ret.std() > 0 else 0,
                'pct_positive':    float((ret > 0).mean() * 100),

                # Volatility statistics
                'vol_mean':        float(vol.mean())        if len(vol) > 0 else np.nan,
                'vol_std':         float(vol.std())         if len(vol) > 0 else np.nan,
                'vol_p25':         float(vol.quantile(.25)) if len(vol) > 0 else np.nan,
                'vol_median':      float(vol.median())      if len(vol) > 0 else np.nan,
                'vol_p75':         float(vol.quantile(.75)) if len(vol) > 0 else np.nan,
                'vol_p95':         float(vol.quantile(.95)) if len(vol) > 0 else np.nan,
                'vol_of_vol':      float(vol.std() / vol.mean())
                                   if len(vol) > 0 and vol.mean() > 0 else np.nan,

                # Technical indicators
                'rsi_mean':        float(rsi.mean())   if len(rsi) > 0 else np.nan,
                'rsi_std':         float(rsi.std())    if len(rsi) > 0 else np.nan,
                'atr_mean':        float(atr.mean())   if len(atr) > 0 else np.nan,
                'bb_width_mean':   float(bb.mean())    if len(bb) > 0 else np.nan,
                'vol_spike_mean':  float(spk.mean())   if len(spk) > 0 else np.nan,

                # Return autocorrelation (persistence of direction)
                'autocorr_lag1':   autocorr(ret, 1),
                'autocorr_lag2':   autocorr(ret, 2),
                'autocorr_lag24':  autocorr(ret, 24),
            }
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2. Transition analysis
    # ------------------------------------------------------------------

    def compute_transition_matrix(self, regimes: np.ndarray) -> pd.DataFrame:
        """
        Compute empirical transition probability matrix.
        Entry [i, j] = P(next regime = j | current regime = i)
        """
        n = self.n_regimes
        counts = np.zeros((n, n), dtype=int)

        for t in range(len(regimes) - 1):
            i = int(regimes[t])
            j = int(regimes[t+1])
            if 0 <= i < n and 0 <= j < n:
                counts[i, j] += 1

        # Normalise rows to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        probs    = np.where(row_sums > 0, counts / row_sums, 0.0)

        labels = [REGIME_NAMES.get(r, f'Regime {r}') for r in range(n)]
        return pd.DataFrame(probs, index=labels, columns=labels)

    def compute_regime_durations(self, regimes: np.ndarray) -> dict:
        """
        Compute duration (consecutive bars) of each regime spell.
        Returns {regime: list of durations}.
        """
        durations = {r: [] for r in range(self.n_regimes)}
        current   = int(regimes[0])
        length    = 1

        for t in range(1, len(regimes)):
            r = int(regimes[t])
            if r == current:
                length += 1
            else:
                durations[current].append(length)
                current = r
                length  = 1
        durations[current].append(length)

        return durations

    def compute_transition_triggers(self, df: pd.DataFrame,
                                     regimes: np.ndarray) -> pd.DataFrame:
        """
        Analyse what feature values are present just before a regime transition.
        Compares pre-transition bar values to regime baseline to identify triggers.

        Returns DataFrame showing mean feature value change before transitions,
        for each from→to regime pair.
        """
        df = df.copy()
        df['regime'] = regimes

        trigger_cols = ['log_return', 'vol_60m', 'vol_ratio_60_240',
                        'rsi', 'volume_spike', 'atr_pct', 'bb_width']
        avail = [c for c in trigger_cols if c in df.columns]

        rows = []
        for t in range(1, len(df) - 1):
            from_r = int(regimes[t - 1])
            to_r   = int(regimes[t])
            if from_r != to_r:
                row = {'from_regime': from_r, 'to_regime': to_r}
                for col in avail:
                    row[col] = float(df[col].iloc[t - 1])
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        transitions_df = pd.DataFrame(rows)

        # Compute mean feature values at transition vs overall regime mean
        summary_rows = []
        for (from_r, to_r), grp in transitions_df.groupby(['from_regime', 'to_regime']):
            if from_r == to_r:
                continue
            row = {
                'from_regime':      from_r,
                'from_name':        REGIME_NAMES.get(from_r, f'R{from_r}'),
                'to_regime':        to_r,
                'to_name':          REGIME_NAMES.get(to_r, f'R{to_r}'),
                'n_transitions':    len(grp),
            }
            for col in avail:
                regime_mean = float(df.loc[df['regime'] == from_r, col].mean())
                trans_mean  = float(grp[col].mean())
                row[f'{col}_at_transition'] = trans_mean
                row[f'{col}_vs_baseline']   = (
                    (trans_mean - regime_mean) / abs(regime_mean)
                    if regime_mean != 0 else 0.0
                )
            summary_rows.append(row)

        return pd.DataFrame(summary_rows).sort_values(
            ['from_regime', 'to_regime']).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_regime_characteristics(self, stats_df: pd.DataFrame,
                                     output_path: Path):
        """Heatmap of regime characteristics normalised for comparison."""
        if stats_df.empty:
            return

        numeric_cols = [
            'return_mean', 'return_std', 'return_skew', 'return_kurtosis',
            'vol_mean', 'vol_of_vol', 'rsi_mean', 'atr_mean',
            'bb_width_mean', 'vol_spike_mean',
            'autocorr_lag1', 'autocorr_lag24', 'sharpe_hourly'
        ]
        avail = [c for c in numeric_cols if c in stats_df.columns]
        heat  = stats_df.set_index('regime_name')[avail].T

        # Normalise each row to [-1, 1] for visual comparison
        heat_norm = heat.apply(
            lambda row: (row - row.mean()) / (row.std() + 1e-10), axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                                 gridspec_kw={'width_ratios': [2, 1]})
        fig.suptitle(f'{self.symbol} — Regime Characteristics',
                     fontsize=13, fontweight='bold')

        # Heatmap
        ax = axes[0]
        im = ax.imshow(heat_norm.values, aspect='auto', cmap='RdYlGn',
                       vmin=-2, vmax=2)
        ax.set_xticks(range(len(heat_norm.columns)))
        ax.set_xticklabels(heat_norm.columns, fontsize=11, fontweight='bold')
        ax.set_yticks(range(len(heat_norm.index)))
        ax.set_yticklabels([c.replace('_', ' ') for c in heat_norm.index],
                           fontsize=9)
        ax.set_title('Normalised Feature Values per Regime\n'
                     '(green = high, red = low)')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Annotate with actual values
        for i, j in product(range(len(heat_norm.index)),
                             range(len(heat_norm.columns))):
            val = heat.values[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=7, color='black')

        # Bar chart of key metrics
        ax2 = axes[1]
        key_metrics = ['vol_mean', 'return_std', 'sharpe_hourly', 'rsi_mean']
        avail_key   = [m for m in key_metrics if m in stats_df.columns]
        x = np.arange(len(avail_key))
        width = 0.25

        for i, (_, row) in enumerate(stats_df.iterrows()):
            r     = int(row['regime'])
            color = REGIME_COLORS.get(r, '#7f8c8d')
            vals  = [float(row[m]) for m in avail_key]
            ax2.bar(x + i * width, vals, width, label=row['regime_name'],
                    color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax2.set_xticks(x + width)
        ax2.set_xticklabels([m.replace('_', '\n') for m in avail_key], fontsize=8)
        ax2.set_title('Key Metrics by Regime')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")

    def plot_transition_matrix(self, trans_matrix: pd.DataFrame,
                                output_path: Path):
        """Heatmap of transition probability matrix."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{self.symbol} — Regime Transition Analysis',
                     fontsize=13, fontweight='bold')

        # Probability heatmap
        ax = axes[0]
        im = ax.imshow(trans_matrix.values, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(trans_matrix.columns)))
        ax.set_yticks(range(len(trans_matrix.index)))
        ax.set_xticklabels(trans_matrix.columns, fontsize=10)
        ax.set_yticklabels(trans_matrix.index,   fontsize=10)
        ax.set_xlabel('Next Regime')
        ax.set_ylabel('Current Regime')
        ax.set_title('Transition Probability Matrix\nP(next | current)')
        plt.colorbar(im, ax=ax)

        for i, j in product(range(len(trans_matrix.index)),
                             range(len(trans_matrix.columns))):
            val = trans_matrix.values[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

        # Persistence probability (diagonal) bar chart
        ax2 = axes[1]
        persistence = np.diag(trans_matrix.values)
        colors      = [REGIME_COLORS.get(r, '#7f8c8d')
                       for r in range(len(persistence))]
        bars = ax2.bar(trans_matrix.columns, persistence,
                       color=colors, edgecolor='black', linewidth=0.8)
        ax2.set_ylabel('P(stay in same regime)')
        ax2.set_title('Regime Persistence\n(diagonal of transition matrix)')
        ax2.set_ylim(0, 1)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5,
                    label='50% threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, persistence):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     val + 0.01, f'{val:.2%}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")

    def plot_duration_distributions(self, durations: dict,
                                     output_path: Path):
        """Histogram of regime spell durations."""
        fig, axes = plt.subplots(1, self.n_regimes,
                                  figsize=(6 * self.n_regimes, 5))
        fig.suptitle(f'{self.symbol} — Regime Duration Distributions',
                     fontsize=13, fontweight='bold')

        if self.n_regimes == 1:
            axes = [axes]

        for r, ax in enumerate(axes):
            d     = durations.get(r, [])
            color = REGIME_COLORS.get(r, '#7f8c8d')
            name  = REGIME_NAMES.get(r, f'Regime {r}')

            if not d:
                ax.set_title(f'Regime {r}: {name}\n(no data)')
                continue

            d_arr = np.array(d)
            ax.hist(d_arr, bins=min(50, len(d_arr)//2 + 1),
                    color=color, alpha=0.8, edgecolor='black', linewidth=0.3)
            ax.axvline(d_arr.mean(),   color='black', lw=2,
                       linestyle='--', label=f'Mean: {d_arr.mean():.1f}h')
            ax.axvline(np.median(d_arr), color='white', lw=2,
                       linestyle=':', label=f'Median: {np.median(d_arr):.1f}h')
            ax.set_xlabel('Duration (hours)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Regime {r}: {name}\n'
                         f'n={len(d)} spells | '
                         f'max={d_arr.max()}h | '
                         f'mean={d_arr.mean():.1f}h')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")

    def plot_transition_triggers(self, triggers_df: pd.DataFrame,
                                  output_path: Path):
        """Bar chart showing which features change most before transitions."""
        if triggers_df.empty:
            print("[VIZ] No transition triggers to plot")
            return

        vs_cols = [c for c in triggers_df.columns if c.endswith('_vs_baseline')]
        feat_names = [c.replace('_vs_baseline', '') for c in vs_cols]

        n_transitions = len(triggers_df)
        fig, axes = plt.subplots(1, n_transitions,
                                  figsize=(7 * n_transitions, 6))
        fig.suptitle(f'{self.symbol} — Feature Changes at Regime Transitions\n'
                     f'(% deviation from regime baseline)',
                     fontsize=12, fontweight='bold')

        if n_transitions == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, triggers_df.iterrows()):
            from_r = int(row['from_regime'])
            to_r   = int(row['to_regime'])
            vals   = [float(row[c]) * 100 for c in vs_cols]  # convert to %
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in vals]

            ax.barh(feat_names, vals, color=colors, edgecolor='black',
                    linewidth=0.5, alpha=0.8)
            ax.axvline(0, color='black', lw=1)
            ax.set_xlabel('% change vs regime baseline')
            from_name = REGIME_NAMES.get(from_r, f'R{from_r}')
            to_name   = REGIME_NAMES.get(to_r,   f'R{to_r}')
            ax.set_title(f'{from_name} → {to_name}\n'
                         f'(n={int(row["n_transitions"])} transitions)',
                         fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")

    def plot_return_distributions(self, df: pd.DataFrame,
                                   regimes: np.ndarray,
                                   output_path: Path):
        """Overlapping return distribution per regime."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{self.symbol} — Return Distributions by Regime',
                     fontsize=13, fontweight='bold')

        df = df.copy()
        df['regime'] = regimes
        ret_col = 'log_return'

        # KDE overlay
        ax = axes[0]
        for r in range(self.n_regimes):
            sub = df.loc[df['regime'] == r, ret_col].dropna()
            if len(sub) < 10:
                continue
            sub.plot.kde(ax=ax, label=REGIME_NAMES.get(r, f'R{r}'),
                         color=REGIME_COLORS.get(r, '#7f8c8d'), linewidth=2)
        ax.set_xlabel('Log Return')
        ax.set_title('Return Distribution (KDE) per Regime')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 0.05)

        # Box plots
        ax2 = axes[1]
        data_by_regime = []
        labels         = []
        colors         = []
        for r in range(self.n_regimes):
            sub = df.loc[df['regime'] == r, ret_col].dropna()
            if len(sub) > 0:
                data_by_regime.append(sub.values)
                labels.append(REGIME_NAMES.get(r, f'R{r}'))
                colors.append(REGIME_COLORS.get(r, '#7f8c8d'))

        bp = ax2.boxplot(data_by_regime, labels=labels, patch_artist=True,
                         showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel('Log Return')
        ax2.set_title('Return Boxplot per Regime\n(outliers hidden)')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(0, color='black', lw=0.8, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")