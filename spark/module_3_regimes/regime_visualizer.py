import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns


class RegimeVisualizer:
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        self.regime_colors = {
            0: '#2ecc71',
            1: '#f39c12',
            2: '#e74c3c',
        }
        self.regime_labels = {
            0: 'Low Volatility',
            1: 'Medium Volatility',
            2: 'High Volatility'
        }

        print("[INFO] Visualizer initialized")

    def _get_vol_col(self, columns):
        """
        Return whichever volatility column is present.
        Prefers vol_60m (alias added by run_hourly_regimes.py),
        falls back to vol_24h (the native hourly column).
        """
        for candidate in ['vol_60m', 'vol_24h', 'vol_168h']:
            if candidate in columns:
                return candidate
        return None

    def _get_cum_return_col(self, columns):
        for candidate in ['cum_return_60m', 'cum_return_24h', 'cum_return_168h']:
            if candidate in columns:
                return candidate
        return None

    def plot_regime_timeseries(self, df_regime, method_name='',
                               output_path=None, sample_size=10000):
        print(f"[VIZ] Creating timeseries plot for {method_name}...")

        # FIX: resolve vol column dynamically instead of hardcoding vol_60m
        if hasattr(df_regime, 'columns'):
            all_cols = df_regime.columns
        else:
            all_cols = df_regime.columns.tolist()

        vol_col = self._get_vol_col(all_cols)
        if vol_col is None:
            print("[VIZ] WARNING: No volatility column found, skipping timeseries plot")
            return

        if hasattr(df_regime, 'toPandas'):
            df_pandas = df_regime.select(['timestamp', 'price_close', 'regime', vol_col]).toPandas()
        else:
            df_pandas = df_regime[['timestamp', 'price_close', 'regime', vol_col]].copy()

        df_pandas = df_pandas.sort_values('timestamp').reset_index(drop=True)

        if len(df_pandas) > sample_size:
            step = len(df_pandas) // sample_size
            df_pandas = df_pandas.iloc[::step].reset_index(drop=True)

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f'Regime Detection: {method_name.upper()}',
                     fontsize=16, fontweight='bold')

        # Plot 1: Price
        ax1 = axes[0]
        for regime in sorted(df_pandas['regime'].unique()):
            mask = df_pandas['regime'] == regime
            ax1.scatter(df_pandas[mask]['timestamp'], df_pandas[mask]['price_close'],
                        c=self.regime_colors[regime], s=1, alpha=0.6,
                        label=self.regime_labels[regime])
        ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Price Colored by Regime', fontsize=12)

        # Plot 2: Volatility
        ax2 = axes[1]
        for regime in sorted(df_pandas['regime'].unique()):
            mask = df_pandas['regime'] == regime
            ax2.scatter(df_pandas[mask]['timestamp'], df_pandas[mask][vol_col],
                        c=self.regime_colors[regime], s=1, alpha=0.6)
        # Label shows the actual column name so it's clear what's being plotted
        vol_label = 'Volatility (24h hourly)' if vol_col == 'vol_24h' else f'Volatility ({vol_col})'
        ax2.set_ylabel(vol_label, fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Realized Volatility Colored by Regime', fontsize=12)

        # Plot 3: Regime timeline
        ax3 = axes[2]
        regime_changes = df_pandas['regime'].ne(df_pandas['regime'].shift()).cumsum()
        for _, group in df_pandas.groupby(regime_changes):
            regime = group['regime'].iloc[0]
            ax3.axvspan(group['timestamp'].iloc[0], group['timestamp'].iloc[-1],
                        color=self.regime_colors[regime], alpha=0.7)
        ax3.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Low', 'Medium', 'High'])
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_title('Regime Timeline', fontsize=12)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved to {output_path}")

        plt.close()

    def plot_regime_characteristics(self, df_regime, method_name='', output_path=None):
        print(f"[VIZ] Creating characteristics plot for {method_name}...")

        if hasattr(df_regime, 'columns'):
            all_cols = df_regime.columns
        else:
            all_cols = df_regime.columns.tolist()

        vol_col      = self._get_vol_col(all_cols) or 'vol_24h'
        cum_ret_col  = self._get_cum_return_col(all_cols) or 'cum_return_24h'

        select_cols = ['regime', 'log_return', vol_col, cum_ret_col, 'volume_spike', 'rsi']
        select_cols = [c for c in select_cols if c in all_cols]

        if hasattr(df_regime, 'toPandas'):
            df_pandas = df_regime.select(select_cols).toPandas()
        else:
            df_pandas = df_regime[select_cols].copy()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Regime Characteristics: {method_name.upper()}',
                     fontsize=16, fontweight='bold')
        axes = axes.flatten()

        features = ['log_return', vol_col, cum_ret_col, 'volume_spike', 'rsi']
        features  = [f for f in features if f in df_pandas.columns]

        for idx, feature in enumerate(features):
            ax = axes[idx]
            data_by_regime = [
                df_pandas[df_pandas['regime'] == r][feature].dropna().values
                for r in sorted(df_pandas['regime'].unique())
            ]
            bp = ax.boxplot(data_by_regime, labels=['Low', 'Medium', 'High'], patch_artist=True)
            for patch, regime in zip(bp['boxes'], sorted(df_pandas['regime'].unique())):
                patch.set_facecolor(self.regime_colors[regime])
                patch.set_alpha(0.7)
            ax.set_ylabel(feature, fontsize=10, fontweight='bold')
            ax.set_xlabel('Regime', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(feature.replace("_", " ").title(), fontsize=11)

        # Regime distribution
        ax = axes[5]
        regime_counts = df_pandas['regime'].value_counts().sort_index()
        colors = [self.regime_colors[r] for r in regime_counts.index]
        ax.bar(regime_counts.index, regime_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Regime', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title('Regime Distribution', fontsize=11)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Low', 'Medium', 'High'])
        ax.grid(True, alpha=0.3, axis='y')
        total = regime_counts.sum()
        for regime, count in regime_counts.items():
            ax.text(regime, count, f'{count/total*100:.1f}%',
                    ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved to {output_path}")

        plt.close()

    def plot_regime_transitions(self, df_regime, method_name='', output_path=None):
        print(f"[VIZ] Creating transition matrix for {method_name}...")

        if hasattr(df_regime, 'toPandas'):
            df_pandas = df_regime.select(['timestamp', 'regime']).orderBy('timestamp').toPandas()
        else:
            df_pandas = df_regime.sort_values('timestamp')

        regimes = df_pandas['regime'].values
        n_regimes = len(np.unique(regimes))
        transition_matrix = np.zeros((n_regimes, n_regimes))

        for i in range(len(regimes) - 1):
            transition_matrix[regimes[i], regimes[i + 1]] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(transition_probs, annot=True, fmt='.3f', cmap='YlOrRd',
                    cbar_kws={'label': 'Probability'},
                    xticklabels=['Low', 'Medium', 'High'],
                    yticklabels=['Low', 'Medium', 'High'], ax=ax)
        ax.set_xlabel('To Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('From Regime', fontsize=12, fontweight='bold')
        ax.set_title(f'Regime Transition Probabilities: {method_name.upper()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved to {output_path}")

        plt.close()

    def plot_comparison_summary(self, comparison_df, output_path=None):
        print("[VIZ] Creating comparison summary...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Regime Detection Method Comparison', fontsize=16, fontweight='bold')

        metrics = [
            ('silhouette_score',    axes[0, 0], 'steelblue',      'Silhouette Score (Higher = Better)'),
            ('davies_bouldin_score',axes[0, 1], 'coral',           'Davies-Bouldin Score (Lower = Better)'),
            ('avg_regime_duration', axes[1, 0], 'mediumseagreen',  'Average Regime Duration (Higher = More Stable)'),
            ('volatility_separation',axes[1,1], 'mediumpurple',    'Volatility Separation (Higher = Better)'),
        ]
        for col, ax, color, title in metrics:
            comparison_df.plot(x='method', y=col, kind='bar', ax=ax, color=color, legend=False)
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved to {output_path}")
        plt.close()

    def plot_detailed_comparison(self, comparison_df, output_path=None):
        print("[VIZ] Creating detailed comparison visualization...")

        fig = plt.figure(figsize=(20, 12))
        gs  = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Regime Detection Method Comparison',
                     fontsize=20, fontweight='bold', y=0.98)

        method_colors = {
            'hmm':       '#3498db',
            'kmeans':    '#2ecc71',
            'gmm':       '#9b59b6',
            'threshold': '#e67e22',
            'patchtst':  '#e74c3c',
            'informer':  '#1abc9c',
        }
        colors = [method_colors.get(m, 'gray') for m in comparison_df['method']]

        def bar_plot(ax, col, title, higher_better=True):
            bars = ax.bar(comparison_df['method'], comparison_df[col],
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h,
                        f'{h:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            best = comparison_df[col].idxmax() if higher_better else comparison_df[col].idxmin()
            bars[best].set_edgecolor('gold')
            bars[best].set_linewidth(3)

        bar_plot(fig.add_subplot(gs[0, 0]), 'silhouette_score',
                 'Silhouette Score\n(Higher = Better)')
        bar_plot(fig.add_subplot(gs[0, 1]), 'calinski_harabasz_score',
                 'Calinski-Harabasz Score\n(Higher = Better)')
        bar_plot(fig.add_subplot(gs[0, 2]), 'davies_bouldin_score',
                 'Davies-Bouldin Score\n(Lower = Better)', higher_better=False)
        bar_plot(fig.add_subplot(gs[0, 3]), 'within_regime_volatility',
                 'Within-Regime Volatility\n(Lower = Better)', higher_better=False)
        bar_plot(fig.add_subplot(gs[1, 0]), 'volatility_separation',
                 'Volatility Separation ⭐\n(Higher = Better)')
        bar_plot(fig.add_subplot(gs[1, 1]), 'regime_balance',
                 'Regime Balance\n(Higher = Better)')
        bar_plot(fig.add_subplot(gs[1, 2]), 'avg_regime_duration',
                 'Avg Regime Duration\n(Stability)')
        bar_plot(fig.add_subplot(gs[1, 3]), 'transition_frequency',
                 'Transition Frequency\n(Lower = More Stable)', higher_better=False)

        # Radar chart
        ax9 = fig.add_subplot(gs[2, :2], projection='polar')
        radar_metrics = {
            'Silhouette':            ('silhouette_score',         False),
            'Calinski-H':            ('calinski_harabasz_score',  False),
            'Davies-B\n(inv)':       ('davies_bouldin_score',     True),
            'Vol Sep':               ('volatility_separation',    False),
            'Balance':               ('regime_balance',           False),
            'Stability':             ('avg_regime_duration',      False),
            'Trans\n(inv)':          ('transition_frequency',     True),
            'Within Vol\n(inv)':     ('within_regime_volatility', True),
        }
        labels_radar = list(radar_metrics.keys())
        angles = np.linspace(0, 2*np.pi, len(labels_radar), endpoint=False).tolist()
        angles += angles[:1]

        for idx, method in enumerate(comparison_df['method']):
            vals = []
            for label, (col, invert) in radar_metrics.items():
                v = comparison_df[col].values
                n = (v - v.min()) / (v.max() - v.min() + 1e-10)
                vals.append(1 - n[idx] if invert else n[idx])
            vals += vals[:1]
            ax9.plot(angles, vals, 'o-', linewidth=2,
                     label=method.upper(), color=method_colors.get(method, 'gray'))
            ax9.fill(angles, vals, alpha=0.15, color=method_colors.get(method, 'gray'))

        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(labels_radar, size=9)
        ax9.set_ylim(0, 1)
        ax9.set_title('Normalised Performance', fontsize=12, fontweight='bold', pad=20)
        ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax9.grid(True)

        # Summary table
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis('off')

        summary_data = []
        for idx, row in comparison_df.iterrows():
            silh_rank    = int(comparison_df['silhouette_score'].rank(ascending=False)[idx])
            vol_sep_rank = int(comparison_df['volatility_separation'].rank(ascending=False)[idx])
            db_rank      = int(comparison_df['davies_bouldin_score'].rank(ascending=True)[idx])
            stab_rank    = int(comparison_df['avg_regime_duration'].rank(ascending=False)[idx])
            avg_rank     = (silh_rank + vol_sep_rank + db_rank) / 3
            summary_data.append([row['method'].upper(),
                                  f"#{silh_rank}", f"#{vol_sep_rank}",
                                  f"#{db_rank}", f"#{stab_rank}", f"{avg_rank:.1f}"])

        summary_data.sort(key=lambda x: float(x[5]))
        table = ax10.table(
            cellText=summary_data,
            colLabels=['Method', 'Silhouette\nRank', 'Vol Sep\nRank ⭐',
                       'Davies-B\nRank', 'Stability\nRank', 'Avg\nRank'],
            cellLoc='center', loc='center',
            colWidths=[0.15]*6
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        for i in range(6):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        for i, row in enumerate(summary_data, 1):
            method = row[0].lower()
            table[(i, 0)].set_facecolor(method_colors.get(method, 'gray'))
            table[(i, 0)].set_text_props(weight='bold', color='white')
            for j in range(1, 6):
                if row[j] == '#1':
                    table[(i, j)].set_facecolor('#f1c40f')
                    table[(i, j)].set_text_props(weight='bold')

        ax10.set_title('Summary Rankings (Lower = Better)', fontsize=12, fontweight='bold', pad=10)
        fig.text(0.5, 0.02,
                 f'⭐ RECOMMENDED: {summary_data[0][0]} shows best overall performance',
                 ha='center', fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved detailed comparison to {output_path}")

        plt.close()
        return summary_data