import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns


class RegimeVisualizer:
    """
    Create visualizations for regime detection results.
    """
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Regime colors
        self.regime_colors = {
            0: '#2ecc71',  # Green - Low volatility
            1: '#f39c12',  # Orange - Medium volatility
            2: '#e74c3c',  # Red - High volatility
        }
        
        self.regime_labels = {
            0: 'Low Volatility',
            1: 'Medium Volatility',
            2: 'High Volatility'
        }
        
        print("[INFO] Visualizer initialized")
    
    def plot_regime_timeseries(self, df_regime, method_name='', 
                               output_path=None, sample_size=10000):
        """
        Plot price and regimes over time.
        
        Args:
            df_regime: Spark or Pandas DataFrame with regime labels
            method_name: Name of detection method
            output_path: Where to save plot
            sample_size: Max number of points to plot
        """
        print(f"[VIZ] Creating timeseries plot for {method_name}...")
        
        # Convert to Pandas if needed
        if hasattr(df_regime, 'toPandas'):
            df_pandas = df_regime.select([
                'timestamp', 'price_close', 'regime', 'vol_60m'
            ]).toPandas()
        else:
            df_pandas = df_regime
        
        # Sort by time
        df_pandas = df_pandas.sort_values('timestamp').reset_index(drop=True)
        
        # Sample if too large
        if len(df_pandas) > sample_size:
            step = len(df_pandas) // sample_size
            df_pandas = df_pandas.iloc[::step].reset_index(drop=True)
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        fig.suptitle(f'Regime Detection: {method_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Price with regime background
        ax1 = axes[0]
        
        # Color background by regime
        for regime in sorted(df_pandas['regime'].unique()):
            regime_mask = df_pandas['regime'] == regime
            regime_data = df_pandas[regime_mask]
            
            ax1.scatter(regime_data['timestamp'], regime_data['price_close'],
                       c=self.regime_colors[regime], s=1, alpha=0.6,
                       label=self.regime_labels[regime])
        
        ax1.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Price Colored by Regime', fontsize=12)
        
        # Plot 2: Volatility with regime background
        ax2 = axes[1]
        
        for regime in sorted(df_pandas['regime'].unique()):
            regime_mask = df_pandas['regime'] == regime
            regime_data = df_pandas[regime_mask]
            
            ax2.scatter(regime_data['timestamp'], regime_data['vol_60m'],
                       c=self.regime_colors[regime], s=1, alpha=0.6)
        
        ax2.set_ylabel('Volatility (60m)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Realized Volatility Colored by Regime', fontsize=12)
        
        # Plot 3: Regime timeline
        ax3 = axes[2]
        
        # Create regime blocks
        regime_changes = df_pandas['regime'].ne(df_pandas['regime'].shift()).cumsum()
        
        for _, group in df_pandas.groupby(regime_changes):
            regime = group['regime'].iloc[0]
            start_time = group['timestamp'].iloc[0]
            end_time = group['timestamp'].iloc[-1]
            
            ax3.axvspan(start_time, end_time, 
                       color=self.regime_colors[regime], alpha=0.7)
        
        ax3.set_ylabel('Regime', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Low', 'Medium', 'High'])
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.set_title('Regime Timeline', fontsize=12)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved to {output_path}")
        
        plt.close()
    
    def plot_regime_characteristics(self, df_regime, method_name='',
                                    output_path=None):
        """
        Plot statistical characteristics of each regime.
        """
        print(f"[VIZ] Creating characteristics plot for {method_name}...")
        
        # Convert to Pandas if needed
        if hasattr(df_regime, 'toPandas'):
            df_pandas = df_regime.select([
                'regime', 'log_return', 'vol_60m', 'cum_return_60m',
                'volume_spike', 'rsi'
            ]).toPandas()
        else:
            df_pandas = df_regime
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Regime Characteristics: {method_name.upper()}',
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        features = ['log_return', 'vol_60m', 'cum_return_60m', 
                   'volume_spike', 'rsi']
        
        for idx, feature in enumerate(features):
            if feature not in df_pandas.columns:
                continue
            
            ax = axes[idx]
            
            # Box plot for each regime
            data_by_regime = [df_pandas[df_pandas['regime'] == r][feature].dropna().values
                             for r in sorted(df_pandas['regime'].unique())]
            
            bp = ax.boxplot(data_by_regime, labels=['Low', 'Medium', 'High'],
                           patch_artist=True)
            
            # Color boxes
            for patch, regime in zip(bp['boxes'], sorted(df_pandas['regime'].unique())):
                patch.set_facecolor(self.regime_colors[regime])
                patch.set_alpha(0.7)
            
            ax.set_ylabel(feature, fontsize=10, fontweight='bold')
            ax.set_xlabel('Regime', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=11)
        
        # Regime distribution in last subplot
        ax = axes[5]
        regime_counts = df_pandas['regime'].value_counts().sort_index()
        colors = [self.regime_colors[r] for r in regime_counts.index]
        
        ax.bar(regime_counts.index, regime_counts.values, 
              color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Regime', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count', fontsize=10, fontweight='bold')
        ax.set_title('Regime Distribution', fontsize=11)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Low', 'Medium', 'High'])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentages
        total = regime_counts.sum()
        for i, (regime, count) in enumerate(regime_counts.items()):
            pct = count / total * 100
            ax.text(regime, count, f'{pct:.1f}%', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved to {output_path}")
        
        plt.close()
    
    def plot_regime_transitions(self, df_regime, method_name='',
                               output_path=None):
        """
        Plot regime transition matrix as heatmap.
        """
        print(f"[VIZ] Creating transition matrix for {method_name}...")
        
        # Convert to Pandas if needed
        if hasattr(df_regime, 'toPandas'):
            df_pandas = df_regime.select(['timestamp', 'regime']) \
                .orderBy('timestamp').toPandas()
        else:
            df_pandas = df_regime.sort_values('timestamp')
        
        # Compute transition matrix
        regimes = df_pandas['regime'].values
        n_regimes = len(np.unique(regimes))
        
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize rows to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_matrix, row_sums, 
                                    where=row_sums != 0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(transition_probs, annot=True, fmt='.3f', 
                   cmap='YlOrRd', cbar_kws={'label': 'Probability'},
                   xticklabels=['Low', 'Medium', 'High'],
                   yticklabels=['Low', 'Medium', 'High'],
                   ax=ax)
        
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
        """
        Create summary comparison plot for all methods.
        """
        print("[VIZ] Creating comparison summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Regime Detection Method Comparison',
                    fontsize=16, fontweight='bold')
        
        # Metric 1: Silhouette Score
        ax1 = axes[0, 0]
        comparison_df.plot(x='method', y='silhouette_score', kind='bar',
                          ax=ax1, color='steelblue', legend=False)
        ax1.set_title('Silhouette Score (Higher = Better)', fontweight='bold')
        ax1.set_xlabel('')
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Metric 2: Davies-Bouldin Score
        ax2 = axes[0, 1]
        comparison_df.plot(x='method', y='davies_bouldin_score', kind='bar',
                          ax=ax2, color='coral', legend=False)
        ax2.set_title('Davies-Bouldin Score (Lower = Better)', fontweight='bold')
        ax2.set_xlabel('')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Metric 3: Regime Stability
        ax3 = axes[1, 0]
        comparison_df.plot(x='method', y='avg_regime_duration', kind='bar',
                          ax=ax3, color='mediumseagreen', legend=False)
        ax3.set_title('Average Regime Duration (Higher = More Stable)', fontweight='bold')
        ax3.set_xlabel('')
        ax3.set_ylabel('Duration (periods)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Metric 4: Volatility Separation
        ax4 = axes[1, 1]
        comparison_df.plot(x='method', y='volatility_separation', kind='bar',
                          ax=ax4, color='mediumpurple', legend=False)
        ax4.set_title('Volatility Separation (Higher = Better)', fontweight='bold')
        ax4.set_xlabel('')
        ax4.set_ylabel('F-statistic')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved to {output_path}")
        
        plt.close()
    
    def plot_detailed_comparison(self, comparison_df, output_path=None):
        """
        Create detailed comparison visualization showing all metrics.
        This helps identify which method is best for different criteria.
        
        Args:
            comparison_df: DataFrame with comparison metrics (from CSV)
            output_path: Where to save the plot
        """
        print("[VIZ] Creating detailed comparison visualization...")
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive Regime Detection Method Comparison', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Color palette for methods
        method_colors = {
            'hmm': '#3498db',      # Blue
            'kmeans': '#2ecc71',   # Green
            'gmm': '#9b59b6',      # Purple
            'threshold': '#e67e22' # Orange
        }
        
        colors = [method_colors.get(m, 'gray') for m in comparison_df['method']]
        
        # ============================================================
        # Row 1: Clustering Quality Metrics
        # ============================================================
        
        # 1. Silhouette Score (Higher = Better)
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(comparison_df['method'], comparison_df['silhouette_score'], 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax1.set_title('Silhouette Score\n(Higher = Better)', 
                     fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Good threshold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Highlight best
        best_idx = comparison_df['silhouette_score'].idxmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # 2. Calinski-Harabasz Score (Higher = Better)
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(comparison_df['method'], comparison_df['calinski_harabasz_score'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_title('Calinski-Harabasz Score\n(Higher = Better)', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        best_idx = comparison_df['calinski_harabasz_score'].idxmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # 3. Davies-Bouldin Score (Lower = Better)
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(comparison_df['method'], comparison_df['davies_bouldin_score'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_title('Davies-Bouldin Score\n(Lower = Better)', 
                     fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Good threshold')
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        best_idx = comparison_df['davies_bouldin_score'].idxmin()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # 4. Within-Regime Volatility (Lower = Better)
        ax4 = fig.add_subplot(gs[0, 3])
        bars = ax4.bar(comparison_df['method'], comparison_df['within_regime_volatility'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_title('Within-Regime Volatility\n(Lower = Better)', 
                     fontsize=12, fontweight='bold')
        ax4.set_ylabel('Std Dev', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        best_idx = comparison_df['within_regime_volatility'].idxmin()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # ============================================================
        # Row 2: Regime Quality Metrics
        # ============================================================
        
        # 5. Volatility Separation (Higher = Better) - MOST IMPORTANT
        ax5 = fig.add_subplot(gs[1, 0])
        bars = ax5.bar(comparison_df['method'], comparison_df['volatility_separation'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax5.set_title('Volatility Separation ⭐\n(Higher = Better)', 
                     fontsize=12, fontweight='bold', color='darkred')
        ax5.set_ylabel('F-statistic', fontweight='bold')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        best_idx = comparison_df['volatility_separation'].idxmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # 6. Regime Balance (Higher = Better)
        ax6 = fig.add_subplot(gs[1, 1])
        bars = ax6.bar(comparison_df['method'], comparison_df['regime_balance'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax6.set_title('Regime Balance\n(Higher = Better)', 
                     fontsize=12, fontweight='bold')
        ax6.set_ylabel('Entropy Score', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim([0.85, 1.0])
        
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        best_idx = comparison_df['regime_balance'].idxmax()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # 7. Average Regime Duration (Stability)
        ax7 = fig.add_subplot(gs[1, 2])
        bars = ax7.bar(comparison_df['method'], comparison_df['avg_regime_duration'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax7.set_title('Avg Regime Duration\n(Stability)', 
                     fontsize=12, fontweight='bold')
        ax7.set_ylabel('Periods', fontweight='bold')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Transition Frequency (Lower = More Stable)
        ax8 = fig.add_subplot(gs[1, 3])
        bars = ax8.bar(comparison_df['method'], comparison_df['transition_frequency'],
                      color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax8.set_title('Transition Frequency\n(Lower = More Stable)', 
                     fontsize=12, fontweight='bold')
        ax8.set_ylabel('Frequency', fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        best_idx = comparison_df['transition_frequency'].idxmin()
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # ============================================================
        # Row 3: Radar Chart and Summary Table
        # ============================================================
        
        # 9. Radar Chart (Normalized Metrics)
        ax9 = fig.add_subplot(gs[2, :2], projection='polar')
        
        # Metrics for radar (normalize to 0-1, invert where lower is better)
        radar_metrics = {
            'Silhouette': 'silhouette_score',
            'Calinski-H': 'calinski_harabasz_score', 
            'Davies-B\n(inverted)': 'davies_bouldin_score',
            'Vol Sep': 'volatility_separation',
            'Balance': 'regime_balance',
            'Stability': 'avg_regime_duration',
            'Trans Freq\n(inverted)': 'transition_frequency',
            'Within Vol\n(inverted)': 'within_regime_volatility'
        }
        
        # Normalize each metric to 0-1
        normalized_data = {}
        for label, col in radar_metrics.items():
            values = comparison_df[col].values
            if 'inverted' in label:
                # Lower is better - invert and normalize
                normalized = 1 - (values - values.min()) / (values.max() - values.min() + 1e-10)
            else:
                # Higher is better - normalize
                normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)
            normalized_data[label] = normalized
        
        # Create radar chart
        labels_radar = list(radar_metrics.keys())
        angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot
        
        for idx, method in enumerate(comparison_df['method']):
            values = [normalized_data[label][idx] for label in labels_radar]
            values += values[:1]  # Close the plot
            
            ax9.plot(angles, values, 'o-', linewidth=2, 
                    label=method.upper(), color=method_colors.get(method, 'gray'))
            ax9.fill(angles, values, alpha=0.15, color=method_colors.get(method, 'gray'))
        
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(labels_radar, size=10)
        ax9.set_ylim(0, 1)
        ax9.set_title('Normalized Performance Across All Metrics', 
                     fontsize=12, fontweight='bold', pad=20)
        ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax9.grid(True)
        
        # 10. Summary Table with Ranks
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis('tight')
        ax10.axis('off')
        
        # Calculate ranks for key metrics (1 = best)
        summary_data = []
        for idx, row in comparison_df.iterrows():
            method = row['method'].upper()
            
            # Rank metrics (1 = best)
            silh_rank = comparison_df['silhouette_score'].rank(ascending=False)[idx]
            vol_sep_rank = comparison_df['volatility_separation'].rank(ascending=False)[idx]
            db_rank = comparison_df['davies_bouldin_score'].rank(ascending=True)[idx]
            stability_rank = comparison_df['avg_regime_duration'].rank(ascending=False)[idx]
            
            # Average rank
            avg_rank = (silh_rank + vol_sep_rank + db_rank) / 3
            
            summary_data.append([
                method,
                f"#{int(silh_rank)}",
                f"#{int(vol_sep_rank)}",
                f"#{int(db_rank)}",
                f"#{int(stability_rank)}",
                f"{avg_rank:.1f}"
            ])
        
        # Sort by average rank
        summary_data.sort(key=lambda x: float(x[5]))
        
        # Create table
        table = ax10.table(cellText=summary_data,
                          colLabels=['Method', 'Silhouette\nRank', 'Vol Sep\nRank ⭐', 
                                   'Davies-B\nRank', 'Stability\nRank', 'Avg\nRank'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(6):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by method
        for i, row in enumerate(summary_data, 1):
            method = row[0].lower()
            color = method_colors.get(method, 'gray')
            table[(i, 0)].set_facecolor(color)
            table[(i, 0)].set_text_props(weight='bold', color='white')
            
            # Highlight best ranks
            for j in range(1, 6):
                if row[j] == '#1':
                    table[(i, j)].set_facecolor('#f1c40f')
                    table[(i, j)].set_text_props(weight='bold')
        
        ax10.set_title('Summary Rankings (Lower Rank = Better)', 
                      fontsize=12, fontweight='bold', pad=10)
        
        # Add overall recommendation
        best_method = summary_data[0][0]
        fig.text(0.5, 0.02, 
                f'⭐ RECOMMENDED: {best_method} shows best overall performance for volatility-based regime detection',
                ha='center', fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[VIZ] Saved detailed comparison to {output_path}")
        
        plt.close()
        
        return summary_data