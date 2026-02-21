# from pyspark.sql import DataFrame
# from pyspark.sql import functions as F
# import pandas as pd
# import numpy as np
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from sklearn.preprocessing import StandardScaler


# class RegimeEvaluator:
#     """
#     Evaluate quality of regime detection.
#     Computes clustering metrics and regime stability measures.
#     """
#     def __init__(self):
#         print("[INFO] Regime evaluator initialized")
    
#     def evaluate(self, df_regime: DataFrame, df_original: DataFrame = None, sample_size=50000) -> dict:
#         """
#         Evaluate regime detection quality.
#         Args:
#             df_regime: DataFrame with regime labels
#             df_original: Optional original DataFrame for comparison
#             sample_size: Maximum number of rows to use for evaluation (for speed)
#         Returns:
#             Dictionary of evaluation metrics
#         """
#         print("[EVAL] Computing evaluation metrics...")
        
#         # Sample if dataset is too large
#         total_count = df_regime.count()
#         if total_count > sample_size:
#             print(f"[EVAL] Sampling {sample_size:,} from {total_count:,} records for faster evaluation...")
#             fraction = sample_size / total_count
#             df_sample = df_regime.sample(fraction=fraction, seed=42)
#         else:
#             df_sample = df_regime
        
#         # Convert to Pandas for sklearn metrics
#         print("[EVAL] Converting to Pandas...")
#         df_pandas = df_sample.select([
#             'timestamp', 'regime', 'log_return', 'vol_60m', 'vol_240m',
#             'cum_return_60m', 'volume_spike'
#         ]).toPandas()
        
#         print(f"[EVAL] Using {len(df_pandas):,} records for evaluation")
        
#         # Feature matrix
#         feature_cols = ['log_return', 'vol_60m', 'vol_240m', 
#                        'cum_return_60m', 'volume_spike']
#         available_features = [c for c in feature_cols if c in df_pandas.columns]
        
#         X = df_pandas[available_features].values
#         X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
#         # Standardize
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         labels = df_pandas['regime'].values
        
#         # Compute metrics
#         metrics = {}
        
#         print("[EVAL] Computing silhouette score...")
#         # 1. Silhouette Score (-1 to 1, higher is better)
#         # Measures how similar points are to their own cluster vs other clusters
#         try:
#             # Silhouette score is very slow on large datasets, sample if needed
#             if len(X_scaled) > 10000:
#                 sample_idx = np.random.choice(len(X_scaled), 10000, replace=False)
#                 metrics['silhouette_score'] = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
#             else:
#                 metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
#         except Exception as e:
#             print(f"[EVAL] Silhouette score failed: {e}")
#             metrics['silhouette_score'] = np.nan
        
#         print("[EVAL] Computing calinski-harabasz score...")
#         # 2. Calinski-Harabasz Score (higher is better)
#         # Ratio of between-cluster to within-cluster dispersion
#         try:
#             metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
#         except:
#             metrics['calinski_harabasz_score'] = np.nan
        
#         print("[EVAL] Computing davies-bouldin score...")
#         # 3. Davies-Bouldin Score (lower is better)
#         # Average similarity between each cluster and its most similar cluster
#         try:
#             metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
#         except:
#             metrics['davies_bouldin_score'] = np.nan
        
#         print("[EVAL] Computing regime stability...")
#         # 4. Regime stability (average regime duration)
#         metrics['avg_regime_duration'] = self._compute_regime_stability(df_pandas)
        
#         print("[EVAL] Computing regime balance...")
#         # 5. Regime balance (how evenly distributed are regimes)
#         metrics['regime_balance'] = self._compute_regime_balance(df_pandas)
        
#         print("[EVAL] Computing volatility separation...")
#         # 6. Volatility separation (how well regimes separate volatility)
#         metrics['volatility_separation'] = self._compute_volatility_separation(df_pandas)
        
#         print("[EVAL] Computing transition frequency...")
#         # 7. Transition frequency
#         metrics['transition_frequency'] = self._compute_transition_frequency(df_pandas)
        
#         print("[EVAL] Computing within-regime volatility...")
#         # 8. Within-regime volatility (lower is better)
#         metrics['within_regime_volatility'] = self._compute_within_regime_volatility(df_pandas)
        
#         print("[EVAL] Metrics computed")
#         return metrics
    
#     def _compute_regime_stability(self, df_pandas):
#         """
#         Compute average regime duration (in number of periods).
#         Longer durations = more stable regimes.
#         """
#         if 'regime' not in df_pandas.columns:
#             return np.nan
        
#         # Find regime changes
#         regime_changes = (df_pandas['regime'] != df_pandas['regime'].shift(1)).astype(int)
#         regime_changes.iloc[0] = 1  # First point is always a "change"
        
#         # Group by regime periods
#         regime_periods = regime_changes.cumsum()
#         period_lengths = df_pandas.groupby(regime_periods).size()
        
#         return period_lengths.mean()
    
#     def _compute_regime_balance(self, df_pandas):
#         """
#         Compute regime balance using entropy.
#         1.0 = perfectly balanced, 0.0 = all in one regime.
#         """
#         if 'regime' not in df_pandas.columns:
#             return np.nan
        
#         regime_counts = df_pandas['regime'].value_counts(normalize=True)
        
#         # Compute entropy
#         entropy = -np.sum(regime_counts * np.log(regime_counts + 1e-10))
        
#         # Normalize by max entropy (log of number of regimes)
#         n_regimes = len(regime_counts)
#         max_entropy = np.log(n_regimes)
        
#         if max_entropy > 0:
#             return entropy / max_entropy
#         else:
#             return 0.0
    
#     def _compute_volatility_separation(self, df_pandas):
#         """
#         Compute how well regimes separate volatility levels.
#         Uses F-statistic from one-way ANOVA.
#         Higher values = better separation.
#         """
#         if 'regime' not in df_pandas.columns or 'vol_60m' not in df_pandas.columns:
#             return np.nan
        
#         from scipy import stats
        
#         # Get volatility for each regime
#         regimes = df_pandas['regime'].unique()
#         vol_by_regime = [df_pandas[df_pandas['regime'] == r]['vol_60m'].values 
#                         for r in regimes]
        
#         # One-way ANOVA
#         try:
#             f_stat, p_value = stats.f_oneway(*vol_by_regime)
#             return f_stat
#         except:
#             return np.nan
    
#     def _compute_transition_frequency(self, df_pandas):
#         """
#         Compute fraction of periods with regime transitions.
#         Lower values = more stable regimes.
#         """
#         if 'regime' not in df_pandas.columns:
#             return np.nan
        
#         transitions = (df_pandas['regime'] != df_pandas['regime'].shift(1)).astype(int)
#         transitions.iloc[0] = 0  # First point is not a transition
        
#         return transitions.mean()
    
#     def _compute_within_regime_volatility(self, df_pandas):
#         """
#         Compute average volatility of returns within each regime.
#         Lower values = more homogeneous regimes.
#         """
#         if 'regime' not in df_pandas.columns or 'log_return' not in df_pandas.columns:
#             return np.nan
        
#         # Standard deviation of returns within each regime
#         regime_vols = df_pandas.groupby('regime')['log_return'].std()
        
#         # Weighted average by regime size
#         regime_sizes = df_pandas.groupby('regime').size()
#         weights = regime_sizes / regime_sizes.sum()
        
#         return (regime_vols * weights).sum()
    
#     def compare_methods(self, results_dict):
#         """
#         Compare multiple regime detection methods.
        
#         Args:
#             results_dict: Dict of {method_name: df_with_regimes}
            
#         Returns:
#             Comparison DataFrame
#         """
#         print("[EVAL] Comparing methods...")
        
#         comparison = []
        
#         for method, df_regime in results_dict.items():
#             if df_regime is None:
#                 continue
            
#             metrics = self.evaluate(df_regime)
#             metrics['method'] = method
#             comparison.append(metrics)
        
#         comparison_df = pd.DataFrame(comparison)
        
#         # Reorder columns
#         cols = ['method'] + [c for c in comparison_df.columns if c != 'method']
#         comparison_df = comparison_df[cols]
        
#         return comparison_df
    
#     def rank_methods(self, comparison_df):
#         """
#         Rank methods based on composite score.
        
#         Higher is better for: silhouette, calinski_harabasz, regime_balance, 
#                              volatility_separation, avg_regime_duration
#         Lower is better for: davies_bouldin, transition_frequency, 
#                            within_regime_volatility
#         """
#         if comparison_df.empty:
#             return None
        
#         df = comparison_df.copy()
        
#         # Normalize each metric to 0-1 scale
#         metrics_higher_better = ['silhouette_score', 'calinski_harabasz_score', 
#                                 'regime_balance', 'volatility_separation',
#                                 'avg_regime_duration']
        
#         metrics_lower_better = ['davies_bouldin_score', 'transition_frequency',
#                                'within_regime_volatility']
        
#         scores = pd.DataFrame()
#         scores['method'] = df['method']
        
#         # Normalize higher-is-better metrics
#         for metric in metrics_higher_better:
#             if metric in df.columns:
#                 vals = df[metric].values
#                 if not np.all(np.isnan(vals)):
#                     min_val, max_val = np.nanmin(vals), np.nanmax(vals)
#                     if max_val > min_val:
#                         scores[metric] = (vals - min_val) / (max_val - min_val)
#                     else:
#                         scores[metric] = 0.5
#                 else:
#                     scores[metric] = np.nan
        
#         # Normalize lower-is-better metrics (invert)
#         for metric in metrics_lower_better:
#             if metric in df.columns:
#                 vals = df[metric].values
#                 if not np.all(np.isnan(vals)):
#                     min_val, max_val = np.nanmin(vals), np.nanmax(vals)
#                     if max_val > min_val:
#                         scores[metric] = 1 - (vals - min_val) / (max_val - min_val)
#                     else:
#                         scores[metric] = 0.5
#                 else:
#                     scores[metric] = np.nan
        
#         # Compute composite score (average of normalized metrics)
#         metric_cols = [c for c in scores.columns if c != 'method']
#         scores['composite_score'] = scores[metric_cols].mean(axis=1)
        
#         # Rank
#         scores = scores.sort_values('composite_score', ascending=False)
#         scores['rank'] = range(1, len(scores) + 1)
        
#         return scores[['rank', 'method', 'composite_score']]

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


class RegimeEvaluator:
    """
    Evaluate quality of regime detection.
    Computes clustering metrics and regime stability measures.
    """
    
    def __init__(self):
        print("[INFO] Regime evaluator initialized")
    
    def evaluate(self, df_regime: DataFrame, df_original: DataFrame = None, sample_size=50000) -> dict:
        """
        Evaluate regime detection quality.
        
        Args:
            df_regime: DataFrame with regime labels
            df_original: Optional original DataFrame for comparison
            sample_size: Maximum number of rows to use for evaluation (for speed)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("[EVAL] Computing evaluation metrics...")
        
        # Sample if dataset is too large
        total_count = df_regime.count()
        if total_count > sample_size:
            print(f"[EVAL] Sampling {sample_size:,} from {total_count:,} records for faster evaluation...")
            fraction = sample_size / total_count
            df_sample = df_regime.sample(fraction=fraction, seed=42)
        else:
            df_sample = df_regime
        
        # Convert to Pandas for sklearn metrics
        print("[EVAL] Converting to Pandas...")
        
        # Auto-detect which volatility features are available
        available_cols = df_regime.columns
        
        # Try to find volatility columns (prioritize shorter timeframes)
        vol_col = None
        vol_long_col = None
        cum_return_col = None
        
        # Volatility columns (in order of preference)
        for col in ['vol_60m', 'vol_24h', 'vol_30m', 'realized_vol_1h']:
            if col in available_cols:
                vol_col = col
                break
        
        # Longer volatility
        for col in ['vol_240m', 'vol_168h', 'vol_720h']:
            if col in available_cols:
                vol_long_col = col
                break
        
        # Cumulative return
        for col in ['cum_return_60m', 'cum_return_24h', 'cum_return_30m']:
            if col in available_cols:
                cum_return_col = col
                break
        
        # Build select list with available features
        select_cols = ['timestamp', 'regime', 'log_return']
        
        if vol_col:
            select_cols.append(vol_col)
        if vol_long_col:
            select_cols.append(vol_long_col)
        if cum_return_col:
            select_cols.append(cum_return_col)
        if 'volume_spike' in available_cols:
            select_cols.append('volume_spike')
        
        print(f"[EVAL] Using features: {select_cols}")
        
        df_pandas = df_sample.select(select_cols).toPandas()
        
        print(f"[EVAL] Using {len(df_pandas):,} records for evaluation")
        
        # Feature matrix - use all numeric columns except timestamp and regime
        feature_cols = [c for c in df_pandas.columns 
                       if c not in ['timestamp', 'regime', 'regime_name']]
        
        X = df_pandas[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        labels = df_pandas['regime'].values
        
        # Compute metrics
        metrics = {}
        
        print("[EVAL] Computing silhouette score...")
        # 1. Silhouette Score (-1 to 1, higher is better)
        # Measures how similar points are to their own cluster vs other clusters
        try:
            # Silhouette score is very slow on large datasets, sample if needed
            if len(X_scaled) > 10000:
                sample_idx = np.random.choice(len(X_scaled), 10000, replace=False)
                metrics['silhouette_score'] = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
            else:
                metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
        except Exception as e:
            print(f"[EVAL] Silhouette score failed: {e}")
            metrics['silhouette_score'] = np.nan
        
        print("[EVAL] Computing calinski-harabasz score...")
        # 2. Calinski-Harabasz Score (higher is better)
        # Ratio of between-cluster to within-cluster dispersion
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
        except:
            metrics['calinski_harabasz_score'] = np.nan
        
        print("[EVAL] Computing davies-bouldin score...")
        # 3. Davies-Bouldin Score (lower is better)
        # Average similarity between each cluster and its most similar cluster
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
        except:
            metrics['davies_bouldin_score'] = np.nan
        
        print("[EVAL] Computing regime stability...")
        # 4. Regime stability (average regime duration)
        metrics['avg_regime_duration'] = self._compute_regime_stability(df_pandas)
        
        print("[EVAL] Computing regime balance...")
        # 5. Regime balance (how evenly distributed are regimes)
        metrics['regime_balance'] = self._compute_regime_balance(df_pandas)
        
        print("[EVAL] Computing volatility separation...")
        # 6. Volatility separation (how well regimes separate volatility)
        metrics['volatility_separation'] = self._compute_volatility_separation(df_pandas)
        
        print("[EVAL] Computing transition frequency...")
        # 7. Transition frequency
        metrics['transition_frequency'] = self._compute_transition_frequency(df_pandas)
        
        print("[EVAL] Computing within-regime volatility...")
        # 8. Within-regime volatility (lower is better)
        metrics['within_regime_volatility'] = self._compute_within_regime_volatility(df_pandas)
        
        print("[EVAL] Metrics computed")
        return metrics
    
    def _compute_regime_stability(self, df_pandas):
        """
        Compute average regime duration (in number of periods).
        Longer durations = more stable regimes.
        """
        if 'regime' not in df_pandas.columns:
            return np.nan
        
        # Find regime changes
        regime_changes = (df_pandas['regime'] != df_pandas['regime'].shift(1)).astype(int)
        regime_changes.iloc[0] = 1  # First point is always a "change"
        
        # Group by regime periods
        regime_periods = regime_changes.cumsum()
        period_lengths = df_pandas.groupby(regime_periods).size()
        
        return period_lengths.mean()
    
    def _compute_regime_balance(self, df_pandas):
        """
        Compute regime balance using entropy.
        1.0 = perfectly balanced, 0.0 = all in one regime.
        """
        if 'regime' not in df_pandas.columns:
            return np.nan
        
        regime_counts = df_pandas['regime'].value_counts(normalize=True)
        
        # Compute entropy
        entropy = -np.sum(regime_counts * np.log(regime_counts + 1e-10))
        
        # Normalize by max entropy (log of number of regimes)
        n_regimes = len(regime_counts)
        max_entropy = np.log(n_regimes)
        
        if max_entropy > 0:
            return entropy / max_entropy
        else:
            return 0.0
    
    def _compute_volatility_separation(self, df_pandas):
        """
        Compute how well regimes separate volatility levels.
        Uses F-statistic from one-way ANOVA.
        Higher values = better separation.
        """
        if 'regime' not in df_pandas.columns:
            return np.nan
        
        # Auto-detect volatility column
        vol_col = None
        for col in ['vol_60m', 'vol_24h', 'vol_168h', 'vol_30m', 'realized_vol_1h']:
            if col in df_pandas.columns:
                vol_col = col
                break
        
        if vol_col is None:
            return np.nan
        
        from scipy import stats
        
        # Get volatility for each regime
        regimes = df_pandas['regime'].unique()
        vol_by_regime = [df_pandas[df_pandas['regime'] == r][vol_col].values 
                        for r in regimes]
        
        # One-way ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*vol_by_regime)
            return f_stat
        except:
            return np.nan
    
    def _compute_transition_frequency(self, df_pandas):
        """
        Compute fraction of periods with regime transitions.
        Lower values = more stable regimes.
        """
        if 'regime' not in df_pandas.columns:
            return np.nan
        
        transitions = (df_pandas['regime'] != df_pandas['regime'].shift(1)).astype(int)
        transitions.iloc[0] = 0  # First point is not a transition
        
        return transitions.mean()
    
    def _compute_within_regime_volatility(self, df_pandas):
        """
        Compute average volatility of returns within each regime.
        Lower values = more homogeneous regimes.
        """
        if 'regime' not in df_pandas.columns or 'log_return' not in df_pandas.columns:
            return np.nan
        
        # Standard deviation of returns within each regime
        regime_vols = df_pandas.groupby('regime')['log_return'].std()
        
        # Weighted average by regime size
        regime_sizes = df_pandas.groupby('regime').size()
        weights = regime_sizes / regime_sizes.sum()
        
        return (regime_vols * weights).sum()
    
    def compare_methods(self, results_dict):
        """
        Compare multiple regime detection methods.
        
        Args:
            results_dict: Dict of {method_name: df_with_regimes}
            
        Returns:
            Comparison DataFrame
        """
        print("[EVAL] Comparing methods...")
        
        comparison = []
        
        for method, df_regime in results_dict.items():
            if df_regime is None:
                continue
            
            metrics = self.evaluate(df_regime)
            metrics['method'] = method
            comparison.append(metrics)
        
        comparison_df = pd.DataFrame(comparison)
        
        # Reorder columns
        cols = ['method'] + [c for c in comparison_df.columns if c != 'method']
        comparison_df = comparison_df[cols]
        
        return comparison_df
    
    def rank_methods(self, comparison_df):
        """
        Rank methods based on composite score.
        
        Higher is better for: silhouette, calinski_harabasz, regime_balance, 
                             volatility_separation, avg_regime_duration
        Lower is better for: davies_bouldin, transition_frequency, 
                           within_regime_volatility
        """
        if comparison_df.empty:
            return None
        
        df = comparison_df.copy()
        
        # Normalize each metric to 0-1 scale
        metrics_higher_better = ['silhouette_score', 'calinski_harabasz_score', 
                                'regime_balance', 'volatility_separation',
                                'avg_regime_duration']
        
        metrics_lower_better = ['davies_bouldin_score', 'transition_frequency',
                               'within_regime_volatility']
        
        scores = pd.DataFrame()
        scores['method'] = df['method']
        
        # Normalize higher-is-better metrics
        for metric in metrics_higher_better:
            if metric in df.columns:
                vals = df[metric].values
                if not np.all(np.isnan(vals)):
                    min_val, max_val = np.nanmin(vals), np.nanmax(vals)
                    if max_val > min_val:
                        scores[metric] = (vals - min_val) / (max_val - min_val)
                    else:
                        scores[metric] = 0.5
                else:
                    scores[metric] = np.nan
        
        # Normalize lower-is-better metrics (invert)
        for metric in metrics_lower_better:
            if metric in df.columns:
                vals = df[metric].values
                if not np.all(np.isnan(vals)):
                    min_val, max_val = np.nanmin(vals), np.nanmax(vals)
                    if max_val > min_val:
                        scores[metric] = 1 - (vals - min_val) / (max_val - min_val)
                    else:
                        scores[metric] = 0.5
                else:
                    scores[metric] = np.nan
        
        # Compute composite score (average of normalized metrics)
        metric_cols = [c for c in scores.columns if c != 'method']
        scores['composite_score'] = scores[metric_cols].mean(axis=1)
        
        # Rank
        scores = scores.sort_values('composite_score', ascending=False)
        scores['rank'] = range(1, len(scores) + 1)
        
        return scores[['rank', 'method', 'composite_score']]