"""
feature_importance.py

Computes feature importance across regimes using both:
  1. Permutation importance — fast, model-agnostic
  2. SHAP values           — slower, more rigorous (requires shap library)

Works on any symbol — just pass the feature DataFrame and regime labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


# Features used by the GARCH regime detector
REGIME_FEATURES = [
    'log_return', 'vol_60m', 'vol_240m', 'vol_ratio_60_240',
    'cum_return_60m', 'return_skew_60m',
    'volume_spike', 'rsi', 'bb_width', 'atr_pct'
]

REGIME_NAMES = {0: 'Low Vol', 1: 'Medium Vol', 2: 'High Vol'}
REGIME_COLORS = {0: '#3498db', 1: '#f39c12', 2: '#e74c3c'}


class FeatureImportanceAnalyzer:
    """
    Analyses which features most strongly define each regime.

    Approach:
      Train a RandomForest to predict regime labels, then compute:
        - Global permutation importance (which features matter overall)
        - Per-regime SHAP values (which features push each regime up/down)
        - Per-regime permutation importance (one-vs-rest per regime)

    Symbol-agnostic: works for any asset as long as the feature
    columns match REGIME_FEATURES.
    """

    def __init__(self, symbol: str = 'BTCUSDT', n_regimes: int = 3):
        self.symbol    = symbol
        self.n_regimes = n_regimes
        self.rf_model  = None
        self.scaler    = StandardScaler()
        self.feature_cols = None
        self.shap_available = False

        try:
            import shap
            self.shap_available = True
            print(f"[FeatureImportance] SHAP available ✓")
        except ImportError:
            print(f"[FeatureImportance] SHAP not installed — "
                  f"using permutation importance only")
            print(f"                   Install with: pip install shap")

    def fit(self, df: pd.DataFrame, regimes: np.ndarray) -> 'FeatureImportanceAnalyzer':
        """
        Fit RandomForest on features → regime labels.

        Args:
            df:      pandas DataFrame with feature columns
            regimes: array of regime labels (0/1/2)
        """
        # Select available feature columns
        self.feature_cols = [c for c in REGIME_FEATURES if c in df.columns]
        print(f"\n[FeatureImportance] Fitting on {len(self.feature_cols)} features, "
              f"{len(df):,} samples...")

        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = regimes.astype(int)

        X_scaled = self.scaler.fit_transform(X)

        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.rf_model.fit(X_scaled, y)

        acc = self.rf_model.score(X_scaled, y)
        print(f"[FeatureImportance] RF training accuracy: {acc:.1%}")
        print(f"[FeatureImportance] (in-sample — used for importance only, not prediction)")

        self._X_scaled = X_scaled
        self._y        = y
        self._df       = df

        return self

    def global_importance(self) -> pd.DataFrame:
        """
        Global feature importance (RF Gini impurity + permutation).
        Returns DataFrame sorted by importance descending.
        """
        # Gini importance
        gini_imp = self.rf_model.feature_importances_

        # Permutation importance (more reliable than Gini for correlated features)
        perm = permutation_importance(
            self.rf_model, self._X_scaled, self._y,
            n_repeats=10, random_state=42, n_jobs=-1
        )

        result = pd.DataFrame({
            'feature':            self.feature_cols,
            'gini_importance':    gini_imp,
            'perm_importance':    perm.importances_mean,
            'perm_std':           perm.importances_std,
        }).sort_values('perm_importance', ascending=False).reset_index(drop=True)

        result['rank'] = range(1, len(result) + 1)
        return result

    def per_regime_importance(self) -> dict:
        """
        One-vs-rest permutation importance per regime.
        Returns {regime: DataFrame sorted by importance}.
        """
        results = {}
        for r in range(self.n_regimes):
            y_binary = (self._y == r).astype(int)

            # Fit one-vs-rest RF
            rf_ovr = RandomForestClassifier(
                n_estimators=100, max_depth=6,
                random_state=42, n_jobs=-1,
                class_weight='balanced'
            )
            rf_ovr.fit(self._X_scaled, y_binary)

            perm = permutation_importance(
                rf_ovr, self._X_scaled, y_binary,
                n_repeats=8, random_state=42, n_jobs=-1
            )

            results[r] = pd.DataFrame({
                'feature':         self.feature_cols,
                'importance':      perm.importances_mean,
                'std':             perm.importances_std,
                'gini':            rf_ovr.feature_importances_,
            }).sort_values('importance', ascending=False).reset_index(drop=True)

        return results

    def shap_importance(self) -> dict:
        """
        SHAP values per regime (requires shap library).
        Returns {regime: mean_abs_shap_per_feature DataFrame}
        """
        if not self.shap_available:
            print("[FeatureImportance] SHAP not available, skipping")
            return {}

        import shap
        print("[FeatureImportance] Computing SHAP values (this may take a minute)...")

        # Use TreeExplainer for RandomForest — fast
        explainer  = shap.TreeExplainer(self.rf_model)
        shap_vals  = explainer.shap_values(self._X_scaled)
        # shap_vals is list of arrays [regime_0, regime_1, regime_2]

        results = {}
        for r in range(self.n_regimes):
            if r >= len(shap_vals):
                continue
            sv = np.abs(shap_vals[r])  # shape (n_samples, n_features)
            results[r] = pd.DataFrame({
                'feature':          self.feature_cols,
                'mean_abs_shap':    sv.mean(axis=0),
                'std_shap':         sv.std(axis=0),
            }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        print("[FeatureImportance] SHAP complete")
        return results

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_global_importance(self, output_path: Path):
        imp_df = self.global_importance()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.symbol} — Global Feature Importance\n'
                     f'(Predicting Volatility Regime)',
                     fontsize=13, fontweight='bold')

        # Permutation importance
        ax = axes[0]
        colors = ['#2ecc71' if i < 3 else '#95a5a6'
                  for i in range(len(imp_df))]
        bars = ax.barh(imp_df['feature'][::-1],
                       imp_df['perm_importance'][::-1],
                       xerr=imp_df['perm_std'][::-1],
                       color=colors[::-1], edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Permutation Importance\n(more reliable for correlated features)')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(0, color='black', lw=0.8)

        # Gini importance
        ax = axes[1]
        imp_sorted = imp_df.sort_values('gini_importance', ascending=False)
        colors2 = ['#e74c3c' if i < 3 else '#95a5a6'
                   for i in range(len(imp_sorted))]
        ax.barh(imp_sorted['feature'][::-1],
                imp_sorted['gini_importance'][::-1],
                color=colors2[::-1], edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Gini Importance')
        ax.set_title('Gini (Impurity) Importance')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")

    def plot_per_regime_importance(self, output_path: Path):
        regime_imp = self.per_regime_importance()

        fig, axes = plt.subplots(1, self.n_regimes,
                                 figsize=(6 * self.n_regimes, 7))
        fig.suptitle(f'{self.symbol} — Per-Regime Feature Importance\n'
                     f'(One-vs-Rest: which features define each regime)',
                     fontsize=13, fontweight='bold')

        if self.n_regimes == 1:
            axes = [axes]

        for r, ax in enumerate(axes):
            df_r  = regime_imp.get(r, pd.DataFrame())
            if df_r.empty:
                continue
            color = REGIME_COLORS.get(r, '#7f8c8d')
            name  = REGIME_NAMES.get(r, f'Regime {r}')

            ax.barh(df_r['feature'][::-1],
                    df_r['importance'][::-1],
                    xerr=df_r['std'][::-1],
                    color=color, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
            ax.set_title(f'Regime {r}: {name}', fontweight='bold', color=color)
            ax.set_xlabel('Permutation Importance')
            ax.grid(True, alpha=0.3, axis='x')
            ax.axvline(0, color='black', lw=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")

    def plot_shap_summary(self, output_path: Path):
        if not self.shap_available:
            print("[FeatureImportance] SHAP not available, skipping plot")
            return

        shap_imp = self.shap_importance()
        if not shap_imp:
            return

        fig, axes = plt.subplots(1, len(shap_imp),
                                 figsize=(6 * len(shap_imp), 7))
        fig.suptitle(f'{self.symbol} — SHAP Feature Importance per Regime',
                     fontsize=13, fontweight='bold')

        if len(shap_imp) == 1:
            axes = [axes]

        for (r, df_r), ax in zip(shap_imp.items(), axes):
            color = REGIME_COLORS.get(r, '#7f8c8d')
            name  = REGIME_NAMES.get(r, f'Regime {r}')

            ax.barh(df_r['feature'][::-1],
                    df_r['mean_abs_shap'][::-1],
                    color=color, alpha=0.8,
                    edgecolor='black', linewidth=0.5)
            ax.set_title(f'Regime {r}: {name}\n(Mean |SHAP|)',
                         fontweight='bold', color=color)
            ax.set_xlabel('Mean |SHAP Value|')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIZ] Saved: {output_path.name}")