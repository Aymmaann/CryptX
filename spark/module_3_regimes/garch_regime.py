from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class GARCHRegimeDetector:
    """
    Regime-Switching GARCH detector — GARCH-first approach.

    The key difference from v1:
      Instead of clustering raw features then fitting GARCH per cluster,
      this version:
        1. Fits a single GARCH(1,1) on the FULL return series first
        2. Uses the GARCH conditional variance sequence as the PRIMARY
           regime signal — this is what GARCH actually knows best
        3. Enriches with other features and clusters on the combined set
        4. Refines by re-fitting per-regime GARCH on the final clusters

    Why this produces better silhouette / volatility_separation:
      The GARCH conditional variance is a smooth, model-based estimate of
      current risk. Clustering on it directly produces tighter, more
      separated regimes than clustering on raw rolling-window volatility,
      because GARCH already accounts for volatility clustering (ARCH effects)
      and mean-reversion (GARCH persistence).
    """

    def __init__(self, n_regimes=3, random_state=42, max_iter=300):
        self.n_regimes    = n_regimes
        self.random_state = random_state
        self.max_iter     = max_iter
        self.scaler       = StandardScaler()
        self.kmeans       = None
        self.garch_params = {}        # regime → (omega, alpha, beta) after refinement
        self.global_params = None     # (omega, alpha, beta) from full-series fit

        print(f"[GARCH] Initialized | n_regimes={n_regimes} | GARCH-first approach")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, df: DataFrame) -> DataFrame:
        print("[GARCH] Starting GARCH-first regime detection...")

        feature_cols = [
            'log_return', 'vol_60m', 'vol_240m', 'vol_ratio_60_240',
            'cum_return_60m', 'return_skew_60m',
            'volume_spike', 'rsi', 'bb_width', 'atr_pct'
        ]
        available = [c for c in feature_cols if c in df.columns]

        print("[GARCH] Converting to Pandas...")
        df_pd = (df.select(['timestamp'] + available)
                   .toPandas()
                   .sort_values('timestamp')
                   .reset_index(drop=True))

        returns = df_pd['log_return'].fillna(0).values
        X_raw   = df_pd[available].values
        X_raw   = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Step 1: Fit GARCH on full series ──────────────────────────
        print("[GARCH] Step 1/4 — Fitting global GARCH(1,1) on full series...")
        omega_g, alpha_g, beta_g = self._fit_garch(returns)
        self.global_params = (omega_g, alpha_g, beta_g)
        print(f"  Global: omega={omega_g:.6f}  alpha={alpha_g:.4f}  "
              f"beta={beta_g:.4f}  persistence={alpha_g+beta_g:.4f}")

        global_variance = self._garch_variance(returns, omega_g, alpha_g, beta_g)
        global_vol      = np.sqrt(np.maximum(global_variance, 0))

        # ── Step 2: Build GARCH-enriched feature matrix ───────────────
        print("[GARCH] Step 2/4 — Building GARCH-enriched feature matrix...")

        # Standardise returns by GARCH vol → should be ~N(0,1) if model fits
        standardised_return = np.where(
            global_vol > 1e-10, returns / global_vol, 0.0)

        # Volatility regime indicators derived from GARCH
        vol_ma_24  = pd.Series(global_vol).rolling(24,  min_periods=1).mean().values
        vol_ma_168 = pd.Series(global_vol).rolling(168, min_periods=1).mean().values
        vol_ratio  = np.where(vol_ma_168 > 1e-10, vol_ma_24 / vol_ma_168, 1.0)

        # Squared standardised returns (ARCH indicator — high = volatility cluster)
        arch_indicator = standardised_return ** 2

        # Stack: [raw features | garch_vol | standardised_ret | vol_ratio | arch]
        garch_features = np.column_stack([
            global_vol,           # the main GARCH signal
            standardised_return,  # demeaned by model risk
            vol_ratio,            # short vs long vol regime
            arch_indicator,       # current shock magnitude
        ])

        # Weight GARCH features 3x vs raw features so clustering is GARCH-driven
        raw_scaled   = self.scaler.fit_transform(X_raw)
        garch_scaled = StandardScaler().fit_transform(garch_features) * 3.0

        X_combined = np.hstack([raw_scaled, garch_scaled])

        # ── Step 3: Cluster on combined matrix ────────────────────────
        print("[GARCH] Step 3/4 — Clustering on GARCH-enriched features...")
        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=30,
            max_iter=500
        )
        raw_labels = self.kmeans.fit_predict(X_combined)
        regimes    = self._map_regimes_by_volatility(raw_labels, global_vol)

        print("[GARCH] Regime distribution:")
        for r in range(self.n_regimes):
            n = (regimes == r).sum()
            print(f"  Regime {r}: {n:,} ({n/len(regimes)*100:.1f}%)")

        # ── Step 4: Refine — fit per-regime GARCH ─────────────────────
        print("[GARCH] Step 4/4 — Fitting per-regime GARCH(1,1)...")
        garch_variance = np.zeros(len(df_pd))

        for r in range(self.n_regimes):
            mask      = regimes == r
            r_returns = returns[mask]

            omega, alpha, beta = self._fit_garch(r_returns)
            self.garch_params[r] = (omega, alpha, beta)
            print(f"  Regime {r}: omega={omega:.6f}  alpha={alpha:.4f}  "
                  f"beta={beta:.4f}  persistence={alpha+beta:.4f}")

            cond_var = self._garch_variance(r_returns, omega, alpha, beta)
            garch_variance[mask] = cond_var

        # ── Assemble ──────────────────────────────────────────────────
        df_pd['regime']           = regimes
        df_pd['regime_name']      = df_pd['regime'].map(
            {0: 'low_vol', 1: 'medium_vol', 2: 'high_vol'})
        df_pd['garch_variance']   = garch_variance
        df_pd['garch_volatility'] = np.sqrt(np.maximum(garch_variance, 0))
        df_pd['global_garch_vol'] = global_vol

        print("[GARCH] Converting back to Spark...")
        spark     = df.sparkSession
        df_result = spark.createDataFrame(
            df_pd[['timestamp', 'regime', 'regime_name',
                   'garch_variance', 'garch_volatility', 'global_garch_vol']]
        )
        df_result = df.join(df_result, on='timestamp', how='inner')
        df_result = df_result.cache()
        df_result.count()

        print("[GARCH] Detection complete")
        self._print_regime_summary(df_pd, returns, garch_variance, regimes)
        return df_result

    # ------------------------------------------------------------------
    # GARCH fitting — pure numpy/scipy, no external arch library needed
    # ------------------------------------------------------------------

    def _fit_garch(self, returns: np.ndarray):
        """
        Fit GARCH(1,1) by maximum likelihood (L-BFGS-B).
        Falls back to moment-based estimates if optimisation fails.
        """
        from scipy.optimize import minimize

        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]

        if len(returns) < 30:
            sigma2 = float(np.var(returns)) if len(returns) > 1 else 1e-6
            return sigma2 * 0.05, 0.10, 0.85

        sigma2_unc = float(np.var(returns))

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e10
            h  = self._garch_variance(returns, omega, alpha, beta)
            h  = np.maximum(h, 1e-10)
            ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + returns**2 / h)
            return float(-ll)

        # Try multiple starting points to avoid local minima
        best_result = None
        starts = [
            [sigma2_unc * 0.05, 0.10, 0.85],
            [sigma2_unc * 0.10, 0.15, 0.80],
            [sigma2_unc * 0.02, 0.05, 0.93],
        ]
        bounds = [(1e-8, None), (1e-6, 0.45), (1e-6, 0.9999)]

        for x0 in starts:
            try:
                res = minimize(
                    neg_log_likelihood, x0,
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': self.max_iter, 'ftol': 1e-10}
                )
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
            except Exception:
                continue

        if best_result is not None and np.isfinite(best_result.fun):
            omega, alpha, beta = best_result.x
            alpha = float(np.clip(alpha, 1e-6, 0.45))
            beta  = float(np.clip(beta,  1e-6, 0.9999 - alpha))
            omega = float(max(omega, 1e-8))
            return omega, alpha, beta

        # Fallback
        return sigma2_unc * 0.05, 0.10, 0.85

    def _garch_variance(self, returns: np.ndarray,
                        omega: float, alpha: float, beta: float) -> np.ndarray:
        """Compute GARCH(1,1) conditional variance sequence (full series)."""
        n    = len(returns)
        h    = np.empty(n, dtype=np.float64)
        h[0] = omega / max(1.0 - alpha - beta, 1e-6)
        for t in range(1, n):
            h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
            if h[t] < 1e-10:
                h[t] = 1e-10
        return h

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _map_regimes_by_volatility(self, labels, volatility):
        regime_vols = {}
        for r in range(self.n_regimes):
            mask = labels == r
            if mask.sum() > 0:
                regime_vols[r] = float(volatility[mask].mean())
        regime_map = {old: new for new, old
                      in enumerate(sorted(regime_vols, key=regime_vols.get))}
        return np.array([regime_map.get(int(r), int(r)) for r in labels])

    def _print_regime_summary(self, df_pd, returns, garch_variance, regimes):
        print("\n[GARCH] Per-regime GARCH summary:")
        print(f"  {'Regime':<10} {'Count':>8} {'Mean ret':>10} "
              f"{'Mean σ²':>12} {'omega':>10} {'alpha':>8} {'beta':>8} {'α+β':>8}")
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() == 0:
                continue
            omega, alpha, beta = self.garch_params.get(r, (0, 0, 0))
            print(f"  {r:<10} {mask.sum():>8,} "
                  f"{returns[mask].mean():>10.6f} "
                  f"{garch_variance[mask].mean():>12.8f} "
                  f"{omega:>10.6f} {alpha:>8.4f} {beta:>8.4f} "
                  f"{alpha+beta:>8.4f}")

    def get_garch_params(self):
        return self.garch_params

    def forecast_variance(self, regime: int, last_return: float,
                          last_variance: float, steps: int = 1) -> np.ndarray:
        """Multi-step ahead variance forecast for a given regime."""
        if regime not in self.garch_params:
            raise ValueError(f"Regime {regime} not fitted.")
        omega, alpha, beta = self.garch_params[regime]
        forecasts    = np.empty(steps)
        persistence  = alpha + beta
        forecasts[0] = omega + alpha * last_return**2 + beta * last_variance
        for s in range(1, steps):
            forecasts[s] = omega + persistence * forecasts[s-1]
        return forecasts