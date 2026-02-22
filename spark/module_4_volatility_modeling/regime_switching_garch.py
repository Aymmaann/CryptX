"""
regime_switching_garch.py

Regime-Switching GARCH volatility model.

Loads the detected regimes from Module 3 (GARCH detector output),
then fits a separate GARCH(1,1) inside each regime.

At each forecast step, the model:
  1. Identifies which regime the current bar belongs to
  2. Uses that regime's GARCH parameters for the forecast

This is the key hypothesis of the research:
  Regime-conditioned volatility forecasts are more accurate than
  a single pooled GARCH because each regime has distinct
  volatility dynamics (different omega/alpha/beta).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path


class RegimeSwitchingGARCH:
    """
    Regime-Switching GARCH(1,1).

    One GARCH model is fitted per regime. During rolling forecast,
    the active regime label determines which model's parameters are used.

    Evaluation is directly comparable to BaselineGARCH — same rolling
    window, same OOS period, same metrics.
    """

    def __init__(self, n_regimes=3, train_frac=0.8,
                 refit_every=168, max_iter=300):
        self.n_regimes   = n_regimes
        self.train_frac  = train_frac
        self.refit_every = refit_every
        self.max_iter    = max_iter

        self.regime_params   = {}   # {regime: (omega, alpha, beta)}
        self.fitted_var      = None
        self.forecasts_1h    = None
        self.returns_        = None
        self.regimes_        = None

        print(f"[RS-GARCH] Initialized | n_regimes={n_regimes} | "
              f"train_frac={train_frac} refit_every={refit_every}h")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray, regimes: np.ndarray):
        """
        Fit per-regime GARCH models and run rolling OOS forecast.

        Args:
            returns:  hourly log-returns (length N)
            regimes:  integer regime labels 0..n_regimes-1 (length N)
        """
        returns = np.asarray(returns, dtype=np.float64)
        regimes = np.asarray(regimes, dtype=np.int32)
        returns = np.where(np.isfinite(returns), returns, 0.0)

        self.returns_ = returns
        self.regimes_ = regimes
        n             = len(returns)
        train_end     = int(n * self.train_frac)

        print(f"[RS-GARCH] Fitting on {n:,} observations "
              f"(train: {train_end:,} | OOS: {n - train_end:,})")

        # Fit one GARCH per regime on training data
        print(f"\n[RS-GARCH] Per-regime GARCH fits (training set):")
        print(f"  {'Regime':<8} {'N':>8} {'omega':>10} "
              f"{'alpha':>8} {'beta':>8} {'α+β':>8}")

        for r in range(self.n_regimes):
            mask      = (regimes[:train_end] == r)
            r_returns = returns[:train_end][mask]

            if len(r_returns) < 30:
                # Fallback for very sparse regimes
                sigma2 = float(np.var(returns[:train_end]))
                self.regime_params[r] = (sigma2 * 0.05, 0.10, 0.85)
                print(f"  {r:<8} {len(r_returns):>8} "
                      f"[fallback — insufficient data]")
                continue

            omega, alpha, beta = self._fit_garch(r_returns)
            self.regime_params[r] = (omega, alpha, beta)
            print(f"  {r:<8} {len(r_returns):>8,} "
                  f"{omega:>10.6f} {alpha:>8.4f} {beta:>8.4f} "
                  f"{alpha+beta:>8.4f}")

        # Compute in-sample conditional variance (for each bar, use its regime's model)
        self.fitted_var = self._compute_insample_variance(
            returns, regimes, train_end)

        # Rolling OOS forecast
        print(f"\n[RS-GARCH] Running rolling OOS forecast...")
        self.forecasts_1h = self._rolling_forecast(returns, regimes)
        print(f"[RS-GARCH] OOS forecast complete "
              f"({np.isfinite(self.forecasts_1h).sum():,} valid points)")

        return self

    def _compute_insample_variance(self, returns, regimes, train_end):
        """Compute in-sample GARCH variance using per-regime parameters."""
        n   = train_end
        var = np.empty(n, dtype=np.float64)

        # Track separate h_prev per regime
        h_prev = {r: (self.regime_params[r][0] /
                      max(1 - self.regime_params[r][1] - self.regime_params[r][2], 1e-6))
                  for r in range(self.n_regimes)}

        for t in range(n):
            r = int(regimes[t])
            if r not in self.regime_params:
                r = 0
            omega, alpha, beta = self.regime_params[r]
            if t == 0:
                var[t] = h_prev[r]
            else:
                var[t] = omega + alpha * returns[t-1]**2 + beta * h_prev[r]
                var[t] = max(var[t], 1e-10)
            h_prev[r] = var[t]

        return var

    def _rolling_forecast(self, returns, regimes):
        """
        Rolling 1-step-ahead forecast using per-regime GARCH parameters.
        Re-fits regime models every `refit_every` steps.
        """
        n         = len(returns)
        train_end = int(n * self.train_frac)
        forecasts = np.full(n, np.nan)

        # Initialise h_prev per regime from training unconditional variance
        h_prev = {r: (self.regime_params[r][0] /
                      max(1 - self.regime_params[r][1] - self.regime_params[r][2], 1e-6))
                  for r in range(self.n_regimes)}

        for t in range(train_end, n):
            r = int(regimes[t])
            if r not in self.regime_params:
                r = 0

            omega, alpha, beta = self.regime_params[r]
            h_forecast    = omega + alpha * returns[t-1]**2 + beta * h_prev[r]
            h_forecast    = max(h_forecast, 1e-10)
            forecasts[t]  = h_forecast
            h_prev[r]     = h_forecast

            # Re-fit on expanding window periodically
            if (t - train_end) % self.refit_every == 0 and t > train_end:
                for regime_id in range(self.n_regimes):
                    mask      = (regimes[:t] == regime_id)
                    r_returns = returns[:t][mask]
                    if len(r_returns) >= 30:
                        omega_new, alpha_new, beta_new = self._fit_garch(r_returns)
                        self.regime_params[regime_id] = (omega_new, alpha_new, beta_new)

        return forecasts

    # ------------------------------------------------------------------
    # GARCH mechanics (same as baseline for fair comparison)
    # ------------------------------------------------------------------

    def _fit_garch(self, returns: np.ndarray):
        returns    = returns[np.isfinite(returns)]
        sigma2_unc = float(np.var(returns)) if len(returns) > 1 else 1e-6

        def nll(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e10
            h  = self._garch_variance(returns, omega, alpha, beta)
            h  = np.maximum(h, 1e-10)
            ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(h) + returns**2 / h)
            return float(-ll)

        bounds = [(1e-8, None), (1e-6, 0.45), (1e-6, 0.9999)]
        starts = [
            [sigma2_unc * 0.05, 0.10, 0.85],
            [sigma2_unc * 0.10, 0.15, 0.80],
            [sigma2_unc * 0.02, 0.05, 0.93],
        ]
        best = None
        for x0 in starts:
            try:
                res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': self.max_iter, 'ftol': 1e-10})
                if best is None or res.fun < best.fun:
                    best = res
            except Exception:
                continue

        if best is not None and np.isfinite(best.fun):
            omega, alpha, beta = best.x
            alpha = float(np.clip(alpha, 1e-6, 0.45))
            beta  = float(np.clip(beta,  1e-6, 0.9999 - alpha))
            return float(max(omega, 1e-8)), alpha, beta

        return sigma2_unc * 0.05, 0.10, 0.85

    def _garch_variance(self, returns, omega, alpha, beta):
        n    = len(returns)
        h    = np.empty(n, dtype=np.float64)
        h[0] = omega / max(1.0 - alpha - beta, 1e-6)
        for t in range(1, n):
            h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
            if h[t] < 1e-10:
                h[t] = 1e-10
        return h

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Compute OOS evaluation metrics — same protocol as BaselineGARCH."""
        if self.forecasts_1h is None:
            raise RuntimeError("Call fit() before evaluate()")

        n         = len(self.returns_)
        train_end = int(n * self.train_frac)

        oos_returns   = self.returns_[train_end:]
        oos_forecasts = self.forecasts_1h[train_end:]

        valid = np.isfinite(oos_forecasts) & np.isfinite(oos_returns)
        r     = oos_returns[valid]
        f     = oos_forecasts[valid]
        rv    = r ** 2

        rmse  = float(np.sqrt(np.mean((f - rv)**2)))
        mae   = float(np.mean(np.abs(f - rv)))
        mape  = float(np.mean(np.abs((f - rv) / np.maximum(rv, 1e-10)))) * 100
        qlike = float(np.mean(np.log(f) + rv / f))
        corr  = float(np.corrcoef(f, rv)[0, 1]) if len(f) > 1 else 0.0

        metrics = {
            'model':   'Regime-Switching GARCH(1,1)',
            'n_oos':    int(valid.sum()),
            'rmse':     rmse,
            'mae':      mae,
            'mape_pct': mape,
            'qlike':    qlike,
            'corr':     corr,
        }

        # Per-regime params
        for r_id, (omega, alpha, beta) in self.regime_params.items():
            metrics[f'regime_{r_id}_omega']       = omega
            metrics[f'regime_{r_id}_alpha']       = alpha
            metrics[f'regime_{r_id}_beta']        = beta
            metrics[f'regime_{r_id}_persistence'] = alpha + beta

        return metrics

    def get_forecast_df(self, timestamps, train_frac=None) -> pd.DataFrame:
        """Return OOS forecasts as a DataFrame."""
        n         = len(self.returns_)
        tf        = train_frac or self.train_frac
        train_end = int(n * tf)
        return pd.DataFrame({
            'timestamp':     timestamps[train_end:],
            'rs_forecast':   self.forecasts_1h[train_end:],
            'rs_vol':        np.sqrt(np.maximum(self.forecasts_1h[train_end:], 0)),
            'realised_var':  self.returns_[train_end:] ** 2,
            'regime':        self.regimes_[train_end:],
        })

    def forecast_ahead(self, steps: int = 24) -> pd.DataFrame:
        """
        Multi-step ahead forecast from the last known bar.
        Uses the most recently active regime's parameters.

        Args:
            steps: forecast horizon in hours

        Returns:
            DataFrame with step, regime, forecast_variance, forecast_vol
        """
        if self.forecasts_1h is None:
            raise RuntimeError("Call fit() before forecast_ahead()")

        # Most recent regime and variance
        last_regime   = int(self.regimes_[-1])
        last_return   = float(self.returns_[-1])
        last_variance = float(self.forecasts_1h[np.where(
            np.isfinite(self.forecasts_1h))[0][-1]])

        if last_regime not in self.regime_params:
            last_regime = 0
        omega, alpha, beta = self.regime_params[last_regime]

        rows = []
        h_prev = last_variance
        r_prev = last_return
        persistence = alpha + beta

        for s in range(1, steps + 1):
            if s == 1:
                h = omega + alpha * r_prev**2 + beta * h_prev
            else:
                # E[r²_{t+s}] = h_{t+s-1} for s > 1
                h = omega + persistence * h_prev
            h = max(h, 1e-10)
            rows.append({
                'step':              s,
                'active_regime':     last_regime,
                'forecast_variance': h,
                'forecast_vol':      np.sqrt(h),
                'forecast_vol_ann':  np.sqrt(h * 8760),  # annualised hourly vol
            })
            h_prev = h

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path):
        """Save fitted model to pickle."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[RS-GARCH] Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'RegimeSwitchingGARCH':
        """Load fitted model from pickle."""
        import pickle
        with open(Path(path), 'rb') as f:
            model = pickle.load(f)
        print(f"[RS-GARCH] Loaded from {path}")
        print(f"  Per-regime params:")
        for r, (omega, alpha, beta) in model.regime_params.items():
            print(f"    Regime {r}: omega={omega:.6f}  "
                  f"alpha={alpha:.4f}  beta={beta:.4f}  "
                  f"persistence={alpha+beta:.4f}")
        return model