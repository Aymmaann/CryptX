"""
baseline_garch.py

Standard GARCH(1,1) volatility model — no regime information.
This is the benchmark that regime-switching GARCH must beat.

Fitted on the full hourly return series, then evaluated via
rolling out-of-sample forecasts so comparison with RS-GARCH is fair.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path


class BaselineGARCH:
    """
    Standard GARCH(1,1) model.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Used as the benchmark in the research comparison:
        Baseline GARCH  vs  Regime-Switching GARCH

    The model is evaluated via a rolling window forecast:
      - Train on first `train_frac` of data
      - Forecast 1-step ahead, roll forward, re-fit every `refit_every` bars
      - This gives a fair out-of-sample comparison
    """

    def __init__(self, train_frac=0.8, refit_every=168, max_iter=300):
        """
        Args:
            train_frac:   Fraction of data used for initial training
            refit_every:  Re-fit model every N bars (168 = weekly)
            max_iter:     Max iterations for MLE optimiser
        """
        self.train_frac  = train_frac
        self.refit_every = refit_every
        self.max_iter    = max_iter

        self.params       = None   # (omega, alpha, beta) from last fit
        self.fitted_var   = None   # in-sample conditional variance
        self.forecasts_1h = None   # 1-step ahead OOS forecasts
        self.returns_     = None   # stored for evaluation

        print(f"[BaselineGARCH] Initialized | "
              f"train_frac={train_frac} refit_every={refit_every}h")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray):
        """
        Fit GARCH(1,1) on full return series (in-sample).
        Also runs rolling OOS forecast.
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = np.where(np.isfinite(returns), returns, 0.0)
        self.returns_ = returns

        print(f"[BaselineGARCH] Fitting on {len(returns):,} observations...")
        omega, alpha, beta = self._fit_garch(returns)
        self.params = (omega, alpha, beta)
        print(f"[BaselineGARCH] Fitted: omega={omega:.6f}  "
              f"alpha={alpha:.4f}  beta={beta:.4f}  "
              f"persistence={alpha+beta:.4f}")

        self.fitted_var = self._garch_variance(returns, omega, alpha, beta)

        # Rolling OOS forecast
        print(f"[BaselineGARCH] Running rolling OOS forecast "
              f"(re-fitting every {self.refit_every}h)...")
        self.forecasts_1h = self._rolling_forecast(returns)
        print(f"[BaselineGARCH] OOS forecast complete "
              f"({len(self.forecasts_1h):,} points)")

        return self

    def _rolling_forecast(self, returns: np.ndarray) -> np.ndarray:
        """
        Rolling 1-step-ahead forecast.
        Re-fits every `refit_every` steps for fair OOS evaluation.
        """
        n         = len(returns)
        train_end = int(n * self.train_frac)
        forecasts = np.full(n, np.nan)

        omega, alpha, beta = self._fit_garch(returns[:train_end])
        h_prev = np.var(returns[:train_end])
        r_prev = returns[train_end - 1]

        for t in range(train_end, n):
            # 1-step forecast
            h_forecast        = omega + alpha * r_prev**2 + beta * h_prev
            forecasts[t]      = max(h_forecast, 1e-10)

            # Update for next step using realised return
            h_prev = forecasts[t]
            r_prev = returns[t]

            # Periodically re-fit on expanding window
            if (t - train_end) % self.refit_every == 0 and t > train_end:
                omega, alpha, beta = self._fit_garch(returns[:t])

        return forecasts

    # ------------------------------------------------------------------
    # GARCH mechanics
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

        bounds  = [(1e-8, None), (1e-6, 0.45), (1e-6, 0.9999)]
        starts  = [
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
        """
        Compute evaluation metrics on OOS forecast period.

        Realised variance proxy: r²_t (standard in GARCH literature)
        """
        if self.forecasts_1h is None:
            raise RuntimeError("Call fit() before evaluate()")

        n         = len(self.returns_)
        train_end = int(n * self.train_frac)

        oos_returns   = self.returns_[train_end:]
        oos_forecasts = self.forecasts_1h[train_end:]

        valid = np.isfinite(oos_forecasts) & np.isfinite(oos_returns)
        r     = oos_returns[valid]
        f     = oos_forecasts[valid]
        rv    = r ** 2   # realised variance proxy

        rmse       = float(np.sqrt(np.mean((f - rv)**2)))
        mae        = float(np.mean(np.abs(f - rv)))
        mape       = float(np.mean(np.abs((f - rv) / np.maximum(rv, 1e-10)))) * 100
        qlike      = float(np.mean(np.log(f) + rv / f))   # QLIKE loss (MFE standard)
        corr       = float(np.corrcoef(f, rv)[0, 1]) if len(f) > 1 else 0.0
        n_oos      = int(valid.sum())

        omega, alpha, beta = self.params
        metrics = {
            'model':       'Baseline GARCH(1,1)',
            'n_oos':        n_oos,
            'rmse':         rmse,
            'mae':          mae,
            'mape_pct':     mape,
            'qlike':        qlike,
            'corr':         corr,
            'omega':        omega,
            'alpha':        alpha,
            'beta':         beta,
            'persistence':  alpha + beta,
        }
        return metrics

    def get_forecast_df(self, timestamps) -> pd.DataFrame:
        """Return OOS forecasts as a DataFrame with timestamps."""
        n         = len(self.returns_)
        train_end = int(n * self.train_frac)
        return pd.DataFrame({
            'timestamp':         timestamps[train_end:],
            'baseline_forecast': self.forecasts_1h[train_end:],
            'baseline_vol':      np.sqrt(np.maximum(self.forecasts_1h[train_end:], 0)),
            'realised_var':      self.returns_[train_end:] ** 2,
        })

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
        print(f"[BaselineGARCH] Saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'BaselineGARCH':
        """Load fitted model from pickle."""
        import pickle
        with open(Path(path), 'rb') as f:
            model = pickle.load(f)
        print(f"[BaselineGARCH] Loaded from {path}")
        print(f"  params: omega={model.params[0]:.6f}  "
              f"alpha={model.params[1]:.4f}  beta={model.params[2]:.4f}")
        return model