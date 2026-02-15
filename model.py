"""
Dynamic Pricing Lab — Demand Model
Gradient Boosting (mean + quantile p10/p90).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class DemandModel:
    """Three GBR models: mean, quantile-10, quantile-90."""

    def __init__(self):
        common = dict(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=30,
            subsample=0.8,
            max_features=0.8,
        )
        self.model_mean = GradientBoostingRegressor(
            loss="squared_error", **common
        )
        self.model_p10 = GradientBoostingRegressor(
            loss="quantile", alpha=0.10, **common
        )
        self.model_p90 = GradientBoostingRegressor(
            loss="quantile", alpha=0.90, **common
        )
        self.is_fitted = False
        self.mae_holdout: float = None
        self.residual_std: float = 1.0
        self._feature_names = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.15):
        """Train all three models and compute holdout MAE."""
        self._feature_names = list(X.columns)
        X_arr = X.values.astype(np.float64)
        y_arr = y.astype(np.float64)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_arr, y_arr, test_size=test_size, random_state=42
        )
        self.model_mean.fit(X_tr, y_tr)
        self.model_p10.fit(X_tr, y_tr)
        self.model_p90.fit(X_tr, y_tr)

        y_pred = self.model_mean.predict(X_te)
        self.mae_holdout = mean_absolute_error(y_te, y_pred)
        residuals = y_te - y_pred
        self.residual_std = max(float(np.std(residuals)), 0.5)
        self.is_fitted = True
        return self

    def _prepare_X(self, X):
        """Ensure X is a 2D numpy array with only the trained feature columns."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        if isinstance(X, pd.DataFrame):
            if self._feature_names is not None:
                # select only trained columns, fill missing with 0
                for col in self._feature_names:
                    if col not in X.columns:
                        X[col] = 0.0
                X = X[self._feature_names]
            return X.values.astype(np.float64)
        return np.asarray(X, dtype=np.float64)

    def predict(self, X) -> tuple:
        """Return (mean, p10, p90) predictions."""
        X_arr = self._prepare_X(X)
        mean = self.model_mean.predict(X_arr)
        p10 = self.model_p10.predict(X_arr)
        p90 = self.model_p90.predict(X_arr)
        # ensure ordering p10 <= mean <= p90
        p10 = np.minimum(p10, mean)
        p90 = np.maximum(p90, mean)
        return np.maximum(mean, 0), np.maximum(p10, 0), np.maximum(p90, 0)

    def predict_single(self, row: dict) -> tuple:
        """Convenience: predict for one observation dict."""
        m, lo, hi = self.predict(row)
        return float(m[0]), float(lo[0]), float(hi[0])
