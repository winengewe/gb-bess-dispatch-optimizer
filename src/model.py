import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


class PriceForecaster:
    """
    XGBoost-based time-series forecaster for Great Britain wholesale electricity
    prices (Settlement Period level: 48 periods/day).
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initializes the XGBoost regression model with default hyperparameters.
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
        )
        self.is_fitted = False

    def fit(
        self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray]
    ) -> "PriceForecaster":
        """
        Fits the XGBoost model on historical feature data.
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def train(
        self, X_train: pd.DataFrame, y_train: Union[pd.Series, np.ndarray]
    ) -> "PriceForecaster":
        """Alias for fit() for backward compatibility."""
        return self.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates day-ahead price forecasts for the given feature matrix.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted prior to running predictions.")
        return self.model.predict(X)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluates model accuracy on out-of-sample test data using standard regression metrics.

        Returns:
            Dict containing MAE (£/MWh), RMSE (£/MWh), and MAPE (%).
        """
        y_pred = self.predict(X_test)
        y_true = np.array(y_test)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Handle zero division safely for MAPE calculation
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = (
                np.mean(
                    np.abs(
                        (y_true[non_zero_mask] - y_pred[non_zero_mask])
                        / y_true[non_zero_mask]
                    )
                )
                * 100.0
            )
        else:
            mape = 0.0

        return {
            "MAE_GBP": round(float(mae), 2),
            "RMSE_GBP": round(float(rmse), 2),
            "MAPE_PCT": round(float(mape), 2),
        }

    def get_feature_importances(
        self, feature_names: Optional[list] = None
    ) -> pd.Series:
        """
        Returns feature importances ranked in descending order.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to retrieve feature importances.")
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)


if __name__ == "__main__":
    from src.data_ingestion import generate_synthetic_gb_prices
    from src.features import FeatureEngineer, get_feature_columns

    # Quick sanity test
    raw_data = generate_synthetic_gb_prices(periods=288)  # 6 days
    fe = FeatureEngineer()
    df_feat = fe.transform(raw_data)

    cols = get_feature_columns()
    X = df_feat[cols]
    y = df_feat["MarketIndexPrice"]

    forecaster = PriceForecaster(n_estimators=50)
    forecaster.fit(X.iloc[:240], y.iloc[:240])

    metrics = forecaster.evaluate(X.iloc[240:], y.iloc[240:])
    print(f"✅ PriceForecaster sanity check passed! Evaluation: {metrics}")
