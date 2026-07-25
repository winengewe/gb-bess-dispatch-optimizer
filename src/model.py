import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb


class PriceForecaster:
    """
    XGBoost Time-Series Forecasting Pipeline with Walk-Forward Cross-Validation.
    """

    def __init__(self, params: dict | None = None):
        self.default_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
        self.params = params if params else self.default_params
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fits the XGBoost Regressor model on training data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates point forecasts for input features.
        """
        return self.model.predict(X)

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        """
        Calculates regression metrics (MAE, RMSE, MAPE).
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Avoid division by zero in MAPE
        non_zero_mask = y_true != 0
        mape = (
            np.mean(
                np.abs(
                    (y_true[non_zero_mask] - y_pred[non_zero_mask])
                    / y_true[non_zero_mask]
                )
            )
            * 100
        )

        return {
            "MAE_GBP_MWh": float(np.round(mae, 2)),
            "RMSE_GBP_MWh": float(np.round(rmse, 2)),
            "MAPE_pct": float(np.round(mape, 2)),
        }

    def walk_forward_cv(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        n_splits: int = 5,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """
        Executes Expanding-Window Walk-Forward Cross Validation across time-series splits.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        predictions_list = []

        X = df[feature_cols]
        y = df[target_col]

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model on training fold
            fold_model = xgb.XGBRegressor(**self.params)
            fold_model.fit(X_train, y_train)

            # Predict on unseen test fold
            preds = fold_model.predict(X_test)

            metrics = self.evaluate(y_test, preds)
            metrics["fold"] = fold + 1
            cv_scores.append(metrics)

            fold_preds_df = pd.DataFrame(
                {"Actual": y_test.values, "Predicted": preds}, index=y_test.index
            )
            predictions_list.append(fold_preds_df)

        results_df = pd.concat(predictions_list)

        # Calculate aggregate metrics across all CV folds
        avg_metrics = {
            "Avg_MAE_GBP_MWh": float(
                np.round(np.mean([s["MAE_GBP_MWh"] for s in cv_scores]), 2)
            ),
            "Avg_RMSE_GBP_MWh": float(
                np.round(np.mean([s["RMSE_GBP_MWh"] for s in cv_scores]), 2)
            ),
        }

        return results_df, avg_metrics

    def get_feature_importances(self, feature_names: list[str]) -> pd.DataFrame:
        """
        Extracts gain feature importance.
        """
        importance = self.model.feature_importances_
        df_imp = (
            pd.DataFrame({"Feature": feature_names, "Importance": importance})
            .sort_values(by="Importance", ascending=False)
            .reset_index(drop=True)
        )
        return df_imp
