import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import Dict, List, Optional, Tuple


class PriceForecaster:
    """
    XGBoost-based time-series regressor for predicting 30-minute 
    GB wholesale electricity market prices.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.feature_names: Optional[List[str]] = None
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PriceForecaster":
        """
        Trains the XGBoost regressor on engineered time-series features.
        """
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates day-ahead wholesale price forecasts (£/MWh).
        """
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet. Call `.fit()` first.")
        
        # Ensure column order matches training
        X_aligned = X[self.feature_names]
        predictions = self.model.predict(X_aligned)
        
        return np.round(predictions, 2)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates forecast accuracy against actual market settlement prices.
        """
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        
        # Calculate Mean Absolute Percentage Error (handling potential near-zero prices)
        non_zero_mask = y_test != 0
        mape = np.mean(np.abs((y_test[non_zero_mask] - preds[non_zero_mask]) / y_test[non_zero_mask])) * 100

        return {
            "MAE_GBP": round(float(mae), 2),
            "RMSE_GBP": round(float(rmse), 2),
            "MAPE_pct": round(float(mape), 2)
        }

    def get_feature_importance(() -> pd.DataFrame:
        """
        Extracts feature importances to inspect model drivers.
        """
        if self.feature_names is None:
            raise ValueError("Model has not been trained yet.")
            
        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        return df_imp


if __name__ == "__main__":
    from src.data_ingestion import generate_synthetic_gb_prices
    from src.features import FeatureEngineer, get_feature_columns

    # 1. Generate 30 days of data (1,440 settlement periods)
    raw_df = generate_synthetic_gb_prices(periods=1440, random_seed=42)
    
    # 2. Build feature matrix
    fe = FeatureEngineer()
    df_features = fe.transform(raw_df)
    
    feature_cols = get_feature_columns()
    X = df_features[feature_cols]
    y = df_features["MarketIndexPrice"]

    # 3. Time-based train/test split (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. Train and evaluate
    forecaster = PriceForecaster()
    forecaster.fit(X_train, y_train)
    metrics = forecaster.evaluate(X_test, y_test)

    print("✅ Model Training & Evaluation Complete:")
    print(f"Test Set Metrics: {metrics}")
    print("\nTop 5 Feature Importances:")
    print(forecaster.get_feature_importance().head())
