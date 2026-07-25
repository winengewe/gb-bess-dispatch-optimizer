"""
src/__init__.py

Package marker for GB BESS Dispatch Optimizer.
Exposes core data ingestion routines, OOP feature and model classes,
optimizer engine, plotting routines, and validation framework.
"""

from .data_ingestion import generate_synthetic_gb_prices
from .features import FeatureEngineer, get_feature_columns
from .model import PriceForecaster
from .optimizer import BESSModularOptimizer
from .visualization import plot_bess_dispatch
from .validation import evaluate_forecaster, validate_dispatch_schedule

__all__ = [
    "generate_synthetic_gb_prices",
    "FeatureEngineer",
    "get_feature_columns",
    "PriceForecaster",
    "BESSModularOptimizer",
    "plot_bess_dispatch",
    "evaluate_forecaster",
    "validate_dispatch_schedule",
]
