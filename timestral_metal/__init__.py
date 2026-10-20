"""Timestral time-series forecasting models on Apple silicon, with MLX."""

from importlib.metadata import version

from timestral_metal.config import TimestralConfig
from timestral_metal.forecaster import Forecast, TimestralForecaster
from timestral_metal.loading import load_model
from timestral_metal.model import TimestralModel

__all__ = ["Forecast", "TimestralConfig", "TimestralForecaster", "TimestralModel", "load_model"]
__version__ = version("timestral-metal")
