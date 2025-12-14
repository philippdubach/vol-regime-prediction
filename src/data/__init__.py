"""
Data sources package for volatility regime prediction.

This package provides a modular, extensible architecture for data collection
from multiple sources with easy swapping capability for future premium sources.
"""

from src.data.base import BaseDataSource
from src.data.data_manager import DataManager

__all__ = ['BaseDataSource', 'DataManager']
