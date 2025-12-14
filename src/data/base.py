"""
Abstract base class for all data sources.

This module defines the interface that all data sources must implement,
enabling easy swapping between free and premium data sources.

Design Principles:
1. All sources implement the same interface
2. Sources are responsible for their own validation
3. Caching is handled at the source level
4. Sources return standardized pandas DataFrames
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """
    Abstract base class for all data sources.
    
    All data sources (FRED, CBOE, yfinance, OptionMetrics, etc.) should
    inherit from this class and implement the required methods.
    
    Attributes:
        name: Human-readable name of the data source
        cache_dir: Directory for caching downloaded data
        cache_enabled: Whether to use caching
        cache_expiry_days: Number of days before cache expires
    """
    
    def __init__(
        self,
        name: str,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        cache_expiry_days: int = 1
    ):
        self.name = name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/raw")
        self.cache_enabled = cache_enabled
        self.cache_expiry_days = cache_expiry_days
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.name} data source")
    
    @abstractmethod
    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from the source.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional source-specific parameters
            
        Returns:
            DataFrame with standardized columns and datetime index
        """
        pass
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate the fetched data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if validation passes, raises exception otherwise
        """
        pass
    
    @abstractmethod
    def get_available_series(self) -> List[str]:
        """
        Get list of available data series from this source.
        
        Returns:
            List of series identifiers
        """
        pass
    
    def get_cache_key(self, **kwargs) -> str:
        """
        Generate a unique cache key for the request.
        
        Args:
            **kwargs: Parameters that affect the data (dates, series, etc.)
            
        Returns:
            MD5 hash string as cache key
        """
        params = {
            'source': self.name,
            **{k: str(v) for k, v in kwargs.items()}
        }
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cached item."""
        return self.cache_dir / f"{self.name}_{cache_key}.parquet"
    
    def is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file exists and is not expired."""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        max_age = self.cache_expiry_days * 24 * 60 * 60
        
        return file_age < max_age
    
    def load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and valid.
        
        Args:
            cache_key: Unique identifier for the cached data
            
        Returns:
            DataFrame if cache hit, None otherwise
        """
        if not self.cache_enabled:
            return None
        
        cache_path = self.get_cache_path(cache_key)
        
        if self.is_cache_valid(cache_path):
            logger.info(f"Cache hit for {self.name}: {cache_key[:8]}...")
            return pd.read_parquet(cache_path)
        
        return None
    
    def save_to_cache(self, df: pd.DataFrame, cache_key: str) -> None:
        """
        Save data to cache.
        
        Args:
            df: DataFrame to cache
            cache_key: Unique identifier for the cached data
        """
        if not self.cache_enabled:
            return
        
        cache_path = self.get_cache_path(cache_key)
        df.to_parquet(cache_path)
        logger.info(f"Cached {self.name} data: {cache_key[:8]}...")
    
    def fetch_with_cache(
        self,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data with caching support.
        
        This is the main method that should be called by users.
        It handles cache lookup, fetching, validation, and cache storage.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional source-specific parameters
            
        Returns:
            Validated DataFrame with requested data
        """
        cache_key = self.get_cache_key(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            **kwargs
        )
        
        # Try cache first
        cached_df = self.load_from_cache(cache_key)
        if cached_df is not None:
            return cached_df
        
        # Fetch fresh data
        logger.info(f"Fetching {self.name} data: {start_date} to {end_date}")
        df = self.fetch(start_date, end_date, **kwargs)
        
        # Validate
        self.validate(df)
        
        # Cache and return
        self.save_to_cache(df, cache_key)
        return df
    
    def clear_cache(self) -> int:
        """
        Clear all cached files for this source.
        
        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob(f"{self.name}_*.parquet"):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cached files for {self.name}")
        return count


class DataSourceError(Exception):
    """Base exception for data source errors."""
    pass


class DataFetchError(DataSourceError):
    """Raised when data fetching fails."""
    pass


class DataValidationError(DataSourceError):
    """Raised when data validation fails."""
    pass


class CacheError(DataSourceError):
    """Raised when cache operations fail."""
    pass
