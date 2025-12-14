"""
Data Manager - Orchestrates all data sources and provides unified interface.

This module is the main entry point for data collection. It:
- Initializes all data sources
- Handles fallbacks between sources
- Merges data into unified datasets
- Manages caching at the dataset level
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import yaml
import logging
from dotenv import load_dotenv

from src.data.base import BaseDataSource, DataFetchError
from src.data.fred import FREDDataSource
from src.data.cboe import CBOEDataSource
from src.data.yfinance_source import YFinanceDataSource
from src.data.alpha_vantage import AlphaVantageDataSource

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates data collection from multiple sources.
    
    This class provides a unified interface for collecting all data
    needed for the volatility regime prediction project.
    
    Example:
        dm = DataManager()
        dataset = dm.collect_all()
        
        # Or collect specific data
        vix = dm.get_vix_data()
        futures = dm.get_vix_futures()
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize DataManager.
        
        Args:
            config_path: Path to config.yaml. Defaults to config/config.yaml.
            cache_dir: Base directory for caching. Defaults to data/raw.
        """
        # Load configuration
        if config_path is None:
            config_path = Path("config/config.yaml")
        
        self.config = self._load_config(config_path)
        
        # Set up directories
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.interim_dir = Path("data/interim")
        
        for dir_path in [self.cache_dir, self.processed_dir, self.interim_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data sources
        self._init_sources()
        
        # Date range from config
        self.start_date = datetime.strptime(
            self.config['data']['start_date'], '%Y-%m-%d'
        )
        self.end_date = datetime.now() if self.config['data']['end_date'] is None else \
                        datetime.strptime(self.config['data']['end_date'], '%Y-%m-%d')
        
        logger.info(f"DataManager initialized: {self.start_date} to {self.end_date}")
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {
                'data': {
                    'start_date': '2006-01-01',
                    'end_date': None,
                    'cache': {
                        'enabled': True,
                        'expiry_days': 1
                    }
                }
            }
    
    def _init_sources(self) -> None:
        """Initialize all data sources."""
        cache_config = self.config.get('data', {}).get('cache', {})
        cache_enabled = cache_config.get('enabled', True)
        cache_expiry = cache_config.get('expiry_days', 1)
        
        # FRED source
        try:
            self.fred = FREDDataSource(
                cache_dir=self.cache_dir / 'fred',
                cache_enabled=cache_enabled,
                cache_expiry_days=cache_expiry
            )
            logger.info("FRED data source initialized")
        except Exception as e:
            logger.warning(f"Could not initialize FRED source: {e}")
            self.fred = None
        
        # CBOE source
        try:
            self.cboe = CBOEDataSource(
                cache_dir=self.cache_dir / 'cboe',
                cache_enabled=cache_enabled,
                cache_expiry_days=cache_expiry
            )
            logger.info("CBOE data source initialized")
        except Exception as e:
            logger.warning(f"Could not initialize CBOE source: {e}")
            self.cboe = None
        
        # Yahoo Finance source
        try:
            self.yfinance = YFinanceDataSource(
                cache_dir=self.cache_dir / 'yfinance',
                cache_enabled=cache_enabled,
                cache_expiry_days=cache_expiry
            )
            logger.info("Yahoo Finance data source initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Yahoo Finance source: {e}")
            self.yfinance = None

        # Alpha Vantage (premium) options/intraday source
        alpha_cfg = self.config.get('data', {}).get('sources', {}).get('alpha_vantage', {})
        alpha_enabled = alpha_cfg.get('enabled', False)
        if alpha_enabled:
            try:
                self.alpha_vantage = AlphaVantageDataSource(
                    cache_dir=self.cache_dir / 'alpha_vantage',
                    cache_enabled=cache_enabled,
                    cache_expiry_days=cache_expiry,
                    api_key=alpha_cfg.get('api_key'),
                )
                logger.info("Alpha Vantage data source initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Alpha Vantage source: {e}")
                self.alpha_vantage = None
        else:
            self.alpha_vantage = None
    
    def get_vix_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'cboe'
    ) -> pd.DataFrame:
        """
        Get VIX index data.
        
        Args:
            start_date: Start date (defaults to config).
            end_date: End date (defaults to config).
            source: 'cboe' or 'fred'.
            
        Returns:
            DataFrame with VIX data.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        if source == 'cboe' and self.cboe:
            try:
                return self.cboe.fetch_vix_index(start, end)
            except DataFetchError as e:
                logger.warning(f"CBOE fetch failed, trying FRED: {e}")
        
        if self.fred:
            return self.fred.fetch_vix(start, end)
        
        raise DataFetchError("No available source for VIX data")
    
    def get_vix_indices(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_skew: bool = True
    ) -> pd.DataFrame:
        """
        Get all VIX-related indices (VIX, VVIX, VIX9D, SKEW, etc.).
        
        Args:
            start_date: Start date.
            end_date: End date.
            include_skew: Whether to include SKEW index.
            
        Returns:
            DataFrame with all VIX indices.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        if not self.cboe:
            raise DataFetchError("CBOE source not available")
        
        # Fetch VIX indices
        series = ['VIX', 'VVIX', 'VIX9D', 'VIX3M', 'VIX6M']
        if include_skew:
            series.append('SKEW')
        
        return self.cboe.fetch_with_cache(start, end, series=series)
    
    def get_putcall_ratios(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get put/call ratio and volume data from CBOE.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with total, index, equity, and VIX put/call data.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        if not self.cboe:
            raise DataFetchError("CBOE source not available")
        
        return self.cboe.fetch_with_cache(
            start, end,
            series=['TOTAL_PC', 'INDEX_PC', 'EQUITY_PC', 'VIX_PC']
        )
    
    def get_vix_futures(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get VIX futures term structure data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with VX1-VX9 and term structure metrics.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        if not self.cboe:
            raise DataFetchError("CBOE source not available")
        
        return self.cboe.fetch_vix_futures(start, end)

    def get_options_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        frequency: str = "W-FRI",
        symbol: str = "SPY",  # SPY has IV/Greeks; SPX does not in API
        require_greeks: bool = True,
        use_cached: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch summarized historical options data (Alpha Vantage premium).

        Note: Use SPY (S&P 500 ETF) not SPX - the API returns IV/Greeks
        only for equity options, not index options.

        Args:
            start_date: Start date.
            end_date: End date.
            frequency: Pandas offset alias to subsample requests (default weekly Friday).
            symbol: Underlying symbol (SPY recommended for IV/Greeks).
            require_greeks: Request greeks/IV fields.
            use_cached: If True, try to load from cached parquet file first.
            
        Returns:
            DataFrame with 26 options features per date (IV, skew, term structure, Greeks).
        """
        if not self.alpha_vantage:
            raise DataFetchError("Alpha Vantage source not available or disabled")

        # Try loading from batch-collected data first
        if use_cached:
            cache_file = self.cache_dir / 'alpha_vantage' / f'{symbol}_options_history.parquet'
            if cache_file.exists():
                logger.info(f"Loading cached options data from {cache_file}")
                df = pd.read_parquet(cache_file)
                # Filter to date range
                start = start_date or self.start_date
                end = end_date or self.end_date
                df = df.loc[start:end]
                return df

        start = (start_date or self.start_date).date()
        end = (end_date or self.end_date).date()
        dates = pd.date_range(start=start, end=end, freq=frequency)

        logger.info(
            f"Fetching Alpha Vantage options for {symbol}: {len(dates)} trading snapshots"
        )
        frames = []
        for dt in dates:
            try:
                frame = self.alpha_vantage.fetch_historical_options(
                    date=dt.to_pydatetime(),
                    symbol=symbol,
                    require_greeks=require_greeks,
                )
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Options fetch failed for {dt.date()}: {e}")

        if not frames:
            raise DataFetchError("No options data retrieved from Alpha Vantage")

        combined = pd.concat(frames).sort_index()
        return combined
    
    def get_spx_prices(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get S&P 500 price data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with S&P 500 OHLCV data.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        if not self.yfinance:
            raise DataFetchError("Yahoo Finance source not available")
        
        return self.yfinance.fetch_spx(start, end)
    
    def get_economic_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get economic indicators from FRED.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with interest rates, spreads, and conditions.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        if not self.fred:
            raise DataFetchError("FRED source not available")
        
        # Fetch different categories
        dfs = []
        
        try:
            rates = self.fred.fetch_interest_rates(start, end)
            dfs.append(rates)
        except Exception as e:
            logger.warning(f"Failed to fetch interest rates: {e}")
        
        try:
            spreads = self.fred.fetch_credit_spreads(start, end)
            dfs.append(spreads)
        except Exception as e:
            logger.warning(f"Failed to fetch credit spreads: {e}")
        
        try:
            conditions = self.fred.fetch_financial_conditions(start, end)
            dfs.append(conditions)
        except Exception as e:
            logger.warning(f"Failed to fetch financial conditions: {e}")
        
        if not dfs:
            raise DataFetchError("No economic data retrieved")
        
        return pd.concat(dfs, axis=1)
    
    def collect_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_futures: bool = True,
        include_economic: bool = True,
        include_putcall: bool = True,
        include_options: bool = False
    ) -> pd.DataFrame:
        """
        Collect all available data and merge into unified dataset.
        
        Args:
            start_date: Start date.
            end_date: End date.
            include_futures: Whether to include VIX futures (slower).
            include_economic: Whether to include economic indicators.
            include_putcall: Whether to include put/call ratios.
            
        Returns:
            Merged DataFrame with all data.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        
        logger.info(f"Collecting all data: {start} to {end}")
        
        datasets = []
        
        # VIX indices (now includes SKEW)
        logger.info("Fetching VIX indices and SKEW...")
        try:
            vix_data = self.get_vix_indices(start, end, include_skew=True)
            datasets.append(('vix_indices', vix_data))
        except Exception as e:
            logger.error(f"Failed to fetch VIX indices: {e}")
        
        # Put/Call ratios
        if include_putcall:
            logger.info("Fetching put/call ratios...")
            try:
                putcall_data = self.get_putcall_ratios(start, end)
                datasets.append(('putcall', putcall_data))
            except Exception as e:
                logger.error(f"Failed to fetch put/call ratios: {e}")
        
        # VIX futures
        if include_futures:
            logger.info("Fetching VIX futures (this may take a while)...")
            try:
                futures_data = self.get_vix_futures(start, end)
                datasets.append(('vix_futures', futures_data))
            except Exception as e:
                logger.error(f"Failed to fetch VIX futures: {e}")
        
        # S&P 500 prices
        logger.info("Fetching S&P 500 prices...")
        try:
            spx_data = self.get_spx_prices(start, end)
            datasets.append(('spx', spx_data))
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500: {e}")

        # Alpha Vantage options (premium)
        if include_options:
            opts_cfg = self.config.get('data', {}).get('sources', {}).get('alpha_vantage', {})
            freq = opts_cfg.get('frequency', 'W-FRI')
            symbol = opts_cfg.get('options_symbol', 'SPX')
            require_greeks = opts_cfg.get('require_greeks', True)

            logger.info(f"Fetching Alpha Vantage options for {symbol} (freq={freq})...")
            try:
                options_data = self.get_options_data(
                    start, end,
                    frequency=freq,
                    symbol=symbol,
                    require_greeks=require_greeks
                )
                datasets.append(('options', options_data))
            except Exception as e:
                logger.error(f"Failed to fetch Alpha Vantage options: {e}")
        
        # Economic data
        if include_economic:
            logger.info("Fetching economic data from FRED...")
            try:
                econ_data = self.get_economic_data(start, end)
                datasets.append(('economic', econ_data))
            except Exception as e:
                logger.error(f"Failed to fetch economic data: {e}")
        
        if not datasets:
            raise DataFetchError("No data collected from any source")
        
        # Merge all datasets
        logger.info("Merging datasets...")
        merged = self._merge_datasets(datasets)
        
        # Filter to trading days only (where we have VIX data)
        if 'VIX_CLOSE' in merged.columns:
            trading_days_mask = merged['VIX_CLOSE'].notna()
            merged = merged[trading_days_mask]
            logger.info(f"Filtered to {len(merged)} trading days")
        
        # Save to interim storage
        interim_path = self.interim_dir / 'raw_merged.parquet'
        merged.to_parquet(interim_path)
        logger.info(f"Saved raw merged data to {interim_path}")
        
        return merged
    
    def _merge_datasets(
        self,
        datasets: List[tuple]
    ) -> pd.DataFrame:
        """
        Merge multiple datasets on date index.
        
        Args:
            datasets: List of (name, DataFrame) tuples.
            
        Returns:
            Merged DataFrame.
        """
        if not datasets:
            return pd.DataFrame()
        
        # Start with first dataset
        name, merged = datasets[0]
        logger.info(f"Starting merge with {name}: {merged.shape}")
        
        # Merge remaining datasets
        for name, df in datasets[1:]:
            logger.info(f"Merging {name}: {df.shape}")
            
            # Align on index
            merged = merged.join(df, how='outer')
        
        # Sort by date
        merged = merged.sort_index()
        
        logger.info(f"Final merged shape: {merged.shape}")
        
        return merged
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for a dataset.
        
        Args:
            df: DataFrame to summarize.
            
        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            'shape': df.shape,
            'date_range': {
                'start': df.index.min().isoformat() if len(df) > 0 else None,
                'end': df.index.max().isoformat() if len(df) > 0 else None,
                'trading_days': len(df)
            },
            'columns': {
                'total': len(df.columns),
                'list': df.columns.tolist()
            },
            'missing_data': {
                col: {
                    'count': int(df[col].isna().sum()),
                    'percent': float(df[col].isna().mean() * 100)
                }
                for col in df.columns
            },
            'statistics': df.describe().to_dict()
        }
        
        return summary
    
    def clear_all_caches(self) -> None:
        """Clear caches for all data sources."""
        if self.fred:
            self.fred.clear_cache()
        if self.cboe:
            self.cboe.clear_cache()
        if self.yfinance:
            self.yfinance.clear_cache()
        if getattr(self, 'alpha_vantage', None):
            self.alpha_vantage.clear_cache()
        
        logger.info("Cleared all caches")


if __name__ == "__main__":
    # Test the DataManager
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    dm = DataManager()
    
    # Collect a small sample
    print("\nCollecting data (small sample for testing)...")
    df = dm.collect_all(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        include_futures=False  # Faster for testing
    )
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample:\n{df.head()}")
