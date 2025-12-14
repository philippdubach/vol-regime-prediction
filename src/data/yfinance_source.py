"""
Yahoo Finance data source via yfinance.

This module provides access to stock prices and basic options data
from Yahoo Finance. Used primarily for:
- S&P 500 index prices (for realized volatility calculation)
- SPY ETF data
- Current options chains (for validation)
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import logging

from src.data.base import BaseDataSource, DataFetchError, DataValidationError

logger = logging.getLogger(__name__)


# Available tickers for volatility research
AVAILABLE_TICKERS = {
    '^GSPC': 'S&P 500 Index',
    '^SPX': 'S&P 500 Index (alt)',
    'SPY': 'SPDR S&P 500 ETF',
    '^VIX': 'CBOE Volatility Index',
    'VXX': 'iPath S&P 500 VIX Short-Term Futures ETN',
    'UVXY': 'ProShares Ultra VIX Short-Term Futures ETF',
    'SVXY': 'ProShares Short VIX Short-Term Futures ETF',
}


class YFinanceDataSource(BaseDataSource):
    """
    Data source for Yahoo Finance via yfinance library.
    
    Provides:
    - Historical price data for indices and ETFs
    - Current options chains
    
    Example:
        source = YFinanceDataSource()
        df = source.fetch_with_cache(
            start_date=datetime(2006, 1, 1),
            end_date=datetime.now(),
            tickers=['^GSPC', 'SPY']
        )
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        cache_expiry_days: int = 1
    ):
        """
        Initialize Yahoo Finance data source.
        
        Args:
            cache_dir: Directory for caching data.
            cache_enabled: Whether to cache downloaded data.
            cache_expiry_days: Days before cache expires.
        """
        super().__init__(
            name="yfinance",
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            cache_expiry_days=cache_expiry_days
        )
        
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("yfinance initialized successfully")
        except ImportError:
            raise DataFetchError(
                "yfinance package not installed. Run: pip install yfinance"
            )
    
    def get_available_series(self) -> List[str]:
        """Get list of commonly used tickers."""
        return list(AVAILABLE_TICKERS.keys())
    
    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch price data from Yahoo Finance.
        
        Args:
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval.
            tickers: List of ticker symbols. Defaults to ['^GSPC'].
            
        Returns:
            DataFrame with OHLCV data for each ticker.
        """
        if tickers is None:
            tickers = ['^GSPC']
        
        logger.info(f"Fetching data for {tickers} from Yahoo Finance")
        
        try:
            # Download data
            df = self.yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,  # Adjust for splits/dividends
                threads=True
            )
            
            if df.empty:
                raise DataFetchError(f"No data returned for {tickers}")
            
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten MultiIndex: (Price, Ticker) -> Ticker_Price
                new_cols = []
                for col in df.columns:
                    price_type = col[0]  # e.g., 'Close', 'Open'
                    ticker = str(col[1]).replace('^', '')  # e.g., 'GSPC'
                    new_cols.append(f"{ticker}_{price_type}")
                df.columns = new_cols
            elif len(tickers) == 1:
                # Single ticker without MultiIndex
                ticker = tickers[0].replace('^', '')
                df.columns = [f"{ticker}_{col}" for col in df.columns]
            
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
            
            logger.info(
                f"Fetched {len(tickers)} tickers: {len(df)} observations "
                f"({df.index.min()} to {df.index.max()})"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch from Yahoo Finance: {e}")
            raise DataFetchError(f"Failed to fetch from Yahoo Finance: {e}")
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate Yahoo Finance data.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            True if valid.
            
        Raises:
            DataValidationError: If validation fails.
        """
        if df.empty:
            raise DataValidationError("Yahoo Finance DataFrame is empty")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("Index is not DatetimeIndex")
        
        # Check for price columns
        price_cols = [col for col in df.columns if 'Close' in col or 'Adj' in col]
        if not price_cols:
            raise DataValidationError("No price columns found")
        
        # Check for negative prices
        for col in price_cols:
            if (df[col].dropna() < 0).any():
                raise DataValidationError(f"Negative prices in {col}")
        
        # Check for reasonable data coverage
        date_range = (df.index.max() - df.index.min()).days
        expected_trading_days = date_range * 252 / 365
        actual_days = len(df)
        coverage = actual_days / expected_trading_days if expected_trading_days > 0 else 0
        
        if coverage < 0.5:
            logger.warning(f"Low data coverage: {coverage:.1%}")
        
        logger.info(f"Yahoo Finance data validation passed: {len(df)} rows")
        return True
    
    def fetch_spx(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch S&P 500 index data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with S&P 500 OHLCV data.
        """
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            tickers=['^GSPC']
        )
    
    def fetch_spy(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch SPY ETF data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with SPY OHLCV data.
        """
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            tickers=['SPY']
        )
    
    def fetch_volatility_etfs(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch volatility-related ETFs.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with VXX, UVXY, SVXY data.
        """
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            tickers=['VXX', 'UVXY', 'SVXY']
        )
    
    def get_current_options_chain(
        self,
        ticker: str = '^SPX'
    ) -> dict:
        """
        Get current options chain for a ticker.
        
        Note: yfinance only provides current chains, not historical.
        For historical options data, use OptionMetrics via WRDS.
        
        Args:
            ticker: Ticker symbol.
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames.
        """
        try:
            tk = self.yf.Ticker(ticker)
            
            # Get available expiration dates
            expirations = tk.options
            
            if not expirations:
                logger.warning(f"No options available for {ticker}")
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            
            # Get the first expiration
            chain = tk.option_chain(expirations[0])
            
            return {
                'calls': chain.calls,
                'puts': chain.puts,
                'expiration': expirations[0],
                'all_expirations': expirations
            }
            
        except Exception as e:
            logger.error(f"Failed to get options chain for {ticker}: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def compute_returns(
        self,
        df: pd.DataFrame,
        price_col: str = 'GSPC_Close'
    ) -> pd.DataFrame:
        """
        Compute returns from price data.
        
        Args:
            df: DataFrame with price data.
            price_col: Column name for close prices.
            
        Returns:
            DataFrame with additional return columns.
        """
        result = df.copy()
        
        # Simple returns
        result[f'{price_col}_Return'] = result[price_col].pct_change()
        
        # Log returns
        result[f'{price_col}_LogReturn'] = np.log(result[price_col]).diff()
        
        return result


if __name__ == "__main__":
    # Test the Yahoo Finance data source
    logging.basicConfig(level=logging.INFO)
    
    source = YFinanceDataSource()
    
    # Fetch S&P 500 data
    spx = source.fetch_spx(
        start_date=datetime(2006, 1, 1),
        end_date=datetime.now()
    )
    
    print(f"\nS&P 500 Data:")
    print(f"Shape: {spx.shape}")
    print(f"Columns: {spx.columns.tolist()}")
    print(f"Date Range: {spx.index.min()} to {spx.index.max()}")
    print(f"\nLast 5 rows:\n{spx.tail()}")
