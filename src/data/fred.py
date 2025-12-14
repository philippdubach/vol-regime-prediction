"""
FRED (Federal Reserve Economic Data) data source.

This module provides access to economic and financial data from FRED,
including VIX, interest rates, and macroeconomic indicators.

API Documentation: https://fred.stlouisfed.org/docs/api/fred/
"""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import logging
from dotenv import load_dotenv

from src.data.base import BaseDataSource, DataFetchError, DataValidationError

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# Available FRED series relevant to volatility research
FRED_SERIES = {
    # Volatility Indices (from CBOE via FRED)
    'VIXCLS': 'CBOE Volatility Index: VIX',
    'VXNCLS': 'CBOE NASDAQ 100 Volatility Index',
    'RVXCLS': 'CBOE Russell 2000 Volatility Index',
    'VXDCLS': 'CBOE DJIA Volatility Index',
    'OVXCLS': 'CBOE Crude Oil ETF Volatility Index',
    'GVZCLS': 'CBOE Gold ETF Volatility Index',
    
    # Interest Rates
    'DFF': 'Federal Funds Effective Rate',
    'DGS1': '1-Year Treasury Constant Maturity Rate',
    'DGS2': '2-Year Treasury Constant Maturity Rate',
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'DGS30': '30-Year Treasury Constant Maturity Rate',
    'T10Y2Y': '10-Year Treasury Minus 2-Year Treasury',
    'T10Y3M': '10-Year Treasury Minus 3-Month Treasury',
    
    # Credit Spreads
    'BAMLH0A0HYM2': 'ICE BofA US High Yield Index Option-Adjusted Spread',
    'BAMLC0A0CM': 'ICE BofA US Corporate Index Option-Adjusted Spread',
    'TEDRATE': 'TED Spread (3-Month LIBOR minus T-Bill)',
    
    # Economic Indicators
    'UMCSENT': 'University of Michigan Consumer Sentiment',
    'UNRATE': 'Unemployment Rate',
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
    
    # Financial Conditions & Stress
    'NFCI': 'Chicago Fed National Financial Conditions Index',
    'STLFSI4': 'St. Louis Fed Financial Stress Index',
    
    # Economic Policy Uncertainty
    'USEPUINDXD': 'Economic Policy Uncertainty Index for United States',
}


class FREDDataSource(BaseDataSource):
    """
    Data source for FRED (Federal Reserve Economic Data).
    
    Uses the fredapi library to fetch data from the FRED API.
    Requires FRED_API_KEY environment variable.
    
    Example:
        source = FREDDataSource()
        df = source.fetch_with_cache(
            start_date=datetime(2006, 1, 1),
            end_date=datetime.now(),
            series=['VIXCLS', 'DFF']
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        cache_expiry_days: int = 1
    ):
        """
        Initialize FRED data source.
        
        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
            cache_dir: Directory for caching data.
            cache_enabled: Whether to cache downloaded data.
            cache_expiry_days: Days before cache expires.
        """
        super().__init__(
            name="fred",
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            cache_expiry_days=cache_expiry_days
        )
        
        # Get API key
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise DataFetchError(
                "FRED API key not found. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize FRED client
        try:
            from fredapi import Fred
            self.fred = Fred(api_key=self.api_key)
            logger.info("FRED API client initialized successfully")
        except ImportError:
            raise DataFetchError(
                "fredapi package not installed. Run: pip install fredapi"
            )
    
    def get_available_series(self) -> List[str]:
        """Get list of available FRED series."""
        return list(FRED_SERIES.keys())
    
    def get_series_info(self) -> Dict[str, str]:
        """Get dictionary of series IDs and descriptions."""
        return FRED_SERIES.copy()
    
    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        series: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from FRED API.
        
        Args:
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval.
            series: List of FRED series IDs. If None, fetches VIXCLS only.
            
        Returns:
            DataFrame with series as columns and date index.
        """
        if series is None:
            series = ['VIXCLS']
        
        # Validate series
        invalid = set(series) - set(FRED_SERIES.keys())
        if invalid:
            logger.warning(f"Unknown FRED series (will attempt anyway): {invalid}")
        
        data_frames = []
        
        for series_id in series:
            try:
                logger.debug(f"Fetching FRED series: {series_id}")
                
                # Fetch the series
                series_data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                
                # Convert to DataFrame
                df = pd.DataFrame({series_id: series_data})
                data_frames.append(df)
                
                logger.info(
                    f"Fetched {series_id}: {len(df)} observations "
                    f"({df.index.min()} to {df.index.max()})"
                )
                
            except Exception as e:
                logger.error(f"Failed to fetch {series_id}: {e}")
                raise DataFetchError(f"Failed to fetch {series_id}: {e}")
        
        # Combine all series
        if not data_frames:
            raise DataFetchError("No data retrieved from FRED")
        
        combined = pd.concat(data_frames, axis=1)
        combined.index = pd.to_datetime(combined.index)
        combined.index.name = 'date'
        
        # Sort by date
        combined = combined.sort_index()
        
        return combined
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate FRED data.
        
        Checks:
        - DataFrame is not empty
        - Index is datetime
        - No completely empty columns
        - Values are numeric
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            True if valid.
            
        Raises:
            DataValidationError: If validation fails.
        """
        if df.empty:
            raise DataValidationError("FRED DataFrame is empty")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("FRED DataFrame index is not DatetimeIndex")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            raise DataValidationError(f"Empty columns in FRED data: {empty_cols}")
        
        # Check that values are numeric
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col].dropna()):
                raise DataValidationError(f"Non-numeric data in column: {col}")
        
        logger.info(f"FRED data validation passed: {len(df)} rows, {len(df.columns)} columns")
        return True
    
    def fetch_vix(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Convenience method to fetch VIX data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with VIX data.
        """
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            series=['VIXCLS']
        )
    
    def fetch_interest_rates(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch interest rate data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with interest rate series.
        """
        rate_series = ['DFF', 'DGS1', 'DGS2', 'DGS10', 'DGS30', 'T10Y2Y', 'T10Y3M']
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            series=rate_series
        )
    
    def fetch_credit_spreads(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch credit spread data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with credit spread series.
        """
        spread_series = ['BAMLH0A0HYM2', 'BAMLC0A0CM']
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            series=spread_series
        )
    
    def fetch_financial_conditions(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch financial conditions indices.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with financial conditions indices.
        """
        fc_series = ['NFCI', 'STLFSI4']
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            series=fc_series
        )


if __name__ == "__main__":
    # Test the FRED data source
    logging.basicConfig(level=logging.INFO)
    
    source = FREDDataSource()
    
    # Fetch VIX data
    df = source.fetch_vix(
        start_date=datetime(2006, 1, 1),
        end_date=datetime.now()
    )
    
    print(f"\nVIX Data Summary:")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"\nStatistics:\n{df.describe()}")
