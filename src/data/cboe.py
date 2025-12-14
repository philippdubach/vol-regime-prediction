"""
CBOE (Chicago Board Options Exchange) data source.

This module provides web scrapers/crawlers to download data from CBOE:
- VIX Index historical data
- VVIX (VIX of VIX)
- VIX9D, VIX3M, VIX6M
- VIX Futures term structure

These are free, publicly available datasets that CBOE provides.
"""

import io
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.data.base import BaseDataSource, DataFetchError, DataValidationError

logger = logging.getLogger(__name__)


# CBOE data URLs - Volatility Indices
CBOE_INDEX_URLS = {
    'VIX': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
    'VVIX': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv',
    'VIX9D': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv',
    'VIX3M': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv',
    'VIX6M': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv',
    'VIX1Y': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX1Y_History.csv',
    'SKEW': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv',
}

# Put/Call Ratio and Volume data URLs
CBOE_PUTCALL_URLS = {
    'TOTAL_PC': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv',
    'INDEX_PC': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpc.csv',
    'EQUITY_PC': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv',
    'VIX_PC': 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/vixpc.csv',
}

# VIX Futures base URL pattern
VIX_FUTURES_BASE_URL = 'https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/'


class CBOEDataSource(BaseDataSource):
    """
    Data source for CBOE public data.
    
    Provides access to:
    - Volatility indices (VIX, VVIX, VIX9D, etc.)
    - VIX Futures historical data
    
    Example:
        source = CBOEDataSource()
        vix = source.fetch_vix_index(
            start_date=datetime(2006, 1, 1),
            end_date=datetime.now()
        )
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        cache_expiry_days: int = 1,
        request_timeout: int = 30
    ):
        """
        Initialize CBOE data source.
        
        Args:
            cache_dir: Directory for caching data.
            cache_enabled: Whether to cache downloaded data.
            cache_expiry_days: Days before cache expires.
            request_timeout: HTTP request timeout in seconds.
        """
        super().__init__(
            name="cboe",
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            cache_expiry_days=cache_expiry_days
        )
        self.timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_available_series(self) -> List[str]:
        """Get list of available CBOE series."""
        return list(CBOE_INDEX_URLS.keys()) + list(CBOE_PUTCALL_URLS.keys()) + ['VX_FUTURES']
    
    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        series: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch data from CBOE.
        
        Args:
            start_date: Start date for data retrieval.
            end_date: End date for data retrieval.
            series: List of series to fetch. Defaults to ['VIX'].
            
        Returns:
            DataFrame with series as columns and date index.
        """
        if series is None:
            series = ['VIX']
        
        data_frames = []
        
        for series_id in series:
            if series_id == 'VX_FUTURES':
                # Handle futures separately
                df = self._fetch_vix_futures(start_date, end_date)
            elif series_id in CBOE_INDEX_URLS:
                df = self._fetch_index(series_id, start_date, end_date)
            elif series_id in CBOE_PUTCALL_URLS:
                df = self._fetch_putcall(series_id, start_date, end_date)
            else:
                logger.warning(f"Unknown CBOE series: {series_id}")
                continue
            
            if df is not None and not df.empty:
                data_frames.append(df)
        
        if not data_frames:
            raise DataFetchError("No data retrieved from CBOE")
        
        # Combine all series
        combined = pd.concat(data_frames, axis=1)
        combined = combined.sort_index()
        
        # Filter to requested date range
        combined = combined.loc[start_date:end_date]
        
        return combined
    
    def _fetch_index(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch a single volatility index from CBOE.
        
        Args:
            series_id: Index identifier (VIX, VVIX, etc.)
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with the index data.
        """
        url = CBOE_INDEX_URLS.get(series_id)
        if not url:
            raise DataFetchError(f"Unknown series: {series_id}")
        
        logger.info(f"Fetching {series_id} from CBOE")
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(
                io.StringIO(response.text),
                parse_dates=['DATE'],
                index_col='DATE'
            )
            
            # Standardize column names
            df.columns = [f"{series_id}_{col}" for col in df.columns]
            
            # Filter to date range
            df = df.loc[start_date:end_date]
            
            logger.info(
                f"Fetched {series_id}: {len(df)} observations "
                f"({df.index.min()} to {df.index.max()})"
            )
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            raise DataFetchError(f"Failed to fetch {series_id}: {e}")
    
    def _fetch_putcall(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch put/call ratio and volume data from CBOE.
        
        Args:
            series_id: Put/call series identifier (TOTAL_PC, INDEX_PC, etc.)
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with put/call data.
        """
        url = CBOE_PUTCALL_URLS.get(series_id)
        if not url:
            raise DataFetchError(f"Unknown put/call series: {series_id}")
        
        logger.info(f"Fetching {series_id} put/call data from CBOE")
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Skip header rows (varies by file)
            lines = response.text.strip().split('\n')
            
            # Find the header row (contains DATE or Date)
            header_idx = 0
            for i, line in enumerate(lines):
                if 'DATE' in line.upper() and 'RATIO' in line.upper():
                    header_idx = i
                    break
            
            # Parse CSV from header row
            csv_text = '\n'.join(lines[header_idx:])
            df = pd.read_csv(io.StringIO(csv_text))
            
            # Standardize date column
            date_col = [c for c in df.columns if 'date' in c.lower()][0]
            df[date_col] = pd.to_datetime(df[date_col], format='mixed')
            df = df.set_index(date_col)
            df.index.name = 'DATE'
            
            # Standardize column names based on series
            prefix = series_id.replace('_PC', '')
            new_cols = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'ratio' in col_lower or 'p/c' in col_lower:
                    new_cols[col] = f"{prefix}_PC_RATIO"
                elif 'put' in col_lower and 'call' not in col_lower:
                    new_cols[col] = f"{prefix}_PUT_VOL"
                elif 'call' in col_lower and 'put' not in col_lower:
                    new_cols[col] = f"{prefix}_CALL_VOL"
                elif 'total' in col_lower:
                    new_cols[col] = f"{prefix}_TOTAL_VOL"
            
            df = df.rename(columns=new_cols)
            
            # Keep only renamed columns
            df = df[[c for c in df.columns if prefix in c]]
            
            # Filter to date range
            df = df.loc[start_date:end_date]
            
            logger.info(
                f"Fetched {series_id}: {len(df)} observations "
                f"({df.index.min()} to {df.index.max()})"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            raise DataFetchError(f"Failed to fetch {series_id}: {e}")
    
    def _fetch_vix_futures(
        self,
        start_date: datetime,
        end_date: datetime,
        max_workers: int = 5
    ) -> pd.DataFrame:
        """
        Fetch VIX futures historical data.
        
        Downloads individual contract files and constructs term structure.
        
        Args:
            start_date: Start date.
            end_date: End date.
            max_workers: Number of parallel download threads.
            
        Returns:
            DataFrame with futures term structure data.
        """
        logger.info("Fetching VIX futures term structure from CBOE")
        
        # Get list of available contracts
        contract_urls = self._get_futures_contract_urls(start_date, end_date)
        
        if not contract_urls:
            logger.warning("No VIX futures contracts found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(contract_urls)} VIX futures contracts to download")
        
        # Download contracts in parallel
        all_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._download_futures_contract, url): url 
                for url in contract_urls
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading VIX futures"):
                url = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")
        
        if not all_data:
            logger.warning("No VIX futures data downloaded")
            return pd.DataFrame()
        
        # Combine all contract data
        combined = pd.concat(all_data, ignore_index=True)
        
        # Process into term structure format
        term_structure = self._process_futures_to_term_structure(combined, start_date, end_date)
        
        return term_structure
    
    def _get_futures_contract_urls(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[str]:
        """
        Generate URLs for VIX futures contracts within date range.
        
        CBOE uses expiration date in filename: VX_YYYY-MM-DD.csv
        We need to fetch contracts that were active during our period.
        """
        urls = []
        
        # Generate monthly expiration dates (3rd Wednesday of each month, approximately)
        current = start_date.replace(day=1)
        
        while current <= end_date + timedelta(days=365):  # Include contracts expiring up to 1 year after end
            # Find 3rd Wednesday
            first_day = current.replace(day=1)
            # Days until first Wednesday
            days_to_wed = (2 - first_day.weekday()) % 7
            first_wed = first_day + timedelta(days=days_to_wed)
            # Third Wednesday
            third_wed = first_wed + timedelta(days=14)
            
            # Build URL
            expiry_str = third_wed.strftime('%Y-%m-%d')
            url = f"{VIX_FUTURES_BASE_URL}VX_{expiry_str}.csv"
            urls.append(url)
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return urls
    
    def _download_futures_contract(self, url: str) -> Optional[pd.DataFrame]:
        """
        Download a single VIX futures contract file.
        
        Args:
            url: URL to the contract CSV.
            
        Returns:
            DataFrame with contract data, or None if failed.
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 404:
                return None  # Contract doesn't exist
            
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            
            # Extract expiry date from URL
            expiry_match = re.search(r'VX_(\d{4}-\d{2}-\d{2})\.csv', url)
            if expiry_match:
                df['Expiry'] = pd.to_datetime(expiry_match.group(1))
            
            return df
            
        except requests.RequestException:
            return None
    
    def _process_futures_to_term_structure(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Process raw futures data into term structure format.
        
        Creates columns for front month (VX1), second month (VX2), etc.
        
        Args:
            df: Raw futures data with all contracts.
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with term structure columns.
        """
        if df.empty:
            return pd.DataFrame()
        
        # Standardize column names
        df.columns = [col.strip().upper() for col in df.columns]
        
        # Find date column
        date_col = None
        for col in ['TRADE DATE', 'DATE', 'TRADE_DATE']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            logger.warning("Could not find date column in futures data")
            return pd.DataFrame()
        
        df['Date'] = pd.to_datetime(df[date_col])
        
        # Find settle column
        settle_col = None
        for col in ['SETTLE', 'SETTLEMENT', 'CLOSE']:
            if col in df.columns:
                settle_col = col
                break
        
        if settle_col is None:
            logger.warning("Could not find settle column in futures data")
            return pd.DataFrame()
        
        # Convert EXPIRY to datetime if not already
        if 'EXPIRY' in df.columns:
            df['Expiry'] = pd.to_datetime(df['EXPIRY'])
        
        # For each date, rank contracts by expiry and create VX1, VX2, etc.
        term_structure_data = []
        
        for date, group in df.groupby('Date'):
            if date < start_date or date > end_date:
                continue
            
            # Sort by expiry and filter to only future expiries
            group = group[group['Expiry'] > date].sort_values('Expiry')
            
            row = {'Date': date}
            for i, (_, contract) in enumerate(group.iterrows()):
                if i >= 9:  # VX1 to VX9
                    break
                row[f'VX{i+1}'] = contract[settle_col]
                row[f'VX{i+1}_Expiry'] = contract['Expiry']
            
            term_structure_data.append(row)
        
        result = pd.DataFrame(term_structure_data)
        
        if not result.empty:
            result = result.set_index('Date').sort_index()
            
            # Add derived term structure metrics
            if 'VX1' in result.columns and 'VX2' in result.columns:
                result['VX_Slope_1_2'] = result['VX2'] - result['VX1']
            if 'VX1' in result.columns and 'VX4' in result.columns:
                result['VX_Slope_1_4'] = result['VX4'] - result['VX1']
        
        logger.info(f"Processed VIX futures term structure: {len(result)} trading days")
        
        return result
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate CBOE data.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            True if valid.
            
        Raises:
            DataValidationError: If validation fails.
        """
        if df.empty:
            raise DataValidationError("CBOE DataFrame is empty")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("CBOE DataFrame index is not DatetimeIndex")
        
        # Check for reasonable values in VIX columns
        vix_cols = [col for col in df.columns if 'VIX' in col.upper() or col.startswith('VX')]
        for col in vix_cols:
            # Skip expiry columns and slope columns (slopes can be negative in backwardation)
            if col.endswith('_Expiry') or 'Slope' in col:
                continue
            
            values = df[col].dropna()
            if len(values) > 0:
                if values.min() < 0:
                    raise DataValidationError(f"Negative values in {col}")
                if values.max() > 200:  # VIX rarely exceeds 100
                    logger.warning(f"Very high values in {col}: max={values.max()}")
        
        logger.info(f"CBOE data validation passed: {len(df)} rows")
        return True
    
    def fetch_vix_index(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Convenience method to fetch VIX index data.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with VIX data.
        """
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            series=['VIX']
        )
    
    def fetch_all_vix_indices(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch all available VIX-related indices.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with VIX, VVIX, VIX9D, VIX3M, VIX6M.
        """
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            series=['VIX', 'VVIX', 'VIX9D', 'VIX3M', 'VIX6M']
        )
    
    def fetch_vix_futures(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch VIX futures term structure.
        
        Args:
            start_date: Start date.
            end_date: End date.
            
        Returns:
            DataFrame with VX1-VX9 and term structure metrics.
        """
        return self.fetch_with_cache(
            start_date=start_date,
            end_date=end_date,
            series=['VX_FUTURES']
        )


if __name__ == "__main__":
    # Test the CBOE data source
    logging.basicConfig(level=logging.INFO)
    
    source = CBOEDataSource()
    
    # Fetch VIX index
    vix = source.fetch_vix_index(
        start_date=datetime(2020, 1, 1),
        end_date=datetime.now()
    )
    
    print(f"\nVIX Index Data:")
    print(f"Shape: {vix.shape}")
    print(f"Columns: {vix.columns.tolist()}")
    print(f"Date Range: {vix.index.min()} to {vix.index.max()}")
