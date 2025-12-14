#!/usr/bin/env python3
"""
Batch Alpha Vantage Data Collection Script

Efficiently collects historical options data from Alpha Vantage while respecting
the 75 requests/minute rate limit. Features:
- Progress tracking with resume capability
- Checkpoint saving (can resume if interrupted)
- Parallel-safe rate limiting
- Comprehensive logging
- Data validation

Usage:
    python src/scripts/batch_alpha_vantage.py --start 2008-01-01 --end 2025-01-01 --freq W-FRI
    python src/scripts/batch_alpha_vantage.py --resume  # Resume from last checkpoint
    python src/scripts/batch_alpha_vantage.py --test    # Test mode (1 month)

Author: Volatility Regime Research
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.data.alpha_vantage import AlphaVantageDataSource, DataFetchError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(project_root / "logs" / "batch_alpha_vantage.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BatchOptionsCollector:
    """
    Batch collector for Alpha Vantage historical options data.
    
    Features:
    - Checkpoint/resume capability
    - Progress tracking
    - Data validation
    - Error handling with retries
    """
    
    def __init__(
        self,
        output_dir: Path,
        checkpoint_file: Path,
        symbol: str = "SPX",
        max_retries: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = Path(checkpoint_file)
        self.symbol = symbol
        self.max_retries = max_retries
        
        # Initialize data source
        self.source = AlphaVantageDataSource(
            cache_dir=self.output_dir / "cache",
            cache_enabled=True,
            cache_expiry_days=365,  # Cache for long time
            max_calls_per_min=65,   # Conservative for batch
        )
        
        # Load checkpoint if exists
        self.checkpoint = self._load_checkpoint()
        
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from file."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint: {len(checkpoint.get('completed_dates', []))} dates completed")
                return checkpoint
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return {"completed_dates": [], "failed_dates": [], "last_updated": None}
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint to file."""
        self.checkpoint["last_updated"] = datetime.now().isoformat()
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def _get_dates_to_process(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "W-FRI"
    ) -> List[datetime]:
        """Generate list of dates to process, excluding completed ones."""
        # Generate all dates based on frequency
        all_dates = pd.date_range(start=start_date, end=end_date, freq=frequency)
        all_dates = [d.to_pydatetime() for d in all_dates]
        
        # Filter out already completed dates
        completed = set(self.checkpoint.get("completed_dates", []))
        dates_to_process = [d for d in all_dates if d.strftime("%Y-%m-%d") not in completed]
        
        logger.info(f"Total dates: {len(all_dates)}, Already completed: {len(completed)}, To process: {len(dates_to_process)}")
        
        return dates_to_process
    
    def collect_options_batch(
        self,
        start_date: datetime,
        end_date: datetime,
        frequency: str = "W-FRI",
        save_interval: int = 10,
    ) -> pd.DataFrame:
        """
        Collect options data for date range.
        
        Args:
            start_date: Start date
            end_date: End date  
            frequency: Pandas frequency string (e.g., "W-FRI" for weekly Friday)
            save_interval: Save checkpoint every N dates
            
        Returns:
            Combined DataFrame of all options data
        """
        dates = self._get_dates_to_process(start_date, end_date, frequency)
        
        if not dates:
            logger.info("No dates to process - all completed")
            return self._load_existing_data()
        
        logger.info(f"Starting batch collection: {len(dates)} dates")
        logger.info(f"Date range: {dates[0].date()} to {dates[-1].date()}")
        
        frames: List[pd.DataFrame] = []
        failed_dates: List[str] = []
        
        # Use tqdm for progress bar
        pbar = tqdm(dates, desc="Collecting options data", unit="date")
        
        for i, date in enumerate(pbar):
            date_str = date.strftime("%Y-%m-%d")
            pbar.set_postfix({"date": date_str, "requests": self.source._request_count})
            
            try:
                df = self._fetch_with_retry(date)
                if df is not None and not df.empty:
                    frames.append(df)
                    self.checkpoint["completed_dates"].append(date_str)
                else:
                    logger.warning(f"No data for {date_str}")
                    failed_dates.append(date_str)
                    
            except Exception as e:
                logger.error(f"Failed to fetch {date_str}: {e}")
                failed_dates.append(date_str)
            
            # Save checkpoint periodically
            if (i + 1) % save_interval == 0:
                self.checkpoint["failed_dates"] = list(set(
                    self.checkpoint.get("failed_dates", []) + failed_dates
                ))
                self._save_checkpoint()
                self._save_intermediate_data(frames)
        
        # Final save
        self.checkpoint["failed_dates"] = list(set(
            self.checkpoint.get("failed_dates", []) + failed_dates
        ))
        self._save_checkpoint()
        
        # Combine all frames
        if frames:
            combined = pd.concat(frames).sort_index()
            combined = combined[~combined.index.duplicated(keep='last')]
            self._save_data(combined)
            return combined
        
        return pd.DataFrame()
    
    def _fetch_with_retry(self, date: datetime) -> Optional[pd.DataFrame]:
        """Fetch options data with retry logic."""
        for attempt in range(self.max_retries):
            try:
                df = self.source.fetch_historical_options(
                    date=date,
                    symbol=self.symbol,
                    raw=False,
                )
                return df
            except DataFetchError as e:
                error_msg = str(e).lower()
                
                # Check for specific error types
                if "no options data" in error_msg or "no data" in error_msg:
                    # This is expected for some dates (weekends, holidays)
                    logger.debug(f"No data available for {date.date()}")
                    return None
                    
                if "rate limit" in error_msg or "exceeded" in error_msg:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {date.date()} in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    raise
                    
        return None
    
    def _save_intermediate_data(self, frames: List[pd.DataFrame]) -> None:
        """Save intermediate data to parquet."""
        if not frames:
            return
        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]
        output_file = self.output_dir / f"{self.symbol}_options_partial.parquet"
        combined.to_parquet(output_file)
        logger.debug(f"Saved intermediate data: {len(combined)} rows")
    
    def _save_data(self, df: pd.DataFrame) -> None:
        """Save final data to parquet and CSV."""
        parquet_file = self.output_dir / f"{self.symbol}_options_history.parquet"
        csv_file = self.output_dir / f"{self.symbol}_options_history.csv"
        
        df.to_parquet(parquet_file)
        df.to_csv(csv_file)
        
        logger.info(f"Saved {len(df)} rows to {parquet_file}")
    
    def _load_existing_data(self) -> pd.DataFrame:
        """Load existing data from parquet."""
        parquet_file = self.output_dir / f"{self.symbol}_options_history.parquet"
        if parquet_file.exists():
            return pd.read_parquet(parquet_file)
        return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate collected data for quality and completeness.
        
        Returns dict with validation results.
        """
        if df.empty:
            return {"valid": False, "error": "Empty DataFrame"}
        
        results = {
            "valid": True,
            "row_count": len(df),
            "date_range": f"{df.index.min()} to {df.index.max()}",
            "columns": list(df.columns),
            "missing_pct": {},
            "statistics": {},
            "warnings": [],
        }
        
        # Check for missing data
        for col in df.columns:
            missing_pct = df[col].isna().mean() * 100
            results["missing_pct"][col] = round(missing_pct, 2)
            if missing_pct > 50:
                results["warnings"].append(f"High missing rate for {col}: {missing_pct:.1f}%")
        
        # Basic statistics for key columns
        key_cols = ["AV_ATM_IV", "AV_PUT_CALL_RATIO_VOL", "AV_IV_SKEW_25D", "AV_TOTAL_VOLUME"]
        for col in key_cols:
            if col in df.columns:
                results["statistics"][col] = {
                    "mean": round(df[col].mean(), 4),
                    "std": round(df[col].std(), 4),
                    "min": round(df[col].min(), 4),
                    "max": round(df[col].max(), 4),
                }
        
        # Sanity checks
        if "AV_ATM_IV" in df.columns:
            # IV should be positive and typically < 2 (200%)
            invalid_iv = (df["AV_ATM_IV"] < 0) | (df["AV_ATM_IV"] > 5)
            if invalid_iv.any():
                results["warnings"].append(f"Invalid IV values: {invalid_iv.sum()} rows")
        
        if "AV_PUT_CALL_RATIO_VOL" in df.columns:
            # Put/Call ratio typically between 0.3 and 3
            pc = df["AV_PUT_CALL_RATIO_VOL"]
            extreme_pc = (pc < 0.1) | (pc > 10)
            if extreme_pc.any():
                results["warnings"].append(f"Extreme put/call ratios: {extreme_pc.sum()} rows")
        
        return results


class BatchIntradayCollector:
    """
    Batch collector for Alpha Vantage intraday data.
    
    Collects 5-min data month by month to compute realized volatility.
    """
    
    def __init__(
        self,
        output_dir: Path,
        symbol: str = "SPY",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        
        self.source = AlphaVantageDataSource(
            cache_dir=self.output_dir / "cache",
            cache_enabled=True,
            cache_expiry_days=365,
            max_calls_per_min=65,
        )
    
    def collect_monthly_rv(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5min",
    ) -> pd.DataFrame:
        """
        Collect intraday data and compute daily realized volatility.
        
        Note: This requires 1 API call per month, which is efficient.
        """
        # Generate month list
        months = pd.date_range(
            start=start_date.replace(day=1),
            end=end_date,
            freq="MS"
        ).strftime("%Y-%m").tolist()
        
        logger.info(f"Collecting intraday RV for {len(months)} months")
        
        all_rv = []
        
        for month in tqdm(months, desc="Collecting intraday data"):
            try:
                df = self.source.fetch_intraday_month(
                    symbol=self.symbol,
                    month=month,
                    interval=interval,
                )
                
                # Compute daily RV
                rv = self.source.compute_realized_vol_from_intraday(df)
                all_rv.append(rv)
                
            except DataFetchError as e:
                logger.warning(f"Failed to fetch {month}: {e}")
                continue
        
        if all_rv:
            combined = pd.concat(all_rv)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            
            # Save
            output_file = self.output_dir / f"{self.symbol}_realized_vol_intraday.parquet"
            combined.to_frame().to_parquet(output_file)
            logger.info(f"Saved RV data: {len(combined)} days")
            
            return combined
        
        return pd.Series(dtype=float)


def estimate_collection_time(start_date: datetime, end_date: datetime, freq: str = "W-FRI") -> Dict[str, Any]:
    """Estimate time required for batch collection."""
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_dates = len(dates)
    
    # At 70 requests/min, time = n_dates / 70 minutes
    est_minutes = n_dates / 70
    est_hours = est_minutes / 60
    
    return {
        "total_dates": n_dates,
        "estimated_requests": n_dates,
        "estimated_minutes": round(est_minutes, 1),
        "estimated_hours": round(est_hours, 2),
        "rate_limit": "75 req/min (using 70 for safety)",
    }


def main():
    parser = argparse.ArgumentParser(description="Batch collect Alpha Vantage options data")
    parser.add_argument("--start", type=str, default="2008-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default=today")
    parser.add_argument("--freq", type=str, default="W-FRI", help="Frequency (e.g., W-FRI, B, D)")
    parser.add_argument("--symbol", type=str, default="SPX", help="Options underlying symbol")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Test mode (1 month only)")
    parser.add_argument("--estimate", action="store_true", help="Just estimate time required")
    parser.add_argument("--validate", action="store_true", help="Validate existing data")
    parser.add_argument("--intraday", action="store_true", help="Collect intraday data for RV")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.now() if args.end is None else datetime.strptime(args.end, "%Y-%m-%d")
    
    # Test mode
    if args.test:
        end_date = start_date + timedelta(days=60)
        logger.info(f"TEST MODE: Using date range {start_date.date()} to {end_date.date()}")
    
    # Estimate mode
    if args.estimate:
        estimate = estimate_collection_time(start_date, end_date, args.freq)
        print("\n=== Collection Time Estimate ===")
        for k, v in estimate.items():
            print(f"  {k}: {v}")
        return
    
    # Set up paths
    output_dir = project_root / "data" / "raw" / "alpha_vantage"
    checkpoint_file = output_dir / "checkpoint.json"
    
    # Intraday mode
    if args.intraday:
        collector = BatchIntradayCollector(output_dir, args.symbol)
        rv = collector.collect_monthly_rv(start_date, end_date)
        print(f"\nCollected RV for {len(rv)} days")
        return
    
    # Options collection
    collector = BatchOptionsCollector(
        output_dir=output_dir,
        checkpoint_file=checkpoint_file,
        symbol=args.symbol,
    )
    
    # Validate mode
    if args.validate:
        df = collector._load_existing_data()
        if df.empty:
            print("No existing data to validate")
            return
        results = collector.validate_data(df)
        print("\n=== Validation Results ===")
        print(json.dumps(results, indent=2))
        return
    
    # Run collection
    print("\n=== Starting Batch Collection ===")
    estimate = estimate_collection_time(start_date, end_date, args.freq)
    print(f"Estimated time: {estimate['estimated_hours']} hours ({estimate['total_dates']} dates)")
    print(f"Symbol: {args.symbol}")
    print(f"Frequency: {args.freq}")
    print()
    
    try:
        df = collector.collect_options_batch(
            start_date=start_date,
            end_date=end_date,
            frequency=args.freq,
        )
        
        # Validate
        results = collector.validate_data(df)
        print("\n=== Collection Complete ===")
        print(f"Rows collected: {len(df)}")
        print(f"Date range: {results['date_range']}")
        
        if results.get("warnings"):
            print("\nWarnings:")
            for w in results["warnings"]:
                print(f"  - {w}")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved to checkpoint.")
        print("Run with --resume to continue.")
    except Exception as e:
        logger.exception(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
