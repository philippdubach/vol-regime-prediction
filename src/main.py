"""
Main script for data collection and feature engineering.

Run this script to collect all data and generate the analysis dataset.
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_manager import DataManager
from src.features.volatility import VolatilityFeatures


def setup_logging():
    """Configure logging for the pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'data_collection.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main pipeline for data collection and feature engineering."""
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting Volatility Regime Prediction Data Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize data manager
        logger.info("Initializing data manager...")
        dm = DataManager()
        
        # Collect all raw data
        logger.info("Collecting raw data from all sources...")
        raw_data = dm.collect_all(
            include_futures=True,
            include_economic=True
        )
        
        logger.info(f"Raw data shape: {raw_data.shape}")
        logger.info(f"Date range: {raw_data.index.min()} to {raw_data.index.max()}")
        
        # Feature engineering
        logger.info("Computing features...")
        vf = VolatilityFeatures()
        
        # Find the right column names
        price_col = None
        for col in ['GSPC_Close', 'GSPC_Adj Close', 'SPY_Close']:
            if col in raw_data.columns:
                price_col = col
                break
        
        vix_col = None
        for col in ['VIX_CLOSE', 'VIXCLS', 'VIX']:
            if col in raw_data.columns:
                vix_col = col
                break
        
        if price_col and vix_col:
            # Get high/low columns
            high_col = price_col.replace('Close', 'High').replace('Adj Close', 'High')
            low_col = price_col.replace('Close', 'Low').replace('Adj Close', 'Low')
            
            features = vf.compute_all(
                raw_data,
                price_col=price_col,
                high_col=high_col if high_col in raw_data.columns else price_col,
                low_col=low_col if low_col in raw_data.columns else price_col,
                vix_col=vix_col
            )
        else:
            logger.warning("Could not find required columns for feature engineering")
            features = raw_data
        
        logger.info(f"Final dataset shape: {features.shape}")
        
        # Save processed dataset
        output_path = Path("data/processed/volatility_dataset.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path)
        logger.info(f"Saved processed dataset to {output_path}")
        
        # Also save as CSV for easy inspection
        csv_path = Path("data/processed/volatility_dataset.csv")
        features.to_csv(csv_path)
        logger.info(f"Saved CSV version to {csv_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATA COLLECTION SUMMARY")
        print("=" * 60)
        print(f"Date Range: {features.index.min().date()} to {features.index.max().date()}")
        print(f"Trading Days: {len(features):,}")
        print(f"Total Columns: {len(features.columns)}")
        print(f"\nColumn Categories:")
        
        # Categorize columns
        vix_cols = [c for c in features.columns if 'VIX' in c.upper() or 'VX' in c.upper()]
        rv_cols = [c for c in features.columns if 'rv' in c.lower() or 'volatility' in c.lower()]
        vrp_cols = [c for c in features.columns if 'vrp' in c.lower()]
        regime_cols = [c for c in features.columns if 'regime' in c.lower()]
        econ_cols = [c for c in features.columns if any(x in c for x in ['DFF', 'DGS', 'BAML', 'NFCI'])]
        
        print(f"  - VIX/Futures: {len(vix_cols)}")
        print(f"  - Realized Vol: {len(rv_cols)}")
        print(f"  - VRP: {len(vrp_cols)}")
        print(f"  - Regime: {len(regime_cols)}")
        print(f"  - Economic: {len(econ_cols)}")
        
        print(f"\nMissing Data (>10% missing):")
        for col in features.columns:
            missing_pct = features[col].isna().mean() * 100
            if missing_pct > 10:
                print(f"  - {col}: {missing_pct:.1f}%")
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)
        
        return features
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
