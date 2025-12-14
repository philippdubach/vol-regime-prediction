#!/usr/bin/env python3
"""
Standalone Feature Computation Script

This script documents and computes all derived features used in the
Volatility Regime Prediction project. It can be run independently
to regenerate features or as a reference for the methodologies.

Key Computed Features:
1. Realized Volatility (close-to-close and Parkinson)
2. Variance Risk Premium (VRP)
3. Term Structure Metrics
4. Regime Indicators
5. Sentiment Indicators (SKEW, Put/Call)

Author: Volatility Research Project
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# REALIZED VOLATILITY
# =============================================================================

def compute_realized_volatility(
    returns: pd.Series,
    windows: list = [5, 10, 21, 63, 126, 252],
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Compute realized volatility using close-to-close estimator.
    
    Formula:
        RV_t^(n) = sqrt(252/n * sum(r_{t-i}^2)) * 100
        
    where r_t are log returns.
    
    Args:
        returns: Series of log returns
        windows: Rolling windows in trading days
        annualization_factor: Trading days per year
        
    Returns:
        DataFrame with RV columns for each window
    """
    result = pd.DataFrame(index=returns.index)
    
    for window in windows:
        col_name = f'rv_{window}'
        result[col_name] = (
            returns
            .rolling(window=window, min_periods=max(1, window // 2))
            .std() * np.sqrt(annualization_factor) * 100
        )
    
    return result


def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    windows: list = [5, 10, 21, 63, 126, 252],
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Compute Parkinson (high-low range) volatility estimator.
    
    Formula:
        σ_P^2 = (1/4*ln(2)) * (ln(H/L))^2
        
    This estimator is more efficient than close-to-close when
    intraday data is available.
    
    Reference:
        Parkinson, M. (1980). "The Extreme Value Method for Estimating 
        the Variance of the Rate of Return." Journal of Business.
        
    Args:
        high: Series of high prices
        low: Series of low prices
        windows: Rolling windows
        annualization_factor: Trading days per year
        
    Returns:
        DataFrame with Parkinson volatility columns
    """
    result = pd.DataFrame(index=high.index)
    
    # Only compute where both high and low are valid
    valid_mask = high.notna() & low.notna()
    log_hl = np.where(valid_mask, np.log(high / low), np.nan)
    parkinson_var = log_hl ** 2 / (4 * np.log(2))
    parkinson_series = pd.Series(parkinson_var, index=high.index)
    
    for window in windows:
        col_name = f'parkinson_rv_{window}'
        result[col_name] = (
            np.sqrt(
                parkinson_series
                .rolling(window=window, min_periods=max(1, window // 2))
                .mean() * annualization_factor
            ) * 100
        )
    
    return result


# =============================================================================
# VARIANCE RISK PREMIUM (VRP)
# =============================================================================

def compute_variance_risk_premium(
    vix: pd.Series,
    realized_vol: pd.Series,
    forward_window: int = 21
) -> pd.DataFrame:
    """
    Compute Variance Risk Premium (VRP).
    
    VRP = IV² - RV²
    
    A positive VRP means implied volatility exceeds realized volatility,
    which is the typical state due to investors' risk aversion.
    
    The VRP can be computed:
    - Forward-looking: Compare IV_t to RV_{t+21} (requires future data)
    - Backward-looking: Compare IV_t to RV_t (available in real-time)
    
    Typical values:
    - VRP is positive ~75-85% of the time
    - Mean VRP ~2-4 volatility points
    - VRP spikes negative during volatility shocks
    
    Args:
        vix: VIX index (in percentage points, e.g., 15.0 for 15%)
        realized_vol: Realized volatility (same units as VIX)
        forward_window: Days ahead for forward-looking VRP
        
    Returns:
        DataFrame with VRP measures
    """
    result = pd.DataFrame(index=vix.index)
    
    # Convert to variance (squared percentage)
    iv_squared = (vix / 100) ** 2
    rv_squared = (realized_vol / 100) ** 2
    
    # Forward-looking VRP (compare to future realized)
    # This is ex-post and requires future data
    result['vrp_forward'] = iv_squared - rv_squared.shift(-forward_window)
    
    # Backward-looking VRP (available in real-time)
    result['vrp_backward'] = iv_squared - rv_squared
    
    # In volatility points (more interpretable)
    result['vrp_vol_points'] = vix - realized_vol
    result['vrp_vol_points_forward'] = vix - realized_vol.shift(-forward_window)
    
    return result


# =============================================================================
# TERM STRUCTURE METRICS
# =============================================================================

def compute_term_structure_features(
    vix_spot: pd.Series,
    vx1: pd.Series,
    vx2: pd.Series,
    vx4: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute VIX term structure features.
    
    The VIX term structure reflects market expectations:
    - Contango (VX1 > VIX): Normal, near-term calm expected
    - Backwardation (VX1 < VIX): Stress, elevated near-term fear
    
    Key features:
    - VIX basis: VX1 - VIX (spot-futures relationship)
    - Term slope: VX2 - VX1 (curve steepness)
    - Contango indicator: Binary flag
    
    Args:
        vix_spot: Spot VIX index
        vx1: Front-month VIX futures
        vx2: Second-month VIX futures
        vx4: Fourth-month VIX futures (optional)
        
    Returns:
        DataFrame with term structure features
    """
    result = pd.DataFrame(index=vix_spot.index)
    
    # VIX futures basis
    result['vix_basis'] = vx1 - vix_spot
    result['vix_basis_pct'] = result['vix_basis'] / vix_spot * 100
    
    # Term structure slope
    result['term_slope_1_2'] = vx2 - vx1
    result['term_slope_1_2_pct'] = result['term_slope_1_2'] / vx1 * 100
    
    if vx4 is not None:
        result['term_slope_1_4'] = vx4 - vx1
        result['term_slope_1_4_pct'] = result['term_slope_1_4'] / vx1 * 100
    
    # Contango indicator
    result['is_contango'] = (result['vix_basis'] > 0).astype(int)
    result.loc[result['vix_basis'].isna(), 'is_contango'] = np.nan
    
    return result


# =============================================================================
# REGIME INDICATORS
# =============================================================================

def compute_regime_indicators(
    vix: pd.Series,
    thresholds: dict = None
) -> pd.DataFrame:
    """
    Compute volatility regime indicators.
    
    Regime classification based on VIX levels:
    - Low: VIX < 15 (calm markets)
    - Medium: 15 <= VIX < 20 (normal)
    - Elevated: 20 <= VIX < 25 (cautious)
    - High: 25 <= VIX < 35 (stressed)
    - Crisis: VIX >= 35 (panic)
    
    Historical regime frequencies (approximate):
    - Low: ~30-35%
    - Medium: ~30%
    - Elevated: ~15-20%
    - High: ~10-15%
    - Crisis: ~5%
    
    Args:
        vix: VIX index series
        thresholds: Dict with threshold values
        
    Returns:
        DataFrame with regime indicators
    """
    result = pd.DataFrame(index=vix.index)
    
    if thresholds is None:
        thresholds = {
            'low': 15,
            'medium': 20,
            'high': 25,
            'crisis': 35
        }
    
    # Binary regime indicators
    result['regime_low_vol'] = (vix < thresholds['low']).astype(int)
    result['regime_medium_vol'] = (
        (vix >= thresholds['low']) & (vix < thresholds['medium'])
    ).astype(int)
    result['regime_elevated_vol'] = (
        (vix >= thresholds['medium']) & (vix < thresholds['high'])
    ).astype(int)
    result['regime_high_vol'] = (
        (vix >= thresholds['high']) & (vix < thresholds['crisis'])
    ).astype(int)
    result['regime_crisis'] = (vix >= thresholds['crisis']).astype(int)
    
    # Categorical regime
    conditions = [
        vix < thresholds['low'],
        vix < thresholds['medium'],
        vix < thresholds['high'],
        vix < thresholds['crisis'],
        vix >= thresholds['crisis']
    ]
    choices = ['low', 'medium', 'elevated', 'high', 'crisis']
    result['regime'] = np.select(conditions, choices, default='unknown')
    
    # Rolling statistics
    result['vix_percentile'] = vix.rank(pct=True)
    rolling_mean = vix.rolling(252, min_periods=63).mean()
    rolling_std = vix.rolling(252, min_periods=63).std()
    result['vix_zscore_252'] = (vix - rolling_mean) / rolling_std
    
    return result


# =============================================================================
# SENTIMENT INDICATORS
# =============================================================================

def compute_skew_features(
    skew: pd.Series,
    vix: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute features from CBOE SKEW index.
    
    SKEW measures tail risk - the probability of outlier returns.
    - SKEW = 100: Normal distribution (no excess tail risk)
    - SKEW > 100: Elevated left-tail risk (crash protection demand)
    - Typical range: 110-140
    
    High SKEW relative to VIX suggests tail hedging demand
    even without elevated overall volatility.
    
    Args:
        skew: CBOE SKEW index
        vix: VIX index (optional, for ratio)
        
    Returns:
        DataFrame with SKEW features
    """
    result = pd.DataFrame(index=skew.index)
    
    # Z-score
    rolling_mean = skew.rolling(252, min_periods=63).mean()
    rolling_std = skew.rolling(252, min_periods=63).std()
    result['skew_zscore'] = (skew - rolling_mean) / rolling_std
    
    # Changes
    result['skew_change_5'] = skew.diff(5)
    result['skew_change_21'] = skew.diff(21)
    
    # Percentile
    result['skew_percentile'] = skew.rolling(252, min_periods=63).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else np.nan, raw=False
    )
    
    # Ratio to VIX
    if vix is not None:
        result['skew_vix_ratio'] = skew / vix
    
    # High SKEW regime
    result['high_skew_regime'] = (skew > 130).astype(int)
    
    return result


def compute_putcall_features(
    pc_ratio: pd.Series,
    name_prefix: str = ''
) -> pd.DataFrame:
    """
    Compute features from put/call ratios.
    
    Put/Call ratio interpretation:
    - P/C > 1: More puts than calls (bearish/hedging)
    - P/C < 1: More calls than puts (bullish)
    - Extreme readings often contrarian signals
    
    Args:
        pc_ratio: Put/call ratio series
        name_prefix: Prefix for column names
        
    Returns:
        DataFrame with P/C features
    """
    result = pd.DataFrame(index=pc_ratio.index)
    prefix = f'{name_prefix}_' if name_prefix else ''
    
    # Moving averages
    result[f'{prefix}pc_ma5'] = pc_ratio.rolling(5, min_periods=3).mean()
    result[f'{prefix}pc_ma21'] = pc_ratio.rolling(21, min_periods=10).mean()
    
    # Z-score (shorter window for P/C)
    rolling_mean = pc_ratio.rolling(63, min_periods=21).mean()
    rolling_std = pc_ratio.rolling(63, min_periods=21).std()
    result[f'{prefix}pc_zscore'] = (pc_ratio - rolling_mean) / rolling_std
    
    # Extreme indicators
    result[f'{prefix}pc_extreme_put'] = (pc_ratio > 1.0).astype(int)
    result[f'{prefix}pc_extreme_call'] = (pc_ratio < 0.5).astype(int)
    
    return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Demonstrate feature computation on actual data."""
    
    # Load raw data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'interim' / 'raw_merged.parquet'
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Run the main pipeline first: python src/main.py")
        return
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data: {df.shape}")
    
    # Filter to trading days
    df = df[df['VIX_CLOSE'].notna()]
    logger.info(f"Trading days: {len(df)}")
    
    # Compute returns
    df['log_return'] = np.log(df['GSPC_Close']).diff()
    
    # 1. Realized Volatility
    rv = compute_realized_volatility(df['log_return'])
    logger.info(f"RV features: {list(rv.columns)}")
    
    # 2. Parkinson Volatility
    parkinson = compute_parkinson_volatility(df['GSPC_High'], df['GSPC_Low'])
    logger.info(f"Parkinson features: {list(parkinson.columns)}")
    
    # 3. VRP
    vrp = compute_variance_risk_premium(df['VIX_CLOSE'], rv['rv_21'])
    logger.info(f"VRP features: {list(vrp.columns)}")
    logger.info(f"VRP positive: {(vrp['vrp_forward'] > 0).mean()*100:.1f}%")
    
    # 4. Term Structure (if futures available)
    if 'VX1' in df.columns and 'VX2' in df.columns:
        term = compute_term_structure_features(
            df['VIX_CLOSE'], df['VX1'], df['VX2'],
            df.get('VX4')
        )
        logger.info(f"Term structure features: {list(term.columns)}")
        contango_pct = term.loc[term['vix_basis'].notna(), 'is_contango'].mean() * 100
        logger.info(f"Contango frequency: {contango_pct:.1f}%")
    
    # 5. Regimes
    regimes = compute_regime_indicators(df['VIX_CLOSE'])
    logger.info(f"Regime distribution:\n{regimes['regime'].value_counts(normalize=True)}")
    
    # 6. SKEW
    if 'SKEW_SKEW' in df.columns:
        skew_features = compute_skew_features(df['SKEW_SKEW'], df['VIX_CLOSE'])
        logger.info(f"SKEW features: {list(skew_features.columns)}")
    
    # 7. Put/Call
    if 'TOTAL_PC_RATIO' in df.columns:
        pc_features = compute_putcall_features(df['TOTAL_PC_RATIO'], 'total')
        logger.info(f"Put/Call features: {list(pc_features.columns)}")
    
    logger.info("Feature computation complete!")


if __name__ == '__main__':
    main()
