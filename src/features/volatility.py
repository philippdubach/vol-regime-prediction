"""
Volatility feature engineering.

This module computes volatility-related features for the research project:
- Realized volatility (multiple windows)
- Variance Risk Premium (VRP)
- Term structure metrics
- Regime indicators
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VolatilityFeatures:
    """
    Computes volatility-related features from raw data.
    
    Features include:
    - Realized volatility (5, 10, 21, 63, 126, 252 day windows)
    - Variance Risk Premium (VIX^2 - RV^2)
    - VIX term structure slope
    - VIX futures basis and roll yield
    - Volatility of volatility
    
    Example:
        vf = VolatilityFeatures()
        features = vf.compute_all(raw_data)
    """
    
    def __init__(
        self,
        rv_windows: List[int] = None,
        annualization_factor: int = 252,
        vrp_forward_window: int = 21
    ):
        """
        Initialize feature engineer.
        
        Args:
            rv_windows: List of windows for realized volatility.
            annualization_factor: Trading days per year.
            vrp_forward_window: Window for forward-looking RV in VRP.
        """
        self.rv_windows = rv_windows or [5, 10, 21, 63, 126, 252]
        self.annualization_factor = annualization_factor
        self.vrp_forward_window = vrp_forward_window
    
    def compute_returns(
        self,
        df: pd.DataFrame,
        price_col: str
    ) -> pd.DataFrame:
        """
        Compute simple and log returns.
        
        Args:
            df: DataFrame with price data.
            price_col: Name of price column.
            
        Returns:
            DataFrame with return columns added.
        """
        result = df.copy()
        
        col_base = price_col.replace('_Close', '').replace('_close', '')
        
        # Simple returns
        result[f'{col_base}_return'] = result[price_col].pct_change()
        
        # Log returns
        result[f'{col_base}_log_return'] = np.log(result[price_col]).diff()
        
        logger.info(f"Computed returns for {price_col}")
        
        return result
    
    def compute_realized_volatility(
        self,
        df: pd.DataFrame,
        return_col: str,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute realized volatility for multiple windows.
        
        Uses close-to-close volatility estimator.
        
        Args:
            df: DataFrame with return data.
            return_col: Name of return column.
            windows: List of rolling windows.
            
        Returns:
            DataFrame with RV columns added.
        """
        result = df.copy()
        windows = windows or self.rv_windows
        
        col_base = return_col.replace('_return', '').replace('_log_return', '')
        
        for window in windows:
            col_name = f'{col_base}_rv_{window}'
            
            # Standard deviation of returns, annualized
            result[col_name] = (
                result[return_col]
                .rolling(window=window, min_periods=max(1, window // 2))
                .std() * np.sqrt(self.annualization_factor) * 100
            )
        
        logger.info(f"Computed realized volatility for windows: {windows}")
        
        return result
    
    def compute_parkinson_volatility(
        self,
        df: pd.DataFrame,
        high_col: str,
        low_col: str,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Compute Parkinson (high-low) volatility estimator.
        
        More efficient than close-to-close for the same window.
        
        Args:
            df: DataFrame with high/low prices.
            high_col: Name of high price column.
            low_col: Name of low price column.
            windows: List of rolling windows.
            
        Returns:
            DataFrame with Parkinson volatility columns.
        """
        result = df.copy()
        windows = windows or self.rv_windows
        
        # Parkinson variance (single period) - handle NaN properly
        # Only compute where both high and low are valid
        valid_mask = result[high_col].notna() & result[low_col].notna()
        log_hl = np.where(
            valid_mask,
            np.log(result[high_col] / result[low_col]),
            np.nan
        )
        parkinson_var = log_hl ** 2 / (4 * np.log(2))
        parkinson_series = pd.Series(parkinson_var, index=result.index)
        
        col_base = high_col.replace('_High', '').replace('_high', '')
        
        for window in windows:
            col_name = f'{col_base}_parkinson_rv_{window}'
            
            # Rolling average of Parkinson variance, annualized
            # Use min_periods to handle NaN gaps
            result[col_name] = (
                np.sqrt(parkinson_series.rolling(window=window, min_periods=max(1, window // 2)).mean() * self.annualization_factor) * 100
            )
        
        logger.info(f"Computed Parkinson volatility for windows: {windows}")
        
        return result
    
    def compute_variance_risk_premium(
        self,
        df: pd.DataFrame,
        vix_col: str,
        rv_col: str,
        forward_window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compute Variance Risk Premium.
        
        VRP = IV² - RV²
        
        A positive VRP means implied volatility exceeds realized,
        which is the typical state (volatility risk premium).
        
        WARNING: Forward-looking VRP (vrp_forward, vrp_vol_points_forward)
        uses future realized volatility and CANNOT be used as a predictor.
        Use only for ex-post analysis. For modeling, use vrp_backward.
        
        Args:
            df: DataFrame with VIX and RV data.
            vix_col: Name of VIX column (in percentage points).
            rv_col: Name of realized volatility column.
            forward_window: Forward window for RV comparison.
            
        Returns:
            DataFrame with VRP columns added:
            - vrp_backward: Real-time available (IV² - past RV²)
            - vrp_forward: Look-ahead (IV² - future RV²) - FOR ANALYSIS ONLY
        """
        result = df.copy()
        forward_window = forward_window or self.vrp_forward_window
        
        # Convert to variance (VIX is in percentage points)
        iv_squared = (result[vix_col] / 100) ** 2
        rv_squared = (result[rv_col] / 100) ** 2
        
        # Forward-looking VRP (compare to future realized)
        result['vrp_forward'] = iv_squared - rv_squared.shift(-forward_window)
        
        # Backward-looking VRP (compare to past realized)
        result['vrp_backward'] = iv_squared - rv_squared
        
        # VRP in volatility points (for interpretability)
        result['vrp_vol_points'] = result[vix_col] - result[rv_col]
        result['vrp_vol_points_forward'] = result[vix_col] - result[rv_col].shift(-forward_window)
        
        logger.info(f"Computed VRP with forward window: {forward_window}")
        
        return result
    
    def compute_term_structure_features(
        self,
        df: pd.DataFrame,
        vix_col: str = 'VIX_CLOSE',
        vx1_col: str = 'VX1',
        vx2_col: str = 'VX2',
        vx4_col: str = 'VX4'
    ) -> pd.DataFrame:
        """
        Compute VIX term structure features.
        
        Args:
            df: DataFrame with VIX and futures data.
            vix_col: Spot VIX column.
            vx1_col: Front month futures column.
            vx2_col: Second month futures column.
            vx4_col: Fourth month futures column.
            
        Returns:
            DataFrame with term structure features.
        """
        result = df.copy()
        
        # Check which columns exist
        available_cols = df.columns.tolist()
        
        # VIX futures basis (VX1 - VIX spot)
        if vx1_col in available_cols and vix_col in available_cols:
            result['vix_basis'] = result[vx1_col] - result[vix_col]
            result['vix_basis_pct'] = result['vix_basis'] / result[vix_col] * 100
        
        # Term structure slope (VX2 - VX1)
        if vx1_col in available_cols and vx2_col in available_cols:
            result['term_slope_1_2'] = result[vx2_col] - result[vx1_col]
            result['term_slope_1_2_pct'] = result['term_slope_1_2'] / result[vx1_col] * 100
        
        # Longer-term slope (VX4 - VX1)
        if vx1_col in available_cols and vx4_col in available_cols:
            result['term_slope_1_4'] = result[vx4_col] - result[vx1_col]
            result['term_slope_1_4_pct'] = result['term_slope_1_4'] / result[vx1_col] * 100
        
        # Contango/backwardation indicator (preserve NaN where vix_basis is missing)
        if 'vix_basis' in result.columns:
            result['is_contango'] = np.where(
                result['vix_basis'].notna(),
                (result['vix_basis'] > 0).astype(float),
                np.nan
            )
        
        logger.info("Computed term structure features")
        
        return result
    
    def compute_volatility_of_volatility(
        self,
        df: pd.DataFrame,
        vix_col: str,
        windows: List[int] = [5, 10, 21]
    ) -> pd.DataFrame:
        """
        Compute volatility of VIX (vol of vol).
        
        Args:
            df: DataFrame with VIX data.
            vix_col: Name of VIX column.
            windows: Rolling windows for vol of vol.
            
        Returns:
            DataFrame with vol of vol features.
        """
        result = df.copy()
        
        # VIX returns
        vix_return = result[vix_col].pct_change()
        
        for window in windows:
            # Standard deviation of VIX returns
            result[f'vix_vol_{window}'] = (
                vix_return.rolling(window=window, min_periods=max(1, window//2)).std() * np.sqrt(self.annualization_factor) * 100
            )
            
            # VIX change (absolute)
            result[f'vix_change_{window}'] = result[vix_col].diff(window)
            
            # VIX high-low range (use min_periods)
            result[f'vix_range_{window}'] = (
                result[vix_col].rolling(window=window, min_periods=max(1, window//2)).max() - 
                result[vix_col].rolling(window=window, min_periods=max(1, window//2)).min()
            )
        
        logger.info(f"Computed volatility of volatility for windows: {windows}")
        
        return result
    
    def compute_regime_indicators(
        self,
        df: pd.DataFrame,
        vix_col: str,
        thresholds: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Compute volatility regime indicators.
        
        Args:
            df: DataFrame with VIX data.
            vix_col: Name of VIX column.
            thresholds: Dictionary with regime thresholds.
            
        Returns:
            DataFrame with regime indicator columns.
        """
        result = df.copy()
        
        if thresholds is None:
            thresholds = {
                'low': 15,
                'medium': 20,
                'high': 25,
                'crisis': 35
            }
        
        # VIX level regimes
        result['regime_low_vol'] = (result[vix_col] < thresholds['low']).astype(int)
        result['regime_medium_vol'] = (
            (result[vix_col] >= thresholds['low']) & 
            (result[vix_col] < thresholds['medium'])
        ).astype(int)
        result['regime_elevated_vol'] = (
            (result[vix_col] >= thresholds['medium']) & 
            (result[vix_col] < thresholds['high'])
        ).astype(int)
        result['regime_high_vol'] = (
            (result[vix_col] >= thresholds['high']) & 
            (result[vix_col] < thresholds['crisis'])
        ).astype(int)
        result['regime_crisis'] = (result[vix_col] >= thresholds['crisis']).astype(int)
        
        # Categorical regime
        conditions = [
            result[vix_col] < thresholds['low'],
            result[vix_col] < thresholds['medium'],
            result[vix_col] < thresholds['high'],
            result[vix_col] < thresholds['crisis'],
            result[vix_col] >= thresholds['crisis']
        ]
        choices = ['low', 'medium', 'elevated', 'high', 'crisis']
        result['regime'] = np.select(conditions, choices, default='unknown')
        
        # Rolling VIX percentile rank (avoids look-ahead bias)
        # Uses expanding window with 252-day minimum for stability
        result['vix_percentile'] = (
            result[vix_col]
            .expanding(min_periods=252)
            .apply(lambda x: (x.iloc[-1] <= x).mean(), raw=False)
        )
        
        # Rolling z-score (use min_periods for robustness with gaps)
        rolling_mean = result[vix_col].rolling(252, min_periods=63).mean()
        rolling_std = result[vix_col].rolling(252, min_periods=63).std()
        result['vix_zscore_252'] = (result[vix_col] - rolling_mean) / rolling_std
        
        logger.info("Computed regime indicators")
        
        return result
    
    def compute_all(
        self,
        df: pd.DataFrame,
        price_col: str = 'GSPC_Close',
        high_col: str = 'GSPC_High',
        low_col: str = 'GSPC_Low',
        vix_col: str = 'VIX_CLOSE'
    ) -> pd.DataFrame:
        """
        Compute all volatility features.
        
        Args:
            df: Raw data DataFrame.
            price_col: S&P 500 close price column.
            high_col: S&P 500 high price column.
            low_col: S&P 500 low price column.
            vix_col: VIX column.
            
        Returns:
            DataFrame with all computed features.
        """
        logger.info("Computing all volatility features...")
        
        result = df.copy()
        
        # Returns
        if price_col in result.columns:
            result = self.compute_returns(result, price_col)
            return_col = price_col.replace('_Close', '').replace('_close', '') + '_log_return'
            
            # Realized volatility
            if return_col in result.columns:
                result = self.compute_realized_volatility(result, return_col)
        
        # Parkinson volatility (if high/low available)
        if high_col in result.columns and low_col in result.columns:
            result = self.compute_parkinson_volatility(result, high_col, low_col)
        
        # VRP
        if vix_col in result.columns:
            # Try both possible RV column names
            rv_col = price_col.replace('_Close', '').replace('_close', '') + '_log_rv_21'
            if rv_col not in result.columns:
                rv_col = price_col.replace('_Close', '').replace('_close', '') + '_rv_21'
            if rv_col in result.columns:
                result = self.compute_variance_risk_premium(result, vix_col, rv_col)
        
        # Term structure (if futures data available)
        result = self.compute_term_structure_features(result)
        
        # Volatility of volatility
        if vix_col in result.columns:
            result = self.compute_volatility_of_volatility(result, vix_col)
        
        # Regime indicators
        if vix_col in result.columns:
            result = self.compute_regime_indicators(result, vix_col)
        
        # SKEW features
        result = self.compute_skew_features(result)
        
        # Put/Call ratio features  
        result = self.compute_putcall_features(result)
        
        logger.info(f"Computed {len(result.columns) - len(df.columns)} new features")
        
        return result
    
    def compute_skew_features(
        self,
        df: pd.DataFrame,
        skew_col: str = 'SKEW_SKEW'
    ) -> pd.DataFrame:
        """
        Compute features from CBOE SKEW index.
        
        SKEW measures tail risk - higher values indicate greater 
        perceived probability of outlier returns (black swan events).
        
        Args:
            df: DataFrame with SKEW data.
            skew_col: Name of SKEW column.
            
        Returns:
            DataFrame with SKEW features.
        """
        result = df.copy()
        
        if skew_col not in result.columns:
            logger.debug(f"SKEW column {skew_col} not found, skipping SKEW features")
            return result
        
        # SKEW level z-score (use min_periods for robustness)
        rolling_mean = result[skew_col].rolling(252, min_periods=63).mean()
        rolling_std = result[skew_col].rolling(252, min_periods=63).std()
        result['skew_zscore'] = (result[skew_col] - rolling_mean) / rolling_std
        
        # SKEW change
        result['skew_change_5'] = result[skew_col].diff(5)
        result['skew_change_21'] = result[skew_col].diff(21)
        
        # SKEW percentile (simplified - use rank over available data)
        result['skew_percentile'] = result[skew_col].rolling(252, min_periods=63).apply(
            lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else np.nan, raw=False
        )
        
        # SKEW / VIX ratio (risk aversion indicator)
        if 'VIX_CLOSE' in result.columns:
            result['skew_vix_ratio'] = result[skew_col] / result['VIX_CLOSE']
        
        # High SKEW regime (elevated tail risk)
        result['high_skew_regime'] = (result[skew_col] > 130).astype(int)
        
        logger.info("Computed SKEW features")
        
        return result
    
    def compute_putcall_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute features from put/call ratio data.
        
        High put/call ratios often indicate fear/hedging activity.
        
        Args:
            df: DataFrame with put/call data.
            
        Returns:
            DataFrame with put/call features.
        """
        result = df.copy()
        
        # Look for put/call ratio columns
        pc_cols = [c for c in result.columns if 'PC_RATIO' in c]
        
        if not pc_cols:
            logger.debug("No put/call ratio columns found, skipping PC features")
            return result
        
        for pc_col in pc_cols:
            prefix = pc_col.replace('_PC_RATIO', '').lower()
            
            # Moving averages
            result[f'{prefix}_pc_ma5'] = result[pc_col].rolling(5, min_periods=3).mean()
            result[f'{prefix}_pc_ma21'] = result[pc_col].rolling(21, min_periods=10).mean()
            
            # Z-score (use min_periods for robustness)
            rolling_mean = result[pc_col].rolling(63, min_periods=21).mean()
            rolling_std = result[pc_col].rolling(63, min_periods=21).std()
            result[f'{prefix}_pc_zscore'] = (result[pc_col] - rolling_mean) / rolling_std
            
            # Extreme sentiment indicator (contrarian signal)
            # High P/C (>1.0) often marks fear peaks
            result[f'{prefix}_pc_extreme_put'] = (result[pc_col] > 1.0).astype(int)
            # Low P/C (<0.5) often marks complacency
            result[f'{prefix}_pc_extreme_call'] = (result[pc_col] < 0.5).astype(int)
        
        # Combined fear index (if multiple series available)
        if 'TOTAL_PC_RATIO' in result.columns and 'INDEX_PC_RATIO' in result.columns:
            # Index P/C typically higher than equity P/C due to hedging
            result['pc_spread'] = result['INDEX_PC_RATIO'] - result.get('EQUITY_PC_RATIO', result['TOTAL_PC_RATIO'])
        
        logger.info(f"Computed put/call features for {len(pc_cols)} series")
        
        return result


if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'GSPC_Close': 3000 + np.cumsum(np.random.randn(len(dates)) * 20),
        'GSPC_High': 3000 + np.cumsum(np.random.randn(len(dates)) * 20) + 10,
        'GSPC_Low': 3000 + np.cumsum(np.random.randn(len(dates)) * 20) - 10,
        'VIX_CLOSE': 15 + np.abs(np.random.randn(len(dates)) * 5),
        'VX1': 16 + np.abs(np.random.randn(len(dates)) * 5),
        'VX2': 17 + np.abs(np.random.randn(len(dates)) * 5),
    }, index=dates)
    
    # Compute features
    vf = VolatilityFeatures()
    features = vf.compute_all(sample_data)
    
    print(f"\nFeatures computed: {len(features.columns)}")
    print(f"\nNew columns:\n{[c for c in features.columns if c not in sample_data.columns]}")
    print(f"\nSample:\n{features[['VIX_CLOSE', 'GSPC_rv_21', 'vrp_vol_points', 'regime']].tail()}")
