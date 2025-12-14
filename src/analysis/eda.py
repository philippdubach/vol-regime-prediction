"""
Exploratory Data Analysis for Volatility Dataset.

This module provides comprehensive analysis of the collected data including:
- Data quality assessment
- Descriptive statistics
- Visualizations
- Correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_data(path: str = "data/processed/volatility_dataset.parquet") -> pd.DataFrame:
    """Load the processed dataset."""
    df = pd.read_parquet(path)
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive data quality report."""
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'min': df.min(numeric_only=True),
        'max': df.max(numeric_only=True),
        'mean': df.mean(numeric_only=True),
        'std': df.std(numeric_only=True)
    })
    return report


def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only include trading days with VIX data.
    This aligns all data to market trading days.
    """
    # Use VIX_CLOSE as the anchor for trading days
    vix_col = None
    for col in ['VIX_CLOSE', 'VIXCLS']:
        if col in df.columns:
            vix_col = col
            break
    
    if vix_col is None:
        print("Warning: No VIX column found, returning original data")
        return df
    
    # Filter to days with VIX data
    trading_days = df[df[vix_col].notna()].copy()
    print(f"Filtered to {len(trading_days):,} trading days with VIX data")
    return trading_days


def plot_vix_history(df: pd.DataFrame, output_dir: Path):
    """Plot VIX historical time series."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # VIX level
    ax1 = axes[0]
    if 'VIX_CLOSE' in df.columns:
        ax1.plot(df.index, df['VIX_CLOSE'], linewidth=0.8, color='blue')
        ax1.set_ylabel('VIX Index')
        ax1.set_title('VIX Index (CBOE Volatility Index)', fontsize=12, fontweight='bold')
        
        # Add regime thresholds
        ax1.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Low Vol (< 20)')
        ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='High Vol (> 30)')
        ax1.legend(loc='upper right')
        
        # Highlight crisis periods
        crisis_periods = [
            ('2008-09-01', '2009-03-31', 'GFC'),
            ('2020-02-01', '2020-04-30', 'COVID'),
            ('2011-08-01', '2011-10-31', 'Euro Crisis'),
            ('2022-01-01', '2022-03-31', 'Rate Hikes'),
        ]
        for start, end, label in crisis_periods:
            if df.index.min() <= pd.to_datetime(start):
                ax1.axvspan(start, end, alpha=0.2, color='red')
    
    # VIX term structure (VIX9D / VIX comparison)
    ax2 = axes[1]
    if 'VIX_CLOSE' in df.columns and 'VIX3M_CLOSE' in df.columns:
        ratio = df['VIX_CLOSE'] / df['VIX3M_CLOSE']
        ax2.plot(df.index, ratio, linewidth=0.8, color='purple')
        ax2.axhline(y=1, color='black', linestyle='-', alpha=0.5)
        ax2.set_ylabel('VIX / VIX3M Ratio')
        ax2.set_title('VIX Term Structure (< 1 = Contango, > 1 = Backwardation)', fontsize=12, fontweight='bold')
        ax2.fill_between(df.index, ratio, 1, where=(ratio > 1), alpha=0.3, color='red', label='Backwardation')
        ax2.fill_between(df.index, ratio, 1, where=(ratio <= 1), alpha=0.3, color='green', label='Contango')
        ax2.legend(loc='upper right')
    
    # VVIX (volatility of volatility)
    ax3 = axes[2]
    if 'VVIX_VVIX' in df.columns:
        ax3.plot(df.index, df['VVIX_VVIX'], linewidth=0.8, color='orange')
        ax3.set_ylabel('VVIX Index')
        ax3.set_title('VVIX (Volatility of VIX)', fontsize=12, fontweight='bold')
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='High Uncertainty (> 80)')
        ax3.legend(loc='upper right')
    
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vix_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'vix_history.png'}")


def plot_spx_returns(df: pd.DataFrame, output_dir: Path):
    """Plot S&P 500 returns and realized volatility."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # S&P 500 Price
    ax1 = axes[0]
    if 'GSPC_Close' in df.columns:
        ax1.plot(df.index, df['GSPC_Close'], linewidth=0.8, color='darkblue')
        ax1.set_ylabel('S&P 500 Index')
        ax1.set_title('S&P 500 Index', fontsize=12, fontweight='bold')
        ax1.set_yscale('log')
    
    # Daily returns
    ax2 = axes[1]
    if 'GSPC_return' in df.columns:
        ax2.plot(df.index, df['GSPC_return'] * 100, linewidth=0.5, color='blue', alpha=0.7)
        ax2.set_ylabel('Daily Return (%)')
        ax2.set_title('S&P 500 Daily Returns', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Highlight extreme returns
        extreme = np.abs(df['GSPC_return']) > 0.03
        ax2.scatter(df.index[extreme], df.loc[extreme, 'GSPC_return'] * 100, 
                   color='red', s=10, zorder=5, label='|Return| > 3%')
        ax2.legend(loc='upper right')
    
    # Realized volatility
    ax3 = axes[2]
    rv_cols = [c for c in df.columns if 'rv_21' in c.lower() and 'parkinson' not in c.lower()]
    if rv_cols:
        ax3.plot(df.index, df[rv_cols[0]] * 100, linewidth=0.8, color='red')
        ax3.set_ylabel('Realized Vol (%)')
        ax3.set_title('21-Day Realized Volatility', fontsize=12, fontweight='bold')
    
    ax3.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spx_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'spx_returns.png'}")


def plot_vix_distribution(df: pd.DataFrame, output_dir: Path):
    """Plot VIX distribution and statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if 'VIX_CLOSE' not in df.columns:
        return
    
    vix = df['VIX_CLOSE'].dropna()
    
    # Histogram
    ax1 = axes[0, 0]
    ax1.hist(vix, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(vix.mean(), color='red', linestyle='--', label=f'Mean: {vix.mean():.1f}')
    ax1.axvline(vix.median(), color='green', linestyle='--', label=f'Median: {vix.median():.1f}')
    ax1.set_xlabel('VIX Level')
    ax1.set_ylabel('Density')
    ax1.set_title('VIX Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Log VIX histogram
    ax2 = axes[0, 1]
    ax2.hist(np.log(vix), bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    ax2.set_xlabel('Log(VIX)')
    ax2.set_ylabel('Density')
    ax2.set_title('Log VIX Distribution (More Normal)', fontsize=12, fontweight='bold')
    
    # VIX by year
    ax3 = axes[1, 0]
    df_vix = df[['VIX_CLOSE']].copy()
    df_vix['year'] = df_vix.index.year
    df_vix.boxplot(column='VIX_CLOSE', by='year', ax=ax3, grid=False)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('VIX')
    ax3.set_title('VIX Distribution by Year', fontsize=12, fontweight='bold')
    plt.suptitle('')
    ax3.tick_params(axis='x', rotation=45)
    
    # VIX percentiles over time
    ax4 = axes[1, 1]
    if 'vix_percentile' in df.columns:
        ax4.plot(df.index, df['vix_percentile'] * 100, linewidth=0.5, color='blue')
        ax4.axhline(y=50, color='black', linestyle='--', alpha=0.5)
        ax4.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90th percentile')
        ax4.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10th percentile')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Percentile Rank')
        ax4.set_title('VIX Percentile (252-day rolling)', fontsize=12, fontweight='bold')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'vix_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'vix_distribution.png'}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    """Plot correlation heatmap for key variables."""
    # Select key columns for correlation analysis
    key_cols = []
    
    # VIX indices
    for col in ['VIX_CLOSE', 'VVIX_VVIX', 'VIX9D_CLOSE', 'VIX3M_CLOSE', 'VIX6M_CLOSE']:
        if col in df.columns:
            key_cols.append(col)
    
    # VIX futures
    for col in ['VX1', 'VX2', 'VX3', 'VX4', 'VX_Slope_1_2']:
        if col in df.columns:
            key_cols.append(col)
    
    # Realized volatility
    for col in ['GSPC_rv_21', 'GSPC_rv_63', 'GSPC_log_rv_21']:
        if col in df.columns:
            key_cols.append(col)
    
    # Economic
    for col in ['DFF', 'DGS10', 'T10Y2Y', 'BAMLH0A0HYM2']:
        if col in df.columns:
            key_cols.append(col)
    
    if len(key_cols) < 3:
        print("Not enough columns for correlation matrix")
        return
    
    # Calculate correlation
    corr_data = df[key_cols].dropna()
    if len(corr_data) < 100:
        print("Not enough data for correlation analysis")
        return
    
    corr = corr_data.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Correlation Matrix: Key Volatility Variables', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'correlation_matrix.png'}")


def plot_regime_analysis(df: pd.DataFrame, output_dir: Path):
    """Analyze volatility regimes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Create vix_regime column if it doesn't exist but 'regime' does
    if 'vix_regime' not in df.columns and 'regime' in df.columns:
        # Map categorical regime to numeric (0=low, 1=normal/medium, 2=high+)
        regime_map = {'low': 0, 'medium': 1, 'elevated': 2, 'high': 2, 'crisis': 2}
        df = df.copy()
        df['vix_regime'] = df['regime'].map(regime_map)
    
    # Alternative: Create from VIX_CLOSE directly if no regime columns
    if 'vix_regime' not in df.columns and 'VIX_CLOSE' in df.columns:
        df = df.copy()
        df['vix_regime'] = np.where(df['VIX_CLOSE'] < 15, 0,
                                    np.where(df['VIX_CLOSE'] < 25, 1, 2))
    
    # Regime indicator over time
    ax1 = axes[0, 0]
    if 'vix_regime' in df.columns and 'VIX_CLOSE' in df.columns:
        regime_colors = {0: 'green', 1: 'orange', 2: 'red'}
        for regime in [0, 1, 2]:
            mask = df['vix_regime'] == regime
            ax1.scatter(df.index[mask], df.loc[mask, 'VIX_CLOSE'] if 'VIX_CLOSE' in df.columns else [0]*mask.sum(),
                       c=regime_colors.get(regime, 'gray'), s=1, alpha=0.5,
                       label=['Low Vol', 'Normal', 'High Vol'][regime])
        ax1.set_ylabel('VIX')
        ax1.set_title('VIX Colored by Regime', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
    
    # Regime distribution
    ax2 = axes[0, 1]
    if 'vix_regime' in df.columns:
        regime_counts = df['vix_regime'].value_counts().sort_index()
        regime_labels = ['Low Vol', 'Normal', 'High Vol']
        colors = ['green', 'orange', 'red']
        ax2.bar(range(len(regime_counts)), regime_counts.values / len(df) * 100, color=colors)
        ax2.set_xticks(range(len(regime_counts)))
        ax2.set_xticklabels(regime_labels[:len(regime_counts)])
        ax2.set_ylabel('Percentage of Days (%)')
        ax2.set_title('Regime Distribution', fontsize=12, fontweight='bold')
        
        for i, v in enumerate(regime_counts.values):
            ax2.text(i, v/len(df)*100 + 1, f'{v/len(df)*100:.1f}%', ha='center')
    
    # VIX distribution by regime
    ax3 = axes[1, 0]
    if 'vix_regime' in df.columns and 'VIX_CLOSE' in df.columns:
        regime_labels = ['Low Vol', 'Normal', 'High Vol']
        for regime in range(3):
            vix_regime = df.loc[df['vix_regime'] == regime, 'VIX_CLOSE'].dropna()
            if len(vix_regime) > 0:
                ax3.hist(vix_regime, bins=30, alpha=0.5, 
                        label=f"{regime_labels[regime]} (n={len(vix_regime):,})")
        ax3.set_xlabel('VIX Level')
        ax3.set_ylabel('Count')
        ax3.set_title('VIX Distribution by Regime', fontsize=12, fontweight='bold')
        ax3.legend()
    
    # Returns by regime
    ax4 = axes[1, 1]
    if 'vix_regime' in df.columns and 'GSPC_return' in df.columns:
        regime_returns = df.groupby('vix_regime')['GSPC_return'].agg(['mean', 'std'])
        regime_labels = ['Low Vol', 'Normal', 'High Vol']
        x = range(len(regime_returns))
        ax4.bar(x, regime_returns['mean'] * 252 * 100, 
               yerr=regime_returns['std'] * np.sqrt(252) * 100,
               capsize=5, color=['green', 'orange', 'red'][:len(regime_returns)])
        ax4.set_xticks(x)
        ax4.set_xticklabels(regime_labels[:len(regime_returns)])
        ax4.set_ylabel('Annualized Return (%)')
        ax4.set_title('Annualized Returns by Regime', fontsize=12, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regime_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'regime_analysis.png'}")


def plot_futures_term_structure(df: pd.DataFrame, output_dir: Path):
    """Analyze VIX futures term structure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Term structure slope over time
    ax1 = axes[0, 0]
    if 'VX_Slope_1_2' in df.columns:
        ax1.plot(df.index, df['VX_Slope_1_2'], linewidth=0.8, color='blue')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax1.fill_between(df.index, df['VX_Slope_1_2'], 0, 
                        where=(df['VX_Slope_1_2'] > 0), alpha=0.3, color='green', label='Contango')
        ax1.fill_between(df.index, df['VX_Slope_1_2'], 0,
                        where=(df['VX_Slope_1_2'] <= 0), alpha=0.3, color='red', label='Backwardation')
        ax1.set_ylabel('VX2 - VX1')
        ax1.set_title('VIX Futures Term Structure Slope', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
    
    # Current term structure snapshot
    ax2 = axes[0, 1]
    vx_cols = [f'VX{i}' for i in range(1, 10) if f'VX{i}' in df.columns]
    if vx_cols:
        latest = df[vx_cols].iloc[-1]
        ax2.plot(range(1, len(latest)+1), latest.values, 'o-', color='blue', markersize=8)
        ax2.set_xlabel('Contract Month')
        ax2.set_ylabel('VIX Futures Price')
        ax2.set_title(f'Term Structure ({df.index[-1].date()})', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(1, len(latest)+1))
    
    # Distribution of slopes
    ax3 = axes[1, 0]
    if 'VX_Slope_1_2' in df.columns:
        slopes = df['VX_Slope_1_2'].dropna()
        ax3.hist(slopes, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.axvline(slopes.mean(), color='green', linestyle='--', label=f'Mean: {slopes.mean():.2f}')
        ax3.set_xlabel('VX2 - VX1')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution of Term Structure Slope', fontsize=12, fontweight='bold')
        ax3.legend()
        
        # Add statistics
        contango_pct = (slopes > 0).sum() / len(slopes) * 100
        ax3.text(0.95, 0.95, f'Contango: {contango_pct:.1f}%\nBackwardation: {100-contango_pct:.1f}%',
                transform=ax3.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # VIX vs VX1 basis
    ax4 = axes[1, 1]
    if 'VIX_CLOSE' in df.columns and 'VX1' in df.columns:
        mask = df[['VIX_CLOSE', 'VX1']].notna().all(axis=1)
        ax4.scatter(df.loc[mask, 'VIX_CLOSE'], df.loc[mask, 'VX1'], alpha=0.3, s=5)
        ax4.plot([0, 100], [0, 100], 'r--', label='VIX = VX1')
        ax4.set_xlabel('VIX (Spot)')
        ax4.set_ylabel('VX1 (Front Month)')
        ax4.set_title('VIX vs Front Month Futures', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.set_xlim(0, 80)
        ax4.set_ylim(0, 80)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'futures_term_structure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'futures_term_structure.png'}")


def plot_economic_variables(df: pd.DataFrame, output_dir: Path):
    """Plot economic variables and their relationship with VIX."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fed Funds Rate
    ax1 = axes[0, 0]
    if 'DFF' in df.columns:
        ax1.plot(df.index, df['DFF'], linewidth=0.8, color='blue')
        ax1.set_ylabel('Fed Funds Rate (%)')
        ax1.set_title('Federal Funds Rate', fontsize=12, fontweight='bold')
    
    # Yield curve spread
    ax2 = axes[0, 1]
    if 'T10Y2Y' in df.columns:
        ax2.plot(df.index, df['T10Y2Y'], linewidth=0.8, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.fill_between(df.index, df['T10Y2Y'], 0, 
                        where=(df['T10Y2Y'] < 0), alpha=0.3, color='red', label='Inverted')
        ax2.set_ylabel('10Y - 2Y Spread (%)')
        ax2.set_title('Yield Curve Spread (10Y - 2Y)', fontsize=12, fontweight='bold')
        ax2.legend()
    
    # Credit spreads
    ax3 = axes[1, 0]
    if 'BAMLH0A0HYM2' in df.columns:
        ax3.plot(df.index, df['BAMLH0A0HYM2'], linewidth=0.8, color='orange', label='High Yield')
    if 'BAMLC0A0CM' in df.columns:
        ax3.plot(df.index, df['BAMLC0A0CM'], linewidth=0.8, color='blue', label='Investment Grade')
    ax3.set_ylabel('Credit Spread (%)')
    ax3.set_title('Corporate Bond Spreads', fontsize=12, fontweight='bold')
    ax3.legend()
    
    # VIX vs Credit spread scatter
    ax4 = axes[1, 1]
    if 'VIX_CLOSE' in df.columns and 'BAMLH0A0HYM2' in df.columns:
        mask = df[['VIX_CLOSE', 'BAMLH0A0HYM2']].notna().all(axis=1)
        ax4.scatter(df.loc[mask, 'BAMLH0A0HYM2'], df.loc[mask, 'VIX_CLOSE'], alpha=0.3, s=5)
        ax4.set_xlabel('HY Credit Spread (%)')
        ax4.set_ylabel('VIX')
        ax4.set_title('VIX vs High Yield Spread', fontsize=12, fontweight='bold')
        
        # Add correlation
        corr = df.loc[mask, ['VIX_CLOSE', 'BAMLH0A0HYM2']].corr().iloc[0, 1]
        ax4.text(0.95, 0.95, f'Correlation: {corr:.3f}',
                transform=ax4.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'economic_variables.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'economic_variables.png'}")


def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the report."""
    stats = {}
    
    # Date range
    stats['start_date'] = df.index.min().date()
    stats['end_date'] = df.index.max().date()
    stats['n_days'] = len(df)
    
    # VIX statistics
    if 'VIX_CLOSE' in df.columns:
        vix = df['VIX_CLOSE'].dropna()
        stats['vix_mean'] = vix.mean()
        stats['vix_std'] = vix.std()
        stats['vix_min'] = vix.min()
        stats['vix_max'] = vix.max()
        stats['vix_median'] = vix.median()
        stats['vix_skew'] = vix.skew()
        stats['vix_kurtosis'] = vix.kurtosis()
        stats['vix_n'] = len(vix)
    
    # Regime statistics
    if 'vix_regime' in df.columns:
        regime_counts = df['vix_regime'].value_counts(normalize=True)
        stats['low_vol_pct'] = regime_counts.get(0, 0) * 100
        stats['normal_vol_pct'] = regime_counts.get(1, 0) * 100
        stats['high_vol_pct'] = regime_counts.get(2, 0) * 100
    
    # Returns statistics
    if 'GSPC_return' in df.columns:
        ret = df['GSPC_return'].dropna()
        stats['return_mean_ann'] = ret.mean() * 252 * 100
        stats['return_std_ann'] = ret.std() * np.sqrt(252) * 100
        stats['return_sharpe'] = (ret.mean() * 252) / (ret.std() * np.sqrt(252))
        stats['return_min'] = ret.min() * 100
        stats['return_max'] = ret.max() * 100
    
    # Term structure
    if 'VX_Slope_1_2' in df.columns:
        slopes = df['VX_Slope_1_2'].dropna()
        stats['contango_pct'] = (slopes > 0).mean() * 100  # VX2 > VX1
        stats['slope_mean'] = slopes.mean()
    
    # VIX basis contango (VX1 > VIX)
    if 'is_contango' in df.columns:
        valid_contango = df['is_contango'].dropna()
        stats['vix_basis_contango_pct'] = valid_contango.mean() * 100
        stats['vix_basis_contango_n'] = len(valid_contango)
    
    # Data quality
    stats['total_columns'] = len(df.columns)
    
    return stats


def run_eda(data_path: str = "data/processed/volatility_dataset.parquet"):
    """Run full exploratory data analysis."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Setup output directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Filter to trading days
    df_trading = filter_trading_days(df)
    
    # Data quality report
    print("\n--- Data Quality Report ---")
    quality = data_quality_report(df_trading)
    quality_path = Path("reports/data_quality.csv")
    quality.to_csv(quality_path)
    print(f"Saved data quality report to {quality_path}")
    
    # Summary statistics
    print("\n--- Summary Statistics ---")
    stats = generate_summary_statistics(df_trading)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save stats
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(Path("reports/summary_statistics.csv"), index=False)
    
    # Generate plots
    print("\n--- Generating Visualizations ---")
    plot_vix_history(df_trading, output_dir)
    plot_spx_returns(df_trading, output_dir)
    plot_vix_distribution(df_trading, output_dir)
    plot_correlation_matrix(df_trading, output_dir)
    plot_regime_analysis(df_trading, output_dir)
    plot_futures_term_structure(df_trading, output_dir)
    plot_economic_variables(df_trading, output_dir)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Figures: {output_dir}")
    print(f"  - Data quality: reports/data_quality.csv")
    print(f"  - Statistics: reports/summary_statistics.csv")
    
    return df_trading, stats


if __name__ == "__main__":
    run_eda()
