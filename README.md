# Volatility Regime Prediction via Causal Discovery

A comprehensive research infrastructure for studying volatility regimes and their predictability using causal discovery methods.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project implements a production-ready data pipeline for volatility research, combining:
- **Multi-source data collection** from CBOE, FRED, Yahoo Finance, and Alpha Vantage Premium
- **165+ engineered features** including VRP, realized volatility, term structure, IV surface, and Greeks
- **Quality-validated dataset** spanning 5,046 trading days (2006-2025)
- **NEW: 896 weekly SPY options snapshots** with IV skew, term structure, and Greeks (2008-2025)

The data pipeline is **production-ready** with checkpoint/resume, rate limiting, and comprehensive validation.

## Key Features

### Data Collection
- **CBOE**: VIX, VVIX, VIX9D, VIX3M, VIX6M, SKEW, VIX futures (VX1-VX9), Put/Call ratios
- **FRED**: Treasury yields, credit spreads, financial conditions indices
- **Yahoo Finance**: S&P 500 OHLCV
- **Alpha Vantage Premium** (✅ Collected): 896 weekly SPY options snapshots with:
  - IV Surface: ATM IV, IV Skew (25-delta, 10-delta), Term Structure
  - Greeks: Net Delta, Total Gamma, Total Vega
  - Volume: Put/Call ratios, Total Volume, Open Interest

### Computed Features
- **Variance Risk Premium (VRP)**: Forward and backward-looking, 84.1% positive historically
- **Realized Volatility**: Close-to-close and Parkinson (high-low) estimators
- **Term Structure**: VIX basis, futures slope, contango/backwardation indicators (76.5% contango)
- **Regime Indicators**: VIX percentile, z-score, categorical regime classification
- **Sentiment**: SKEW features, put/call ratio dynamics
- **Options Surface (NEW)**: IV skew, term structure slope, Greeks aggregates from Alpha Vantage

### Quality Assurance
- All computations validated against expected statistical properties
- VIX-SPX correlation: -0.81 (daily % change) ✓
- Contango frequency: 76.5% (when VX1 data available, n=3,259) ✓
- VRP positive: 84.1% (of valid observations) ✓

> **⚠️ Important**: Some features are forward-looking (`vrp_forward`, `vrp_vol_points_forward`) and cannot be used as predictors. Use only for ex-post analysis. For modeling, use backward-looking versions.

## Project Structure

```
vol-regime-prediction/
├── config/
│   └── config.yaml          # Date ranges, cache settings
├── data/
│   ├── raw/                 # Cached API responses
│   │   └── alpha_vantage/   # SPY options history (parquet)
│   ├── interim/             # Intermediate merged data
│   └── processed/           # Final dataset (parquet + csv)
├── docs/
│   └── data_implementation_status.md  # Detailed data documentation
├── reports/
│   ├── figures/             # EDA visualizations
│   └── data_report.tex      # LaTeX methodology report
├── src/
│   ├── data/
│   │   ├── base.py          # Abstract data source interface
│   │   ├── alpha_vantage.py # Alpha Vantage Premium integration
│   │   ├── fred.py          # FRED API integration
│   │   ├── cboe.py          # CBOE web scraper
│   │   ├── yfinance_source.py
│   │   └── data_manager.py  # Pipeline orchestration
│   ├── features/
│   │   └── volatility.py    # Feature engineering
│   ├── analysis/
│   │   └── eda.py           # Exploratory data analysis
│   ├── scripts/
│   │   ├── batch_alpha_vantage.py  # Batch options collection
│   │   ├── sanity_check.py         # Data validation
│   │   └── compute_features.py     # Feature computation
│   └── main.py              # Entry point
├── .env.example             # API key template
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/philippdubach/vol-regime-prediction.git
cd vol-regime-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env: Add your FRED_API_KEY (free from https://fred.stlouisfed.org)
```

### Run the Pipeline

```bash
# Collect data and compute features
python src/main.py

# Output:
# - data/processed/volatility_dataset.parquet (5,046 rows × 139 columns)
# - data/processed/volatility_dataset.csv
```

### Generate Analysis

```bash
# Run exploratory data analysis
python src/analysis/eda.py

# Compile LaTeX report
cd reports && pdflatex data_report.tex
```

## Data Sources

| Source | Variables | Coverage | Cost |
|--------|-----------|----------|------|
| CBOE | VIX, VVIX, VIX9D/3M/6M, SKEW, VX futures | 2006-present | Free |
| CBOE | Put/Call ratios (total, index, equity, VIX) | 2006-2019 | Free |
| FRED | DFF, Treasury yields, credit spreads, NFCI | 2006-present | Free |
| Yahoo | S&P 500 OHLCV | 2006-present | Free |
| **Alpha Vantage** | **SPY options IV/Greeks (26 features)** | **2008-2025 ✅** | Premium |

### Feature Summary

| Category | Features | Description |
|----------|----------|-------------|
| Realized Vol | `GSPC_log_rv_*`, `GSPC_parkinson_rv_*` | 5, 10, 21, 63, 126, 252-day windows |
| VRP | `vrp_forward`, `vrp_backward`, `vrp_vol_points` | Variance risk premium measures |
| Term Structure | `vix_basis`, `term_slope_*`, `is_contango` | Futures curve metrics |
| Regimes | `regime_*`, `vix_zscore_252`, `vix_percentile` | Volatility state classification |
| SKEW | `skew_zscore`, `skew_percentile`, `skew_vix_ratio` | Tail risk indicators |
| Sentiment | `*_pc_zscore`, `*_pc_extreme_*` | Put/call ratio features |
| **Options (✅ Collected)** | `AV_*` (26 features) | IV skew, term structure, Greeks |

## Usage Examples

### Load the Dataset

```python
import pandas as pd

# Load processed dataset
df = pd.read_parquet('data/processed/volatility_dataset.parquet')

print(f"Shape: {df.shape}")  # (5046, 139)
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

### Access Raw Data Sources

```python
from src.data.data_manager import DataManager

dm = DataManager()

# Individual data sources
vix_data = dm.get_vix_indices()
futures = dm.get_vix_futures()
economic = dm.get_economic_data()

# Full collection
full_dataset = dm.collect_all()
```

### Compute Custom Features

```python
from src.features.volatility import VolatilityFeatures

vf = VolatilityFeatures(
    rv_windows=[5, 10, 21],
    vrp_forward_window=21
)

features = vf.compute_all(raw_data)
```

## Configuration

### config/config.yaml

```yaml
data:
  start_date: "2006-01-01"
  end_date: null  # null = today
  cache:
    enabled: true
    expiry_days: 1
```

### Environment Variables

```bash
# .env
FRED_API_KEY=your_fred_api_key_here
ALPHAVANTAGE_API_KEY=your_alphavantage_key_here  # Optional: Premium features
```

### Alpha Vantage Premium Options (✅ Collected)

Full historical SPY options dataset has been collected:

| Metric | Value |
|--------|-------|
| Rows | 896 weekly observations |
| Date Range | 2008-03-07 to 2025-12-12 |
| Features | 26 columns |
| File | `data/raw/alpha_vantage/SPY_options_history.parquet` |

**Key Features**:
- `AV_ATM_IV`: ATM implied volatility
- `AV_IV_SKEW_25D`, `AV_IV_SKEW_10D`: Put-call IV skew
- `AV_IV_TERM_SLOPE`: IV term structure slope
- `AV_NET_DELTA`, `AV_TOTAL_GAMMA`, `AV_TOTAL_VEGA`: Greeks aggregates
- `AV_PUT_CALL_RATIO_VOL`, `AV_PUT_CALL_RATIO_OI`: Volume/OI ratios

**To re-collect or update**:
```bash
python src/scripts/batch_alpha_vantage.py --start 2008-01-01 --symbol SPY
```

## Data Quality

### Validation Results

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| VRP Backward Positive % | 75-85% | 84.1% | ✅ |
| VRP Forward Positive % | 75-85% | 82.0% | ✅ |
| VIX-SPX Correlation (% change) | -0.7 to -0.8 | -0.72 | ✅ |
| VIX-SPX Correlation (pt change) | ~-0.8 | -0.81 | ✅ |
| Contango Frequency | 75-80% | 76.5% | ✅ |
| VIX Mean | 18-20 | 19.46 | ✅ |
| VIX Skewness | 2-3 | 2.50 | ✅ |

### Known Limitations

1. **Forward-looking features** - `vrp_forward` and `vrp_vol_points_forward` use future RV (for analysis only)
2. **Put/Call data ends Oct 2019** - CBOE discontinued free distribution
   - ✅ **Mitigated**: Alpha Vantage provides Put/Call ratios 2008-2025
3. **Weekly economic data** - NFCI/STLFSI4 forward-filled to daily
4. **No intraday data** - Using daily bars with Parkinson estimator
5. **VX1 futures from 2013** - Full term structure limited before 2012
6. **SPY vs SPX** - Using SPY options (SPX lacks IV/Greeks in Alpha Vantage API)

## Extending the Pipeline

### Adding New Data Sources

```python
from src.data.base import BaseDataSource

class MyDataSource(BaseDataSource):
    def __init__(self, **kwargs):
        super().__init__(name="my_source", **kwargs)
    
    def get_available_series(self):
        return ["SERIES1", "SERIES2"]
    
    def _fetch_data(self, start_date, end_date):
        # Implement data fetching
        pass
    
    def _validate_data(self, df):
        # Implement validation
        pass
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{volatility_regime_prediction,
  title={Volatility Regime Prediction via Causal Discovery},
  author={Your Name},
  year={2024},
  url={https://github.com/philippdubach/vol-regime-prediction}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

