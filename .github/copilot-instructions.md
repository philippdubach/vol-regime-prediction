# Copilot Instructions for Volatility Regime Prediction Project

## Project Context

This is a quantitative finance research project focused on **volatility regime prediction** using causal discovery methods. The codebase provides infrastructure for:

1. **Data Collection**: Multi-source pipeline (CBOE, FRED, Yahoo Finance)
2. **Feature Engineering**: Volatility measures, term structure, sentiment indicators
3. **Research Support**: Quality-validated dataset for ML/causal analysis

## Key Technical Details

### Language & Environment
- **Python 3.9+** with virtual environment at `./venv`
- **Run commands** with: `./venv/bin/python` (not system Python)
- **Key packages**: pandas, numpy, scipy, requests, fredapi, yfinance, beautifulsoup4

### Architecture Patterns

#### Data Sources (src/data/)
All data sources inherit from `BaseDataSource` (abstract class):
```python
class MySource(BaseDataSource):
    def get_available_series(self) -> List[str]: ...
    def _fetch_data(self, start, end) -> pd.DataFrame: ...
    def _validate_data(self, df) -> pd.DataFrame: ...
```

Built-in caching (file-based, 1-day expiry) via base class.

#### Feature Engineering (src/features/)
- `VolatilityFeatures` class with composable methods
- Each method takes DataFrame, returns DataFrame with new columns
- Rolling windows use `min_periods` for robustness with gaps

### Data Conventions

| Convention | Details |
|------------|---------|
| Index | DatetimeIndex (trading days only) |
| VIX units | Percentage points (e.g., 15.0 = 15%) |
| Returns | Log returns (`np.log(p_t/p_{t-1})`) |
| Realized Vol | Annualized (×√252), in percentage |
| VRP | Variance units ((vol/100)²) |

### Important File Locations

| Purpose | Path |
|---------|------|
| Main entry | `src/main.py` |
| Data manager | `src/data/data_manager.py` |
| Feature engineering | `src/features/volatility.py` |
| Config | `config/config.yaml` |
| Output data | `data/processed/volatility_dataset.parquet` |
| API keys | `.env` (FRED_API_KEY required) |

### Key Computed Features

```python
# Variance Risk Premium - BACKWARD (real-time available)
vrp_backward = (vix/100)² - (realized_vol/100)²  # Use for modeling

# Variance Risk Premium - FORWARD (look-ahead! analysis only)
vrp_forward = (vix/100)² - (realized_vol/100)².shift(-21)  # DO NOT use as predictor

# Realized Volatility (21-day)
rv_21 = returns.rolling(21).std() * √252 * 100

# Parkinson Volatility
parkinson = √(Σ(ln(H/L))² / (4*ln(2)*n)) * √252 * 100

# Term Structure
vix_basis = VX1 - VIX  # Positive = contango
is_contango = (vix_basis > 0).astype(int)

# VIX Percentile (rolling, no look-ahead)
vix_percentile = vix.expanding(min_periods=252).apply(
    lambda x: (x.iloc[-1] <= x).mean()
)
```

**⚠️ CRITICAL**: `vrp_forward` and `vrp_vol_points_forward` use future realized volatility and CANNOT be used as predictors in ML models. Use `vrp_backward` for real-time available features.

### Expected Statistical Properties

When validating data/features, check:
- **VRP backward positive**: ~84.1% of observations
- **VRP forward positive**: ~82% of observations  
- **VIX-SPX correlation** (daily % change): ~ -0.81
- **Contango frequency**: ~76.5% (when VX1 data available, n=3,259)
- **VIX**: Mean ~19.5, skewness ~2.5
- **is_contango**: Preserves NaN when VX1 unavailable (1,787 NaN)

### Common Tasks

#### Add New Data Source
1. Create class in `src/data/` inheriting `BaseDataSource`
2. Implement `get_available_series()`, `_fetch_data()`, `_validate_data()`
3. Register in `DataManager.__init__()`
4. Add to `collect_all()` merge

#### Add New Feature
1. Add method to `VolatilityFeatures` class
2. Use `min_periods` in rolling calculations
3. Call from `compute_all()`
4. Update documentation

#### Run Pipeline
```bash
./venv/bin/python src/main.py
```

### Code Style

- Type hints on all function signatures
- Docstrings with Args/Returns sections
- Logging via `logger = logging.getLogger(__name__)`
- Validation warnings, not hard failures on data issues

### Known Gotchas

1. **Forward-looking features** - `vrp_forward`, `vrp_vol_points_forward` use future data (ex-post analysis only)
2. **Put/Call data ends Oct 2019** - historical only, many NaN after
3. **Weekly FRED data** (NFCI) - appears as NaN on non-Friday dates
4. **VIX futures term structure** - VX7-VX9 have sparse coverage; VX1 from 2013+
5. **Parkinson RV** - requires both High and Low; NaN handling important
6. **is_contango column** - includes NaN periods, use `df[df['VX1'].notna()]` for true frequency

### Testing

```bash
# Run feature computation standalone
./venv/bin/python src/scripts/compute_features.py

# Run EDA
./venv/bin/python src/analysis/eda.py
```

## Response Guidelines

When helping with this project:

1. **Validate data statistically** - Check means, correlations, % positive
2. **Use proper finance conventions** - Annualization, log returns
3. **Handle missing data** - Use `min_periods`, don't fail silently
4. **Document assumptions** - Especially for computed features
5. **Reference authoritative sources** - Academic papers for methods

## Domain Knowledge

### VIX & Term Structure
- VIX measures 30-day implied volatility from SPX options
- Contango (VIX < VX1 < VX2): Normal state, ~80% of time
- Backwardation: Stress indicator, short-term fear elevated

### Variance Risk Premium (VRP)
- VRP = IV² - RV² (realized variance)
- Typically positive due to risk aversion
- Negative during volatility shocks

### SKEW Index
- Measures S&P 500 tail risk
- 100 = normal distribution
- \>130 = elevated crash probability perception

### Regime Classification
| VIX Level | Regime | Typical Frequency |
|-----------|--------|-------------------|
| < 15 | Low | ~35% |
| 15-20 | Medium | ~30% |
| 20-25 | Elevated | ~17% |
| 25-35 | High | ~13% |
| ≥ 35 | Crisis | ~5% |
