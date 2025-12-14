"""
Alpha Vantage premium data source (options + intraday + sentiment).

Supports:
- Historical options chains with IV and Greeks (HISTORICAL_OPTIONS)
- Intraday OHLCV for realized volatility (TIME_SERIES_INTRADAY with extended history)
- Market news sentiment (NEWS_SENTIMENT)

The class returns summarized daily aggregates for options chains to avoid
storing very large raw responses. Raw chains can be returned if needed.

Rate Limit: 75 requests/minute (Premium tier)
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import requests
import pandas as pd
import numpy as np
from io import StringIO

from src.data.base import BaseDataSource, DataFetchError, DataValidationError

logger = logging.getLogger(__name__)


class AlphaVantageDataSource(BaseDataSource):
    """
    Alpha Vantage premium data source for volatility research.
    
    Premium Features Used:
    - HISTORICAL_OPTIONS: Full SPX options chain with IV/Greeks (15+ years history)
    - TIME_SERIES_INTRADAY: 5-min bars for computing realized volatility (20+ years)
    - NEWS_SENTIMENT: Market news sentiment for sentiment features
    
    Rate Limiting:
    - Configured for 75 req/min (Premium tier)
    - Automatic throttling with backoff
    - Progress tracking for batch operations
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_enabled: bool = True,
        cache_expiry_days: int = 1,
        request_timeout: int = 60,
        max_calls_per_min: int = 70,  # Conservative buffer under 75 limit
    ):
        super().__init__(
            name="alpha_vantage",
            cache_dir=cache_dir,
            cache_enabled=cache_enabled,
            cache_expiry_days=cache_expiry_days,
        )
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise DataFetchError(
                "Alpha Vantage API key missing. Set ALPHAVANTAGE_API_KEY or pass api_key."
            )

        self.timeout = request_timeout
        self.max_calls_per_min = max_calls_per_min
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "vol-regime-research/1.0",
                "Accept": "application/json",
            }
        )
        self._call_timestamps: List[float] = []
        self._request_count = 0
        self._last_progress_log = time.time()

    def get_available_series(self) -> List[str]:
        return ["HISTORICAL_OPTIONS", "TIME_SERIES_INTRADAY", "NEWS_SENTIMENT"]

    def _throttle(self) -> None:
        """Respect documented 75 req/min by keeping < max_calls_per_min."""
        now = time.time()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
        if len(self._call_timestamps) >= self.max_calls_per_min:
            sleep_for = 60 - (now - self._call_timestamps[0]) + 0.5
            logger.info(f"Rate limit: sleeping {sleep_for:.1f}s ({len(self._call_timestamps)} calls in last minute)")
            time.sleep(max(0.0, sleep_for))
        self._call_timestamps.append(time.time())
        self._request_count += 1

    def _query(
        self,
        params: Dict[str, Any],
        datatype: str = "json",
        max_retries: int = 3
    ) -> Any:
        """Execute API query with retry logic and rate limiting."""
        params = {**params, "apikey": self.api_key, "datatype": datatype}
        
        for attempt in range(max_retries):
            self._throttle()
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
            except requests.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise DataFetchError("Request timeout after retries")
            except requests.RequestException as exc:
                raise DataFetchError(f"Alpha Vantage request failed: {exc}") from exc

            if datatype == "csv":
                return response.text

            payload = response.json()
            
            # Check for API errors
            if "Error Message" in payload:
                raise DataFetchError(payload.get("Error Message", "Unknown error"))
            if "Note" in payload:
                # Rate limit warning - wait and retry
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit, waiting 60s...")
                    time.sleep(60)
                    continue
                raise DataFetchError(payload.get("Note"))
            if "Information" in payload:
                # API info message (sometimes indicates no data)
                info = payload.get("Information", "")
                if "Invalid API call" in info or "premium" in info.lower():
                    raise DataFetchError(info)
                logger.debug(f"API info: {info}")
            
            return payload
        
        raise DataFetchError("Max retries exceeded")

    def get_request_stats(self) -> Dict[str, Any]:
        """Return request statistics for monitoring."""
        return {
            "total_requests": self._request_count,
            "requests_last_minute": len(self._call_timestamps),
            "max_per_minute": self.max_calls_per_min,
        }

    # ------------------------------------------------------------------
    # Options - Enhanced with more features for ML
    # ------------------------------------------------------------------
    def fetch_historical_options(
        self,
        date: datetime,
        symbol: str = "SPX",
        require_greeks: bool = True,  # Kept for API compat but not used
        raw: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical options chain for a specific date.

        Note: The HISTORICAL_OPTIONS endpoint always returns IV and Greeks
        when available - there is no require_greeks parameter unlike REALTIME_OPTIONS.

        Returns a one-row DataFrame indexed by date with comprehensive metrics:
        - Total put/call volume & open interest
        - Put/call ratios (volume & OI)
        - Mean/median IV by side
        - ATM IV (closest to ATM by delta or strike)
        - IV skew metrics (OTM put vs OTM call IV)
        - IV term structure (near vs far expiry)
        - Greeks aggregates (net delta, gamma)
        - Underlying price
        """
        params = {
            "function": "HISTORICAL_OPTIONS",
            "symbol": symbol,
            "date": date.strftime("%Y-%m-%d"),
        }
        # Note: require_greeks is NOT a valid parameter for HISTORICAL_OPTIONS
        # (only REALTIME_OPTIONS supports it). Historical always returns Greeks.

        payload = self._query(params, datatype="json")
        df_raw = self._parse_options_payload(payload)
        if df_raw.empty:
            raise DataFetchError(f"No options data returned for {symbol} on {date.date()}")

        if raw:
            return df_raw

        summary = self._summarize_options_enhanced(df_raw, date)
        self.validate(summary)
        return summary

    def _parse_options_payload(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Parse Alpha Vantage HISTORICAL_OPTIONS response.
        
        The API returns a flat list of options contracts, each with fields:
        - contractID, symbol, expiration, strike, type (call/put)
        - last, mark, bid, bid_size, ask, ask_size, volume, open_interest
        - implied_volatility, delta, gamma, theta, vega, rho
        """
        options: List[Dict[str, Any]] = []
        
        # Get underlying price if provided (may be in root or absent)
        underlying_price = self._safe_float(
            payload.get("underlying_price")
            or payload.get("underlyingPrice")
            or payload.get("underlying_last_price")
        )

        # The "data" field is a flat list of option contracts
        data_list = payload.get("data", [])
        
        for opt in data_list:
            # Each option is a flat dict with all fields at top level
            options.append(
                {
                    "type": opt.get("type", "").lower(),  # "call" or "put"
                    "expiration": opt.get("expiration"),
                    "strike": self._safe_float(opt.get("strike")),
                    "iv": self._safe_float(opt.get("implied_volatility")),
                    "delta": self._safe_float(opt.get("delta")),
                    "gamma": self._safe_float(opt.get("gamma")),
                    "theta": self._safe_float(opt.get("theta")),
                    "vega": self._safe_float(opt.get("vega")),
                    "rho": self._safe_float(opt.get("rho")),
                    "open_interest": self._safe_float(opt.get("open_interest")),
                    "volume": self._safe_float(opt.get("volume")),
                    "last": self._safe_float(opt.get("last")),
                    "bid": self._safe_float(opt.get("bid")),
                    "ask": self._safe_float(opt.get("ask")),
                    "underlying_price": underlying_price,
                }
            )

        if not options:
            return pd.DataFrame()

        df = pd.DataFrame(options)
        df["expiration"] = pd.to_datetime(df["expiration"], errors="coerce")
        return df

    def _select_atm_iv(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Pick an ATM option by delta, fallback to closest strike."""
        underlying = df["underlying_price"].dropna().median() if not df.empty else np.nan
        if not df.empty:
            calls = df[df["type"] == "call"].copy()
            puts = df[df["type"] == "put"].copy()
            for subset, target_delta in ((calls, 0.5), (puts, -0.5)):
                subset["delta_score"] = np.abs(subset["delta"] - target_delta)
                subset["strike_score"] = np.abs(subset["strike"] - underlying) if not np.isnan(underlying) else np.inf
            candidates = pd.concat([calls, puts], axis=0)
            candidates["overall_score"] = candidates[["delta_score", "strike_score"]].min(axis=1)
            candidates = candidates.sort_values(["overall_score"])
            if not candidates.empty:
                row = candidates.iloc[0]
                return (
                    self._safe_float(row.get("iv")),
                    self._safe_float(row.get("delta")),
                    self._safe_float(row.get("strike")),
                )
        return (np.nan, np.nan, np.nan)

    def _summarize_options_enhanced(self, df: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """
        Create comprehensive options summary with features for ML.
        
        Features computed:
        - Volume & OI metrics (put/call totals and ratios)
        - IV metrics (mean, median, ATM by side)
        - IV skew (25-delta put vs 25-delta call)
        - IV term structure (near-term vs far-term)
        - Greeks aggregates (net delta, total gamma, total vega)
        - Moneyness distribution
        """
        calls = df[df["type"] == "call"]
        puts = df[df["type"] == "put"]
        underlying_price = df["underlying_price"].dropna().median() if not df.empty else np.nan

        # Basic volume/OI metrics
        call_vol = calls["volume"].sum()
        put_vol = puts["volume"].sum()
        call_oi = calls["open_interest"].sum()
        put_oi = puts["open_interest"].sum()

        # IV metrics
        call_iv_mean = calls["iv"].mean()
        call_iv_median = calls["iv"].median()
        put_iv_mean = puts["iv"].mean()
        put_iv_median = puts["iv"].median()
        atm_iv, atm_delta, atm_strike = self._select_atm_iv(df)

        # IV Skew: 25-delta put IV minus 25-delta call IV
        # Positive skew = puts more expensive (fear)
        iv_skew_25d = self._compute_iv_skew(calls, puts, delta_target=0.25)
        iv_skew_10d = self._compute_iv_skew(calls, puts, delta_target=0.10)

        # IV Term structure: near-term (< 30d) vs far-term (30-90d)
        iv_term_near, iv_term_far = self._compute_iv_term_structure(df, date)
        iv_term_slope = iv_term_far - iv_term_near if not np.isnan(iv_term_near) and not np.isnan(iv_term_far) else np.nan

        # Greeks aggregates (volume-weighted where applicable)
        net_delta = self._compute_net_delta(df)
        total_gamma = df["gamma"].sum() if "gamma" in df else np.nan
        total_vega = df["vega"].sum() if "vega" in df else np.nan
        
        # Volume-weighted average IV
        vw_iv = self._compute_volume_weighted_iv(df)

        # Moneyness distribution (% OTM options by volume)
        otm_vol_ratio = self._compute_otm_volume_ratio(df, underlying_price)

        out = pd.DataFrame(
            {
                # Volume & OI
                "AV_CALL_VOLUME": call_vol,
                "AV_PUT_VOLUME": put_vol,
                "AV_PUT_CALL_RATIO_VOL": put_vol / call_vol if call_vol > 0 else np.nan,
                "AV_CALL_OI": call_oi,
                "AV_PUT_OI": put_oi,
                "AV_PUT_CALL_RATIO_OI": put_oi / call_oi if call_oi > 0 else np.nan,
                "AV_TOTAL_VOLUME": call_vol + put_vol,
                "AV_TOTAL_OI": call_oi + put_oi,
                
                # IV metrics
                "AV_CALL_IV_MEAN": call_iv_mean,
                "AV_CALL_IV_MEDIAN": call_iv_median,
                "AV_PUT_IV_MEAN": put_iv_mean,
                "AV_PUT_IV_MEDIAN": put_iv_median,
                "AV_ATM_IV": atm_iv,
                "AV_ATM_STRIKE": atm_strike,
                "AV_VW_IV": vw_iv,  # Volume-weighted IV
                
                # IV Skew (put premium over call - fear gauge)
                "AV_IV_SKEW_25D": iv_skew_25d,
                "AV_IV_SKEW_10D": iv_skew_10d,
                
                # IV Term structure
                "AV_IV_TERM_NEAR": iv_term_near,  # <30 days
                "AV_IV_TERM_FAR": iv_term_far,    # 30-90 days
                "AV_IV_TERM_SLOPE": iv_term_slope, # Contango/backwardation in IV
                
                # Greeks
                "AV_NET_DELTA": net_delta,
                "AV_TOTAL_GAMMA": total_gamma,
                "AV_TOTAL_VEGA": total_vega,
                
                # Moneyness
                "AV_OTM_VOL_RATIO": otm_vol_ratio,  # % volume in OTM options
                
                # Underlying
                "AV_UNDERLYING": underlying_price,
                "AV_CONTRACT_COUNT": len(df),
            },
            index=[pd.to_datetime(date.date())],
        )
        out.index.name = "date"
        return out

    def _compute_iv_skew(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        delta_target: float = 0.25
    ) -> float:
        """Compute IV skew: put IV at delta vs call IV at delta."""
        # Find put closest to -delta_target
        if not puts.empty:
            puts_sorted = puts.copy()
            puts_sorted["delta_diff"] = np.abs(puts_sorted["delta"].abs() - delta_target)
            put_iv = puts_sorted.nsmallest(3, "delta_diff")["iv"].mean()
        else:
            put_iv = np.nan
            
        # Find call closest to +delta_target
        if not calls.empty:
            calls_sorted = calls.copy()
            calls_sorted["delta_diff"] = np.abs(calls_sorted["delta"] - delta_target)
            call_iv = calls_sorted.nsmallest(3, "delta_diff")["iv"].mean()
        else:
            call_iv = np.nan
            
        if not np.isnan(put_iv) and not np.isnan(call_iv):
            return put_iv - call_iv
        return np.nan

    def _compute_iv_term_structure(
        self,
        df: pd.DataFrame,
        date: datetime
    ) -> Tuple[float, float]:
        """Compute near-term (<30d) and far-term (30-90d) average IV."""
        if df.empty or "expiration" not in df.columns:
            return np.nan, np.nan
            
        df = df.copy()
        df["dte"] = (df["expiration"] - pd.Timestamp(date)).dt.days
        
        near = df[(df["dte"] > 0) & (df["dte"] <= 30)]
        far = df[(df["dte"] > 30) & (df["dte"] <= 90)]
        
        near_iv = near["iv"].mean() if not near.empty else np.nan
        far_iv = far["iv"].mean() if not far.empty else np.nan
        
        return near_iv, far_iv

    def _compute_net_delta(self, df: pd.DataFrame) -> float:
        """Compute net delta (calls positive, puts negative) weighted by OI."""
        if df.empty:
            return np.nan
        calls = df[df["type"] == "call"]
        puts = df[df["type"] == "put"]
        
        call_delta = (calls["delta"] * calls["open_interest"]).sum()
        put_delta = (puts["delta"] * puts["open_interest"]).sum()
        total_oi = df["open_interest"].sum()
        
        if total_oi > 0:
            return (call_delta + put_delta) / total_oi
        return np.nan

    def _compute_volume_weighted_iv(self, df: pd.DataFrame) -> float:
        """Compute volume-weighted average IV."""
        if df.empty or df["volume"].sum() == 0:
            return np.nan
        return (df["iv"] * df["volume"]).sum() / df["volume"].sum()

    def _compute_otm_volume_ratio(self, df: pd.DataFrame, underlying: float) -> float:
        """Compute ratio of OTM option volume to total volume."""
        if df.empty or np.isnan(underlying) or df["volume"].sum() == 0:
            return np.nan
            
        df = df.copy()
        # OTM: calls with strike > underlying, puts with strike < underlying
        df["is_otm"] = (
            ((df["type"] == "call") & (df["strike"] > underlying)) |
            ((df["type"] == "put") & (df["strike"] < underlying))
        )
        
        return df.loc[df["is_otm"], "volume"].sum() / df["volume"].sum()

    # ------------------------------------------------------------------
    # Intraday data for Realized Volatility
    # ------------------------------------------------------------------
    def fetch_intraday_month(
        self,
        symbol: str,
        month: str,
        interval: str = "5min",
        adjusted: bool = True,
        extended_hours: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV for a specific month.
        
        Args:
            symbol: Stock symbol (e.g., "SPY")
            month: Month in YYYY-MM format (e.g., "2024-01")
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            adjusted: Whether to adjust for splits/dividends
            extended_hours: Include pre/post market hours
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "month": month,
            "outputsize": "full",
            "adjusted": str(adjusted).lower(),
            "extended_hours": str(extended_hours).lower(),
        }

        text = self._query(params, datatype="csv")
        if not text or text.strip() == "":
            raise DataFetchError(f"No intraday data for {symbol} {month}")
            
        df = pd.read_csv(StringIO(text))
        if df.empty:
            raise DataFetchError(f"Empty intraday data for {symbol} {month}")
            
        # Parse timestamp column (may be named "timestamp" or "time")
        time_col = "timestamp" if "timestamp" in df.columns else "time"
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
        df = df.sort_index()
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        return df

    def compute_realized_vol_from_intraday(
        self,
        df: pd.DataFrame,
        annualize: bool = True
    ) -> pd.Series:
        """
        Compute realized volatility from intraday returns.
        
        Uses 5-min squared returns summed per day (RV estimator).
        
        Args:
            df: Intraday OHLCV DataFrame
            annualize: Whether to annualize (multiply by sqrt(252))
            
        Returns:
            Series of daily realized volatility
        """
        if df.empty:
            return pd.Series(dtype=float)
            
        # Compute log returns
        df = df.copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["squared_return"] = df["log_return"] ** 2
        
        # Sum squared returns per day
        daily_rv = df.groupby(df.index.date)["squared_return"].sum()
        daily_rv = np.sqrt(daily_rv)
        
        if annualize:
            # Annualize: multiply by sqrt(252)
            daily_rv = daily_rv * np.sqrt(252)
        
        daily_rv.index = pd.to_datetime(daily_rv.index)
        daily_rv.name = "realized_vol_intraday"
        
        return daily_rv

    # ------------------------------------------------------------------
    # News Sentiment
    # ------------------------------------------------------------------
    def fetch_news_sentiment(
        self,
        tickers: Optional[str] = None,
        topics: Optional[str] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        sort: str = "LATEST",
        limit: int = 50,
    ) -> pd.DataFrame:
        """
        Fetch market news sentiment data.
        
        Args:
            tickers: Comma-separated tickers (e.g., "SPY,^VIX")
            topics: Comma-separated topics (e.g., "financial_markets,economy_macro")
            time_from: Start time in YYYYMMDDTHHMM format
            time_to: End time in YYYYMMDDTHHMM format
            sort: LATEST, EARLIEST, or RELEVANCE
            limit: Max results (up to 1000)
            
        Returns:
            DataFrame with news articles and sentiment scores
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "sort": sort,
            "limit": limit,
        }
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        payload = self._query(params, datatype="json")
        
        articles = payload.get("feed", [])
        if not articles:
            return pd.DataFrame()
        
        records = []
        for article in articles:
            record = {
                "title": article.get("title"),
                "url": article.get("url"),
                "time_published": article.get("time_published"),
                "source": article.get("source"),
                "overall_sentiment_score": self._safe_float(article.get("overall_sentiment_score")),
                "overall_sentiment_label": article.get("overall_sentiment_label"),
            }
            
            # Extract ticker-specific sentiment if available
            ticker_sentiment = article.get("ticker_sentiment", [])
            if ticker_sentiment and tickers:
                primary_ticker = tickers.split(",")[0]
                for ts in ticker_sentiment:
                    if ts.get("ticker") == primary_ticker:
                        record["ticker_sentiment_score"] = self._safe_float(ts.get("ticker_sentiment_score"))
                        record["ticker_relevance"] = self._safe_float(ts.get("relevance_score"))
                        break
            
            records.append(record)
        
        df = pd.DataFrame(records)
        if "time_published" in df.columns:
            df["time_published"] = pd.to_datetime(df["time_published"], format="%Y%m%dT%H%M%S", errors="coerce")
            df = df.set_index("time_published")
        
        return df

    def aggregate_daily_sentiment(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment to daily level.
        
        Args:
            df: News sentiment DataFrame (from fetch_news_sentiment)
            
        Returns:
            Daily aggregated sentiment metrics
        """
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        df["date"] = df.index.date
        
        daily = df.groupby("date").agg({
            "overall_sentiment_score": ["mean", "std", "count"],
            "ticker_sentiment_score": ["mean", "std"] if "ticker_sentiment_score" in df.columns else [],
        })
        
        # Flatten column names
        daily.columns = ["_".join(col).strip("_") for col in daily.columns.values]
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "date"
        
        # Rename for clarity
        rename_map = {
            "overall_sentiment_score_mean": "AV_SENTIMENT_MEAN",
            "overall_sentiment_score_std": "AV_SENTIMENT_STD",
            "overall_sentiment_score_count": "AV_NEWS_COUNT",
        }
        daily = daily.rename(columns=rename_map)
        
        return daily

    # ------------------------------------------------------------------
    # Base interface
    # ------------------------------------------------------------------
    def fetch(self, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Not implemented - use specific fetch methods."""
        raise NotImplementedError("Use fetch_historical_options, fetch_intraday_month, or fetch_news_sentiment")

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame output."""
        if df.empty:
            raise DataValidationError("Alpha Vantage DataFrame is empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError("Index is not DatetimeIndex")
        return True

    @staticmethod
    def _safe_float(val: Any) -> float:
        """Safely convert value to float, returning NaN on failure."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan
