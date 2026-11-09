"""Next-week (5-session) variance: HAR with Parkinson range inputs, rescaled by its own recent track record.

HAR-PK regresses the next 5 sessions' realized variance (sum of squared daily log returns) on today's,
this week's and this month's Parkinson range variance, over the trailing 500 sessions. Raw HAR forecasts
run biased, so each ticker gets a scale factor k: the mean ratio of realized to forecast variance over
the 52 most recent weeks whose outcome is known. In a pre-registered test on 10 US large caps and ETFs
(2024-2026) this beat EWMA, GARCH(1,1) and a zero-shot foundation model on QLIKE.

Short histories fall back to a 5-session EWMA (lambda 0.94) with no rescaling, and say so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HORIZON = 5
WINDOW = 500
K_WEEKS = 52
LAMBDA = 0.94
MIN_SESSIONS = WINDOW + 22 + HORIZON * (K_WEEKS + 2)


@dataclass
class Forecast:
    var5: float          # calibrated 5-session log-return variance
    method: str          # "HAR-PK" or "EWMA (short history)"
    k: float | None      # scale factor applied to the raw forecast
    weeks: int           # weeks behind k

    @property
    def sd5(self) -> float:
        return float(np.sqrt(self.var5))


def ewma_5day(lr: np.ndarray) -> float:
    lr = lr[np.isfinite(lr)]
    s2 = np.var(lr[:60], ddof=1)
    for r in lr[60:]:
        s2 = LAMBDA * s2 + (1 - LAMBDA) * r * r
    return HORIZON * s2


def parkinson(df: pd.DataFrame) -> np.ndarray:
    return np.log(df.high.values / df.low.values) ** 2 / (4 * np.log(2))


def har_raw(df: pd.DataFrame) -> float:
    """Raw HAR-PK forecast at the last row of df (nothing after it is visible)."""
    lr = np.diff(np.log(df.close.values), prepend=np.nan)
    pk = pd.Series(parkinson(df))
    X = np.column_stack([np.ones(len(pk)), pk, pk.rolling(5).mean(), pk.rolling(22).mean()])
    y5 = pd.Series(lr ** 2).rolling(HORIZON).sum().shift(-HORIZON).values
    rows = slice(len(pk) - WINDOW, len(pk) - HORIZON)
    beta, *_ = np.linalg.lstsq(X[rows], y5[rows], rcond=None)
    f = float(X[-1] @ beta)
    return f if f > 0 else ewma_5day(lr)


def weekly_origins(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each ISO week."""
    iso = dates.isocalendar()
    last = pd.Series(dates, index=dates).groupby([iso.year.values, iso.week.values]).max()
    return pd.DatetimeIndex(sorted(last.values))


def forecast(df: pd.DataFrame) -> Forecast:
    if len(df) < MIN_SESSIONS:
        lr = np.diff(np.log(df.close.values), prepend=np.nan)
        return Forecast(ewma_5day(lr), "EWMA (short history)", None, 0)
    dates = df.index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    last = len(dates) - 1
    done = [o for o in weekly_origins(dates) if pos[o] + HORIZON <= last and pos[o] >= WINDOW + 22][-K_WEEKS:]
    lr = np.diff(np.log(df.close.values), prepend=np.nan)
    ratios = []
    for o in done:
        i = pos[o]
        realized = float(np.sum(lr[i + 1:i + 1 + HORIZON] ** 2))
        ratios.append(realized / har_raw(df.iloc[:i + 1]))
    k = float(np.mean(ratios))
    return Forecast(har_raw(df) * k, "HAR-PK", k, len(done))


def correlation(frames: dict[str, pd.DataFrame], tickers: list[str], sessions: int = 250) -> pd.DataFrame:
    """Correlation of daily log returns over the last `sessions` shared sessions."""
    closes = pd.concat({t: frames[t].close for t in tickers}, axis=1).dropna()
    lr = np.log(closes).diff().dropna().iloc[-sessions:]
    return lr.corr()
