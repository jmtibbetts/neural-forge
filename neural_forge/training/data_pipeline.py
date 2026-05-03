"""
NeuralForge - Financial Data Pipeline
Downloads OHLCV data via yfinance, computes indicators, builds
sliding-window sequences labeled BUY / HOLD / SELL.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# ─── Indicator helpers ──────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False).mean()
    rs    = avg_g / (avg_l + 1e-9)
    return 100 - (100 / (1 + rs))

def _macd(close: pd.Series):
    fast = _ema(close, 12)
    slow = _ema(close, 26)
    macd = fast - slow
    sig  = _ema(macd, 9)
    hist = macd - sig
    return macd, sig, hist

def _bollinger(close: pd.Series, window: int = 20):
    mid  = close.rolling(window).mean()
    std  = close.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    pct_b = (close - lower) / (upper - lower + 1e-9)
    return mid, upper, lower, pct_b

def _atr(high, low, close, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    df["rsi"]       = _rsi(c) / 100.0          # normalised 0-1
    macd, sig, hist = _macd(c)
    df["macd"]      = macd / c                  # pct of price
    df["macd_sig"]  = sig  / c
    df["macd_hist"] = hist / c
    _, _, _, df["pct_b"] = _bollinger(c)
    df["atr_pct"]   = _atr(h, l, c) / c        # ATR as % of price

    # Price momentum
    df["ret_1"]  = c.pct_change(1)
    df["ret_5"]  = c.pct_change(5)
    df["ret_20"] = c.pct_change(20)

    # Volume
    df["vol_ma20"] = v.rolling(20).mean()
    df["vol_ratio"] = v / (df["vol_ma20"] + 1e-9)

    # Candle body
    df["body"]     = (c - df["Open"]) / (h - l + 1e-9)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "ret_1", "ret_5", "ret_20",
    "rsi", "macd", "macd_sig", "macd_hist",
    "pct_b", "atr_pct",
    "vol_ratio", "body",
]

# Add close normalised later => 12 features total if we include close_norm
ALL_FEATURES = FEATURE_COLS  # 11 features


def label_signals(close: pd.Series, forward: int = 5, thresh: float = 0.015):
    """
    Label each candle:
        BUY  (0) : next `forward` bars return > +thresh
        SELL (2) : next `forward` bars return < -thresh
        HOLD (1) : otherwise
    """
    fwd_ret = close.shift(-forward) / close - 1
    labels  = np.where(fwd_ret > thresh, 0,
               np.where(fwd_ret < -thresh, 2, 1))
    return pd.Series(labels, index=close.index)


# ─── Dataset ────────────────────────────────────────────────

class FinancialDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequences(features: np.ndarray, labels: np.ndarray, window: int = 30):
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i - window: i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ─── Main loader ────────────────────────────────────────────

def load_financial_data(
    ticker:      str   = "SPY",
    period:      str   = "5y",
    window:      int   = 30,
    forward:     int   = 5,
    thresh:      float = 0.015,
    batch_size:  int   = 64,
    val_split:   float = 0.15,
    test_split:  float = 0.10,
):
    """
    Downloads data, computes features and labels, returns
    (train_loader, val_loader, test_loader, feature_dim, class_weights)
    """
    if not HAS_YF:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    print(f"  Downloading {ticker} ({period})...")
    raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = build_features(raw.copy())

    feat_arr  = df[ALL_FEATURES].values.astype(np.float32)
    label_arr = label_signals(df["Close"], forward=forward, thresh=thresh).values

    # Clip label_arr to same length as feat_arr
    label_arr = label_arr[:len(feat_arr)]

    # Normalize features (z-score per feature)
    mean = feat_arr.mean(axis=0)
    std  = feat_arr.std(axis=0) + 1e-9
    feat_arr = (feat_arr - mean) / std

    X, y = build_sequences(feat_arr, label_arr, window=window)

    # Remove any HOLD-labelled rows at the tail where forward window is invalid
    valid = y != -1  # sentinel (shouldn't exist but safety check)
    X, y = X[valid], y[valid]

    # Split: train / val / test  (chronological — no shuffling)
    n      = len(X)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)
    n_train = n - n_val - n_test

    X_train, y_train = X[:n_train],               y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val],  y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],         y[n_train+n_val:]

    print(f"  Sequences  : {n} total | {n_train} train | {n_val} val | {n_test} test")
    print(f"  Features   : {X.shape[2]}")
    dist = np.bincount(y_train, minlength=3)
    print(f"  Label dist : BUY={dist[0]} HOLD={dist[1]} SELL={dist[2]}")

    # Class weights (inverse frequency) for imbalanced labels
    freq   = dist / dist.sum()
    weights = torch.tensor(1.0 / (freq + 1e-9), dtype=torch.float32)
    weights = weights / weights.sum()

    train_ds = FinancialDataset(X_train, y_train)
    val_ds   = FinancialDataset(X_val,   y_val)
    test_ds  = FinancialDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X.shape[2], weights
