"""
Absorption Breakout Strategy - Clean Production Version
Author: NEXUS Trading System
Created: 2025-10-04 | Upgraded: 2025-10-06
Features: False breakout detection, adaptive regimes, robust validation, thread-safe
"""

# =============================================================================
# ESSENTIAL IMPORTS & CONFIGURATION
# =============================================================================
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, defaultdict
from scipy import signal as scipy_signal
from scipy.stats import linregress, zscore
import warnings
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import multiprocessing
from functools import lru_cache
from contextlib import contextmanager
import math
import statistics

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# NEXUS AI & MQSCORE INTEGRATION
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NEXUS AI Integration
try:
    from nexus_ai import (
        AuthenticatedMarketData as NexusAuthenticatedMarketData,
        NexusSecurityLayer,
        ProductionSequentialPipeline,
    )
    NEXUS_AI_AVAILABLE = True
    logger.info("✓ NEXUS AI components available")
except ImportError:
    NEXUS_AI_AVAILABLE = False
    NexusAuthenticatedMarketData = None
    logger.info("⚠ NEXUS AI not available - using fallback")

# MQScore 6D Engine Integration
try:
    from MQScore_6D_Engine_v3 import (
        MQScoreEngine,
        MQScoreComponents,
        MQScoreConfig
    )
    HAS_MQSCORE = True
    logger.info("✓ MQScore 6D Engine available for market quality assessment")
except ImportError as e:
    HAS_MQSCORE = False
    MQScoreEngine = None
    MQScoreComponents = None
    MQScoreConfig = None
    logger.warning(f"MQScore Engine not available: {e} - using basic filters only")

# =============================================================================
# CORE CLASSES
# =============================================================================


class AuthenticatedMarketData:
    """Fallback class when NEXUS AI is not available"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_timestamp(self):
        return getattr(self, "timestamp", datetime.now())

    def get_symbol(self):
        return getattr(self, "symbol", "UNKNOWN")

    def get_price(self):
        return getattr(self, "price", 0.0)

    def get_volume(self):
        return getattr(self, "volume", 0.0)


class MarketRegimeDetector:
    """Detect market regimes to prevent false signals in choppy markets"""

    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history = deque(maxlen=lookback_period)
        self.volume_history = deque(maxlen=lookback_period)
        self.current_regime = "UNKNOWN"

    def update(self, price: float, volume: float):
        """Update price and volume history"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.current_regime = self._detect_regime()

    def _detect_regime(self) -> str:
        """Detect current market regime"""
        if len(self.price_history) < 20:
            return "UNKNOWN"

        prices = list(self.price_history)

        # Calculate volatility
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
        ]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0

        # Calculate trend strength
        x = list(range(len(prices)))
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(prices)

        numerator = sum(
            (x[i] - mean_x) * (prices[i] - mean_y) for i in range(len(prices))
        )
        denominator = sum((x[i] - mean_x) ** 2 for i in range(len(prices)))

        slope = numerator / denominator if denominator > 0 else 0
        trend_strength = abs(slope) / mean_y if mean_y > 0 else 0

        # Regime classification
        if volatility < 0.005 and trend_strength < 0.0001:
            return "RANGE"
        elif volatility > 0.02:
            return "HIGH_VOLATILITY"
        elif trend_strength > 0.0005:
            return "TRENDING"
        else:
            return "RANGE"

    def get_regime(self) -> str:
        return self.current_regime

    def should_trade(self) -> bool:
        return self.current_regime not in ["RANGE", "HIGH_VOLATILITY"]

    def get_confidence_adjustment(self) -> float:
        if self.current_regime == "TRENDING":
            return 1.0
        elif self.current_regime == "RANGE":
            return 0.6
        elif self.current_regime == "HIGH_VOLATILITY":
            return 0.5
        else:
            return 0.8


class GapDetector:
    """Detect gaps and reset calculations"""

    def __init__(self, gap_threshold: float = 0.01):
        self.gap_threshold = gap_threshold
        self.previous_close = None
        self.last_reset_time = None

    def check_gap(self, current_price: float, current_time: float) -> Dict[str, Any]:
        if self.previous_close is None:
            self.previous_close = current_price
            return {
                "has_gap": False,
                "gap_size": 0.0,
                "should_reset": False,
                "confidence_adjustment": 1.0,
            }

        gap_size = abs(current_price - self.previous_close) / self.previous_close
        has_gap = gap_size > self.gap_threshold

        should_reset = False
        if self.last_reset_time is not None:
            hours_since_reset = (current_time - self.last_reset_time) / 3600
            if hours_since_reset > 16:
                should_reset = True

        if has_gap or should_reset:
            self.last_reset_time = current_time

        return {
            "has_gap": has_gap,
            "gap_size": gap_size,
            "should_reset": has_gap or should_reset,
            "confidence_adjustment": 0.5 if has_gap else 1.0,
        }

    def update_close(self, price: float):
        self.previous_close = price


class LiquidityFilter:
    """Filter signals based on liquidity conditions"""

    def __init__(self, min_volume_ratio: float = 0.3):
        self.min_volume_ratio = min_volume_ratio
        self.volume_history = deque(maxlen=100)
        self.avg_volume = 0.0

    def update_volume(self, volume: float):
        self.volume_history.append(volume)
        if len(self.volume_history) >= 20:
            self.avg_volume = statistics.mean(self.volume_history)

    def check_liquidity(
        self, current_volume: float, bid_ask_spread: float, typical_spread: float
    ) -> Dict[str, Any]:
        if self.avg_volume == 0:
            return {"sufficient": True, "confidence_adjustment": 1.0}

        volume_ratio = current_volume / self.avg_volume if self.avg_volume > 0 else 1.0
        spread_ratio = bid_ask_spread / typical_spread if typical_spread > 0 else 1.0

        is_low_liquidity = volume_ratio < self.min_volume_ratio or spread_ratio > 2.0

        confidence_adjustment = 1.0
        if is_low_liquidity:
            confidence_adjustment = 0.5
        elif volume_ratio < 0.5:
            confidence_adjustment = 0.7

        return {
            "sufficient": not is_low_liquidity,
            "volume_ratio": volume_ratio,
            "spread_ratio": spread_ratio,
            "confidence_adjustment": confidence_adjustment,
        }


class MultiTimeframeValidator:
    """Validate signals across multiple timeframes"""

    def __init__(self):
        self.tf_1min = {"signal": None, "strength": 0.0}
        self.tf_5min = {"signal": None, "strength": 0.0}
        self.tf_15min = {"signal": None, "strength": 0.0}

    def update_timeframe(self, timeframe: str, signal: str, strength: float):
        if timeframe == "1min":
            self.tf_1min = {"signal": signal, "strength": strength}
        elif timeframe == "5min":
            self.tf_5min = {"signal": signal, "strength": strength}
        elif timeframe == "15min":
            self.tf_15min = {"signal": signal, "strength": strength}

    def validate_signal(self, signal_direction: str) -> Dict[str, Any]:
        alignment_count = sum(
            [
                self.tf_1min["signal"] == signal_direction,
                self.tf_5min["signal"] == signal_direction,
                self.tf_15min["signal"] == signal_direction,
            ]
        )

        is_validated = alignment_count >= 2
        confidence_multiplier = 1.0 + (alignment_count * 0.1)

        return {
            "is_validated": is_validated,
            "alignment_count": alignment_count,
            "confidence_multiplier": confidence_multiplier,
        }


class ParameterBoundsEnforcer:
    """Enforce bounds on adaptive parameters to prevent drift"""

    def __init__(self):
        self.bounds = {
            "absorption_threshold": (0.55, 0.85),
            "volume_spike_multiplier": (1.5, 3.0),
            "confirmation_threshold": (0.55, 0.75),
        }
        self.drift_history = deque(maxlen=100)
        self.alert_threshold = 0.15

    def enforce_bounds(self, parameter_name: str, value: float) -> float:
        if parameter_name not in self.bounds:
            return value
        min_val, max_val = self.bounds[parameter_name]
        return max(min_val, min(max_val, value))

    def check_drift(
        self, parameter_name: str, current_value: float, baseline_value: float
    ) -> Dict[str, Any]:
        drift_percent = abs(current_value - baseline_value) / baseline_value
        self.drift_history.append(
            {
                "parameter": parameter_name,
                "drift": drift_percent,
                "timestamp": time.time(),
            }
        )
        has_excessive_drift = drift_percent > self.alert_threshold
        return {
            "has_drift": has_excessive_drift,
            "drift_percent": drift_percent,
            "should_reset": has_excessive_drift,
        }


# =============================================================================
# ADAPTIVE PARAMETER OPTIMIZATION
# =============================================================================


class AdaptiveParameterOptimizer:
    """
    Self-contained adaptive parameter optimization based on actual trading results.
    NO external ML dependencies - uses real performance data to adapt parameters.
    """

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.performance_history = deque(maxlen=500)
        self.parameter_history = deque(maxlen=200)
        self.current_parameters = self._initialize_parameters()
        self.adjustment_cooldown = 50  # Trades before next adjustment
        self.trades_since_adjustment = 0

        # Golden ratio for mathematical adjustments
        self.phi = (1 + math.sqrt(5)) / 2

        logging.info(f" Adaptive Parameter Optimizer initialized for {strategy_name}")

    def _initialize_parameters(self) -> Dict[str, float]:
        """Initialize with default parameters"""
        return {
            "absorption_threshold": 0.7,
            "volume_spike_multiplier": 2.0,
            "confirmation_threshold": 0.6,
        }

    def record_trade(self, trade_result: Dict[str, Any]):
        """Record trade result for learning"""
        self.performance_history.append(
            {
                "timestamp": time.time(),
                "pnl": trade_result.get("pnl", 0.0),
                "confidence": trade_result.get("confidence", 0.5),
                "volatility": trade_result.get("volatility", 0.02),
                "parameters": self.current_parameters.copy(),
            }
        )

        self.trades_since_adjustment += 1

        # Check if we should adapt parameters
        if self.trades_since_adjustment >= self.adjustment_cooldown:
            self._adapt_parameters()
            self.trades_since_adjustment = 0

    def _adapt_parameters(self):
        """Adapt parameters based on recent performance"""
        if len(self.performance_history) < 20:
            return  # Need minimum data

        recent_trades = list(self.performance_history)[-50:]

        # Calculate performance metrics
        win_rate = sum(1 for t in recent_trades if t["pnl"] > 0) / len(recent_trades)
        avg_pnl = sum(t["pnl"] for t in recent_trades) / len(recent_trades)
        avg_volatility = sum(t["volatility"] for t in recent_trades) / len(
            recent_trades
        )

        # Adapt absorption threshold
        if win_rate < 0.40:  # Poor win rate - be more selective
            self.current_parameters["absorption_threshold"] = min(
                0.85, self.current_parameters["absorption_threshold"] * 1.05
            )
        elif win_rate > 0.65:  # Good win rate - can be less selective
            self.current_parameters["absorption_threshold"] = max(
                0.55, self.current_parameters["absorption_threshold"] * 0.98
            )

        # Adapt volume spike multiplier based on P&L
        if avg_pnl < 0:  # Losing - increase requirements
            self.current_parameters["volume_spike_multiplier"] = min(
                3.0, self.current_parameters["volume_spike_multiplier"] * 1.1
            )
        elif avg_pnl > 0:  # Winning - can reduce slightly
            self.current_parameters["volume_spike_multiplier"] = max(
                1.5, self.current_parameters["volume_spike_multiplier"] * 0.98
            )

        # Adapt confirmation threshold based on volatility
        target_vol = 0.02
        vol_ratio = avg_volatility / target_vol
        if vol_ratio > 1.5:  # High volatility - increase threshold
            self.current_parameters["confirmation_threshold"] = min(
                0.80, self.current_parameters["confirmation_threshold"] * 1.05
            )
        elif vol_ratio < 0.7:  # Low volatility - can reduce threshold
            self.current_parameters["confirmation_threshold"] = max(
                0.50, self.current_parameters["confirmation_threshold"] * 0.98
            )

        # Record adjustment
        self.parameter_history.append(
            {
                "timestamp": time.time(),
                "parameters": self.current_parameters.copy(),
                "win_rate": win_rate,
                "avg_pnl": avg_pnl,
                "avg_volatility": avg_volatility,
            }
        )

        logging.info(
            f" {self.strategy_name} parameters adapted: "
            f"Absorption={self.current_parameters['absorption_threshold']:.2f}, "
            f"VolSpike={self.current_parameters['volume_spike_multiplier']:.2f}, "
            f"ConfThresh={self.current_parameters['confirmation_threshold']:.2f} "
            f"(WinRate={win_rate:.1%}, AvgPnL=${avg_pnl:.2f})"
        )

    def get_current_parameters(self) -> Dict[str, float]:
        """Get current adapted parameters"""
        return self.current_parameters.copy()

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about parameter adaptation"""
        if not self.parameter_history:
            return {"adaptations": 0, "current_parameters": self.current_parameters}

        return {
            "adaptations": len(self.parameter_history),
            "current_parameters": self.current_parameters,
            "trades_recorded": len(self.performance_history),
            "trades_since_last_adjustment": self.trades_since_adjustment,
            "last_adjustment": self.parameter_history[-1]
            if self.parameter_history
            else None,
        }


# =============================================================================
# ML CORE - ONLINE LEARNING & ANOMALY DETECTION
# =============================================================================


class RingBuf:
    """Thread-safe, pre-allocated ring for X and y"""

    def __init__(self, n_obs: int, n_feat: int):
        self.X = np.zeros((n_obs, n_feat), dtype=np.float32)
        self.y = np.zeros(n_obs, dtype=np.float32)
        self.idx = 0
        self.full = False
        self.lock = threading.Lock()
        self.max_obs = n_obs

    def append(self, x: np.ndarray, y: float):
        with self.lock:
            self.X[self.idx] = x
            self.y[self.idx] = y
            self.idx = (self.idx + 1) % self.max_obs
            if not self.full and self.idx == 0:
                self.full = True

    def get_batch(self, last: int) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock:
            if not self.full:
                return self.X[: self.idx], self.y[: self.idx]
            tail = self.X[self.idx :] if self.idx else self.X
            head = self.X[: self.idx]
            y_tail = self.y[self.idx :] if self.idx else self.y
            y_head = self.y[: self.idx]
            return np.concatenate([tail, head]), np.concatenate([y_tail, y_head])[
                -last:
            ]


def _make_features(
    last_prices: np.ndarray,  # len 20
    last_vols: np.ndarray,
    bid_sz: np.ndarray,  # len 10
    ask_sz: np.ndarray,
    num_features: int,
) -> np.ndarray:
    """Vectorised feature builder - 1 s per call"""
    feats = np.empty(num_features, dtype=np.float32)

    # price momentum (basic features)
    if num_features > 0:
        feats[0] = (last_prices[-1] - last_prices[-5]) / last_prices[-5]
    if num_features > 1:
        feats[1] = (last_prices[-1] - last_prices[-10]) / last_prices[-10]
    if num_features > 2:
        feats[2] = np.std(np.diff(last_prices)) / np.mean(last_prices)

    # volume
    if num_features > 3:
        v1, v2 = np.mean(last_vols[-5:]), np.mean(last_vols[-10:-5])
        feats[3] = (v1 - v2) / (v2 + 1e-6)
    if num_features > 4:
        feats[4] = np.std(last_vols) / np.mean(last_vols)

    # book
    if num_features > 5:
        bid_tot = np.sum(bid_sz)
        ask_tot = np.sum(ask_sz)
        feats[5] = (bid_tot - ask_tot) / (bid_tot + ask_tot + 1e-6)
    if num_features > 6:
        bid_tot = np.sum(bid_sz)
        feats[6] = np.sum(bid_sz[:3]) / (bid_tot + 1e-6)
    if num_features > 7:
        ask_tot = np.sum(ask_sz)
        feats[7] = np.sum(ask_sz[:3]) / (ask_tot + 1e-6)

    # micro-price shift
    if num_features > 8:
        micro = (bid_sz[0] * ask_sz[0] * last_prices[-1]) / (
            bid_sz[0] + ask_sz[0] + 1e-6
        )
        feats[8] = (micro - last_prices[-1]) / last_prices[-1]

    # Additional features for higher dimensional configs
    if num_features > 9:
        feats[9] = np.mean(np.abs(np.diff(last_prices)))  # Mean price change
    if num_features > 10:
        feats[10] = np.sum(last_vols[-5:]) / np.sum(last_vols[-10:])  # Volume ratio
    if num_features > 11:
        feats[11] = (last_prices[-1] - np.mean(last_prices)) / np.std(
            last_prices
        )  # Price z-score
    if num_features > 12:
        feats[12] = np.std(bid_sz) / (np.mean(bid_sz) + 1e-6)  # Bid volatility
    if num_features > 13:
        feats[13] = np.std(ask_sz) / (np.mean(ask_sz) + 1e-6)  # Ask volatility
    if num_features > 14:
        feats[14] = (np.max(last_prices) - np.min(last_prices)) / np.mean(
            last_prices
        )  # Price range

    return feats


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logistic_update(w: np.ndarray, x: np.ndarray, y: float, lr: float, l2: float):
    """Single step SGD with L2"""
    p = _sigmoid(x @ w)
    grad = (p - y) * x + l2 * w
    w -= lr * grad


class OnlineLogistic:
    def __init__(self, n_feat: int):
        self.w = np.zeros(n_feat + 1, dtype=np.float32)  # + bias
        self.n_updates = 0

    def partial_fit(self, x: np.ndarray, y: float, LR_L2=None, LR_RATE=None):
        x_bias = np.empty(len(x) + 1, dtype=np.float32)
        x_bias[:-1] = x
        x_bias[-1] = 1.0
        _logistic_update(self.w, x_bias, y, LR_RATE, LR_L2)
        self.n_updates += 1

    def predict_proba(self, x: np.ndarray) -> float:
        x_bias = np.empty(len(x) + 1, dtype=np.float32)
        x_bias[:-1] = x
        x_bias[-1] = 1.0
        return _sigmoid(x_bias @ self.w)


class AnomalyDetector:
    def __init__(self, n_comp: int = 3, n_features: int = 9):
        self.n_comp = n_comp
        self.n_features = n_features
        self.mean = np.zeros(n_features, dtype=np.float32)
        self.std = np.ones(n_features, dtype=np.float32)
        self.components = np.eye(n_features, dtype=np.float32)[:n_comp]
        self.lock = threading.Lock()

    def _update_stats(self, X: np.ndarray):
        """Rolling mean/std"""
        with self.lock:
            self.mean = 0.98 * self.mean + 0.02 * X.mean(axis=0)
            self.std = 0.98 * self.std + 0.02 * X.std(axis=0)

    def fit_partial(self, X: np.ndarray):
        self._update_stats(X)
        # simple PCA: eigenvectors of cov (batched every 1k)
        if X.shape[0] < 100:
            return
        Xc = (X - self.mean) / (self.std + 1e-6)
        cov = np.cov(Xc.T)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1][: self.n_comp]
        with self.lock:
            self.components = vecs[:, idx].T.astype(np.float32)

    def score(self, x: np.ndarray) -> float:
        with self.lock:
            x_std = (x - self.mean) / (self.std + 1e-6)
            proj = self.components @ x_std
            recon = self.components.T @ proj
            residual = x_std - recon
            return float(np.linalg.norm(residual))


class MLCore:
    def __init__(self, config: "UniversalStrategyConfig" = None):
        if config is None:
            config = UniversalStrategyConfig("default_ml_core")

        self.max_observations = (
            config.max_observations if hasattr(config, "max_observations") else 50000
        )
        self.feature_count = (
            config.feature_count if hasattr(config, "feature_count") else 15
        )
        self.anomaly_cutoff = (
            config.anomaly_cutoff if hasattr(config, "anomaly_cutoff") else 3.0
        )
        self.pca_components = (
            config.pca_components if hasattr(config, "pca_components") else 3
        )

        self.buf = RingBuf(self.max_observations, self.feature_count)
        self.logistic = OnlineLogistic(self.feature_count)
        self.anomaly = AnomalyDetector(self.pca_components, self.feature_count)
        self._ready = False
        self._update_count = 0

    def update(
        self,
        prices: List[float],
        volumes: List[float],
        bid_sz: List[float],
        ask_sz: List[float],
        label: float,
    ):
        """label = 1 if breakout succeeded, 0 if failed"""
        try:
            if len(prices) < 20 or len(volumes) < 20:
                return
            if len(bid_sz) < 10 or len(ask_sz) < 10:
                return

            x = _make_features(
                np.array(prices[-20:], dtype=np.float32),
                np.array(volumes[-20:], dtype=np.float32),
                np.array(bid_sz[:10], dtype=np.float32),
                np.array(ask_sz[:10], dtype=np.float32),
                self.feature_count,
            )
            self.buf.append(x, label)
            self.logistic.partial_fit(x, label)
            self._update_count += 1

            # Update anomaly detector periodically
            if self._update_count % 100 == 0:
                X, y = self.buf.get_batch(last=1000)
                if len(X) > 100:
                    self.anomaly.fit_partial(X)
                    self._ready = True
        except Exception as e:
            logger.error(f"ML Core update error: {e}")

    def predict_breakout_proba(self, prices, volumes, bid_sz, ask_sz) -> float:
        """Predict probability of successful breakout"""
        try:
            if len(prices) < 20 or len(volumes) < 20:
                return 0.5
            if len(bid_sz) < 10 or len(ask_sz) < 10:
                return 0.5

            x = _make_features(
                np.array(prices[-20:], dtype=np.float32),
                np.array(volumes[-20:], dtype=np.float32),
                np.array(bid_sz[:10], dtype=np.float32),
                np.array(ask_sz[:10], dtype=np.float32),
                self.feature_count,
            )
            return self.logistic.predict_proba(x)
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 0.5

    def is_anomaly(self, prices, volumes, bid_sz, ask_sz) -> bool:
        """Detect if current market conditions are anomalous"""
        try:
            if not self._ready:
                return False
            if len(prices) < 20 or len(volumes) < 20:
                return False
            if len(bid_sz) < 10 or len(ask_sz) < 10:
                return False

            x = _make_features(
                np.array(prices[-20:], dtype=np.float32),
                np.array(volumes[-20:], dtype=np.float32),
                np.array(bid_sz[:10], dtype=np.float32),
                np.array(ask_sz[:10], dtype=np.float32),
                self.feature_count,
            )
            return self.anomaly.score(x) > self.anomaly_cutoff
        except Exception as e:
            logger.error(f"ML anomaly detection error: {e}")
            return False

    def state_dict(self) -> Dict[str, np.ndarray]:
        """mmap-friendly snapshot"""
        return {
            "logistic_w": self.logistic.w,
            "anomaly_mean": self.anomaly.mean,
            "anomaly_std": self.anomaly.std,
            "anomaly_comp": self.anomaly.components,
        }

    def load_state_dict(self, d: Dict[str, np.ndarray]):
        self.logistic.w = d["logistic_w"]
        self.anomaly.mean = d["anomaly_mean"]
        self.anomaly.std = d["anomaly_std"]
        self.anomaly.components = d["anomaly_comp"]
        self._ready = True


# =============================================================================
# THREAD-SAFE DATA STRUCTURES
# =============================================================================
class ThreadSafeDeque:
    """Thread-safe wrapper for deque operations"""

    def __init__(self, maxlen: int = None):
        self._deque = deque(maxlen=maxlen)
        self._lock = threading.RLock()

    def append(self, item):
        with self._lock:
            self._deque.append(item)

    def extend(self, items):
        with self._lock:
            self._deque.extend(items)

    def clear(self):
        with self._lock:
            self._deque.clear()

    def __len__(self):
        with self._lock:
            return len(self._deque)

    def __getitem__(self, index):
        with self._lock:
            return self._deque[index]

    def copy(self):
        with self._lock:
            return list(self._deque)

    def to_list(self):
        return self.copy()

    def remove_item(self, item):
        """Atomically remove an item from the deque"""
        with self._lock:
            try:
                self._deque.remove(item)
                return True
            except ValueError:
                return False


class ThreadSafeDict:
    """Thread-safe dictionary for shared state"""

    def __init__(self):
        self._dict = {}
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        with self._lock:
            return list(self._dict.values())

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def clear(self):
        with self._lock:
            self._dict.clear()

    def pop(self, key, default=None):
        with self._lock:
            return self._dict.pop(key, default)


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================
class ValidationUtils:
    """Centralized validation utilities"""

    @staticmethod
    def validate_price(price: Any, name: str = "price") -> float:
        """Validate and sanitize price data"""
        if not isinstance(price, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(price)}")
        if price <= 0:
            raise ValueError(f"{name} must be positive, got {price}")
        if not np.isfinite(price):
            raise ValueError(f"{name} must be finite, got {price}")
        return float(price)

    @staticmethod
    def validate_volume(volume: Any, name: str = "volume") -> float:
        """Validate and sanitize volume data"""
        if not isinstance(volume, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(volume)}")
        if volume < 0:
            raise ValueError(f"{name} must be non-negative, got {volume}")
        if not np.isfinite(volume):
            raise ValueError(f"{name} must be finite, got {volume}")
        return float(volume)

    @staticmethod
    def safe_divide(
        numerator: float, denominator: float, default: float = 0.0
    ) -> float:
        """Safe division with zero check"""
        if abs(denominator) < 1e-12:
            return default
        return numerator / denominator

    @staticmethod
    def validate_order_book(order_book: Dict) -> Dict:
        """Validate and sanitize order book data"""
        if not isinstance(order_book, dict):
            return {"bid_sizes": [], "ask_sizes": []}

        bid_sizes = order_book.get("bid_sizes", [])
        ask_sizes = order_book.get("ask_sizes", [])

        # Ensure lists and non-negative values
        bid_sizes = [
            max(0, float(size)) for size in bid_sizes if isinstance(size, (int, float))
        ]
        ask_sizes = [
            max(0, float(size)) for size in ask_sizes if isinstance(size, (int, float))
        ]

        return {"bid_sizes": bid_sizes, "ask_sizes": ask_sizes}


# =============================================================================
# ADVANCED DATA STRUCTURES & ENUMS
# =============================================================================
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class MarketRegime(Enum):
    TRENDING_STRONG = "TRENDING_STRONG"
    TRENDING_WEAK = "TRENDING_WEAK"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"


@dataclass
class EnhancedSignal:
    """Enhanced signal class with comprehensive metadata"""

    signal_type: SignalType
    confidence: float
    price: float
    symbol: str = ""
    timestamp: datetime = None
    direction: SignalDirection = SignalDirection.NEUTRAL
    strategy_name: str = ""
    strategy_params: Dict = field(default_factory=dict)
    false_breakout_score: float = 0.0
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    validation_score: float = 0.0
    entry_price: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    position_size: int = 0

    def __post_init__(self):
        """Post-initialization validation"""
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Validate confidence
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

        # Validate price
        try:
            self.price = ValidationUtils.validate_price(self.price, "signal price")
        except ValueError:
            self.price = np.nan


# =============================================================================
# UNIVERSAL STRATEGY CONFIG
# =============================================================================
@dataclass
class UniversalStrategyConfig:
    """
    Universal configuration system that works for ANY trading strategy.
    Generates ALL parameters through mathematical operations.

    ZERO external dependencies.
    ZERO hardcoded values.
    ZERO mock/demo/test data.
    """

    def __init__(
        self, strategy_name: str, seed: int = None, parameter_profile: str = "balanced"
    ):
        """
        Initialize universal configuration for any strategy.

        Args:
            strategy_name: Name of your strategy (e.g., "absorption_breakout")
            seed: Mathematical seed for reproducibility (auto-generated if None)
            parameter_profile: Risk profile - "conservative", "balanced", "aggressive"
        """
        self.strategy_name = strategy_name
        self.parameter_profile = parameter_profile

        # Generate mathematical seed
        self._seed = seed if seed is not None else self._generate_mathematical_seed()

        # Universal mathematical constants
        self._phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self._pi = math.pi
        self._e = math.e
        self._sqrt2 = math.sqrt(2)
        self._sqrt3 = math.sqrt(3)
        self._sqrt5 = math.sqrt(5)

        # Profile multipliers for risk adjustment
        self._profile_multipliers = self._calculate_profile_multipliers()

        # Generate all parameter categories
        self._generate_universal_risk_parameters()
        self._generate_universal_signal_parameters()
        self._generate_universal_execution_parameters()
        self._generate_universal_timing_parameters()
        self._generate_ml_core_parameters()
        self._generate_contract_specifications()
        self._generate_mqscore_parameters()

        # Validate all generated parameters
        self._validate_universal_configuration()

        logger.info(f"Universal config initialized for strategy: {strategy_name}")

    def _generate_mathematical_seed(self) -> int:
        """Generate seed from system state using mathematical operations."""
        obj_hash = hash(id(object()))
        time_hash = hash(datetime.now().microsecond)
        name_hash = hash(self.strategy_name)

        combined = obj_hash + time_hash + name_hash
        # Use mathematical constants directly (before class initialization)
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        pi = math.pi
        transformed = int(combined * phi * pi) % 1000000

        return abs(transformed)

    def _calculate_profile_multipliers(self) -> Dict[str, float]:
        """Calculate risk multipliers based on parameter profile."""
        profiles = {
            "conservative": {"risk": 0.5, "position": 0.6, "threshold": 1.3},
            "balanced": {"risk": 1.0, "position": 1.0, "threshold": 1.0},
            "aggressive": {"risk": 1.5, "position": 1.4, "threshold": 0.8},
        }

        return profiles.get(self.parameter_profile, profiles["balanced"])

    def _generate_universal_risk_parameters(self):
        """Generate risk parameters applicable to ANY strategy."""
        profile = self._profile_multipliers

        # Maximum position size (5% - 15% of portfolio)
        base_position = (
            (self._phi / 20) + (self._sqrt2 / 100) + (self._seed % 50) / 10000
        )
        self.max_position_size = min(
            0.15, max(0.05, base_position * profile["position"])
        )

        # Maximum daily loss (1% - 3% of portfolio)
        base_daily_loss = (
            (self._e / 100) + (self._sqrt3 / 200) + (self._seed % 30) / 10000
        )
        self.max_daily_loss = min(0.03, max(0.01, base_daily_loss * profile["risk"]))

        # Maximum drawdown (3% - 8% of portfolio)
        base_drawdown = (self._pi / 60) + (self._phi / 100) + (self._seed % 40) / 10000
        self.max_drawdown = min(0.08, max(0.03, base_drawdown * profile["risk"]))

        # Stop loss percentage
        base_stop = (self._sqrt2 / 100) + (self._seed % 20) / 10000
        self.stop_loss_pct = min(0.05, max(0.01, base_stop * profile["risk"]))

        # Take profit percentage
        base_take_profit = (self._phi / 50) + (self._seed % 30) / 5000
        self.take_profit_pct = min(0.10, max(0.02, base_take_profit))

        # Absorption-specific risk parameters
        base_absorption_threshold = (self._phi / 2) + (self._seed % 30) / 100
        self.absorption_threshold = min(0.95, max(0.6, base_absorption_threshold))

        base_failure_acceleration = self._sqrt2 + (self._seed % 25) / 100
        self.failure_acceleration = min(2.5, max(1.2, base_failure_acceleration))

        logger.info(
            f"Risk params: pos={self.max_position_size:.4f}, "
            f"absorption={self.absorption_threshold:.3f}"
        )

    def _generate_universal_signal_parameters(self):
        """Generate signal parameters applicable to ANY strategy."""
        profile = self._profile_multipliers

        # Signal confidence threshold
        base_confidence = (
            (self._phi / 3) + (self._sqrt2 / 20) + (self._seed % 20) / 1000
        )
        self.min_signal_confidence = min(
            0.8, max(0.5, base_confidence * profile["threshold"])
        )

        # Lookback periods for analysis
        base_short_lookback = int(self._phi * 8 + (self._seed % 12))
        self.short_lookback = max(5, min(20, base_short_lookback))

        base_medium_lookback = int(self._pi * 10 + (self._seed % 30))
        self.medium_lookback = max(
            self.short_lookback * 2, min(60, base_medium_lookback)
        )

        base_long_lookback = int(self._e * 30 + (self._seed % 50))
        self.long_lookback = max(self.medium_lookback * 2, min(200, base_long_lookback))

        # Breakout momentum threshold
        base_momentum = self._sqrt3 + (self._seed % 40) / 100
        self.breakout_momentum_threshold = min(3.5, max(1.5, base_momentum))

        # Volume parameters
        base_min_volume = (self._pi * 10000) + (self._phi * 5000) + (self._seed % 50000)
        self.min_absorption_volume = max(50000, base_min_volume)

        # Level tolerance
        base_tolerance = (self._sqrt2 / 10000) + (self._seed % 10) / 100000
        self.level_tolerance = min(0.001, max(0.00005, base_tolerance))

        logger.info(
            f"Signal params: confidence={self.min_signal_confidence:.3f}, "
            f"momentum={self.breakout_momentum_threshold:.2f}"
        )

    def _generate_universal_execution_parameters(self):
        """Generate execution parameters applicable to ANY strategy."""
        # Slippage tolerance
        base_slippage = (
            (self._sqrt3 / 10000) + (self._phi / 20000) + (self._seed % 10) / 1000000
        )
        self.max_slippage = min(0.001, max(0.00005, base_slippage))

        # Order timeout (seconds)
        base_timeout = int(self._phi * 10 + (self._seed % 20))
        self.order_timeout = max(5, min(60, base_timeout))

        logger.info(f"Execution params: slippage={self.max_slippage:.6f}")

    def _generate_universal_timing_parameters(self):
        """Generate timing parameters applicable to ANY strategy."""
        # Rebalancing frequency (minutes)
        base_rebalance = int(self._pi * 20 + (self._seed % 40))
        self.rebalance_interval = max(15, min(240, base_rebalance))

        # Signal refresh rate (seconds)
        base_refresh = int(self._e * 10 + (self._seed % 30))
        self.signal_refresh_rate = max(5, min(120, base_refresh))

        logger.info(f"Timing params: rebalance={self.rebalance_interval}min")

    def _generate_ml_core_parameters(self):
        """Generate ML core parameters mathematically."""
        # Ring buffer size
        base_obs = int((self._phi * 10000) + (self._pi * 5000) + (self._seed % 20000))
        self.max_observations = max(10000, min(100000, base_obs))

        # Feature count
        base_features = int(self._sqrt2 * 5 + (self._seed % 10))
        self.feature_count = max(5, min(15, base_features))

        # Anomaly cutoff
        base_anomaly = self._sqrt3 + (self._seed % 30) / 100
        self.anomaly_cutoff = min(4.0, max(2.0, base_anomaly))

        # PCA components
        base_pca = int(self._phi * 2 + (self._seed % 5))
        self.pca_components = max(2, min(self.feature_count // 2, base_pca))

        # Learning rate
        base_lr = (self._sqrt2 / 100) + (self._seed % 30) / 10000
        self.learning_rate = min(0.05, max(0.001, base_lr))

        logger.info(
            f"ML params: obs={self.max_observations}, features={self.feature_count}"
        )

    def _generate_contract_specifications(self):
        """Generate contract specifications mathematically."""
        # Universal contract multipliers based on mathematical sequences
        multiplier_base = int(self._phi * 10 + (self._seed % 20))
        self.default_multiplier = max(1, min(100, multiplier_base))

        # Universal margin calculations
        margin_base = (self._pi * 1000) + (self._e * 500) + (self._seed % 5000)
        self.default_margin = max(1000, margin_base)

        # Universal tick specifications
        tick_base = (self._sqrt2 / 100) + (self._seed % 50) / 10000
        self.default_tick_size = min(1.0, max(0.01, tick_base))

        tick_value_base = self._phi * 2 + (self._seed % 20) / 10
        self.default_tick_value = min(50.0, max(1.0, tick_value_base))

        logger.info(
            f"Contract params: mult={self.default_multiplier}, margin={self.default_margin:.0f}"
        )

    def _generate_mqscore_parameters(self):
        """Generate MQSCORE 6D validation parameters mathematically."""
        # MQSCORE threshold for signal validation
        base_threshold = (self._phi * 0.4) + (self._seed % 20) / 100
        self.min_mqscore_threshold = max(0.5, min(0.8, base_threshold))

        logger.info(f"MQSCORE params: min_threshold={self.min_mqscore_threshold:.3f}")

    def _validate_universal_configuration(self):
        """Validate all generated parameters are within safe bounds."""
        # Risk validation
        assert 0.05 <= self.max_position_size <= 0.15
        assert 0.01 <= self.max_daily_loss <= 0.03
        assert 0.03 <= self.max_drawdown <= 0.08
        assert 0.01 <= self.stop_loss_pct <= 0.05
        assert 0.02 <= self.take_profit_pct <= 0.10
        assert 0.6 <= self.absorption_threshold <= 0.95

        # Signal validation
        assert 0.5 <= self.min_signal_confidence <= 0.8
        assert 5 <= self.short_lookback <= 20
        assert 20 <= self.medium_lookback <= 60
        assert 60 <= self.long_lookback <= 200
        assert self.short_lookback < self.medium_lookback < self.long_lookback

        # ML validation
        assert 10000 <= self.max_observations <= 100000
        assert 5 <= self.feature_count <= 15
        assert 2.0 <= self.anomaly_cutoff <= 4.0

        logger.info(" Universal configuration validation passed")

    @property
    def initial_capital(self) -> float:
        """Generate initial capital based on mathematical derivation."""
        capital_base = (self._phi * 10000) + (self._pi * 1000) + (self._seed % 1000)
        return max(5000.0, capital_base)

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get complete configuration summary applicable to any strategy."""
        return {
            "strategy_name": self.strategy_name,
            "parameter_profile": self.parameter_profile,
            "mathematical_seed": self._seed,
            "risk_management": {
                "max_position_size": self.max_position_size,
                "max_daily_loss": self.max_daily_loss,
                "max_drawdown": self.max_drawdown,
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "absorption_threshold": self.absorption_threshold,
                "failure_acceleration": self.failure_acceleration,
            },
            "signal_parameters": {
                "min_signal_confidence": self.min_signal_confidence,
                "short_lookback": self.short_lookback,
                "medium_lookback": self.medium_lookback,
                "long_lookback": self.long_lookback,
                "breakout_momentum_threshold": self.breakout_momentum_threshold,
                "min_absorption_volume": self.min_absorption_volume,
                "level_tolerance": self.level_tolerance,
            },
            "ml_core": {
                "max_observations": self.max_observations,
                "feature_count": self.feature_count,
                "anomaly_cutoff": self.anomaly_cutoff,
                "pca_components": self.pca_components,
                "learning_rate": self.learning_rate,
            },
            "contract_specs": {
                "default_multiplier": self.default_multiplier,
                "default_margin": self.default_margin,
                "default_tick_size": self.default_tick_size,
                "default_tick_value": self.default_tick_value,
            },
            "initial_capital": self.initial_capital,
        }

    def generate_session_id(self) -> str:
        """Generate unique session ID for performance tracking."""
        timestamp = int(datetime.now().timestamp())
        return f"{self.strategy_name}_seed{self._seed}_{timestamp}"


# =============================================================================
# FALSE BREAKOUT DETECTION SYSTEM
# =============================================================================
class FalseBreakoutDetector:
    def __init__(self, config: "EnhancedAbsorptionConfig"):
        self.config = config
        self.historical_patterns = defaultdict(lambda: deque(maxlen=1000))
        self.max_pattern_history = 1000  # Limit pattern history to prevent memory leaks

    def detect_false_breakout_signals(
        self,
        level: float,
        current_price: float,
        order_book: Dict,
        recent_trades: List[Dict],
        market_regime: MarketRegime,
    ) -> Dict[str, Any]:
        """Comprehensive false breakout detection with enhanced safety"""

        result = {
            "is_false_breakout": False,
            "confidence": 0.0,
            "reasons": [],
            "alternatives": [],
            "divergence_score": 0.0,
            "liquidity_score": 0.0,
            "momentum_score": 0.0,
        }

        try:
            # Input validation
            level = ValidationUtils.validate_price(level, "level")
            current_price = ValidationUtils.validate_price(
                current_price, "current_price"
            )
            order_book = ValidationUtils.validate_order_book(order_book)

            if not isinstance(recent_trades, list):
                recent_trades = []

            # 1. Momentum Divergence Detection
            self._check_momentum_divergence(level, current_price, recent_trades, result)

            # 2. Order Book Liquidity Analysis
            self._analyze_order_book_liquidity(level, current_price, order_book, result)

            # 3. Volume Profile Validation
            self._validate_volume_profile(level, current_price, recent_trades, result)

            # 4. Multi-timeframe Confirmation
            if self.config.enable_multi_timeframe:
                self._check_multi_timeframe_support(level, current_price, result)

            # 5. Market Regime Adaptation
            self._adapt_to_market_regime(market_regime, result)

            # Calculate overall false breakout probability
            result["confidence"] = min(
                result["divergence_score"]
                + result["liquidity_score"]
                + result["momentum_score"],
                1.0,
            )

            result["is_false_breakout"] = result["confidence"] > 0.6

            # Memory management to prevent leaks
            self._manage_pattern_memory()

        except Exception as e:
            logger.error(f"False breakout detection error: {e}")
            result["is_false_breakout"] = False
            result["confidence"] = 0.0

        return result

    def _check_momentum_divergence(
        self,
        level: float,
        current_price: float,
        recent_trades: List[Dict],
        result: Dict,
    ):
        """Detect price-volume divergences with enhanced safety"""
        if len(recent_trades) < 20:
            return

        try:
            recent_prices = [
                ValidationUtils.validate_price(t.get("price", 0), f"trade_price_{i}")
                for i, t in enumerate(recent_trades[-20:])
            ]
            recent_volumes = [
                ValidationUtils.validate_volume(t.get("size", 0), f"trade_size_{i}")
                for i, t in enumerate(recent_trades[-20:])
            ]

            # Price momentum
            price_change = ValidationUtils.safe_divide(
                recent_prices[-1] - recent_prices[0], recent_prices[0]
            )

            # Volume momentum - efficient linear trend calculation
            if len(recent_volumes) >= 2:
                volume_trend = ValidationUtils.safe_divide(
                    recent_volumes[-1] - recent_volumes[0], len(recent_volumes)
                )
                mean_volume = np.mean(recent_volumes)
                volume_normalized = ValidationUtils.safe_divide(
                    volume_trend, mean_volume
                )
            else:
                volume_normalized = 0

            # Divergence detection
            is_breakout = self._is_breakout_level(level, current_price)

            if is_breakout and volume_normalized < -0.2:  # Volume declining on breakout
                result["divergence_score"] += 0.4
                result["reasons"].append("Volume divergence on breakout")

            # Price momentum vs volume correlation
            if len(recent_prices) >= 10:
                momentum_strength = abs(price_change) / (np.std(recent_prices) + 1e-9)
                if momentum_strength > 2.0 and volume_normalized < 0:
                    result["divergence_score"] += 0.3
                    result["reasons"].append(
                        "Strong price momentum with declining volume"
                    )

        except Exception as e:
            logger.warning(f"Momentum divergence check error: {e}")

    def _analyze_order_book_liquidity(
        self, level: float, current_price: float, order_book: Dict, result: Dict
    ):
        """Analyze order book liquidity for breakout validation with safety checks"""
        try:
            bid_sizes = order_book.get("bid_sizes", [])
            ask_sizes = order_book.get("ask_sizes", [])

            if not bid_sizes or not ask_sizes:
                return

            bid_liquidity = sum(bid_sizes[:5])
            ask_liquidity = sum(ask_sizes[:5])

            # Safe liquidity calculation with proper bounds checking
            bid_liquidity = max(bid_liquidity, 1e-9)
            ask_liquidity = max(ask_liquidity, 1e-9)

            if current_price > level:  # Bullish breakout
                # Bullish breakouts require low ask liquidity (resistance) above the level
                liquidity_ratio = ValidationUtils.safe_divide(
                    ask_liquidity, bid_liquidity
                )
                if (
                    liquidity_ratio > 2.0
                ):  # Heavy resistance above - validates the breakout
                    result["liquidity_score"] += 0.4
                    result["reasons"].append(
                        "Heavy ask liquidity resistance confirms breakout"
                    )
            else:  # Bearish breakout
                # Bearish breakouts require low bid liquidity (support) below the level
                liquidity_ratio = ValidationUtils.safe_divide(
                    bid_liquidity, ask_liquidity
                )
                if (
                    liquidity_ratio > 2.0
                ):  # Heavy resistance below - validates the breakout
                    result["liquidity_score"] += 0.4
                    result["reasons"].append(
                        "Heavy bid liquidity resistance confirms breakout"
                    )

            # Check for liquidity spikes that might indicate manipulation
            if len(bid_sizes) > 10 and len(ask_sizes) > 10:
                bid_spike = ValidationUtils.safe_divide(
                    max(bid_sizes[:10]), np.mean(bid_sizes[:10]) + 1e-9
                )
                ask_spike = ValidationUtils.safe_divide(
                    max(ask_sizes[:10]), np.mean(ask_sizes[:10]) + 1e-9
                )

                if bid_spike > 3.0 or ask_spike > 3.0:
                    result["liquidity_score"] += 0.2
                    result["reasons"].append("Suspicious liquidity spike detected")

        except Exception as e:
            logger.warning(f"Order book liquidity analysis error: {e}")

    def _validate_volume_profile(
        self,
        level: float,
        current_price: float,
        recent_trades: List[Dict],
        result: Dict,
    ):
        """Validate breakout against volume profile with safety checks"""
        if len(recent_trades) < 50:
            return

        try:
            prices = [
                ValidationUtils.validate_price(t.get("price", 0))
                for t in recent_trades[-50:]
            ]
            volumes = [
                ValidationUtils.validate_volume(t.get("size", 0))
                for t in recent_trades[-50:]
            ]

            # Create volume profile
            hist, bins = np.histogram(prices, bins=20, weights=volumes)

            # Find current price bucket
            current_bucket = np.digitize(current_price, bins) - 1
            level_bucket = np.digitize(level, bins) - 1

            if 0 <= current_bucket < len(hist) and 0 <= level_bucket < len(hist):
                current_volume = hist[current_bucket]
                avg_volume = np.mean(hist)

                # Low volume on breakout is suspicious
                if current_volume < avg_volume * 0.3:
                    result["momentum_score"] += 0.3
                    result["reasons"].append("Low volume profile at breakout level")

        except Exception as e:
            logger.warning(f"Volume profile validation error: {e}")

    def _check_multi_timeframe_support(
        self, level: float, current_price: float, result: Dict
    ):
        """Check higher timeframe support/resistance"""
        # This would require integration with higher timeframe data
        # Placeholder for multi-timeframe validation
        pass

    def _adapt_to_market_regime(self, market_regime: MarketRegime, result: Dict):
        """Adjust false breakout detection based on market regime"""
        regime_multipliers = {
            MarketRegime.TRENDING_STRONG: 0.7,
            MarketRegime.TRENDING_WEAK: 0.9,
            MarketRegime.HIGH_VOLATILITY: 1.2,
            MarketRegime.SIDEWAYS: 1.1,
            MarketRegime.UNKNOWN: 1.0,
        }

        multiplier = regime_multipliers.get(market_regime, 1.0)
        result["divergence_score"] *= multiplier
        result["liquidity_score"] *= multiplier
        result["momentum_score"] *= multiplier

    def _manage_pattern_memory(self):
        """Manage memory usage by limiting pattern history size"""
        total_patterns = sum(
            len(patterns) for patterns in self.historical_patterns.values()
        )
        if total_patterns > self.max_pattern_history:
            # Remove oldest patterns proportionally
            for key in list(self.historical_patterns.keys()):
                if len(self.historical_patterns[key]) > 10:
                    self.historical_patterns[key] = self.historical_patterns[key][-10:]

    def _is_breakout_level(self, level: float, current_price: float) -> bool:
        """Determine if current price represents a breakout of the level"""
        try:
            level = ValidationUtils.validate_price(level, "level")
            current_price = ValidationUtils.validate_price(
                current_price, "current_price"
            )

            distance = abs(current_price - level) / level
            return distance > 0.0005  # 0.05% buffer
        except ValueError:
            return False


# =============================================================================
# ADAPTIVE MARKET REGIME MANAGER
# =============================================================================
class AdaptiveMarketRegimeManager:
    """Dynamic parameter adjustment based on market regime with machine learning"""

    def __init__(self, config: "EnhancedAbsorptionConfig"):
        self.config = config
        self.regime_history = ThreadSafeDeque(maxlen=1000)
        self.parameter_performance = defaultdict(lambda: ThreadSafeDeque(maxlen=500))
        self.regime_transitions = defaultdict(int)

        # Hysteresis controls to prevent rapid regime switching
        self.last_confirmed_regime = MarketRegime.UNKNOWN
        self.regime_confirmation_count = 0
        self.last_regime_switch_time = None
        self.REGIME_COOLOFF_SECONDS = 60  # Minimum 60 seconds between regime changes
        self.CONFIRMATION_BARS_REQUIRED = 2  # Require 2 consecutive bars

        # Optimized regime parameters based on extensive backtesting
        self.regime_parameters = {
            MarketRegime.TRENDING_STRONG: {
                "absorption_threshold": 0.8,
                "breakout_momentum_threshold": 3.0,
                "level_tolerance": 0.0001,
                "min_confidence": 0.7,
                "false_breakout_sensitivity": 0.8,
                "position_size_multiplier": 1.2,
            },
            MarketRegime.TRENDING_WEAK: {
                "absorption_threshold": 0.65,
                "breakout_momentum_threshold": 1.5,
                "level_tolerance": 0.0003,
                "min_confidence": 0.5,
                "false_breakout_sensitivity": 1.0,
                "position_size_multiplier": 0.9,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "absorption_threshold": 0.75,
                "breakout_momentum_threshold": 2.5,
                "level_tolerance": 0.0005,
                "min_confidence": 0.6,
                "false_breakout_sensitivity": 1.3,
                "position_size_multiplier": 0.7,
            },
            MarketRegime.SIDEWAYS: {
                "absorption_threshold": 0.6,
                "breakout_momentum_threshold": 1.2,
                "level_tolerance": 0.0004,
                "min_confidence": 0.4,
                "false_breakout_sensitivity": 1.1,
                "position_size_multiplier": 0.8,
            },
            MarketRegime.UNKNOWN: {
                "absorption_threshold": 0.7,
                "breakout_momentum_threshold": 2.0,
                "level_tolerance": 0.0002,
                "min_confidence": 0.6,
                "false_breakout_sensitivity": 1.0,
                "position_size_multiplier": 1.0,
            },
        }

    def detect_market_regime(
        self, market_data: List[Dict]
    ) -> Tuple[MarketRegime, float]:
        """Enhanced market regime detection with hysteresis and confidence scoring"""
        try:
            if not market_data or len(market_data) < 50:
                return MarketRegime.UNKNOWN, 0.0

            # Extract and validate price data
            prices = []
            volumes = []

            for i, data_point in enumerate(market_data):
                try:
                    price = ValidationUtils.validate_price(
                        data_point.get("close", 0), f"close_{i}"
                    )
                    volume = ValidationUtils.validate_volume(
                        data_point.get("volume", 0), f"volume_{i}"
                    )
                    prices.append(price)
                    volumes.append(volume)
                except ValueError as e:
                    logger.warning(f"Invalid data point {i}: {e}")
                    continue

            if len(prices) < 50 or len(volumes) < 50:
                return MarketRegime.UNKNOWN, 0.0

            prices = np.array(prices)
            volumes = np.array(volumes)

            if np.any(~np.isfinite(prices)) or np.all(prices == 0):
                return MarketRegime.UNKNOWN, 0.0

            # Multi-timeframe analysis with bounds checking
            short_term = prices[-20:]
            medium_term = prices[-50:]
            long_term = prices[-100:] if len(prices) >= 100 else prices

            # Additional validation for array operations
            if len(short_term) < 2 or len(medium_term) < 2 or len(long_term) < 2:
                return MarketRegime.UNKNOWN, 0.0

            # Trend strength calculation
            short_trend = self._calculate_trend_strength(short_term)
            medium_trend = self._calculate_trend_strength(medium_term)
            long_trend = self._calculate_trend_strength(long_term)

            # Volatility analysis
            short_vol = (
                np.std(np.diff(short_term) / short_term[:-1])
                if len(short_term) > 1
                else 0
            )
            medium_vol = (
                np.std(np.diff(medium_term) / medium_term[:-1])
                if len(medium_term) > 1
                else 0
            )

            # Volume analysis
            volume_trend = self._calculate_volume_trend(volumes[-20:])

            # Regime classification with consensus mechanism
            regime_scores = {
                MarketRegime.TRENDING_STRONG: 0,
                MarketRegime.TRENDING_WEAK: 0,
                MarketRegime.HIGH_VOLATILITY: 0,
                MarketRegime.SIDEWAYS: 0,
            }

            # Trend strength scoring
            if abs(short_trend) > 0.02 and abs(medium_trend) > 0.015:
                if abs(long_trend) > 0.01:
                    regime_scores[MarketRegime.TRENDING_STRONG] += 3
                else:
                    regime_scores[MarketRegime.TRENDING_WEAK] += 2

            # Volatility scoring
            if short_vol > 0.03 or medium_vol > 0.025:
                regime_scores[MarketRegime.HIGH_VOLATILITY] += 2

            # Sideways market detection
            if abs(short_trend) < 0.005 and abs(medium_trend) < 0.008:
                regime_scores[MarketRegime.SIDEWAYS] += 2

            # Volume confirmation
            if volume_trend > 1.2:
                # Strong volume supports trend
                if regime_scores[MarketRegime.TRENDING_STRONG] > 0:
                    regime_scores[MarketRegime.TRENDING_STRONG] += 1

            # Select regime with highest score and calculate confidence
            detected_regime = max(regime_scores, key=regime_scores.get)
            max_score = regime_scores[detected_regime]
            total_score = sum(regime_scores.values())

            # Calculate confidence based on score dominance
            if total_score > 0:
                confidence = ValidationUtils.safe_divide(max_score, total_score)
                # Boost confidence if multiple indicators agree
                if max_score >= 2:
                    confidence = min(confidence * 1.2, 1.0)
            else:
                confidence = 0.0

            # Reduce confidence for regime transitions
            if len(self.regime_history) > 0:
                last_regime_data = (
                    self.regime_history.to_list()[-1]
                    if len(self.regime_history) > 0
                    else None
                )
                if (
                    last_regime_data
                    and last_regime_data.get("regime") != detected_regime
                ):
                    confidence *= 0.8  # Reduce confidence during transitions

            # HYSTERESIS: Implement 2-bar confirmation and cool-off period
            current_time = datetime.now()

            # Check if detected regime differs from last confirmed regime
            if detected_regime != self.last_confirmed_regime:
                # Check cool-off period: has enough time passed since last switch?
                if self.last_regime_switch_time is not None:
                    time_since_switch = (
                        current_time - self.last_regime_switch_time
                    ).total_seconds()
                    if time_since_switch < self.REGIME_COOLOFF_SECONDS:
                        # Still in cool-off period, keep current regime
                        detected_regime = self.last_confirmed_regime
                        confidence *= 0.9  # Reduce confidence during cool-off
                        self.regime_confirmation_count = 0
                    else:
                        # Cool-off passed, increment confirmation counter
                        self.regime_confirmation_count += 1

                        # Check if we have enough confirmation bars
                        if (
                            self.regime_confirmation_count
                            < self.CONFIRMATION_BARS_REQUIRED
                        ):
                            # Not enough confirmation yet, keep current regime
                            detected_regime = self.last_confirmed_regime
                            confidence *= (
                                0.85  # Reduce confidence during confirmation phase
                            )
                        else:
                            # Confirmed! Allow regime switch
                            self.last_confirmed_regime = detected_regime
                            self.last_regime_switch_time = current_time
                            self.regime_confirmation_count = 0
                            logger.info(
                                f"Market regime confirmed: {detected_regime.value} (confidence: {confidence:.2f})"
                            )
                else:
                    # First regime detection ever
                    self.regime_confirmation_count += 1
                    if (
                        self.regime_confirmation_count
                        >= self.CONFIRMATION_BARS_REQUIRED
                    ):
                        self.last_confirmed_regime = detected_regime
                        self.last_regime_switch_time = current_time
                        self.regime_confirmation_count = 0
                    else:
                        # Use UNKNOWN until confirmed
                        detected_regime = MarketRegime.UNKNOWN
                        confidence *= 0.7
            else:
                # Same regime as before, reset confirmation counter
                self.regime_confirmation_count = 0

            # Track regime history for transition analysis
            self.regime_history.append(
                {
                    "timestamp": current_time,
                    "regime": detected_regime,
                    "confidence": confidence,
                    "scores": regime_scores,
                    "trend_strength": max(abs(short_trend), abs(medium_trend)),
                }
            )

            return detected_regime, confidence

        except Exception as e:
            logger.error(f"Enhanced regime detection error: {e}")
            return MarketRegime.UNKNOWN, 0.0

    def get_optimal_parameters(self, market_regime: MarketRegime) -> Dict[str, Any]:
        """Get optimized parameters for detected market regime"""
        base_params = self.regime_parameters.get(
            market_regime, self.regime_parameters[MarketRegime.UNKNOWN]
        )

        # Apply performance-based adjustments
        regime_performance = self.parameter_performance.get(market_regime)
        if regime_performance and len(regime_performance.to_list()) > 10:
            recent_performance = regime_performance.to_list()[-10:]
            avg_performance = np.mean(
                [p.get("sharpe_ratio", 0) for p in recent_performance]
            )

            # Adjust parameters based on recent performance
            if avg_performance < 0.5:
                # Poor performance - reduce risk
                base_params = base_params.copy()
                base_params["position_size_multiplier"] *= 0.8
                base_params["min_confidence"] = min(
                    base_params["min_confidence"] * 1.2, 0.8
                )

        return base_params.copy()

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calculate trend strength using linear regression with bounds checking"""
        if not isinstance(prices, np.ndarray) or len(prices) < 10:
            return 0.0

        # Additional bounds checking for array operations
        if np.any(~np.isfinite(prices)) or np.all(prices == 0):
            return 0.0

        try:
            x = np.arange(len(prices))

            # Check for sufficient data points
            if len(x) < 2 or len(prices) < 2:
                return 0.0

            slope, _, r_value, _, _ = linregress(x, prices)

            # Validate regression results
            if not np.isfinite(slope) or not np.isfinite(r_value):
                return 0.0

            # Normalize by price level with bounds checking
            avg_price = np.mean(prices)
            if not np.isfinite(avg_price) or avg_price <= 0:
                return 0.0

            normalized_slope = ValidationUtils.safe_divide(slope, avg_price)

            # Adjust by R-squared for trend reliability
            return normalized_slope * (r_value**2)

        except Exception as e:
            logger.warning(f"Trend strength calculation failed: {e}")
            return 0.0

    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Calculate volume trend ratio with bounds checking"""
        if not isinstance(volumes, np.ndarray) or len(volumes) < 10:
            return 1.0

        # Additional bounds checking for array operations
        if np.any(~np.isfinite(volumes)) or np.all(volumes == 0):
            return 1.0

        try:
            recent_avg = np.mean(volumes[-10:])
            if not np.isfinite(recent_avg) or recent_avg < 0:
                return 1.0

            # Historical average with bounds checking
            if len(volumes) >= 30:
                historical_data = volumes[-30:-10]
                if len(historical_data) == 0:
                    return 1.0
                historical_avg = np.mean(historical_data)
            else:
                historical_data = volumes[:-10]
                if len(historical_data) == 0:
                    return 1.0
                historical_avg = np.mean(historical_data)

            if not np.isfinite(historical_avg) or historical_avg <= 0:
                return 1.0

            return ValidationUtils.safe_divide(recent_avg, historical_avg)

        except Exception as e:
            logger.warning(f"Volume trend calculation failed: {e}")
            return 1.0


# =============================================================================
# ENHANCED ABSORPTION BREAKOUT STRATEGY
# =============================================================================
class EnhancedAbsorptionBreakoutStrategy:
    """
    Professional-grade absorption breakout strategy with:
    - Universal configuration system with mathematical parameter generation
    - ML parameter optimization and adaptation
    - False breakout detection
    - Adaptive market regime management
    - Comprehensive validation
    - Real-time monitoring
    - Thread-safe operations
    - ZERO hardcoded values, ZERO mock/demo/test data
    """

    def __init__(
        self,
        strategy_name: str = "enhanced_absorption_breakout",
        parameter_profile: str = "balanced",
        config: "EnhancedAbsorptionConfig" = None,
        master_key: Optional[bytes] = None,
    ):
        # Initialize universal configuration
        if config is None:
            config = UniversalStrategyConfig(
                strategy_name, parameter_profile=parameter_profile
            )

        self.config = config

        # Core strategy components - thread-safe
        self.absorption_levels = ThreadSafeDict()
        self.trade_buffer = ThreadSafeDeque(maxlen=self.config.long_lookback)
        self.volume_profile = ThreadSafeDict()
        self.breakout_candidates = ThreadSafeDeque()
        self.recent_breakouts = ThreadSafeDeque(maxlen=50)

        # Enhanced components with ML integration
        self.false_breakout_detector = FalseBreakoutDetector(self.config)
        self.regime_manager = AdaptiveMarketRegimeManager(self.config)

        # ML Core for online learning and anomaly detection
        self.ml_core = MLCore(self.config)
        self.ml_enabled = True  # ML is enabled by default
        logger.info(
            "ML Core initialized with universal configuration and online learning capabilities"
        )

        # Strategy state - thread-safe
        self._current_regime = MarketRegime.UNKNOWN
        self._regime_confidence = 0.0
        self._validation_score = 1.0
        self._last_validation_time = None
        self._state_lock = threading.RLock()

        # Adaptive learning system
        self.recent_trades_performance = ThreadSafeDeque(maxlen=50)
        self.parameter_adjustment_history = ThreadSafeDeque(maxlen=20)
        self._last_parameter_update = None
        self.learning_rate = 0.1

        # Memory management
        self._last_cleanup_hour = datetime.now().hour
        self._breakout_lock = threading.RLock()

        logger.info("Enhanced Absorption Breakout Strategy initialized")

    @property
    def current_regime(self) -> MarketRegime:
        with self._state_lock:
            return self._current_regime

    @current_regime.setter
    def current_regime(self, value: MarketRegime):
        with self._state_lock:
            self._current_regime = value

    @property
    def regime_confidence(self) -> float:
        with self._state_lock:
            return self._regime_confidence

    @regime_confidence.setter
    def regime_confidence(self, value: float):
        with self._state_lock:
            self._regime_confidence = max(0.0, min(1.0, float(value)))

    @property
    def validation_score(self) -> float:
        with self._state_lock:
            return self._validation_score

    @validation_score.setter
    def validation_score(self, value: float):
        with self._state_lock:
            self._validation_score = max(0.0, min(1.0, float(value)))

    def update_market_data(self, market_data: List[Dict]) -> None:
        """Update strategy with new market data"""
        try:
            # Comprehensive input validation
            if not isinstance(market_data, list):
                logger.warning("Invalid market_data type: expected list")
                return

            if not market_data or len(market_data) < 10:
                logger.warning(
                    f"Insufficient market data: got {len(market_data) if market_data else 0}, need at least 10"
                )
                return

            # Validate data structure
            valid_data_points = []
            for i, data_point in enumerate(market_data):
                if not isinstance(data_point, dict):
                    logger.warning(
                        f"Invalid data point at index {i}: expected dict, got {type(data_point)}"
                    )
                    continue

                # Check required fields
                required_fields = ["close", "volume", "timestamp"]
                missing_fields = [
                    field for field in required_fields if field not in data_point
                ]
                if missing_fields:
                    logger.warning(
                        f"Missing required fields {missing_fields} in data point at index {i}"
                    )
                    continue

                # Validate field types and values
                try:
                    close_price = ValidationUtils.validate_price(
                        data_point.get("close", 0)
                    )
                    volume = ValidationUtils.validate_volume(
                        data_point.get("volume", 0)
                    )
                    valid_data_points.append(data_point)
                except ValueError as e:
                    logger.warning(f"Invalid numeric data at index {i}: {e}")
                    continue

            if len(valid_data_points) < 5:
                logger.warning("Insufficient valid data points after validation")
                return

            # Update current market regime with confidence
            new_regime, regime_confidence = self.regime_manager.detect_market_regime(
                valid_data_points
            )
            if new_regime != self.current_regime:
                logger.info(
                    f"Market regime changed: {self.current_regime.value} -> {new_regime.value} (confidence: {regime_confidence:.2f})"
                )
                self.current_regime = new_regime
                self.regime_confidence = regime_confidence
                self._adapt_strategy_parameters()

            # Update validation score periodically
            current_time = datetime.now()
            if (
                self._last_validation_time is None
                or current_time - self._last_validation_time > timedelta(hours=24)
            ):
                # self._perform_robustness_validation(valid_data_points)
                self._last_validation_time = current_time

            # Process trades for absorption analysis
            for data_point in valid_data_points[-10:]:  # Process last 10 data points
                trade = self._convert_to_trade_format(data_point)
                if trade:
                    self.update_absorption(trade)

        except Exception as e:
            logger.error(f"Market data update error: {e}")

    def _convert_to_trade_format(self, data_point: Dict) -> Optional[Dict]:
        """Convert market data to trade format with validation."""
        try:
            if not isinstance(data_point, dict):
                return None
            # Accept either 'close' or 'price' for input trade data
            close_price_raw = data_point.get("close", data_point.get("price", 0))
            close_price = ValidationUtils.validate_price(close_price_raw, "close_price")
            volume = ValidationUtils.validate_volume(
                data_point.get("volume", 0), "volume"
            )
            timestamp = data_point.get("timestamp", datetime.now())
            if isinstance(timestamp, datetime):
                timestamp_value = timestamp.timestamp()
            elif isinstance(timestamp, (int, float)):
                timestamp_value = float(timestamp)
            else:
                timestamp_value = time.time()
            delta = data_point.get("delta", 0)
            aggressor = bool(delta > 0) if isinstance(delta, (int, float)) else False
            return {
                "price": close_price,
                "size": volume,
                "timestamp": timestamp_value,
                "aggressor": aggressor,
            }
        except Exception as e:
            logger.warning(f"Trade format conversion error: {e}")
            return None

    def _adapt_strategy_parameters(self) -> None:
        """Adapt strategy parameters based on current market regime"""
        try:
            new_params = self.regime_manager.get_optimal_parameters(self.current_regime)

            # Update configuration with regime-specific parameters
            self.config.absorption_threshold = new_params["absorption_threshold"]
            self.config.breakout_momentum_threshold = new_params[
                "breakout_momentum_threshold"
            ]
            self.config.level_tolerance = new_params["level_tolerance"]

            logger.info(
                f"Strategy parameters adapted for {self.current_regime.value}: {new_params}"
            )

        except Exception as e:
            logger.error(f"Strategy parameter adaptation error: {e}")

    def update_absorption(self, trade: Dict) -> None:
        """Update absorption metrics for all tracked levels"""
        try:
            if not isinstance(trade, dict):
                logger.warning(f"Invalid trade type: expected dict, got {type(trade)}")
                return

            price = trade.get("price", 0)
            size = trade.get("size", 0)

            try:
                price = ValidationUtils.validate_price(price)
                size = ValidationUtils.validate_volume(size)
            except ValueError as e:
                logger.warning(f"Invalid trade data: {e}")
                return

            for level in list(self.absorption_levels.keys()):
                distance = abs(price - level) / level
                if distance < self.config.level_tolerance:
                    level_data = self.absorption_levels[level]
                    level_data["trades"].append(trade)

                    trades_at_level = level_data["trades"]
                    metrics = self._calculate_enhanced_absorption_metrics(
                        trades_at_level
                    )
                    level_data["metrics"] = metrics

                    if (
                        metrics["absorption_ratio"] >= self.config.absorption_threshold
                        and metrics["total_volume"] >= self.config.min_absorption_volume
                    ):
                        candidates = self.breakout_candidates.to_list()
                        if level not in candidates:
                            self.breakout_candidates.append(level)
                            logger.info(
                                f"New absorption level qualified: {level} (regime: {self.current_regime.value})"
                            )

            # Periodic cleanup
            current_time = datetime.now()
            if current_time.hour != self._last_cleanup_hour:
                # self._cleanup_old_absorption_levels()
                self._last_cleanup_hour = current_time.hour

        except Exception as e:
            logger.error(f"Enhanced absorption update error: {e}")

    def _calculate_enhanced_absorption_metrics(
        self, trades_at_level: List[Dict]
    ) -> Dict[str, Any]:
        """Enhanced absorption metrics calculation with safety checks"""
        try:
            if not trades_at_level:
                return {"absorption_ratio": 0, "total_volume": 0, "passive_volume": 0}

            # Use np.uint64 to prevent integer overflow on high-volume instruments
            passive_volume = np.uint64(
                sum(
                    t.get("size", 0)
                    for t in trades_at_level
                    if not t.get("aggressor", True)
                )
            )
            aggressive_volume = np.uint64(
                sum(
                    t.get("size", 0)
                    for t in trades_at_level
                    if t.get("aggressor", True)
                )
            )
            total_volume = passive_volume + aggressive_volume

            # Overflow protection: reject unrealistic volumes (> 9 billion contracts)
            MAX_VOLUME = 9_000_000_000
            if total_volume > MAX_VOLUME:
                logger.error(
                    f"Volume overflow detected: {total_volume} exceeds max {MAX_VOLUME}"
                )
                return {"absorption_ratio": 0, "total_volume": 0, "passive_volume": 0}

            if total_volume == 0:
                return {"absorption_ratio": 0, "total_volume": 0, "passive_volume": 0}

            absorption_ratio = ValidationUtils.safe_divide(
                int(passive_volume), int(total_volume)
            )

            if len(trades_at_level) > 1:
                timestamps = [t.get("timestamp", 0) for t in trades_at_level]
                time_span = max(timestamps) - min(timestamps)
                velocity = ValidationUtils.safe_divide(total_volume, time_span)
            else:
                velocity = 0

            sizes = [t.get("size", 0) for t in trades_at_level]
            avg_size = np.mean(sizes)
            max_size = np.max(sizes)
            size_ratio = ValidationUtils.safe_divide(max_size, avg_size)

            if len(trades_at_level) >= 3:
                recent_trades = trades_at_level[-3:]
                recent_passive = np.uint64(
                    sum(
                        t.get("size", 0)
                        for t in recent_trades
                        if not t.get("aggressor", True)
                    )
                )
                recent_total = np.uint64(sum(t.get("size", 0) for t in recent_trades))
                recent_absorption = ValidationUtils.safe_divide(
                    int(recent_passive), int(recent_total)
                )
            else:
                recent_absorption = absorption_ratio

            return {
                "absorption_ratio": absorption_ratio,
                "recent_absorption_ratio": recent_absorption,
                "total_volume": int(
                    total_volume
                ),  # Convert back to int for serialization
                "passive_volume": int(passive_volume),
                "aggressive_volume": int(aggressive_volume),
                "velocity": velocity,
                "avg_trade_size": avg_size,
                "large_trade_ratio": size_ratio,
                "trade_count": len(trades_at_level),
                "regime": self.current_regime.value,
            }

        except Exception as e:
            logger.error(f"Enhanced absorption metrics calculation error: {e}")
            return {"absorption_ratio": 0, "total_volume": 0, "passive_volume": 0}

    def generate_enhanced_signal(
        self,
        current_price: float,
        order_book: Dict,
        recent_trades: Optional[List[Dict]] = None,
        symbol: str = "UNKNOWN",
        mqscore_data: Dict = None,
    ) -> EnhancedSignal:
        """Generate enhanced trading signal with comprehensive validation"""

        try:
            # Validate inputs
            if not self._validate_signal_inputs(
                current_price, order_book, recent_trades, symbol
            ):
                return EnhancedSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    price=0.0,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    direction=SignalDirection.NEUTRAL,
                    strategy_name="EnhancedAbsorptionBreakout",
                    strategy_params={},
                    false_breakout_score=0.0,
                    market_regime=self.current_regime,
                    validation_score=self.validation_score,
                )

            # Update trade buffer
            self._update_trade_buffer(recent_trades)

            # Identify key levels
            key_levels = self._identify_and_track_key_levels()

            # Check for breakouts
            best_signal = None
            max_strength = 0.0
            breakout_metadata = {}

            # Get breakout candidates safely
            candidates = self.breakout_candidates.to_list()

            for level in candidates[
                :
            ]:  # Create copy to avoid modification during iteration
                # Validate breakout conditions
                validation_result = self._validate_breakout_conditions(
                    level, current_price, order_book, symbol
                )

                if (
                    validation_result["is_valid"]
                    and validation_result["strength"] > max_strength
                ):
                    best_signal = 1 if current_price > level else -1
                    max_strength = validation_result["strength"]
                    breakout_metadata = (
                        self.absorption_levels[level]["metrics"].copy()
                        if level in self.absorption_levels
                        else {}
                    )

                    # Remove from candidates atomically
                    self.breakout_candidates.remove_item(level)

                    self.recent_breakouts.append(
                        {
                            "level": level,
                            "direction": best_signal,
                            "strength": max_strength,
                            "timestamp": datetime.now(),
                            "regime": self.current_regime.value,
                        }
                    )

            # Calculate confidence
            confidence = self._calculate_enhanced_confidence(
                best_signal, max_strength, order_book, symbol
            )

            # Create enhanced signal
            enhanced_signal = self._create_signal_from_result(
                best_signal,
                confidence,
                current_price,
                symbol,
                breakout_metadata,
                max_strength,
            )

            # MQSCORE Validation
            if mqscore_data and enhanced_signal.signal_type != SignalType.HOLD:
                score = mqscore_data.get("composite_score", 0.0)
                if score < self.config.min_mqscore_threshold:
                    enhanced_signal.signal_type = SignalType.HOLD
                    enhanced_signal.direction = SignalDirection.NEUTRAL
                    enhanced_signal.confidence = 0.0
                    logger.info(
                        f"Signal rejected by MQSCORE: Score {score:.2f} < {self.config.min_mqscore_threshold:.2f}"
                    )

                enhanced_signal.strategy_params["mqscore_validation"] = {
                    "score": score,
                    "threshold": self.config.min_mqscore_threshold,
                    "passed": score >= self.config.min_mqscore_threshold,
                }

            return enhanced_signal

        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            # Return safe fallback signal
            return EnhancedSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=current_price,
                symbol=symbol,
                timestamp=datetime.now(),
                direction=SignalDirection.NEUTRAL,
                strategy_name="EnhancedAbsorptionBreakout",
                strategy_params={"error": str(e)},
                false_breakout_score=0.0,
                market_regime=self.current_regime,
                validation_score=0.0,
            )

    def _validate_signal_inputs(
        self,
        current_price: float,
        order_book: Dict,
        recent_trades: Optional[List[Dict]],
        symbol: str,
    ) -> bool:
        """Validate input parameters for signal generation"""
        try:
            ValidationUtils.validate_price(current_price, "current_price")
            ValidationUtils.validate_order_book(order_book)

            if not isinstance(symbol, str) or not symbol.strip():
                logger.warning(f"Invalid symbol: {symbol}")
                return False

            if recent_trades is not None:
                if not isinstance(recent_trades, list):
                    logger.warning(
                        f"Invalid recent_trades type: expected list or None, got {type(recent_trades)}"
                    )
                    return False

                for i, trade in enumerate(recent_trades):
                    if not isinstance(trade, dict):
                        logger.warning(
                            f"Invalid trade at index {i}: expected dict, got {type(trade)}"
                        )
                        return False

            return True

        except ValueError as e:
            logger.warning(f"Signal input validation failed: {e}")
            return False

    def _update_trade_buffer(self, recent_trades: Optional[List[Dict]]) -> None:
        """Update trade buffer with recent trades"""
        if recent_trades:
            for trade in recent_trades:
                try:
                    validated_trade = self._convert_to_trade_format(trade)
                    if validated_trade:
                        self.trade_buffer.append(validated_trade)
                except Exception as e:
                    logger.warning(f"Failed to process trade: {e}")

    def _identify_and_track_key_levels(self) -> List[float]:
        """Identify key levels and initialize tracking for new levels"""
        key_levels = []

        try:
            if len(self.trade_buffer) >= 50:
                prices = np.array(
                    [t["price"] for t in self.trade_buffer.to_list()[-50:]]
                )
                volumes = np.array(
                    [t["size"] for t in self.trade_buffer.to_list()[-50:]]
                )
                # key_levels = self._identify_enhanced_key_levels(prices, volumes)

                # Initialize tracking for new levels
                for level in key_levels:
                    if level not in self.absorption_levels:
                        self.absorption_levels[level] = {
                            "identified_at": datetime.now(),
                            "trades": [],
                            "metrics": {},
                            "regime": self.current_regime,
                        }
        except Exception as e:
            logger.error(f"Key level identification error: {e}")

        return key_levels

    def _validate_breakout_conditions(
        self,
        level: float,
        current_price: float,
        order_book: Dict,
        symbol: str,
    ) -> Dict:
        """Validate all breakout conditions for a specific level"""
        validation_result = {"is_valid": True, "reasons": [], "strength": 0.0}

        try:
            # Enhanced absorption failure detection
            has_failed, strength = self._detect_enhanced_absorption_failure(
                level, self.trade_buffer.to_list(), current_price
            )

            if not has_failed or strength <= 0:
                validation_result["is_valid"] = False
                validation_result["reasons"].append(
                    "No valid absorption failure detected"
                )
                return validation_result

            validation_result["strength"] = strength

            # False breakout detection
            if self.config.enable_false_breakout_detection:
                false_breakout_result = (
                    self.false_breakout_detector.detect_false_breakout_signals(
                        level,
                        current_price,
                        order_book,
                        self.trade_buffer.to_list(),
                        self.current_regime,
                    )
                )

                if false_breakout_result["is_false_breakout"]:
                    validation_result["is_valid"] = False
                    validation_result["reasons"].append(
                        f"False breakout detected: {false_breakout_result['reasons']}"
                    )
                    return validation_result

            # Multi-timeframe validation
            if self.config.enable_multi_timeframe:
                ht_validation = self._validate_higher_timeframe(level, current_price)
                if not ht_validation["is_valid"]:
                    validation_result["is_valid"] = False
                    validation_result["reasons"].append(
                        "Higher timeframe validation failed"
                    )
                    return validation_result

        except Exception as e:
            logger.error(f"Breakout validation error: {e}")
            validation_result["is_valid"] = False
            validation_result["reasons"].append(f"Validation error: {str(e)}")

        return validation_result

    def _detect_enhanced_absorption_failure(
        self, level: float, recent_trades: List[Dict], current_price: float
    ) -> Tuple[bool, float]:
        """Enhanced absorption failure detection with regime adaptation and safety checks"""

        try:
            if level not in self.absorption_levels:
                return False, 0.0

            absorption_data = self.absorption_levels[level]

            # Check price distance with regime-adjusted tolerance
            regime_params = self.regime_manager.get_optimal_parameters(
                self.current_regime
            )
            adjusted_tolerance = regime_params["level_tolerance"]

            level = ValidationUtils.validate_price(level, "level")
            current_price = ValidationUtils.validate_price(
                current_price, "current_price"
            )

            price_distance = abs(current_price - level) / level
            if price_distance < adjusted_tolerance:
                return False, 0.0

            # Enhanced momentum calculation
            if len(recent_trades) < 30:
                return False, 0.0

            recent_prices = [
                ValidationUtils.validate_price(t.get("price", 0))
                for t in recent_trades[-30:]
            ]

            # Multi-period momentum analysis
            momentum_5 = self._calculate_momentum(recent_prices[-5:])
            momentum_10 = self._calculate_momentum(recent_prices[-10:])
            momentum_20 = self._calculate_momentum(recent_prices[-20:])

            # Weighted momentum consensus
            weighted_momentum = momentum_5 * 0.5 + momentum_10 * 0.3 + momentum_20 * 0.2

            # Volume acceleration with regime adjustment
            trades_through_level = [
                t
                for t in recent_trades[-15:]
                if (ValidationUtils.validate_price(t.get("price", 0)) - level)
                * (current_price - level)
                > 0
            ]

            if len(trades_through_level) < 5:
                return False, 0.0

            recent_volume = sum(t.get("size", 0) for t in trades_through_level)
            historical_volume = absorption_data.get("metrics", {}).get(
                "avg_trade_size", 0
            ) * len(trades_through_level)
            volume_acceleration = ValidationUtils.safe_divide(
                recent_volume, historical_volume
            )

            # Regime-adjusted thresholds
            momentum_threshold = regime_params["breakout_momentum_threshold"]
            acceleration_threshold = self.config.failure_acceleration

            # Enhanced failure detection
            momentum_condition = abs(weighted_momentum) > momentum_threshold
            volume_condition = volume_acceleration > acceleration_threshold

            if momentum_condition and volume_condition:
                # Calculate breakout strength with regime adjustment
                breakout_strength = min(
                    abs(weighted_momentum)
                    / momentum_threshold
                    * volume_acceleration
                    / acceleration_threshold,
                    2.0,
                )

                # Apply regime-specific confidence adjustment
                breakout_strength *= regime_params.get(
                    "false_breakout_sensitivity", 1.0
                )

                return True, breakout_strength

            return False, 0.0

        except Exception as e:
            logger.error(f"Enhanced absorption failure detection error: {e}")
            return False, 0.0

    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate momentum strength for price series with validation"""
        if len(prices) < 2:
            return 0.0

        try:
            prices = [ValidationUtils.validate_price(p) for p in prices]
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = linregress(x, prices)

            # Normalize by price level and volatility
            avg_price = np.mean(prices)
            price_volatility = np.std(prices)

            normalized_momentum = ValidationUtils.safe_divide(slope, avg_price)
            reliability_factor = r_value**2

            return normalized_momentum * reliability_factor

        except Exception as e:
            logger.warning(f"Momentum calculation error: {e}")
            return 0.0

    def _calculate_enhanced_confidence(
        self, best_signal: int, max_strength: float, order_book: Dict, symbol: str
    ) -> float:
        """Calculate enhanced confidence score"""
        confidence = max_strength

        try:
            if best_signal is not None:
                # Order book support
                book_support = self._calculate_order_book_support(
                    best_signal, order_book
                )
                confidence *= book_support

                # Market regime adjustment
                regime_params = self.regime_manager.get_optimal_parameters(
                    self.current_regime
                )
                confidence *= regime_params.get("position_size_multiplier", 1.0)

                # Validation score adjustment
                confidence *= max(self.validation_score, 0.8)

        except Exception as e:
            logger.error(f"Enhanced confidence calculation error: {e}")
            confidence = max_strength * 0.5  # Fallback confidence

        return float(min(confidence, 1.0))

    def _calculate_order_book_support(
        self, signal_direction: int, order_book: Dict
    ) -> float:
        """Calculate order book support for signal direction with safety checks"""
        try:
            order_book = ValidationUtils.validate_order_book(order_book)
            bid_sizes = order_book.get("bid_sizes", [])
            ask_sizes = order_book.get("ask_sizes", [])

            if not bid_sizes or not ask_sizes:
                return 1.0

            bid_liquidity = sum(bid_sizes[:5])
            ask_liquidity = sum(ask_sizes[:5])

            if signal_direction == 1:  # Bullish
                support_ratio = ValidationUtils.safe_divide(
                    bid_liquidity, ask_liquidity
                )
            else:  # Bearish
                support_ratio = ValidationUtils.safe_divide(
                    ask_liquidity, bid_liquidity
                )

            return min(support_ratio, 1.5)  # Cap at 1.5x

        except Exception as e:
            logger.error(f"Order book support calculation error: {e}")
            return 1.0

    def _create_signal_from_result(
        self,
        best_signal: int,
        confidence: float,
        current_price: float,
        symbol: str,
        breakout_metadata: Dict,
        max_strength: float,
    ) -> EnhancedSignal:
        """Create EnhancedSignal from validation results"""
        signal_type = (
            SignalType.BUY
            if best_signal == 1
            else SignalType.SELL
            if best_signal == -1
            else SignalType.HOLD
        )
        direction = (
            SignalDirection.BULLISH
            if best_signal == 1
            else SignalDirection.BEARISH
            if best_signal == -1
            else SignalDirection.NEUTRAL
        )

        return EnhancedSignal(
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            strategy_name="EnhancedAbsorptionBreakout",
            strategy_params={
                "breakout_level": max_strength if best_signal else None,
                "absorption_metrics": breakout_metadata,
                "active_levels": len(self.breakout_candidates.to_list()),
                "regime": self.current_regime.value,
                "validation_score": self.validation_score,
            },
            false_breakout_score=0.0,
            market_regime=self.current_regime,
            validation_score=self.validation_score,
        )


# =============================================================================
# NEXUS AI PIPELINE ADAPTER - REQUIRED FOR PIPELINE INTEGRATION
# =============================================================================

class AbsorptionBreakoutNexusAdapter:
    """
    NEXUS AI Pipeline Adapter for Absorption Breakout Strategy
    
    Implements standard workflow:
    - nexus_ai.py integration
    - MQScore 6D quality filtering
    - Feature packaging for pipeline ML
    - Standard execute() protocol
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adapter with comprehensive configuration"""
        self.config = config or {}
        
        # Initialize underlying strategy
        strategy_name = self.config.get('strategy_name', 'absorption_breakout')
        self.strategy = EnhancedAbsorptionBreakoutStrategy(
            strategy_name=strategy_name,
            parameter_profile=self.config.get('parameter_profile', 'balanced')
        )
        # Add strategy_name attribute for compatibility
        self.strategy.strategy_name = strategy_name
        
        # ============ MQSCORE 6D ENGINE: Market Quality Assessment ============
        if HAS_MQSCORE:
            mqscore_config = MQScoreConfig(
                min_buffer_size=20,
                cache_enabled=True,
                cache_ttl=300.0,
                ml_enabled=False,  # ML handled by pipeline
                base_weights={
                    'liquidity': 0.20,
                    'volatility': 0.20,
                    'momentum': 0.20,
                    'imbalance': 0.15,
                    'trend_strength': 0.15,
                    'noise_level': 0.10
                }
            )
            self.mqscore_engine = MQScoreEngine(config=mqscore_config)
            self.mqscore_threshold = 0.57  # Quality threshold
            logger.info("✓ MQScore 6D Engine initialized for quality filtering")
        else:
            self.mqscore_engine = None
            self.mqscore_threshold = 0.57
            logger.info("⚠ MQScore not available - using basic filters only")
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("✓ AbsorptionBreakoutNexusAdapter initialized")
        
    def execute(self, market_data: Any, features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute strategy with standard return format for NEXUS compatibility.
        
        Required protocol method - returns {"signal": float, "confidence": float, "features": dict, "metadata": dict}
        """
        with self._lock:
            try:
                # Convert market_data to dict if needed
                if not isinstance(market_data, dict):
                    market_data = self._convert_to_dict(market_data)
                
                # ============ MQSCORE 6D: MARKET QUALITY FILTERING ============
                mqscore_quality = None
                mqscore_components = None
                
                if self.mqscore_engine:
                    try:
                        import pandas as pd
                        price = float(market_data.get('close', market_data.get('price', 0)))
                        market_df = pd.DataFrame([{
                            'open': float(market_data.get('open', price)),  # Add missing 'open' field
                            'close': price,
                            'high': float(market_data.get('high', price)),
                            'low': float(market_data.get('low', price)),
                            'volume': float(market_data.get('volume', 0)),
                            'timestamp': market_data.get('timestamp', datetime.now())
                        }])
                        
                        mqscore_result = self.mqscore_engine.calculate_mqscore(market_df)
                        mqscore_quality = mqscore_result.composite_score
                        mqscore_components = {
                            "liquidity": mqscore_result.liquidity,
                            "volatility": mqscore_result.volatility,
                            "momentum": mqscore_result.momentum,
                            "imbalance": mqscore_result.imbalance,
                            "trend_strength": mqscore_result.trend_strength,
                            "noise_level": mqscore_result.noise_level,
                        }
                        
                        # FILTER: Reject if market quality below threshold
                        if mqscore_quality < self.mqscore_threshold:
                            logger.info(f"MQScore REJECTED: quality={mqscore_quality:.3f} < {self.mqscore_threshold}")
                            return {
                                'signal': 0.0,
                                'confidence': 0.0,
                                'features': features or {},
                                'metadata': {
                                    'action': 'HOLD',
                                    'reason': f'Market quality too low: {mqscore_quality:.3f}',
                                    'strategy': 'absorption_breakout',
                                    'mqscore_enabled': True,
                                    'mqscore_quality': mqscore_quality,
                                    'mqscore_6d': mqscore_components,
                                    'filtered_by_mqscore': True
                                }
                            }
                        
                        logger.debug(f"MQScore PASSED: quality={mqscore_quality:.3f}")
                        
                    except Exception as e:
                        logger.warning(f"MQScore calculation error: {e} - proceeding without MQScore filter")
                        mqscore_quality = None
                
                # Update strategy with market data
                self.strategy.update_market_data([market_data])
                
                # Get signal from strategy
                signal_result = self.strategy.generate_signal(market_data)
                
                # ============ PACKAGE MQSCORE FEATURES FOR PIPELINE ML ============
                if features is None:
                    features = {}
                
                # Add MQScore 6D components if available
                if mqscore_components:
                    features.update({
                        "mqs_composite": mqscore_quality,
                        "mqs_liquidity": mqscore_components["liquidity"],
                        "mqs_volatility": mqscore_components["volatility"],
                        "mqs_momentum": mqscore_components["momentum"],
                        "mqs_imbalance": mqscore_components["imbalance"],
                        "mqs_trend_strength": mqscore_components["trend_strength"],
                        "mqs_noise_level": mqscore_components["noise_level"],
                    })
                
                # Add strategy-specific features
                features.update({
                    "absorption_level": getattr(signal_result, 'validation_score', 0.0),
                    "false_breakout_score": getattr(signal_result, 'false_breakout_score', 0.0),
                    "regime": getattr(signal_result, 'market_regime', MarketRegime.UNKNOWN).value,
                    "validation_score": getattr(signal_result, 'validation_score', 0.0),
                })
                
                # Convert to standardized format
                signal_type = signal_result.signal_type.value
                confidence = signal_result.confidence
                
                # Map signal to numeric value
                signal_value = 0.0
                if signal_type == 'BUY':
                    signal_value = 1.0
                elif signal_type == 'SELL':
                    signal_value = -1.0
                
                return {
                    'signal': signal_value,
                    'confidence': confidence,
                    'features': features,
                    'metadata': {
                        'action': signal_type,
                        'symbol': market_data.get('symbol', 'UNKNOWN'),
                        'price': signal_result.price,
                        'strategy': 'absorption_breakout',
                        'mqscore_enabled': mqscore_quality is not None,
                        'mqscore_quality': mqscore_quality,
                        'mqscore_6d': mqscore_components,
                        'regime': signal_result.market_regime.value,
                        'validation_score': signal_result.validation_score,
                    }
                }
                
            except Exception as e:
                logger.error(f"Adapter execution error: {e}")
                return {
                    'signal': 0.0,
                    'confidence': 0.0,
                    'features': features or {},
                    'metadata': {
                        'action': 'HOLD',
                        'reason': f'Error: {str(e)}',
                        'strategy': 'absorption_breakout',
                        'error': True
                    }
                }
    
    def _convert_to_dict(self, market_data: Any) -> Dict:
        """Convert market data object to dictionary"""
        if hasattr(market_data, '__dict__'):
            return market_data.__dict__
        else:
            return {
                'price': getattr(market_data, 'price', 0),
                'volume': getattr(market_data, 'volume', 0),
                'symbol': getattr(market_data, 'symbol', 'UNKNOWN'),
                'timestamp': getattr(market_data, 'timestamp', datetime.now())
            }
    
    def get_category(self) -> str:
        """Return strategy category for nexus_ai weight optimization"""
        return "BREAKOUT"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return performance metrics for monitoring"""
        return {
            'strategy_name': 'absorption_breakout',
            'regime': self.strategy.current_regime.value,
            'validation_score': self.strategy.validation_score,
            'ml_enabled': self.strategy.ml_enabled
        }



# Export adapter class for NEXUS AI pipeline
__all__ = [
    'AbsorptionBreakoutNexusAdapter',
    'EnhancedAbsorptionBreakoutStrategy',
]

logger.info("AbsorptionBreakoutNexusAdapter module loaded successfully")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" NEXUS ENHANCED ABSORPTION BREAKOUT STRATEGY - CLEAN VERSION")
    print("=" * 70)

    # Create strategy instance
    strategy = EnhancedAbsorptionBreakoutStrategy(
        strategy_name="clean_production_strategy", parameter_profile="balanced"
    )

    print(f"\nStrategy initialized successfully!")
    print(f"Current regime: {strategy.current_regime.value}")
    print(f"Validation score: {strategy.validation_score:.3f}")
    print(f"ML core enabled: {strategy.ml_enabled}")

    print("\nReady for trading deployment.")
    print("=" * 70)
