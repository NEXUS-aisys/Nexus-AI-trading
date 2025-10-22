"""
MQScore 6D Engine v3.0 - Production-Grade Market Quality Scoring System
Complete rebuild with enhanced performance, security, and maintainability

Author: MQScore Team
Version: 3.0.0
License: MIT
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import warnings
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MQScore3")

# Optional imports with graceful fallbacks
try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available - using fallback for signal processing")
    
    def savgol_filter(x, window_length=None, polyorder=None):
        """Fallback for Savitzky-Golay filter"""
        return x

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    talib = None
    logger.info("TA-Lib not available - using built-in indicators")

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available - ML features disabled")

try:
    from lightgbm import LGBMClassifier
    from xgboost import XGBRegressor
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    logger.info("LightGBM/XGBoost not available - using fallback models")

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.info("ONNX Runtime not available - pre-trained models disabled")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    import pickle
    logger.info("Joblib not available - using pickle for model persistence")

# Suppress warnings in production
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MQScoreConfig:
    """Centralized configuration for MQScore Engine v3.0"""
    
    # Performance Settings
    cache_enabled: bool = True
    cache_ttl: float = 300.0  # 5 minutes
    cache_max_size: int = 1000
    max_workers: int = 6
    batch_size: int = 100
    
    # Data Requirements
    min_buffer_size: int = 20  # Minimum data points needed
    max_buffer_size: int = 1000  # Maximum buffer size
    calibration_size: int = 1000
    
    # ML Settings
    ml_enabled: bool = True
    ml_n_estimators: int = 100
    ml_max_depth: int = 6
    ml_learning_rate: float = 0.1
    ml_random_state: int = 42
    outlier_contamination: float = 0.1
    
    # ONNX Model Paths
    onnx_models_base_path: str = "BEST_UNIQUE_MODELS"
    onnx_classifier_path: str = "06_CLASSIFICATION/Classifier_lightgbm_optimized.onnx"
    onnx_regressor_path: str = "07_REGRESSION/Regressor_lightgbm_optimized.onnx"
    onnx_expected_features: int = 65  # Expected feature count for ONNX models
    
    # Dimension Weights
    liquidity_weights: List[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.15, 0.10]
    )
    momentum_weights: List[float] = field(
        default_factory=lambda: [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10]
    )
    imbalance_weights: List[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.20, 0.15, 0.15]
    )
    trend_strength_weights: List[float] = field(
        default_factory=lambda: [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
    )
    noise_level_weights: List[float] = field(
        default_factory=lambda: [0.20, 0.15, 0.15, 0.15, 0.10, 0.15, 0.10]
    )
    
    # Lookback Periods
    lookback_periods: Dict[str, int] = field(
        default_factory=lambda: {
            "short": 20,
            "medium": 50,
            "long": 100
        }
    )
    
    # Base Dimension Weights
    base_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "liquidity": 0.15,
            "volatility": 0.15,
            "momentum": 0.15,
            "imbalance": 0.15,
            "trend_strength": 0.20,
            "noise_level": 0.20
        }
    )
    
    # Signal Thresholds
    signal_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "hold_range_min": 0.4,
            "hold_range_max": 0.6
        }
    )
    
    # Grade Thresholds
    grade_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "A+": 0.9, "A": 0.8, "A-": 0.75,
            "B+": 0.7, "B": 0.65, "B-": 0.6,
            "C+": 0.55, "C": 0.5, "C-": 0.45,
            "D+": 0.4, "D": 0.35, "D-": 0.3,
            "F": 0.0
        }
    )
    
    # Regime Classification Thresholds
    regime_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "high_volatility": 0.7,
            "low_liquidity": 0.3,
            "strong_trend": 0.7,
            "weak_trend": 0.3,
            "momentum_threshold": 0.6,
            "low_noise": 0.4,
            "high_noise": 0.7,
            "high_composite": 0.7,
            "low_composite": 0.3,
            "trend_threshold": 0.6,
            "balanced_threshold": 0.3
        }
    )
    
    # Regime Probabilities (for rule-based classification)
    regime_probabilities: Dict[str, float] = field(
        default_factory=lambda: {
            "high_vol_low_liq": 0.8,
            "strong_trend": 0.8,
            "ranging": 0.7,
            "high_quality_trend": 0.9,
            "low_quality_choppy": 0.7,
            "balanced": 0.6
        }
    )
    
    # Security Settings
    max_file_size_mb: int = 100
    allowed_model_extensions: List[str] = field(
        default_factory=lambda: [".pkl", ".joblib", ".onnx"]
    )
    allowed_model_paths: List[str] = field(
        default_factory=lambda: ["models", "BEST_UNIQUE_MODELS"]
    )
    
    # Adaptive Settings
    weight_adaptation_enabled: bool = True
    weight_adjustment_rate: float = 0.1
    min_performance_history: int = 20
    recalibration_frequency: int = 100
    
    # Error Recovery
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    fallback_score: float = 0.5
    
    # Performance Monitoring
    performance_history_size: int = 1000
    dimension_performance_history_size: int = 500


# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Thread-safe LRU cache with TTL support and O(1) operations"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.lock = RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with O(1) complexity"""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            entry = self.cache.pop(key)
            
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            
            # Move to end (most recently used)
            self.cache[key] = entry
            self.stats["hits"] += 1
            return entry["value"]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with O(1) complexity"""
        with self.lock:
            # Remove if exists
            if key in self.cache:
                self.cache.pop(key)
            
            # Add new entry
            self.cache[key] = {
                "value": value,
                "timestamp": time.time()
            }
            
            # Evict if over capacity
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)  # Remove oldest
                self.stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total if total > 0 else 0.0
            
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self.stats["evictions"],
                "expired": self.stats["expired"],
                "size": len(self.cache),
                "max_size": self.max_size
            }


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MQScoreComponents:
    """Container for MQScore calculation results"""
    liquidity: float
    volatility: float
    momentum: float
    imbalance: float
    trend_strength: float
    noise_level: float
    composite_score: float
    grade: str
    confidence: float
    timestamp: pd.Timestamp
    processing_time: float = 0.0
    cache_used: bool = False
    regime_probability: Dict[str, float] = field(default_factory=dict)
    dimension_rankings: Dict[str, int] = field(default_factory=dict)
    quality_indicators: Dict[str, float] = field(default_factory=dict)
    adaptive_weights: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# SECURITY VALIDATOR
# ============================================================================

class SecurityValidator:
    """Security validation for model loading and data processing"""
    
    def __init__(self, config: MQScoreConfig):
        self.config = config
    
    def validate_model_path(self, path: str) -> bool:
        """Validate model file path for security"""
        try:
            # Resolve to absolute path
            p = Path(path).resolve()
            
            # Check if path is within allowed directories
            allowed_dirs = [Path(d).resolve() for d in self.config.allowed_model_paths]
            if not any(str(p).startswith(str(allowed)) for allowed in allowed_dirs):
                logger.error(f"Path {p} outside allowed directories")
                return False
            
            # Check file extension
            if p.suffix not in self.config.allowed_model_extensions:
                logger.error(f"Invalid file extension: {p.suffix}")
                return False
            
            # Check file size if it exists
            if p.exists():
                size_mb = p.stat().st_size / (1024 * 1024)
                if size_mb > self.config.max_file_size_mb:
                    logger.error(f"File too large: {size_mb:.2f} MB")
                    return False
            
            # Create parent directory if needed
            p.parent.mkdir(parents=True, exist_ok=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Path validation error: {e}")
            return False
    
    def validate_input_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input market data"""
        errors = []
        
        # Check for required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        for col in required_cols:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                errors.append(f"Column {col} must be numeric")
        
        # Check for NaN/Inf values
        if data[required_cols].isnull().any().any():
            errors.append("Data contains NaN values")
        
        if np.isinf(data[required_cols].values).any():
            errors.append("Data contains infinite values")
        
        # Validate OHLC relationships
        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            invalid_ohlc = (
                (data["high"] < data["low"]) |
                (data["high"] < data["open"]) |
                (data["high"] < data["close"]) |
                (data["low"] > data["open"]) |
                (data["low"] > data["close"])
            )
            if invalid_ohlc.any():
                errors.append("Invalid OHLC relationships detected")
        
        # Check value ranges
        if "close" in data.columns:
            if (data["close"] <= 0).any():
                errors.append("Close prices must be positive")
            if (data["close"] > 1e6).any():
                errors.append("Close prices exceed reasonable bounds")
        
        if "volume" in data.columns:
            if (data["volume"] < 0).any():
                errors.append("Volume cannot be negative")
        
        return len(errors) == 0, errors


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Performance monitoring and metrics tracking"""
    
    def __init__(self):
        self.calculation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.total_calculations = 0
        self.start_time = time.time()
        self.lock = RLock()
    
    def record_calculation_time(self, operation: str, duration: float) -> None:
        """Record calculation time for an operation"""
        with self.lock:
            self.calculation_times[operation].append(duration)
    
    def record_error(self, operation: str) -> None:
        """Record an error for an operation"""
        with self.lock:
            self.error_counts[operation] += 1
    
    def increment_calculations(self) -> None:
        """Increment total calculation counter"""
        with self.lock:
            self.total_calculations += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            avg_times = {}
            for op, times in self.calculation_times.items():
                if times:
                    avg_times[op] = np.mean(list(times))
            
            return {
                "uptime_seconds": uptime,
                "total_calculations": self.total_calculations,
                "calculations_per_second": self.total_calculations / uptime if uptime > 0 else 0,
                "avg_calculation_times": avg_times,
                "error_counts": dict(self.error_counts),
                "error_rate": sum(self.error_counts.values()) / max(1, self.total_calculations)
            }


# ============================================================================
# ONNX HANDLER
# ============================================================================

class ONNXHandler:
    """Safe ONNX model handler with validation"""
    
    def __init__(self, model_path: str, expected_features: int):
        self.model_path = Path(model_path)
        self.expected_features = expected_features
        self.session: Optional[ort.InferenceSession] = None
        
        if HAS_ONNX and self.model_path.exists():
            try:
                self._load_model()
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
    
    def _load_model(self) -> None:
        """Load and verify ONNX model"""
        # Verify file size
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:  # Max 100MB
            raise ValueError(f"Model file too large: {size_mb:.2f} MB")
        
        # Load model
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"]
        )
        
        # Verify input shape
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 2 and input_shape[1] != self.expected_features:
            logger.warning(
                f"Feature count mismatch: expected {self.expected_features}, "
                f"model expects {input_shape[1]}"
            )
    
    def predict(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Run prediction with validation"""
        if self.session is None:
            return None
        
        try:
            # Validate input shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Ensure correct number of features
            if features.shape[1] != self.expected_features:
                # Pad or truncate as needed
                if features.shape[1] < self.expected_features:
                    padding = np.zeros((features.shape[0], self.expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :self.expected_features]
            
            # Convert to float32
            if features.dtype != np.float32:
                features = features.astype(np.float32)
            
            # Check for NaN/Inf
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("Features contain NaN or Inf values")
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: features})
            
            return outputs[0] if outputs else None
            
        except Exception as e:
            logger.error(f"ONNX prediction error: {e}")
            return None


# ============================================================================
# DIMENSION CALCULATORS
# ============================================================================

class DimensionCalculator:
    """Base class for dimension calculations"""
    
    def __init__(self, config: MQScoreConfig):
        self.config = config
        self.lookback = config.lookback_periods
    
    def calculate(self, data: pd.DataFrame) -> float:
        """Calculate dimension score (to be overridden)"""
        raise NotImplementedError


class LiquidityCalculator(DimensionCalculator):
    """Calculate liquidity dimension score"""
    
    def calculate(self, data: pd.DataFrame) -> float:
        try:
            # Volume-based liquidity
            volume = data["volume"].rolling(self.lookback["short"]).mean()
            volume_cv = data["volume"].rolling(self.lookback["short"]).std() / (volume + 1e-10)
            volume_consistency = 1.0 - volume_cv.fillna(0.5)
            
            # Price range as liquidity proxy
            price_range = (data["high"] - data["low"]) / (data["close"] + 1e-10)
            relative_spread = price_range.rolling(self.lookback["short"]).mean()
            
            # Price impact approximation
            returns = data["close"].pct_change()
            volume_normalized = data["volume"] / (volume + 1e-10)
            price_impact = np.abs(returns) / (volume_normalized + 1e-10)
            price_impact_score = 1.0 / (1.0 + price_impact.rolling(self.lookback["short"]).mean())
            
            # Spread consistency
            spread_consistency = 1.0 - relative_spread.rolling(self.lookback["short"]).std()
            
            # Market depth indicator
            q25 = data["volume"].rolling(self.lookback["medium"]).quantile(0.25)
            q75 = data["volume"].rolling(self.lookback["medium"]).quantile(0.75)
            volume_depth = (q75 - q25) / (volume + 1e-10)
            
            # Combine components
            components = [
                1.0 - relative_spread.fillna(0.5).iloc[-1],
                volume_consistency.fillna(0.5).iloc[-1],
                price_impact_score.fillna(0.5).iloc[-1],
                spread_consistency.fillna(0.5).iloc[-1],
                np.clip(volume_depth.fillna(0.5).iloc[-1], 0, 1)
            ]
            
            # Weighted average
            weights = self.config.liquidity_weights
            score = sum(w * c for w, c in zip(weights, components))
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Liquidity calculation error: {e}")
            return 0.5


class VolatilityCalculator(DimensionCalculator):
    """Calculate volatility dimension score"""
    
    def calculate(self, data: pd.DataFrame) -> float:
        try:
            returns = data["close"].pct_change()
            
            # Multi-timeframe volatility
            vol_short = returns.rolling(self.lookback["short"]).std() * np.sqrt(252)
            vol_medium = returns.rolling(self.lookback["medium"]).std() * np.sqrt(252)
            vol_long = returns.rolling(self.lookback["long"]).std() * np.sqrt(252)
            
            # Volatility ratios
            vol_ratio_sm = vol_short / (vol_medium + 1e-10)
            vol_ratio_ml = vol_medium / (vol_long + 1e-10)
            
            # Volatility persistence
            squared_returns = returns ** 2
            vol_persistence = squared_returns.rolling(self.lookback["short"]).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0
            )
            
            # Intraday volatility
            intraday_vol = (data["high"] - data["low"]) / (data["close"] + 1e-10)
            intraday_consistency = 1.0 - intraday_vol.rolling(self.lookback["short"]).std()
            
            # Combine components
            vol_predictability = 1.0 - np.abs(vol_ratio_sm - 1.0).fillna(0.5)
            clustering_score = np.abs(vol_persistence).fillna(0.5)
            regime_stability = 1.0 - np.abs(vol_ratio_ml - 1.0).fillna(0.5)
            
            components = [
                vol_predictability.iloc[-1],
                clustering_score.iloc[-1],
                intraday_consistency.fillna(0.5).iloc[-1],
                regime_stability.iloc[-1]
            ]
            
            score = np.mean(components)
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return 0.5


class MomentumCalculator(DimensionCalculator):
    """Calculate momentum dimension score"""
    
    def calculate(self, data: pd.DataFrame) -> float:
        try:
            # Price momentum
            returns_short = data["close"].pct_change(self.lookback["short"])
            returns_medium = data["close"].pct_change(self.lookback["medium"])
            
            # Volume momentum
            volume_sma = data["volume"].rolling(self.lookback["medium"]).mean()
            volume_momentum = (data["volume"] / (volume_sma + 1e-10) - 1).fillna(0)
            
            # Trend alignment
            ma_short = data["close"].rolling(self.lookback["short"]).mean()
            ma_medium = data["close"].rolling(self.lookback["medium"]).mean()
            ma_long = data["close"].rolling(self.lookback["long"]).mean()
            
            trend_alignment_sm = (ma_short > ma_medium).astype(float)
            trend_alignment_ml = (ma_medium > ma_long).astype(float)
            trend_consistency = (trend_alignment_sm == trend_alignment_ml).astype(float)
            
            # RSI momentum
            rsi = self._calculate_rsi(data["close"].values, 14)
            rsi_momentum = np.abs(rsi - 50) / 50
            
            # MACD momentum
            macd_hist = self._calculate_macd_histogram(data["close"].values)
            macd_momentum = np.abs(macd_hist) / (np.abs(macd_hist).max() + 1e-10)
            
            # Momentum consistency
            returns_1d = data["close"].pct_change(1)
            momentum_consistency = 1.0 - np.abs(returns_1d).rolling(self.lookback["short"]).std()
            
            # Combine components with robust NaN handling
            components = [
                np.abs(returns_short.iloc[-1]) if not np.isnan(returns_short.iloc[-1]) else 0.0,
                np.abs(returns_medium.iloc[-1]) if not np.isnan(returns_medium.iloc[-1]) else 0.0,
                float(np.abs(volume_momentum.iloc[-1])) if not np.isnan(volume_momentum.iloc[-1]) else 0.0,
                float(trend_consistency.iloc[-1]) if not np.isnan(trend_consistency.iloc[-1]) else 0.5,
                float(rsi_momentum[-1]) if len(rsi_momentum) > 0 and not np.isnan(rsi_momentum[-1]) else 0.5,
                float(macd_momentum[-1]) if len(macd_momentum) > 0 and not np.isnan(macd_momentum[-1]) else 0.5,
                float(momentum_consistency.fillna(0.5).iloc[-1]) if not np.isnan(momentum_consistency.iloc[-1]) else 0.5
            ]
            
            # Ensure no NaN in components
            components = [0.5 if np.isnan(c) or np.isinf(c) else c for c in components]
            
            weights = self.config.momentum_weights
            score = sum(w * c for w, c in zip(weights, components))
            
            # Final safety check
            if np.isnan(score) or np.isinf(score):
                logger.warning(f"Momentum score is NaN/Inf, returning default 0.5")
                return 0.5
            
            return float(np.clip(score, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return 0.5
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        if HAS_TALIB and talib is not None:
            return talib.RSI(prices, timeperiod=period)
        
        # Manual RSI calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean()
        avg_loss = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi.fillna(50).values])
    
    def _calculate_macd_histogram(self, prices: np.ndarray) -> np.ndarray:
        """Calculate MACD histogram"""
        if HAS_TALIB and talib is not None:
            _, _, hist = talib.MACD(prices)
            return hist
        
        # Manual MACD calculation
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=12, adjust=False).mean()
        ema_slow = prices_series.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return histogram.fillna(0).values


class ImbalanceCalculator(DimensionCalculator):
    """Calculate order flow imbalance dimension score"""
    
    def calculate(self, data: pd.DataFrame) -> float:
        try:
            # VWAP calculation
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            vwap = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()
            price_vs_vwap = (data["close"] - vwap) / (vwap + 1e-10)
            vwap_deviation = np.abs(price_vs_vwap.rolling(self.lookback["short"]).mean())
            
            # Volume imbalance
            up_volume = data["volume"].where(data["close"] > data["open"], 0)
            down_volume = data["volume"].where(data["close"] < data["open"], 0)
            total_directional = up_volume + down_volume + 1e-10
            volume_imbalance = (up_volume - down_volume) / total_directional
            
            # Order flow strength
            flow_strength = np.abs(data["close"].pct_change()) * data["volume"]
            flow_mean = flow_strength.rolling(self.lookback["short"]).mean()
            flow_std = flow_strength.rolling(self.lookback["short"]).std()
            flow_consistency = 1.0 - flow_std / (flow_mean + 1e-10)
            
            # Price level concentration
            price_bins = pd.cut(data["close"], bins=20, labels=False)
            level_counts = pd.Series(price_bins).value_counts(normalize=True)
            level_concentration = 1.0 - level_counts.max() if len(level_counts) > 0 else 0.5
            
            # Tick imbalance
            price_changes = data["close"].diff()
            volume_weighted_changes = price_changes * data["volume"]
            tick_imbalance = volume_weighted_changes.rolling(self.lookback["short"]).sum()
            tick_balance_score = 1.0 / (1.0 + np.abs(tick_imbalance))
            
            # Combine components
            components = [
                1.0 - vwap_deviation.fillna(0.5).iloc[-1],
                1.0 - np.abs(volume_imbalance.rolling(self.lookback["short"]).mean()).fillna(0).iloc[-1],
                flow_consistency.fillna(0.5).iloc[-1],
                level_concentration,
                tick_balance_score.fillna(0.5).iloc[-1]
            ]
            
            weights = self.config.imbalance_weights
            score = sum(w * c for w, c in zip(weights, components))
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Imbalance calculation error: {e}")
            return 0.5


class TrendStrengthCalculator(DimensionCalculator):
    """Calculate trend strength dimension score"""
    
    def calculate(self, data: pd.DataFrame) -> float:
        try:
            # ADX calculation
            adx_value = self._calculate_adx(data)
            
            # Moving average convergence
            ma_fast = data["close"].rolling(self.lookback["short"]).mean()
            ma_slow = data["close"].rolling(self.lookback["long"]).mean()
            ma_convergence = np.abs(ma_fast - ma_slow) / (ma_slow + 1e-10)
            
            # Trend consistency
            price_above_ma = (data["close"] > ma_fast).astype(float)
            trend_consistency = price_above_ma.rolling(self.lookback["short"]).mean()
            trend_consistency = np.abs(trend_consistency - 0.5) * 2
            
            # Breakout strength
            bb_period = self.lookback["short"]
            bb_middle = data["close"].rolling(bb_period).mean()
            bb_std = data["close"].rolling(bb_period).std()
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            
            breakout_upper = np.maximum(0, (data["close"] - bb_upper) / (bb_upper + 1e-10))
            breakout_lower = np.maximum(0, (bb_lower - data["close"]) / (bb_lower + 1e-10))
            breakout_strength = np.maximum(breakout_upper, breakout_lower)
            
            # Volume-confirmed trend
            volume_trend = data["volume"].rolling(self.lookback["short"]).mean()
            price_trend = data["close"].rolling(self.lookback["short"]).mean()
            
            price_direction = (price_trend > price_trend.shift(5)).astype(float)
            volume_confirmation = (data["volume"] > volume_trend).astype(float)
            volume_confirmed = (price_direction == volume_confirmation).astype(float)
            volume_trend_score = volume_confirmed.rolling(self.lookback["short"]).mean()
            
            # Directional movement
            high_diff = data["high"].diff()
            low_diff = -data["low"].diff()
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_di = pd.Series(plus_dm).rolling(14).mean()
            minus_di = pd.Series(minus_dm).rolling(14).mean()
            directional_strength = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            
            # Combine components
            components = [
                adx_value / 100.0,
                directional_strength.fillna(0.5).iloc[-1],
                ma_convergence.fillna(0.5).iloc[-1],
                trend_consistency.fillna(0.5).iloc[-1],
                breakout_strength.fillna(0).iloc[-1],
                volume_trend_score.fillna(0.5).iloc[-1]
            ]
            
            weights = self.config.trend_strength_weights
            score = sum(w * c for w, c in zip(weights, components))
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Trend strength calculation error: {e}")
            return 0.5
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        if HAS_TALIB and talib is not None:
            adx = talib.ADX(data["high"].values, data["low"].values, data["close"].values, timeperiod=period)
            return adx[-1] if len(adx) > 0 and not np.isnan(adx[-1]) else 50.0
        
        # Manual ADX calculation
        try:
            high = data["high"].values
            low = data["low"].values
            close = data["close"].values
            
            # True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = pd.Series(tr).rolling(period).mean()
            
            # Directional Movement
            up_move = high[1:] - high[:-1]
            down_move = low[:-1] - low[1:]
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-10)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx = dx.rolling(period).mean()
            
            return adx.iloc[-1] if not adx.empty and not np.isnan(adx.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0


class NoiseLevelCalculator(DimensionCalculator):
    """Calculate noise level dimension score"""
    
    def calculate(self, data: pd.DataFrame) -> float:
        try:
            price_changes = data["close"].diff().fillna(0)
            price_returns = data["close"].pct_change().fillna(0)
            
            # Signal-to-noise ratio using Savitzky-Golay filter
            snr_sg = self._calculate_snr_sg(data["close"].values)
            
            # Moving average based SNR
            ma_signal = data["close"].rolling(self.lookback["short"]).mean()
            ma_noise = data["close"] - ma_signal
            snr_ma = ma_signal.var() / (ma_noise.var() + 1e-10)
            
            # Frequency analysis
            high_freq_changes = price_changes.rolling(5).std()
            low_freq_changes = price_changes.rolling(self.lookback["short"]).std()
            frequency_ratio = low_freq_changes / (high_freq_changes + 1e-10)
            
            # Volume noise
            volume_changes = data["volume"].pct_change().fillna(0)
            volume_noise = volume_changes.rolling(self.lookback["short"]).std()
            volume_signal = data["volume"].rolling(self.lookback["short"]).mean()
            volume_snr = volume_signal / (volume_noise + 1e-10)
            
            # Price-volume correlation
            price_volume_corr = data["close"].rolling(self.lookback["short"]).corr(data["volume"])
            correlation_score = np.abs(price_volume_corr)
            
            # Microstructure noise
            microstructure_noise = (data["high"] - data["low"]) / (data["close"] + 1e-10)
            microstructure_consistency = 1.0 - microstructure_noise.rolling(self.lookback["short"]).std()
            
            # Autocorrelation
            returns_autocorr = price_returns.rolling(self.lookback["short"]).apply(
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
                raw=False
            )
            autocorr_score = np.abs(returns_autocorr)
            
            # Combine components (higher score = lower noise)
            components = [
                np.tanh(snr_sg / 10.0),
                np.tanh(snr_ma / 10.0) if not np.isnan(snr_ma) else 0.5,
                np.tanh(frequency_ratio.fillna(1).iloc[-1] / 5.0),
                np.tanh(volume_snr.fillna(1).iloc[-1] / 10.0),
                correlation_score.fillna(0.5).iloc[-1],
                microstructure_consistency.fillna(0.5).iloc[-1],
                autocorr_score.fillna(0.5).iloc[-1]
            ]
            
            weights = self.config.noise_level_weights
            score = sum(w * c for w, c in zip(weights, components))
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Noise level calculation error: {e}")
            return 0.5
    
    def _calculate_snr_sg(self, prices: np.ndarray) -> float:
        """Calculate signal-to-noise ratio using Savitzky-Golay filter"""
        try:
            if not HAS_SCIPY or len(prices) < 21:
                return 1.0
            
            window_length = min(21, len(prices) // 2 * 2 - 1)
            if window_length < 5:
                return 1.0
            
            smooth_price = savgol_filter(prices, window_length=window_length, polyorder=3)
            price_signal = smooth_price[-self.lookback["short"]:]
            price_noise = prices[-self.lookback["short"]:] - price_signal
            
            signal_var = np.var(price_signal)
            noise_var = np.var(price_noise)
            
            return signal_var / (noise_var + 1e-10) if noise_var > 0 else 10.0
            
        except Exception:
            return 1.0


# ============================================================================
# MAIN ENGINE
# ============================================================================

class MQScoreEngine:
    """Main MQScore 6D Engine v3.0"""
    
    def __init__(self, config: Optional[MQScoreConfig] = None):
        # Configuration
        self.config = config or MQScoreConfig()
        
        # Core components
        self.cache = CacheManager(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl
        )
        self.security_validator = SecurityValidator(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Dimension calculators
        self.dimensions = {
            "liquidity": LiquidityCalculator(self.config),
            "volatility": VolatilityCalculator(self.config),
            "momentum": MomentumCalculator(self.config),
            "imbalance": ImbalanceCalculator(self.config),
            "trend_strength": TrendStrengthCalculator(self.config),
            "noise_level": NoiseLevelCalculator(self.config)
        }
        
        # ML components
        self.onnx_classifier: Optional[ONNXHandler] = None
        self.onnx_regressor: Optional[ONNXHandler] = None
        self.regime_classifier = None
        self.outlier_detector = None
        
        # Adaptive weights
        self.adaptive_weights = self.config.base_weights.copy()
        
        # Calibration data
        self.calibration_data: deque = deque(maxlen=self.config.calibration_size)
        self.dimension_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.dimension_performance_history_size)
        )
        
        # Thread pool for async execution
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Thread safety
        self._lock = RLock()
        
        # Initialize ML components
        self._initialize_ml_components()
        
        logger.info("MQScore Engine v3.0 initialized successfully")
    
    def _initialize_ml_components(self) -> None:
        """Initialize ML and ONNX components"""
        # Load ONNX models if available
        if HAS_ONNX:
            classifier_path = Path(self.config.onnx_models_base_path) / self.config.onnx_classifier_path
            if classifier_path.exists():
                self.onnx_classifier = ONNXHandler(
                    str(classifier_path),
                    self.config.onnx_expected_features
                )
                logger.info(f"Loaded ONNX classifier: {classifier_path}")
            
            regressor_path = Path(self.config.onnx_models_base_path) / self.config.onnx_regressor_path
            if regressor_path.exists():
                self.onnx_regressor = ONNXHandler(
                    str(regressor_path),
                    self.config.onnx_expected_features
                )
                logger.info(f"Loaded ONNX regressor: {regressor_path}")
        
        # Initialize sklearn models if available
        if HAS_SKLEARN:
            try:
                self.outlier_detector = IsolationForest(
                    contamination=self.config.outlier_contamination,
                    random_state=self.config.ml_random_state,
                    n_estimators=self.config.ml_n_estimators
                )
            except Exception as e:
                logger.warning(f"Failed to initialize outlier detector: {e}")
    
    def calculate_mqscore(self, market_data: pd.DataFrame) -> MQScoreComponents:
        """Calculate MQScore for given market data"""
        start_time = time.time()
        
        try:
            # Validate input data
            is_valid, errors = self.security_validator.validate_input_data(market_data)
            if not is_valid:
                logger.error(f"Input validation failed: {errors}")
                return self._get_default_mqscore()
            
            # Check data sufficiency
            if len(market_data) < self.config.min_buffer_size:
                logger.warning(f"Insufficient data: {len(market_data)} < {self.config.min_buffer_size}")
                return self._get_default_mqscore()
            
            # Check cache
            cache_key = self._generate_cache_key(market_data)
            if self.config.cache_enabled:
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    cached_result.cache_used = True
                    return cached_result
            
            # Calculate dimensions
            dimension_scores = self._calculate_dimensions(market_data)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(dimension_scores)
            
            # Classify market regime
            regime_probability = self._classify_market_regime(market_data, dimension_scores)
            
            # Calculate confidence
            confidence = self._calculate_confidence(market_data, dimension_scores, composite_score)
            
            # Determine grade
            grade = self._score_to_grade(composite_score)
            
            # Calculate quality indicators
            quality_indicators = self._calculate_quality_indicators(dimension_scores)
            
            # Rank dimensions
            dimension_rankings = self._rank_dimensions(dimension_scores)
            
            # Create result
            processing_time = time.time() - start_time
            
            result = MQScoreComponents(
                liquidity=dimension_scores["liquidity"],
                volatility=dimension_scores["volatility"],
                momentum=dimension_scores["momentum"],
                imbalance=dimension_scores["imbalance"],
                trend_strength=dimension_scores["trend_strength"],
                noise_level=dimension_scores["noise_level"],
                composite_score=composite_score,
                grade=grade,
                confidence=confidence,
                timestamp=market_data.index[-1] if not market_data.empty else pd.Timestamp.now(),
                processing_time=processing_time,
                cache_used=False,
                regime_probability=regime_probability,
                dimension_rankings=dimension_rankings,
                quality_indicators=quality_indicators,
                adaptive_weights=self.adaptive_weights.copy()
            )
            
            # Cache result
            if self.config.cache_enabled:
                self.cache.set(cache_key, result)
            
            # Update calibration data
            self._update_calibration(result)
            
            # Record performance metrics
            self.performance_monitor.record_calculation_time("total", processing_time)
            self.performance_monitor.increment_calculations()
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating MQScore: {e}")
            self.performance_monitor.record_error("calculation")
            return self._get_default_mqscore()
    
    def _calculate_dimensions(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all dimension scores"""
        dimension_scores = {}
        
        for name, calculator in self.dimensions.items():
            try:
                start = time.time()
                score = calculator.calculate(market_data)
                duration = time.time() - start
                
                dimension_scores[name] = score
                self.performance_monitor.record_calculation_time(f"dim_{name}", duration)
                
                # Record dimension performance
                self.dimension_performance[name].append({
                    "score": score,
                    "time": duration,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                dimension_scores[name] = 0.5
                self.performance_monitor.record_error(f"dim_{name}")
        
        return dimension_scores
    
    def _calculate_composite_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        try:
            # Use adaptive weights if enabled
            weights = self.adaptive_weights if self.config.weight_adaptation_enabled else self.config.base_weights
            
            # Weighted average
            total = sum(weights[dim] * score for dim, score in dimension_scores.items())
            
            return np.clip(total, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {e}")
            return 0.5
    
    def _classify_market_regime(
        self,
        market_data: pd.DataFrame,
        dimension_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Classify market regime using ML or rule-based approach"""
        try:
            # Try ONNX classifier first
            if self.onnx_classifier is not None:
                features = self._prepare_ml_features(market_data, dimension_scores)
                predictions = self.onnx_classifier.predict(features)
                
                if predictions is not None:
                    # Map predictions to regime probabilities
                    regime_classes = [
                        "HIGH_VOLATILITY_LOW_LIQUIDITY",
                        "STRONG_TREND",
                        "RANGING",
                        "HIGH_QUALITY_TREND",
                        "LOW_QUALITY_CHOPPY",
                        "BALANCED"
                    ]
                    
                    if predictions.ndim == 2 and predictions.shape[1] == len(regime_classes):
                        # Probabilities returned
                        return dict(zip(regime_classes, predictions[0]))
                    elif predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
                        # Class index returned
                        class_idx = int(predictions.flatten()[0])
                        if 0 <= class_idx < len(regime_classes):
                            probs = [0.0] * len(regime_classes)
                            probs[class_idx] = 1.0
                            return dict(zip(regime_classes, probs))
            
            # Fallback to rule-based classification
            return self._classify_regime_rule_based(dimension_scores)
            
        except Exception as e:
            logger.error(f"Error in regime classification: {e}")
            return {"BALANCED": 1.0}
    
    def _classify_regime_rule_based(self, dimension_scores: Dict[str, float]) -> Dict[str, float]:
        """Rule-based regime classification"""
        regimes = {
            "HIGH_VOLATILITY_LOW_LIQUIDITY": 0.0,
            "STRONG_TREND": 0.0,
            "RANGING": 0.0,
            "HIGH_QUALITY_TREND": 0.0,
            "LOW_QUALITY_CHOPPY": 0.0,
            "BALANCED": 0.0
        }
        
        vol = dimension_scores.get("volatility", 0.5)
        liq = dimension_scores.get("liquidity", 0.5)
        trend = dimension_scores.get("trend_strength", 0.5)
        momentum = dimension_scores.get("momentum", 0.5)
        noise = dimension_scores.get("noise_level", 0.5)
        composite = np.mean(list(dimension_scores.values()))
        
        # High volatility, low liquidity
        if vol > self.config.regime_thresholds["high_volatility"] and \
           liq < self.config.regime_thresholds["low_liquidity"]:
            regimes["HIGH_VOLATILITY_LOW_LIQUIDITY"] = self.config.regime_probabilities["high_vol_low_liq"]
        
        # Strong trend
        if trend > self.config.regime_thresholds["strong_trend"] and \
           momentum > self.config.regime_thresholds["momentum_threshold"]:
            regimes["STRONG_TREND"] = self.config.regime_probabilities["strong_trend"]
        
        # Ranging market
        if trend < self.config.regime_thresholds["weak_trend"] and \
           noise < self.config.regime_thresholds["low_noise"]:
            regimes["RANGING"] = self.config.regime_probabilities["ranging"]
        
        # High quality trending
        if composite > self.config.regime_thresholds["high_composite"] and \
           trend > self.config.regime_thresholds["trend_threshold"]:
            regimes["HIGH_QUALITY_TREND"] = self.config.regime_probabilities["high_quality_trend"]
        
        # Low quality/choppy
        if composite < self.config.regime_thresholds["low_composite"] or \
           noise > self.config.regime_thresholds["high_noise"]:
            regimes["LOW_QUALITY_CHOPPY"] = self.config.regime_probabilities["low_quality_choppy"]
        
        # Balanced (default)
        if max(regimes.values()) < self.config.regime_thresholds["balanced_threshold"]:
            regimes["BALANCED"] = self.config.regime_probabilities["balanced"]
        
        # Normalize probabilities
        total = sum(regimes.values())
        if total > 0:
            regimes = {k: v / total for k, v in regimes.items()}
        else:
            regimes["BALANCED"] = 1.0
        
        return regimes
    
    def _prepare_ml_features(
        self,
        market_data: pd.DataFrame,
        dimension_scores: Dict[str, float]
    ) -> np.ndarray:
        """Prepare features for ML models (89 features total)"""
        features = []
        
        # Dimension scores (6 features)
        for dim in ["liquidity", "volatility", "momentum", "imbalance", "trend_strength", "noise_level"]:
            features.append(dimension_scores.get(dim, 0.5))
        
        # Market data features
        if len(market_data) >= 20:
            returns = market_data["close"].pct_change()
            
            # Rolling statistics (4 features)
            features.extend([
                returns.iloc[-20:].mean(),
                returns.iloc[-20:].std(),
                returns.iloc[-20:].skew() if len(returns.iloc[-20:]) > 2 else 0,
                returns.iloc[-20:].kurt() if len(returns.iloc[-20:]) > 3 else 0
            ])
            
            # Price level features (2 features)
            current_price = market_data["close"].iloc[-1]
            ma_20 = market_data["close"].rolling(20).mean().iloc[-1]
            features.extend([
                (current_price / ma_20 - 1) if ma_20 > 0 else 0,
                (market_data["high"].iloc[-1] - market_data["low"].iloc[-1]) / current_price if current_price > 0 else 0
            ])
            
            # Volume features (1 feature)
            avg_volume = market_data["volume"].rolling(20).mean().iloc[-1]
            current_volume = market_data["volume"].iloc[-1]
            features.append((current_volume / avg_volume - 1) if avg_volume > 0 else 0)
            
            # Extended features to reach 89 total
            # Add rolling window features for different timeframes
            for window in [5, 10, 15, 25, 30]:
                if len(market_data) >= window:
                    # Price features (2 per window = 10 total)
                    price_ma = market_data["close"].rolling(window).mean().iloc[-1]
                    price_std = market_data["close"].rolling(window).std().iloc[-1]
                    features.extend([
                        (current_price / price_ma - 1) if price_ma > 0 else 0,
                        price_std / current_price if current_price > 0 else 0
                    ])
                    
                    # Volume features (2 per window = 10 total)
                    vol_ma = market_data["volume"].rolling(window).mean().iloc[-1]
                    vol_std = market_data["volume"].rolling(window).std().iloc[-1]
                    features.extend([
                        (current_volume / vol_ma - 1) if vol_ma > 0 else 0,
                        vol_std / vol_ma if vol_ma > 0 else 0
                    ])
                    
                    # Return features (4 per window = 20 total)
                    window_returns = returns.iloc[-window:]
                    features.extend([
                        window_returns.mean() if len(window_returns) > 0 else 0,
                        window_returns.std() if len(window_returns) > 0 else 0,
                        window_returns.skew() if len(window_returns) > 2 else 0,
                        window_returns.kurt() if len(window_returns) > 3 else 0
                    ])
            
            # RSI features (3 features)
            for period in [9, 14, 21]:
                if len(market_data) >= period:
                    delta = market_data["close"].diff()
                    gain = delta.where(delta > 0, 0).rolling(period).mean().iloc[-1]
                    loss = (-delta.where(delta < 0, 0)).rolling(period).mean().iloc[-1]
                    rs = gain / loss if loss > 0 else 100
                    rsi = 100 - (100 / (1 + rs))
                    features.append(rsi)
            
            # Momentum features (5 features)
            for lag in [1, 3, 5, 10, 15]:
                if len(market_data) > lag:
                    momentum = (current_price / market_data["close"].iloc[-lag-1]) - 1
                    features.append(momentum)
            
            # Spread proxy (1 feature)
            features.append(0.0)  # Placeholder for bid-ask spread
            
            # Price range features (3 features)
            for window in [5, 10, 20]:
                if len(market_data) >= window:
                    high_low_ratio = (market_data["high"].iloc[-window:].max() /
                                    market_data["low"].iloc[-window:].min()) - 1
                    features.append(high_low_ratio)
        
        # Ensure exactly 89 features
        while len(features) < self.config.onnx_expected_features:
            features.append(0.0)
        
        if len(features) > self.config.onnx_expected_features:
            features = features[:self.config.onnx_expected_features]
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_confidence(
        self,
        market_data: pd.DataFrame,
        dimension_scores: Dict[str, float],
        composite_score: float
    ) -> float:
        """Calculate confidence score"""
        try:
            confidence_factors = []
            
            # Data quality
            data_completeness = 1.0 - (market_data.isnull().sum().sum() /
                                      (len(market_data) * len(market_data.columns)))
            confidence_factors.append(data_completeness)
            
            # Score consistency
            if len(self.calibration_data) > 10:
                recent_scores = [item.composite_score for item in list(self.calibration_data)[-10:]]
                score_stability = 1.0 - np.std(recent_scores)
                confidence_factors.append(score_stability)
            else:
                confidence_factors.append(0.7)
            
            # Dimension agreement
            dimension_values = list(dimension_scores.values())
            dimension_agreement = 1.0 - np.std(dimension_values)
            confidence_factors.append(dimension_agreement)
            
            # Market activity
            if len(market_data) >= 20:
                vol_std = market_data["volume"].rolling(20).std().iloc[-1]
                vol_mean = market_data["volume"].rolling(20).mean().iloc[-1]
                volume_consistency = 1.0 - (vol_std / (vol_mean + 1e-10))
                confidence_factors.append(np.clip(volume_consistency, 0, 1))
            else:
                confidence_factors.append(0.5)
            
            # Performance factor
            metrics = self.performance_monitor.get_metrics()
            error_rate = metrics.get("error_rate", 0)
            performance_factor = 1.0 - error_rate
            confidence_factors.append(performance_factor)
            
            confidence = np.mean(confidence_factors)
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_quality_indicators(self, dimension_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate quality indicators"""
        try:
            indicators = {}
            
            # Overall quality
            indicators["overall_quality"] = np.mean(list(dimension_scores.values()))
            
            # Quality consistency
            indicators["quality_consistency"] = 1.0 - np.std(list(dimension_scores.values()))
            
            # Market efficiency
            efficiency_components = [
                dimension_scores.get("liquidity", 0.5),
                dimension_scores.get("noise_level", 0.5),
                1.0 - dimension_scores.get("volatility", 0.5)
            ]
            indicators["market_efficiency"] = np.mean(efficiency_components)
            
            # Trading favorability
            trading_components = [
                dimension_scores.get("trend_strength", 0.5),
                dimension_scores.get("momentum", 0.5),
                dimension_scores.get("liquidity", 0.5)
            ]
            indicators["trading_favorability"] = np.mean(trading_components)
            
            # Risk level
            risk_components = [
                dimension_scores.get("volatility", 0.5),
                1.0 - dimension_scores.get("liquidity", 0.5),
                1.0 - dimension_scores.get("noise_level", 0.5)
            ]
            indicators["risk_level"] = np.mean(risk_components)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating quality indicators: {e}")
            return {}
    
    def _rank_dimensions(self, dimension_scores: Dict[str, float]) -> Dict[str, int]:
        """Rank dimensions by score"""
        sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
        return {dim: rank + 1 for rank, (dim, _) in enumerate(sorted_dims)}
    
    def _generate_cache_key(self, market_data: pd.DataFrame) -> str:
        """Generate cache key for market data"""
        try:
            # Use last N rows for cache key
            recent_data = market_data.tail(self.config.lookback_periods["long"])
            
            # Create hash from key data points
            key_data = f"{recent_data['close'].iloc[-1]:.4f}"
            key_data += f"_{recent_data['volume'].iloc[-1]:.0f}"
            key_data += f"_{len(recent_data)}"
            key_data += f"_{recent_data['close'].mean():.4f}"
            
            return hashlib.sha256(key_data.encode()).hexdigest()
            
        except Exception:
            return str(time.time())
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        for grade, threshold in sorted(self.config.grade_thresholds.items(), 
                                      key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return grade
        return "F"
    
    def _update_calibration(self, result: MQScoreComponents) -> None:
        """Update calibration data"""
        with self._lock:
            self.calibration_data.append(result)
            
            # Periodic recalibration
            if len(self.calibration_data) % self.config.recalibration_frequency == 0:
                self._recalibrate()
    
    def _recalibrate(self) -> None:
        """Recalibrate adaptive weights"""
        if not self.config.weight_adaptation_enabled:
            return
        
        if len(self.calibration_data) < self.config.min_performance_history:
            return
        
        try:
            # Analyze recent performance
            recent_data = list(self.calibration_data)[-50:]
            
            # Find best performing weight combinations
            best_score = 0
            best_weights = self.config.base_weights.copy()
            
            for item in recent_data:
                if item.composite_score > best_score:
                    best_score = item.composite_score
                    if item.adaptive_weights:
                        best_weights = item.adaptive_weights
            
            # Gradually adjust weights
            rate = self.config.weight_adjustment_rate
            for dim in self.adaptive_weights:
                current = self.adaptive_weights[dim]
                target = best_weights[dim]
                self.adaptive_weights[dim] = current + rate * (target - current)
            
            # Normalize weights
            total = sum(self.adaptive_weights.values())
            self.adaptive_weights = {k: v/total for k, v in self.adaptive_weights.items()}
            
            logger.info(f"Weights recalibrated: {self.adaptive_weights}")
            
        except Exception as e:
            logger.error(f"Error in recalibration: {e}")
    
    def _get_default_mqscore(self) -> MQScoreComponents:
        """Return default MQScore for error cases"""
        return MQScoreComponents(
            liquidity=0.5,
            volatility=0.5,
            momentum=0.5,
            imbalance=0.5,
            trend_strength=0.5,
            noise_level=0.5,
            composite_score=0.5,
            grade="C",
            confidence=0.3,
            timestamp=pd.Timestamp.now(),
            processing_time=0.0,
            cache_used=False,
            regime_probability={"BALANCED": 1.0},
            dimension_rankings={dim: 1 for dim in ["liquidity", "volatility", "momentum", 
                                                  "imbalance", "trend_strength", "noise_level"]},
            quality_indicators={},
            adaptive_weights=self.config.base_weights.copy()
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.performance_monitor.get_metrics()
        cache_stats = self.cache.get_stats()
        
        return {
            **metrics,
            "cache": cache_stats,
            "adaptive_weights": self.adaptive_weights.copy(),
            "calibration_size": len(self.calibration_data),
            "ml_enabled": self.onnx_classifier is not None or self.onnx_regressor is not None
        }
    
    def save_model(self, path: Optional[str] = None) -> bool:
        """Save model to disk"""
        try:
            model_path = path or "models/mqscore_v3_model.pkl"
            
            if not self.security_validator.validate_model_path(model_path):
                logger.error("Model path validation failed")
                return False
            
            model_data = {
                "config": self.config,
                "adaptive_weights": self.adaptive_weights,
                "calibration_data": list(self.calibration_data)[-100:],
                "version": "3.0.0",
                "timestamp": datetime.now().isoformat()
            }
            
            if HAS_JOBLIB:
                import joblib
                joblib.dump(model_data, model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load model from disk"""
        try:
            model_path = path or "models/mqscore_v3_model.pkl"
            
            if not self.security_validator.validate_model_path(model_path):
                logger.error("Model path validation failed")
                return False
            
            if not Path(model_path).exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            if HAS_JOBLIB:
                import joblib
                model_data = joblib.load(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Load components
            if "adaptive_weights" in model_data:
                self.adaptive_weights = model_data["adaptive_weights"]
            
            if "calibration_data" in model_data:
                for item in model_data["calibration_data"]:
                    self.calibration_data.append(item)
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            self.executor.shutdown(wait=True)
            self.cache.clear()
            logger.info("MQScore Engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Alias for backward compatibility
EnhancedMQScoreEngine = MQScoreEngine
EnhancedMQScoreComponents = MQScoreComponents

def create_enhanced_mqscore_engine(config: Optional[Dict[str, Any]] = None) -> MQScoreEngine:
    """Create MQScore engine (backward compatibility)"""
    if config:
        # Convert dict config to MQScoreConfig
        config_obj = MQScoreConfig()
        for key, value in config.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
        return MQScoreEngine(config_obj)
    return MQScoreEngine()


# ============================================================================
# ASYNC WRAPPER
# ============================================================================

class AsyncMQScoreEngine:
    """Async wrapper for MQScore engine"""
    
    def __init__(self, config: Optional[MQScoreConfig] = None):
        self.engine = MQScoreEngine(config)
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    async def calculate_mqscore_enhanced(self, market_data: pd.DataFrame) -> MQScoreComponents:
        """Async calculation of MQScore"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.engine.calculate_mqscore,
            market_data
        )
    
    async def cleanup(self) -> None:
        """Async cleanup"""
        self._executor.shutdown(wait=True)
        self.engine.cleanup()


# ============================================================================
# STRATEGY ADAPTER
# ============================================================================

class MQScore6DStrategy:
    """Strategy adapter for orchestrator compatibility"""
    
    def __init__(self):
        self.engine = MQScoreEngine()
        self.data_buffer = []
        self.config = self.engine.config
    
    def analyze(self, market_data: Any) -> Any:
        """Analyze market data and generate signal"""
        try:
            # Convert input to dict
            if hasattr(market_data, "__dict__"):
                data_dict = market_data.__dict__
            else:
                data_dict = market_data
            
            # Extract price data
            close_price = float(data_dict.get("close", data_dict.get("price", 100.0)))
            
            # Add to buffer
            current_data = {
                "close": close_price,
                "high": float(data_dict.get("high", close_price)),
                "low": float(data_dict.get("low", close_price)),
                "open": float(data_dict.get("open", close_price)),
                "volume": float(data_dict.get("volume", 1000.0)),
                "timestamp": data_dict.get("timestamp", time.time())
            }
            
            self.data_buffer.append(current_data)
            
            # Limit buffer size
            if len(self.data_buffer) > self.config.max_buffer_size:
                self.data_buffer = self.data_buffer[-self.config.max_buffer_size:]
            
            # Check minimum data requirement
            if len(self.data_buffer) < self.config.min_buffer_size:
                return self._create_neutral_signal(close_price)
            
            # Create DataFrame
            df = pd.DataFrame(self.data_buffer)
            df.index = pd.DatetimeIndex([pd.Timestamp.fromtimestamp(d["timestamp"]) 
                                        for d in self.data_buffer])
            
            # Calculate MQScore
            result = self.engine.calculate_mqscore(df)
            
            # Generate signal
            return self._create_signal(result, close_price)
            
        except Exception as e:
            logger.error(f"Strategy analysis error: {e}")
            return self._create_error_signal(str(e))
    
    def _create_signal(self, mqscore: MQScoreComponents, price: float) -> Any:
        """Create trading signal from MQScore"""
        class Signal:
            def __init__(self, score, confidence, price, config):
                buy_threshold = config.signal_thresholds["buy_threshold"]
                sell_threshold = config.signal_thresholds["sell_threshold"]
                
                self.signal_type = (
                    "BUY" if score > buy_threshold else
                    "SELL" if score < sell_threshold else
                    "HOLD"
                )
                self.confidence = confidence
                self.entry_price = price
                self.price = price
                self.timestamp = time.time()
                self.strategy_name = "mqscore_6d_engine"
                self.symbol = "ES"
                self.direction = (
                    "BULLISH" if score > buy_threshold else
                    "BEARISH" if score < sell_threshold else
                    "NEUTRAL"
                )
                self.metadata = {"mqscore": score}
        
        return Signal(mqscore.composite_score, mqscore.confidence, price, self.config)
    
    def _create_neutral_signal(self, price: float) -> Any:
        """Create neutral signal for insufficient data"""
        class NeutralSignal:
            def __init__(self, price, buffer_size):
                self.signal_type = "HOLD"
                self.confidence = 0.1
                self.entry_price = price
                self.price = price
                self.timestamp = time.time()
                self.strategy_name = "mqscore_6d_engine"
                self.symbol = "ES"
                self.direction = "NEUTRAL"
                self.metadata = {"insufficient_data": True, "buffer_size": buffer_size}
        
        return NeutralSignal(price, len(self.data_buffer))
    
    def _create_error_signal(self, error_msg: str) -> Any:
        """Create error signal"""
        class ErrorSignal:
            def __init__(self, error_msg):
                self.signal_type = "HOLD"
                self.confidence = 0.05
                self.entry_price = 100.0
                self.price = 100.0
                self.timestamp = time.time()
                self.strategy_name = "mqscore_6d_engine"
                self.symbol = "ES"
                self.direction = "NEUTRAL"
                self.metadata = {"error": error_msg, "fallback": True}
        
        return ErrorSignal(error_msg)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage
    import sys
    
    print("MQScore 6D Engine v3.0")
    print("-" * 50)
    
    # Create sample data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    sample_data = pd.DataFrame({
        "open": 100 + np.random.randn(100).cumsum(),
        "high": 102 + np.random.randn(100).cumsum(),
        "low": 98 + np.random.randn(100).cumsum(),
        "close": 100 + np.random.randn(100).cumsum(),
        "volume": 1000 + np.random.randint(0, 500, 100)
    }, index=dates)
    
    # Ensure OHLC relationships
    sample_data["high"] = sample_data[["open", "high", "low", "close"]].max(axis=1)
    sample_data["low"] = sample_data[["open", "low", "close"]].min(axis=1)
    
    # Create engine
    engine = MQScoreEngine()
    
    # Calculate MQScore
    result = engine.calculate_mqscore(sample_data)
    
    # Display results
    print(f"Composite Score: {result.composite_score:.3f}")
    print(f"Grade: {result.grade}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"\nDimension Scores:")
    for dim in ["liquidity", "volatility", "momentum", "imbalance", "trend_strength", "noise_level"]:
        score = getattr(result, dim)
        rank = result.dimension_rankings.get(dim, 0)
        print(f"  {dim:15s}: {score:.3f} (Rank #{rank})")
    
    print(f"\nMarket Regime:")
    for regime, prob in result.regime_probability.items():
        if prob > 0.1:
            print(f"  {regime:30s}: {prob:.1%}")
    
    print(f"\nQuality Indicators:")
    for indicator, value in result.quality_indicators.items():
        print(f"  {indicator:20s}: {value:.3f}")
    
    print(f"\nProcessing Time: {result.processing_time:.3f}s")
    print(f"Cache Used: {result.cache_used}")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    metrics = engine.get_performance_metrics()
    print(f"  Total Calculations: {metrics['total_calculations']}")
    print(f"  Cache Hit Rate: {metrics['cache']['hit_rate']:.1%}")
    print(f"  Error Rate: {metrics['error_rate']:.1%}")
    
    # Cleanup
    engine.cleanup()
    
    print("\nMQScore Engine v3.0 - Test Complete")
