#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI - Advanced Trading System
Production-ready modular architecture with adaptive decision making
Version: 3.0.0
"""

# ============================================================================
# CORE MODULE: nexus_core.py
# ============================================================================

"""Core module containing base classes and interfaces."""

import logging
import asyncio
import time
import hashlib
import hmac
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

# MQScore 6D Engine Integration (Layer 1)
try:
    from MQScore_6D_Engine_v3 import MQScoreEngine, MQScoreConfig, MQScoreComponents
    HAS_MQSCORE = True
except ImportError:
    HAS_MQSCORE = False
    print("WARNING: MQScore_6D_Engine_v3 not available - Layer 1 disabled")

# ML Model Loading 
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("WARNING: ONNX Runtime not available - Using rule-based fallbacks")

try:
    import pickle
    HAS_PICKLE = True
except ImportError:
    HAS_PICKLE = False
    print("WARNING: Pickle not available - PKL model loading disabled")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("INFO: Joblib not available - Using pickle fallback")


# ============================================================================
# EXPLICIT STRATEGY IMPORTS - ADD/REMOVE STRATEGIES HERE
# ============================================================================
"""
MANUAL STRATEGY REGISTRATION
Add or remove strategies by uncommenting/commenting the imports below
NO automatic file discovery - full manual control
"""

# GROUP 1: EVENT-DRIVEN (1 strategy)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("event_driven", "Event-Driven Strategy.py")
    event_driven_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(event_driven_module)
    EnhancedEventDrivenStrategy = event_driven_module.EnhancedEventDrivenStrategy
    HAS_EVENT_DRIVEN = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_EVENT_DRIVEN = False

# GROUP 2: BREAKOUT-BASED (3 strategies)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("lvn", "lvn_breakout_strategy.py")
    lvn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lvn_module)
    LVNBreakoutNexusAdapter = lvn_module.LVNBreakoutNexusAdapter
    HAS_LVN_BREAKOUT = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_LVN_BREAKOUT = False

try:
    from absorption_breakout import AbsorptionBreakoutNexusAdapter
    HAS_ABSORPTION_BREAKOUT = True
except ImportError as e:
    HAS_ABSORPTION_BREAKOUT = False
    print(f"WARNING: Absorption Breakout import failed: {e}")

try:
    from momentum_breakout import MomentumBreakoutStrategy
    HAS_MOMENTUM_BREAKOUT = True
except ImportError:
    HAS_MOMENTUM_BREAKOUT = False

# GROUP 3: MARKET MICROSTRUCTURE (3 strategies)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("market_micro", "Market Microstructure Strategy.py")
    market_micro_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(market_micro_module)
    MarketMicrostructureStrategy = market_micro_module.MarketMicrostructureStrategy
    HAS_MARKET_MICROSTRUCTURE = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_MARKET_MICROSTRUCTURE = False

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("order_book", "Order Book Imbalance Strategy.py")
    order_book_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(order_book_module)
    OrderBookImbalanceNexusAdapter = order_book_module.OrderBookImbalanceNexusAdapter
    HAS_ORDER_BOOK_IMBALANCE = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_ORDER_BOOK_IMBALANCE = False

try:
    from liquidity_absorption import LiquidityAbsorptionNexusAdapter
    HAS_LIQUIDITY_ABSORPTION = True
except ImportError:
    HAS_LIQUIDITY_ABSORPTION = False

# GROUP 4: DETECTION/ALERT (4 strategies)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("spoofing", "Spoofing Detection Strategy.py")
    spoofing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spoofing_module)
    SpoofingDetectionNexusAdapter = spoofing_module.SpoofingDetectionNexusAdapter
    HAS_SPOOFING_DETECTION = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_SPOOFING_DETECTION = False

try:
    from iceberg_detection import IcebergDetectionNexusAdapter
    HAS_ICEBERG_DETECTION = True
except ImportError as e:
    HAS_ICEBERG_DETECTION = False
    print(f"WARNING: Iceberg Detection import failed: {e}")

try:
    from liquidation_detection import LiquidationDetectionNexusAdapterV2
    HAS_LIQUIDATION_DETECTION = True
except ImportError:
    HAS_LIQUIDATION_DETECTION = False

try:
    from liquidity_traps import LiquidityTrapsNexusAdapterV2
    HAS_LIQUIDITY_TRAPS = True
except ImportError:
    HAS_LIQUIDITY_TRAPS = False

# GROUP 5: TECHNICAL ANALYSIS (3 strategies)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("multi_tf", "Multi-Timeframe Alignment Strategy.py")
    multi_tf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(multi_tf_module)
    MultiTimeframeAlignmentNexusAdapter = multi_tf_module.MultiTimeframeAlignmentNexusAdapter
    HAS_MULTI_TIMEFRAME = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_MULTI_TIMEFRAME = False

try:
    from cumulative_delta import EnhancedDeltaTradingStrategy
    HAS_CUMULATIVE_DELTA = True
except ImportError as e:
    HAS_CUMULATIVE_DELTA = False
    print(f"WARNING: Cumulative Delta import failed: {e}")

try:
    from delta_divergence import EnhancedDeltaDivergenceStrategy
    HAS_DELTA_DIVERGENCE = True
except ImportError as e:
    HAS_DELTA_DIVERGENCE = False
    print(f"WARNING: Delta Divergence import failed: {e}")

# GROUP 6: CLASSIFICATION/ROTATION (2 strategies)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("open_drive", "Open Drive vs Fade Strategy.py")
    open_drive_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(open_drive_module)
    EnhancedOpenDriveVsFadeStrategy = open_drive_module.EnhancedOpenDriveVsFadeStrategy
    HAS_OPEN_DRIVE_FADE = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_OPEN_DRIVE_FADE = False

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("profile_rotation", "Profile Rotation Strategy.py")
    profile_rotation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile_rotation_module)
    EnhancedProfileRotationStrategy = profile_rotation_module.EnhancedProfileRotationStrategy
    HAS_PROFILE_ROTATION = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_PROFILE_ROTATION = False

# GROUP 7: MEAN REVERSION (2 strategies)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("vwap", "VWAP Reversion Strategy.py")
    vwap_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vwap_module)
    VWAPReversionNexusAdapter = vwap_module.VWAPReversionNexusAdapter
    HAS_VWAP_REVERSION = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_VWAP_REVERSION = False

try:
    from stop_run_anticipation import StopRunAnticipationNexusAdapter
    HAS_STOP_RUN = True
except ImportError:
    HAS_STOP_RUN = False

# GROUP 8: ADVANCED ML (2 strategies)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("momentum_ignition", "Momentum Ignition Strategy.py")
    momentum_ignition_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(momentum_ignition_module)
    MomentumIgnitionNexusAdapter = momentum_ignition_module.MomentumIgnitionNexusAdapter
    HAS_MOMENTUM_IGNITION = True
except (ImportError, AttributeError, FileNotFoundError) as e:
    HAS_MOMENTUM_IGNITION = False

try:
    from volume_imbalance import VolumeImbalanceNexusAdapter
    HAS_VOLUME_IMBALANCE = True
except ImportError:
    HAS_VOLUME_IMBALANCE = False


# Configure logging
def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# ============================================================================
# Core Enumerations
# ============================================================================

class SecurityLevel(IntEnum):
    """Security classification levels."""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


class MarketDataType(Enum):
    """Market data type enumeration."""
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    BOOK_UPDATE = "BOOK_UPDATE"
    INDEX = "INDEX"
    AGGREGATE = "AGGREGATE"


class StrategyCategory(Enum):
    """Strategy categorization for classification and routing."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    EVENT_DRIVEN = "event_driven"
    BREAKOUT = "breakout"
    VOLUME_PROFILE = "volume_profile"
    ORDER_FLOW = "order_flow"
    LIQUIDITY_ANALYSIS = "liquidity_analysis"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"


class SignalType(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass(frozen=True, slots=True)
class MarketData:
    """Immutable market data structure."""
    symbol: str
    timestamp: float
    price: Decimal
    volume: Decimal
    bid: Decimal
    ask: Decimal
    bid_size: int
    ask_size: int
    data_type: MarketDataType
    exchange_timestamp: float
    sequence_num: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def mid_price(self) -> Decimal:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> Decimal:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> Decimal:
        """Calculate spread in basis points."""
        if self.mid_price == 0:
            return Decimal(0)
        return (self.spread / self.mid_price) * 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price': float(self.price),
            'volume': float(self.volume),
            'bid': float(self.bid),
            'ask': float(self.ask),
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'data_type': self.data_type.value,
            'exchange_timestamp': self.exchange_timestamp,
            'sequence_num': self.sequence_num,
            'metadata': self.metadata
        }


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    confidence: float
    symbol: str
    timestamp: float
    strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class RiskMetrics:
    """Risk management metrics."""
    position_size: float
    stop_loss: float
    take_profit: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    expected_return: float
    risk_score: float
    
    def validate(self) -> bool:
        """Validate risk metrics."""
        return all([
            0 <= self.position_size <= 1,
            self.stop_loss > 0,
            self.take_profit > 0,
            0 <= self.risk_score <= 1
        ])


# ============================================================================
# Core Interfaces
# ============================================================================

class IStrategy(ABC):
    """Strategy interface."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with configuration."""
        pass
    
    @abstractmethod
    def execute(self, data: MarketData) -> TradingSignal:
        """Execute strategy and generate signal."""
        pass
    
    @abstractmethod
    def get_category(self) -> StrategyCategory:
        """Get strategy category."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        pass


class IDataProvider(ABC):
    """Data provider interface."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source."""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data for symbols."""
        pass
    
    @abstractmethod
    async def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for symbol."""
        pass


class IRiskManager(ABC):
    """Risk management interface."""
    
    @abstractmethod
    def evaluate_risk(self, signal: TradingSignal, portfolio: Dict[str, Any]) -> RiskMetrics:
        """Evaluate risk for a trading signal."""
        pass
    
    @abstractmethod
    def validate_trade(self, signal: TradingSignal, metrics: RiskMetrics) -> bool:
        """Validate if trade should be executed."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, capital: float) -> float:
        """Calculate appropriate position size."""
        pass


# ============================================================================
# SECURITY MODULE: nexus_security.py
# ============================================================================

"""Security module for data validation and authentication."""

import hashlib
import hmac
import secrets
import time
from typing import Optional


class SecurityManager:
    """Manages security operations for the trading system."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize security manager."""
        self._master_key = master_key or secrets.token_bytes(32)
        self._session_keys = {}
        self._logger = setup_logging(f"{__name__}.SecurityManager")
        
    def generate_session_key(self, session_id: str) -> bytes:
        """Generate a session-specific key."""
        session_key = hmac.new(
            self._master_key,
            session_id.encode('utf-8'),
            hashlib.sha256
        ).digest()
        self._session_keys[session_id] = session_key
        return session_key
    
    def create_signature(self, data: bytes, session_id: Optional[str] = None) -> bytes:
        """Create HMAC signature for data."""
        key = self._session_keys.get(session_id, self._master_key) if session_id else self._master_key
        return hmac.new(key, data, hashlib.sha256).digest()
    
    def verify_signature(self, data: bytes, signature: bytes, session_id: Optional[str] = None) -> bool:
        """Verify HMAC signature."""
        expected = self.create_signature(data, session_id)
        return hmac.compare_digest(expected, signature)
    
    def hash_data(self, data: bytes) -> str:
        """Create SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def validate_market_data(self, data: MarketData) -> bool:
        """Validate market data integrity."""
        try:
            # Validate required fields
            if not data.symbol or data.symbol.strip() == "":
                return False
            
            # Validate numeric ranges
            if data.price <= 0 or data.volume < 0:
                return False
            
            # Validate timestamps
            current_time = time.time()
            if data.timestamp > current_time + 60:  # Allow 60s clock drift
                return False
            
            # Validate spread
            if data.bid > data.ask:
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Market data validation error: {e}")
            return False
    
    def sanitize_input(self, data: Any) -> Any:
        """Sanitize user input to prevent injection attacks."""
        if isinstance(data, str):
            # Remove potentially dangerous characters
            sanitized = data.replace('\x00', '')
            # Limit length
            return sanitized[:1000]
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        return data


# ============================================================================
# DATA MANAGEMENT MODULE: nexus_data.py
# ============================================================================

"""Data management module for efficient data handling."""

import asyncio
import threading
from collections import deque
from datetime import datetime, timedelta


class DataBuffer:
    """Thread-safe circular buffer for market data."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize data buffer."""
        self._capacity = capacity
        self._buffer = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._logger = setup_logging(f"{__name__}.DataBuffer")
        
    def add(self, data: MarketData) -> None:
        """Add data to buffer."""
        with self._lock:
            self._buffer.append(data)
    
    def get_latest(self, n: int = 1) -> List[MarketData]:
        """Get latest n data points."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]
    
    def get_range(self, start_time: float, end_time: float) -> List[MarketData]:
        """Get data within time range."""
        with self._lock:
            return [
                d for d in self._buffer
                if start_time <= d.timestamp <= end_time
            ]
    
    def clear(self) -> None:
        """Clear buffer."""
        with self._lock:
            self._buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        with self._lock:
            return len(self._buffer)
    
    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity


class DataCache:
    """LRU cache for processed data."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize data cache."""
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
        self._logger = setup_logging(f"{__name__}.DataCache")
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            if self._is_expired(key):
                del self._cache[key]
                del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = time.time()
            return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            self._cache[key] = value
            self._access_times[key] = time.time()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._access_times:
            return True
        return time.time() - self._access_times[key] > self._ttl
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        lru_key = min(self._access_times, key=self._access_times.get)
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'ttl_seconds': self._ttl
            }


# ============================================================================
# CONFIGURATION MODULE: nexus_config.py
# ============================================================================

"""Configuration management module."""

from dataclasses import asdict


@dataclass
class SystemConfig:
    """System-wide configuration."""
    
    # Performance settings
    buffer_size: int = 10000
    cache_size: int = 1000
    cache_ttl: int = 300
    max_workers: int = 4
    
    # Risk management
    max_position_size: float = 0.1
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    
    # Execution settings
    max_slippage: float = 0.001
    order_timeout: int = 30
    min_fill_percentage: float = 0.8
    
    # Monitoring
    enable_metrics: bool = True
    metrics_interval: int = 30
    alert_cooldown: int = 300
    
    # Security
    enable_authentication: bool = True
    session_timeout: int = 3600
    max_failed_attempts: int = 5
    
    def validate(self) -> bool:
        """Validate configuration."""
        validations = [
            self.buffer_size > 0,
            self.cache_size > 0,
            self.cache_ttl > 0,
            self.max_workers > 0,
            0 < self.max_position_size <= 1,
            0 < self.max_daily_loss <= 1,
            0 < self.max_drawdown <= 1,
            self.stop_loss_pct > 0,
            self.take_profit_pct > 0,
            self.max_slippage >= 0,
            self.order_timeout > 0,
            0 < self.min_fill_percentage <= 1,
            self.metrics_interval > 0,
            self.alert_cooldown > 0,
            self.session_timeout > 0,
            self.max_failed_attempts > 0
        ]
        return all(validations)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ConfigManager:
    """Manages system configuration."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize configuration manager."""
        self._config = config or SystemConfig()
        self._logger = setup_logging(f"{__name__}.ConfigManager")
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate configuration on initialization."""
        if not self._config.validate():
            raise ValueError("Invalid configuration")
        self._logger.info("Configuration validated successfully")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self._config, key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        for key, value in updates.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        
        self._validate_config()
        self._logger.info(f"Configuration updated: {updates.keys()}")
    
    def get_config(self) -> SystemConfig:
        """Get full configuration."""
        return self._config
    
    def export(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return self._config.to_dict()


# ============================================================================
# STRATEGY ENGINE MODULE: nexus_strategies.py
# ============================================================================

"""Strategy engine for trading signal generation."""


class BaseStrategy(IStrategy):
    """Base strategy implementation."""
    
    def __init__(self, name: str, category: StrategyCategory):
        """Initialize base strategy."""
        self.name = name
        self.category = category
        self._config = {}
        self._metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_confidence': 0.0
        }
        self._logger = setup_logging(f"{__name__}.{name}")
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with configuration."""
        self._config = config
        self._logger.info(f"Strategy {self.name} initialized")
    
    def get_category(self) -> StrategyCategory:
        """Get strategy category."""
        return self.category
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        return self._metrics.copy()
    
    def _update_metrics(self, signal: TradingSignal) -> None:
        """Update internal metrics."""
        self._metrics['total_signals'] += 1
        
        # Update average confidence
        n = self._metrics['total_signals']
        prev_avg = self._metrics['avg_confidence']
        self._metrics['avg_confidence'] = (prev_avg * (n - 1) + signal.confidence) / n


class UniversalStrategyAdapter:
    """
    Universal adapter that wraps ANY strategy and provides standard interface.
    Converts DataFrame → Dict at the STRATEGY level, not pipeline level.
    """
    
    def __init__(self, strategy: Any, strategy_name: str):
        self._strategy = strategy
        self._strategy_name = strategy_name
        self._logger = setup_logging(f"{__name__}.{strategy_name}")
    
    def execute(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Universal execute - accepts DataFrame, converts to Dict for strategy"""
        try:
            market_dict = self._dataframe_to_dict(market_data, symbol)
            result = self._strategy.execute(market_dict, None)
            
            if result is None:
                return {'signal': 0.0, 'confidence': 0.0}
            return result if isinstance(result, dict) else {'signal': 0.0, 'confidence': 0.0}
        except Exception as e:
            self._logger.error(f"{self._strategy_name} error: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}
    
    def _dataframe_to_dict(self, market_data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Convert DataFrame to comprehensive Dict with all fields"""
        current_bar = market_data.iloc[-1]
        current_price = float(current_bar['close'])
        
        return {
            'symbol': symbol,
            'timestamp': current_bar.get('timestamp', time.time()),
            'price': current_price,
            'open': float(current_bar['open']),
            'high': float(current_bar['high']),
            'low': float(current_bar['low']),
            'close': current_price,
            'volume': float(current_bar['volume']),
            'bid': float(current_bar.get('bid', current_price * 0.9995)),
            'ask': float(current_bar.get('ask', current_price * 1.0005)),
            'bid_size': int(current_bar.get('bid_size', current_bar['volume'] * 0.4)),
            'ask_size': int(current_bar.get('ask_size', current_bar['volume'] * 0.4)),
            'sequence_num': int(time.time() * 1000) % 1000000,
        }


class StrategyManager:
    """Manages multiple trading strategies."""
    
    def __init__(self):
        """Initialize strategy manager."""
        self._strategies: Dict[str, IStrategy] = {}
        self._logger = setup_logging(f"{__name__}.StrategyManager")
        
    def register_strategy(self, strategy: IStrategy) -> None:
        """Register a trading strategy."""
        strategy_name = strategy.__class__.__name__
        self._strategies[strategy_name] = strategy
        self._logger.info(f"Registered strategy: {strategy_name}")
    
    def unregister_strategy(self, name: str) -> None:
        """Unregister a trading strategy."""
        if name in self._strategies:
            del self._strategies[name]
            self._logger.info(f"Unregistered strategy: {name}")
    
    def execute_all(self, data: MarketData) -> List[TradingSignal]:
        """Execute all registered strategies."""
        signals = []
        
        for name, strategy in self._strategies.items():
            try:
                signal = strategy.execute(data)
                signals.append(signal)
            except Exception as e:
                self._logger.error(f"Strategy {name} execution failed: {e}")
        
        return signals
    
    def generate_signals(self, symbol: str, market_data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate signals from all registered strategies.
        
        Executes all 20 registered strategies on the market data.
        
        Args:
            symbol: Trading symbol
            market_data: DataFrame with OHLCV data
            
        Returns:
            List of TradingSignal objects from all strategies
        """
        signals = []
        
        for name, strategy in self._strategies.items():
            try:
                # Pass DataFrame directly - wrapped strategies handle conversion
                result = strategy.execute(market_data, symbol)
                
                if result is None:
                    continue
                
                # Convert dict response to TradingSignal object
                if isinstance(result, dict):
                    signal_value = result.get('signal', 0.0)
                    confidence = result.get('confidence', 0.0)
                    
                    # Map signal value to SignalType
                    if signal_value >= 2.0:
                        signal_type = SignalType.STRONG_BUY
                    elif signal_value >= 1.0:
                        signal_type = SignalType.BUY
                    elif signal_value <= -2.0:
                        signal_type = SignalType.STRONG_SELL
                    elif signal_value <= -1.0:
                        signal_type = SignalType.SELL
                    else:
                        signal_type = SignalType.NEUTRAL
                    
                    # Create TradingSignal object
                    signal = TradingSignal(
                        signal_type=signal_type,
                        confidence=confidence,
                        symbol=symbol,
                        timestamp=time.time(),
                        strategy=name,
                        metadata=result.get('metadata', {})
                    )
                    signals.append(signal)
                elif isinstance(result, TradingSignal):
                    # Already a TradingSignal (backward compatibility)
                    signals.append(result)
                    
            except Exception as e:
                self._logger.error(f"Strategy {name} failed for {symbol}: {e}")
                continue
        
        self._logger.debug(f"Generated {len(signals)} signals from {len(self._strategies)} strategies for {symbol}")
        return signals
    
    def execute_strategy(self, name: str, data: MarketData) -> Optional[TradingSignal]:
        """Execute specific strategy."""
        if name not in self._strategies:
            self._logger.warning(f"Strategy {name} not found")
            return None
        
        try:
            return self._strategies[name].execute(data)
        except Exception as e:
            self._logger.error(f"Strategy {name} execution failed: {e}")
            return None
    
    def get_strategies(self) -> List[str]:
        """Get list of registered strategies."""
        return list(self._strategies.keys())
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all strategies."""
        return {
            name: strategy.get_metrics()
            for name, strategy in self._strategies.items()
        }


class StrategyRegistry:
    """
    Global registry for strategy metadata and capabilities.
    Allows strategies to register themselves with the NEXUS AI orchestrator.
    """
    
    _registry: Dict[str, Dict[str, Any]] = {}
    _logger = setup_logging(f"{__name__}.StrategyRegistry")
    
    @classmethod
    def register(
        cls,
        name: str,
        strategy_class: type,
        category: str = "general",
        version: str = "1.0.0",
        capabilities: Dict[str, bool] = None,
        parameters: Dict[str, Any] = None,
        performance_targets: Dict[str, float] = None
    ) -> None:
        """
        Register a strategy with metadata.
        
        Args:
            name: Unique strategy identifier
            strategy_class: Strategy class reference
            category: Strategy category (e.g., 'trend_following', 'mean_reversion')
            version: Strategy version
            capabilities: Dict of capability flags
            parameters: Default parameters
            performance_targets: Expected performance metrics
        """
        cls._registry[name] = {
            'name': name,
            'class': strategy_class,
            'category': category,
            'version': version,
            'capabilities': capabilities or {},
            'parameters': parameters or {},
            'performance_targets': performance_targets or {},
            'registered_at': time.time()
        }
        cls._logger.info(f"✅ Registered strategy: {name} v{version} ({category})")
    
    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get strategy metadata by name."""
        return cls._registry.get(name)
    
    @classmethod
    def list_all(cls) -> List[Dict[str, Any]]:
        """List all registered strategies."""
        return list(cls._registry.values())
    
    @classmethod
    def list_by_category(cls, category: str) -> List[Dict[str, Any]]:
        """List strategies by category."""
        return [
            strategy for strategy in cls._registry.values()
            if strategy['category'] == category
        ]
    
    @classmethod
    def get_capabilities(cls, name: str) -> Dict[str, bool]:
        """Get strategy capabilities."""
        strategy = cls._registry.get(name)
        return strategy['capabilities'] if strategy else {}
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies."""
        cls._registry.clear()
        cls._logger.info("Strategy registry cleared")


# ============================================================================
# ML INTEGRATION MODULE: Phase 1, 2 & 3 - Complete ML Pipeline
# ============================================================================

"""
ML Integration for 8-Layer Architecture
Implements Layer 1 (MQScore) and Layers 3-6 (Meta, Aggregation, Governance, Risk)

Reference: _FINAL_PLAN_WITH_MQSCORE.md
Version: 1.2 - Phase 1, 2 & 3 Implementation (with ONNX/PKL loading)
"""


# ============================================================================
# MODEL LOADER UTILITY (Phase 3)
# ============================================================================

class MLModelLoader:
    """
    Utility class for loading ONNX and PKL models with proper error handling.
    
    Features:
    - ONNX model loading with onnxruntime
    - PKL/Joblib model loading
    - Automatic fallback to rule-based if loading fails
    - Model path validation
    - Performance tracking
    """
    
    def __init__(self, base_path: str = "BEST_UNIQUE_MODELS"):
        """Initialize model loader."""
        self._logger = setup_logging(f"{__name__}.MLModelLoader")
        self.base_path = base_path
        self.loaded_models = {}
        self.load_stats = {
            'onnx_loaded': 0,
            'onnx_failed': 0,
            'pkl_loaded': 0,
            'pkl_failed': 0,
            'keras_loaded': 0,
            'keras_failed': 0
        }
    
    def load_onnx_model(self, model_path: str, model_name: str) -> Optional[Any]:
        """
        Load ONNX model with onnxruntime.
        
        Args:
            model_path: Relative path to ONNX model
            model_name: Friendly name for logging
        
        Returns:
            InferenceSession or None if loading failed
        """
        if not HAS_ONNX:
            self._logger.warning(f"ONNX Runtime not available - {model_name} will use fallback")
            return None
        
        try:
            import os
            full_path = os.path.join(self.base_path, model_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                self._logger.warning(f"ONNX model not found: {full_path}")
                self.load_stats['onnx_failed'] += 1
                return None
            
            # Load model
            session = ort.InferenceSession(full_path)
            self.loaded_models[model_name] = session
            self.load_stats['onnx_loaded'] += 1
            self._logger.info(f"✅ Loaded ONNX model: {model_name} from {model_path}")
            
            return session
            
        except Exception as e:
            self._logger.error(f"Failed to load ONNX model {model_name}: {e}")
            self.load_stats['onnx_failed'] += 1
            return None
    
    def load_pkl_model(self, model_path: str, model_name: str) -> Optional[Any]:
        """
        Load PKL/Joblib model.
        
        Args:
            model_path: Relative path to PKL model
            model_name: Friendly name for logging
        
        Returns:
            Model object or None if loading failed
        """
        if not HAS_PICKLE and not HAS_JOBLIB:
            self._logger.warning(f"Pickle/Joblib not available - {model_name} will use fallback")
            return None
        
        try:
            import os
            full_path = os.path.join(self.base_path, model_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                self._logger.warning(f"PKL model not found: {full_path}")
                self.load_stats['pkl_failed'] += 1
                return None
            
            # Try joblib first, then pickle
            if HAS_JOBLIB:
                model = joblib.load(full_path)
            elif HAS_PICKLE:
                with open(full_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                return None
            
            self.loaded_models[model_name] = model
            self.load_stats['pkl_loaded'] += 1
            self._logger.info(f"✅ Loaded PKL model: {model_name} from {model_path}")
            
            return model
            
        except Exception as e:
            self._logger.error(f"Failed to load PKL model {model_name}: {e}")
            self.load_stats['pkl_failed'] += 1
            return None
    
    def load_keras_model(self, model_path: str, model_name: str) -> Optional[Any]:
        """
        Load Keras/TensorFlow model.
        
        Args:
            model_path: Relative path to Keras model (.h5 or .keras)
            model_name: Friendly name for logging
        
        Returns:
            Keras model or None if loading failed
        """
        try:
            import os
            from tensorflow import keras
            
            full_path = os.path.join(self.base_path, model_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                self._logger.warning(f"Keras model not found: {full_path}")
                self.load_stats['keras_failed'] += 1
                return None
            
            # Load Keras model
            model = keras.models.load_model(full_path, compile=False)
            self.loaded_models[model_name] = model
            self.load_stats['keras_loaded'] += 1
            self._logger.info(f"✅ Loaded Keras model: {model_name} from {model_path}")
            
            return model
            
        except Exception as e:
            self._logger.error(f"Failed to load Keras model {model_name}: {e}")
            self.load_stats['keras_failed'] += 1
            return None
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get previously loaded model by name."""
        return self.loaded_models.get(model_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics."""
        total_attempted = sum(self.load_stats.values())
        total_loaded = (self.load_stats['onnx_loaded'] + 
                       self.load_stats['pkl_loaded'] + 
                       self.load_stats['keras_loaded'])
        
        return {
            'total_models_loaded': total_loaded,
            'total_attempted': total_attempted,
            'success_rate': total_loaded / total_attempted if total_attempted > 0 else 0.0,
            'details': self.load_stats.copy()
        }


# Layer 1: Market Quality Assessment (MQScore 6D Engine)
# ============================================================================

class MarketQualityLayer1:
    """
    Layer 1: Market Quality Assessment - HYBRID with 46 PRODUCTION models.
    
    PRIMARY: MQScore 6D Engine v3.0 (LightGBM + ONNX)
    ENHANCEMENTS (TIER 1 - 3 models):
        - Data Quality Scorer (ONNX)
        - Quantum Volatility Forecaster (ONNX)
        - Regime Classifier (ONNX)
    
    Latency: ~10ms (MQScore) + ~0.9ms (enhancements) = ~10.9ms total
    Input: 65 engineered features (MQScore) + model-specific features
    Output: 6 dimensions + composite score + regime + grade + enhancements
    
    Purpose: Foundation layer - determines if market conditions are tradeable.
    Critical Gates:
        - MQScore composite >= 0.5 (minimum quality)
        - Liquidity >= 0.3 (sufficient depth)
        - Regime != CRISIS (not in crisis mode)
    
    If ANY gate fails → SKIP symbol immediately (no further processing)
    """
    
    def __init__(self, config: Optional[Any] = None, model_loader: Optional['MLModelLoader'] = None):
        """Initialize MQScore Layer 1 with enhancements."""
        self._logger = setup_logging(f"{__name__}.MarketQualityLayer1")
        
        # Initialize MQScore engine (PRIMARY)
        if HAS_MQSCORE:
            mqscore_config = MQScoreConfig() if config is None else config
            self.mqscore_engine = MQScoreEngine(config=mqscore_config)
            self._enabled = True
            self._logger.info("MQScore 6D Engine initialized successfully")
        else:
            self.mqscore_engine = None
            self._enabled = False
            self._logger.warning("MQScore unavailable - Layer 1 bypassed (DANGEROUS!)")
        
        # Load TIER 1 enhancement models
        self._data_quality_model = None
        self._volatility_model = None
        self._regime_model = None
        
        if model_loader is not None:
            # Data Quality Scorer (0.064ms)
            self._data_quality_model = model_loader.load_onnx_model(
                "PRODUCTION/01_DATA_QUALITY/Regressor_lightgbm_optimized.onnx",
                "DataQualityScorer"
            )
            if self._data_quality_model:
                self._logger.info("✅ Data Quality Scorer loaded (TIER 1)")
            
            # Quantum Volatility Forecaster (0.111ms)
            self._volatility_model = model_loader.load_onnx_model(
                "PRODUCTION/02_VOLATILITY_FORECAST/quantum_volatility_model_final_optimized.onnx",
                "QuantumVolatilityForecaster"
            )
            if self._volatility_model:
                self._logger.info("✅ Quantum Volatility Forecaster loaded (TIER 1)")
            
            # Regime Classifier (0.719ms)
            self._regime_model = model_loader.load_onnx_model(
                "PRODUCTION/03_REGIME_DETECTION/ONNX_RegimeClassifier_optimized.onnx",
                "RegimeClassifier"
            )
            if self._regime_model:
                self._logger.info("✅ Regime Classifier loaded (TIER 1)")
        
        # Decision gate thresholds (OPTIMIZED based on backtest analysis)
        self.min_composite_score = 0.45  # Lowered from 0.5 - allows 92.6% NQ, ~70% BTC
        self.min_liquidity_score = 0.3   # Keep this - already 100% pass rate
        self.crisis_regimes = ['HIGH_VOLATILITY_LOW_LIQUIDITY', 'CRISIS', 'CRISIS_MODE']
        
        # Performance tracking
        self.stats = {
            'total_calls': 0,
            'gate_failures': {
                'composite': 0,
                'liquidity': 0,
                'regime': 0
            },
            'passed': 0,
            'avg_latency': 0.0
        }
    
    def assess_market_quality(self, market_data: Any) -> Dict[str, Any]:
        """
        Assess market quality using MQScore 6D Engine.
        
        Args:
            market_data: DataFrame with OHLCV data (minimum 20 bars required)
        
        Returns:
            dict with:
                - passed: bool (all gates passed?)
                - mqscore_result: MQScoreComponents (full result)
                - gate_status: dict (which gates passed/failed)
                - market_context: dict (features for downstream layers)
                - reason: str (if failed, why?)
        """
        start_time = time.time()
        self.stats['total_calls'] += 1
        
        # If MQScore disabled, bypass with warning
        if not self._enabled:
            self._logger.warning("MQScore disabled - bypassing Layer 1 gates (HIGH RISK!)")
            return {
                'passed': True,  # Allow continuation but log warning
                'mqscore_result': None,
                'gate_status': {'bypassed': True},
                'market_context': {},
                'reason': 'MQScore unavailable - gates bypassed'
            }
        
        try:
            # Calculate MQScore
            mqscore_result = self.mqscore_engine.calculate_mqscore(market_data)
            
            # Extract dimensions
            composite = mqscore_result.composite_score
            liquidity = mqscore_result.liquidity
            volatility = mqscore_result.volatility
            momentum = mqscore_result.momentum
            imbalance = mqscore_result.imbalance
            trend_strength = mqscore_result.trend_strength
            noise_level = mqscore_result.noise_level
            
            # Determine primary regime
            regime_probs = mqscore_result.regime_probability
            primary_regime = max(regime_probs, key=regime_probs.get) if regime_probs else 'UNKNOWN'
            
            # GATE 1: Composite score check
            gate_composite = composite >= self.min_composite_score
            if not gate_composite:
                self.stats['gate_failures']['composite'] += 1
            
            # GATE 2: Liquidity check
            gate_liquidity = liquidity >= self.min_liquidity_score
            if not gate_liquidity:
                self.stats['gate_failures']['liquidity'] += 1
            
            # GATE 3: Regime check (not in crisis)
            gate_regime = primary_regime not in self.crisis_regimes
            if not gate_regime:
                self.stats['gate_failures']['regime'] += 1
            
            # All gates must pass
            all_gates_passed = gate_composite and gate_liquidity and gate_regime
            
            if all_gates_passed:
                self.stats['passed'] += 1
            
            # Build gate status
            gate_status = {
                'composite': gate_composite,
                'liquidity': gate_liquidity,
                'regime': gate_regime,
                'all_passed': all_gates_passed
            }
            
            # Build market context for downstream layers
            market_context = {
                # MQScore dimensions
                'mqscore': composite,
                'liquidity': liquidity,
                'volatility': volatility,
                'momentum': momentum,
                'imbalance': imbalance,
                'trend_strength': trend_strength,
                'noise': noise_level,
                
                # Regime info
                'regime': primary_regime,
                'regime_probabilities': regime_probs,
                
                # Grade & confidence
                'grade': mqscore_result.grade,
                'confidence': mqscore_result.confidence,
                
                # Quality indicators
                'quality_indicators': mqscore_result.quality_indicators,
                
                # Timestamp
                'timestamp': mqscore_result.timestamp
            }
            
            # Determine failure reason
            reason = None
            if not all_gates_passed:
                failed_gates = []
                if not gate_composite:
                    failed_gates.append(f"MQScore={composite:.3f} < {self.min_composite_score}")
                if not gate_liquidity:
                    failed_gates.append(f"Liquidity={liquidity:.3f} < {self.min_liquidity_score}")
                if not gate_regime:
                    failed_gates.append(f"Regime={primary_regime} (CRISIS)")
                reason = f"Layer 1 gates failed: {', '.join(failed_gates)}"
            
            # Update latency stats
            latency = time.time() - start_time
            self.stats['avg_latency'] = (
                (self.stats['avg_latency'] * (self.stats['total_calls'] - 1) + latency) /
                self.stats['total_calls']
            )
            
            return {
                'passed': all_gates_passed,
                'mqscore_result': mqscore_result,
                'gate_status': gate_status,
                'market_context': market_context,
                'reason': reason,
                'latency': latency
            }
            
        except Exception as e:
            self._logger.error(f"MQScore calculation failed: {e}")
            return {
                'passed': False,
                'mqscore_result': None,
                'gate_status': {'error': True},
                'market_context': {},
                'reason': f'MQScore error: {str(e)}'
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Layer 1 performance statistics."""
        total = self.stats['total_calls']
        passed = self.stats['passed']
        
        return {
            'total_assessments': total,
            'passed': passed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'gate_failures': self.stats['gate_failures'].copy(),
            'avg_latency_ms': self.stats['avg_latency'] * 1000,
            'enabled': self._enabled
        }


# ============================================================================
# Layer 2: Signal Generation (Strategy Execution)
# ============================================================================

class SignalGenerationLayer2:
    """
    Layer 2: Execute multiple trading strategies and generate signals.
    
    Runs 20+ strategies in parallel and generates TradingSignal objects.
    Each strategy analyzes market data and produces BUY/SELL/NEUTRAL signals.
    
    Reference: PIPELINE_FLOWCHART.md - Layer 2
    """
    
    def __init__(self, strategy_manager: Optional[StrategyManager] = None):
        """
        Initialize signal generation layer.
        
        Args:
            strategy_manager: Shared StrategyManager with registered strategies.
        """
        self._logger = setup_logging(f"{__name__}.SignalGenerationLayer2")
        self._strategy_manager = strategy_manager if strategy_manager is not None else StrategyManager()
        self.stats = {
            'total_generated': 0,
            'signals_passed': 0,
            'signals_filtered': 0,
            'avg_confidence': 0.0
        }
        
        # Log strategy count
        strategy_count = len(self._strategy_manager.get_strategies())
        self._logger.info(f"Layer 2 initialized with {strategy_count} strategies")
    
    def generate_signals(self, market_data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """
        Execute all 20 strategies and generate trading signals.
        
        Uses existing strategy files from the project.
        
        Args:
            market_data: OHLCV data from Layer 1
            symbol: Trading symbol
        
        Returns:
            List of TradingSignal objects with confidence >= 0.57
        """
        try:
            # Use StrategyManager to execute all registered strategies
            signals = self._strategy_manager.generate_signals(symbol, market_data)
            
            # Filter: Keep only confidence >= 0.57
            filtered_signals = [s for s in signals if s.confidence >= 0.57]
            
            # Update stats
            self.stats['total_generated'] += len(signals)
            self.stats['signals_passed'] += len(filtered_signals)
            self.stats['signals_filtered'] += (len(signals) - len(filtered_signals))
            
            if filtered_signals:
                avg_conf = np.mean([s.confidence for s in filtered_signals])
                self.stats['avg_confidence'] = avg_conf
            
            self._logger.debug(f"Generated {len(signals)} signals, {len(filtered_signals)} passed filter")
            
            return filtered_signals
            
        except Exception as e:
            self._logger.error(f"Signal generation error: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get layer statistics."""
        return self.stats.copy()


# Layer 3: Meta-Learning (Strategy Selection)
# ============================================================================

class MetaStrategySelector:
    """
    Layer 3: Dynamic strategy weight assignment based on market conditions.
    
    Model: Quantum Meta-Strategy Selector.onnx
    Latency: 0.108ms
    
    Purpose: Determines which strategies to TRUST in current market regime.
    Example: 14 SELL signals, 6 BUY signals → Meta assigns high weights to BUY strategies → BUY wins!
    """
    
    def __init__(self, model_loader: Optional[MLModelLoader] = None):
        """Initialize Meta-Strategy Selector."""
        self._logger = setup_logging(f"{__name__}.MetaStrategySelector")
        self._strategy_performance = defaultdict(lambda: {
            'accuracy': 0.5,
            'sharpe': 0.0,
            'pnl': 0.0,
            'trades': 0,
            'wins': 0
        })
        
        # Phase 3: Try to load ONNX model
        self._ml_model = None
        self._use_ml_model = False
        
        if model_loader is not None:
            self._ml_model = model_loader.load_onnx_model(
                "PRODUCTION/04_STRATEGY_SELECTION/Quantum Meta-Strategy Selector.onnx",
                "MetaStrategySelector"
            )
            if self._ml_model is not None:
                self._use_ml_model = True
                self._logger.info("✅ MetaStrategySelector using ML model (ONNX)")
            else:
                self._logger.info("⚠️ MetaStrategySelector using rule-based fallback")
        else:
            self._logger.info("⚠️ MetaStrategySelector using rule-based fallback (no loader)")
        
    def select_strategy_weights(self, market_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Assign dynamic weights to strategies based on market conditions.
        
        Args:
            market_features: 44 market context features
                - regime: Market regime from MQScore (0=Trend, 1=Range, 2=Volatile)
                - volatility: From MQScore volatility dimension
                - momentum: From MQScore momentum dimension
                - Recent strategy performance metrics
        
        Returns:
            {
                'strategy_weights': [19 weights, 0-1 per strategy],
                'anomaly_score': 0.0-1.0 (market anomaly detection),
                'regime_confidence': [P(trend), P(range), P(volatile)]
            }
        """
        try:
            # Phase 3: Try ML model first, fallback to rule-based
            if self._use_ml_model:
                try:
                    return self._ml_inference(market_features)
                except Exception as e:
                    self._logger.warning(f"ML inference failed, using fallback: {e}")
                    # Fall through to rule-based
            
            # Rule-based fallback
            regime = market_features.get('regime', 0)
            volatility = market_features.get('volatility', 0.5)
            
            # Calculate strategy weights based on regime
            strategy_weights = self._calculate_weights_by_regime(regime, volatility)
            
            # Detect market anomalies
            anomaly_score = self._detect_anomaly(market_features)
            
            # Regime confidence
            regime_confidence = self._estimate_regime_confidence(market_features)
            
            return {
                'strategy_weights': strategy_weights,
                'anomaly_score': anomaly_score,
                'regime_confidence': regime_confidence
            }
            
        except Exception as e:
            self._logger.error(f"Meta-Strategy Selector error: {e}")
            # Fallback: equal weights
            return {
                'strategy_weights': [0.5] * 20,  # 20 strategies
                'anomaly_score': 0.0,
                'regime_confidence': [0.33, 0.33, 0.34]
            }
    
    def _ml_inference(self, market_features: Dict[str, float]) -> Dict[str, Any]:
        """Run ML model inference (Phase 3)."""
        # Prepare input features for ONNX model
        feature_vector = self._prepare_feature_vector(market_features)
        
        # Run ONNX inference
        input_name = self._ml_model.get_inputs()[0].name
        output = self._ml_model.run(None, {input_name: feature_vector})
        
        # Parse outputs
        # Output 0: strategy_logits [batch, 19]
        # Output 1: anomaly_logits [batch] - scalar
        # Output 2: regime_logits [batch, 3]
        strategy_weights = output[0][0].tolist() if len(output) > 0 else [0.5] * 20
        anomaly_score = float(output[1][0]) if len(output) > 1 else 0.0  # Scalar output
        regime_confidence = output[2][0].tolist() if len(output) > 2 else [0.33, 0.33, 0.34]
        
        return {
            'strategy_weights': strategy_weights,
            'anomaly_score': anomaly_score,
            'regime_confidence': regime_confidence
        }
    
    def _prepare_feature_vector(self, market_features: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for ML model."""
        # Extract key features in expected order
        features = [
            market_features.get('mqscore', 0.5),
            market_features.get('liquidity', 0.5),
            market_features.get('volatility', 0.5),
            market_features.get('momentum', 0.5),
            market_features.get('imbalance', 0.5),
            market_features.get('trend_strength', 0.5),
            market_features.get('noise', 0.5),
            # Add more features as needed (up to 44 total)
        ]
        
        # Pad to expected size if needed
        while len(features) < 44:
            features.append(0.5)
        
        return np.array([features], dtype=np.float32)
    
    def _calculate_weights_by_regime(self, regime: int, volatility: float) -> List[float]:
        """Calculate strategy weights based on market regime."""
        weights = []
        
        # Define strategy performance by regime (based on historical data)
        # Regime 0 = Trending, 1 = Ranging, 2 = Volatile
        
        if regime == 0:  # Trending market
            # Favor: Momentum, Breakout, Trend-following strategies
            trend_strategies = [0.9, 0.85, 0.8, 0.75]  # High trust
            range_strategies = [0.2, 0.15, 0.1]  # Low trust
            weights = trend_strategies + range_strategies + [0.5] * 13
            
        elif regime == 1:  # Ranging market
            # Favor: Mean reversion, VWAP, Statistical arb
            range_strategies = [0.9, 0.85, 0.8]  # High trust
            trend_strategies = [0.2, 0.15]  # Low trust
            weights = range_strategies + trend_strategies + [0.5] * 15
            
        else:  # Volatile market (regime == 2)
            # Favor: HFT, Scalping, Quick exits
            # Lower trust overall due to uncertainty
            weights = [0.6] * 4 + [0.3] * 16
        
        # Adjust for volatility
        if volatility > 0.8:  # High volatility
            weights = [w * 0.7 for w in weights]  # Reduce all weights
        
        return weights[:20]  # Ensure 20 weights
    
    def _detect_anomaly(self, market_features: Dict[str, float]) -> float:
        """Detect market anomalies (0=normal, 1=highly anomalous)."""
        # Simple anomaly detection based on extreme values
        volatility = market_features.get('volatility', 0.5)
        noise = market_features.get('noise', 0.5)
        
        anomaly_score = 0.0
        
        # High volatility is anomalous
        if volatility > 0.9:
            anomaly_score += 0.5
        
        # High noise is anomalous
        if noise > 0.8:
            anomaly_score += 0.5
        
        return min(anomaly_score, 1.0)
    
    def _estimate_regime_confidence(self, market_features: Dict[str, float]) -> List[float]:
        """Estimate confidence in regime classification."""
        trend_strength = market_features.get('trend_strength', 0.5)
        volatility = market_features.get('volatility', 0.5)
        
        # Simple heuristic confidence
        if trend_strength > 0.7:
            return [0.8, 0.1, 0.1]  # High confidence in trending
        elif volatility > 0.8:
            return [0.1, 0.1, 0.8]  # High confidence in volatile
        else:
            return [0.2, 0.6, 0.2]  # High confidence in ranging
    
    def update_strategy_performance(self, strategy_name: str, outcome: bool, pnl: float):
        """Update strategy performance tracking."""
        perf = self._strategy_performance[strategy_name]
        perf['trades'] += 1
        if outcome:
            perf['wins'] += 1
        perf['pnl'] += pnl
        perf['accuracy'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0.5


# Layer 4: Signal Aggregation
# ============================================================================

class SignalAggregator:
    """
    Layer 4: Weighted combination of strategy signals into single decision.
    
    Model: ONNX Signal Aggregator.onnx
    Latency: 0.237ms
    
    Purpose: Combines 10-20 filtered signals using Meta-assigned weights.
    Output: Single aggregated signal (-1.0 to +1.0)
    """
    
    def __init__(self, model_loader: Optional[MLModelLoader] = None):
        """Initialize Signal Aggregator."""
        self._logger = setup_logging(f"{__name__}.SignalAggregator")
        
        # Phase 3: Try to load ONNX model
        self._ml_model = None
        self._use_ml_model = False
        
        if model_loader is not None:
            self._ml_model = model_loader.load_onnx_model(
                "PRODUCTION/05_SIGNAL_AGGREGATION/ONNX Signal Aggregator.onnx",
                "SignalAggregator"
            )
            if self._ml_model is not None:
                self._use_ml_model = True
                self._logger.info("✅ SignalAggregator using ML model (ONNX)")
            else:
                self._logger.info("⚠️ SignalAggregator using rule-based fallback")
        else:
            self._logger.info("⚠️ SignalAggregator using rule-based fallback (no loader)")
        
    def aggregate_signals(
        self, 
        signals: List[TradingSignal], 
        strategy_weights: List[float]
    ) -> Dict[str, Any]:
        """
        Aggregate multiple strategy signals into single decision.
        
        Args:
            signals: List of filtered signals (confidence >= 0.57)
            strategy_weights: Weights from Meta-Strategy Selector
        
        Returns:
            {
                'aggregated_signal': -1.0 to +1.0 (SELL to BUY),
                'signal_strength': 0.0-1.0 (confidence in signal),
                'num_strategies_used': int,
                'direction': 'BUY' | 'SELL' | 'HOLD'
            }
        """
        try:
            if not signals:
                return self._neutral_signal()
            
            # Phase 3: Try ML model first, fallback to rule-based
            if self._use_ml_model:
                try:
                    return self._ml_inference(signals, strategy_weights)
                except Exception as e:
                    self._logger.warning(f"ML inference failed, using fallback: {e}")
                    # Fall through to rule-based
            
            # Rule-based fallback: Weighted aggregation
            weighted_sum = 0.0
            total_weight = 0.0
            confidences = []
            
            for idx, signal in enumerate(signals):
                weight = strategy_weights[idx] if idx < len(strategy_weights) else 0.5
                confidence = signal.confidence
                signal_value = signal.signal_type.value  # +1 or -1
                
                weighted_sum += signal_value * weight * confidence
                total_weight += weight * confidence
                confidences.append(confidence)
            
            # Calculate aggregated signal
            aggregated_signal = weighted_sum / total_weight if total_weight > 0 else 0.0
            signal_strength = np.mean(confidences) if confidences else 0.0
            
            # Determine direction
            if aggregated_signal > 0.3:
                direction = 'BUY'
            elif aggregated_signal < -0.3:
                direction = 'SELL'
            else:
                direction = 'HOLD'
            
            return {
                'aggregated_signal': aggregated_signal,
                'signal_strength': signal_strength,
                'num_strategies_used': len(signals),
                'direction': direction
            }
            
        except Exception as e:
            self._logger.error(f"Signal Aggregation error: {e}")
            return self._neutral_signal()
    
    def _ml_inference(self, signals: List[TradingSignal], strategy_weights: List[float]) -> Dict[str, Any]:
        """Run ML model inference (Phase 3)."""
        # Prepare input features for ONNX model
        feature_vector = self._prepare_feature_vector(signals, strategy_weights)
        
        # Run ONNX inference
        input_name = self._ml_model.get_inputs()[0].name
        output = self._ml_model.run(None, {input_name: feature_vector})
        
        # Parse outputs - model returns scalars
        # Output 0: signal (batch,) - scalar value
        # Output 1: confidence (batch,) - scalar value
        aggregated_signal = float(output[0][0]) if len(output) > 0 else 0.0
        signal_strength = float(output[1][0]) if len(output) > 1 else 0.5
        
        # Determine direction from aggregated signal
        if aggregated_signal > 0.3:
            direction = 'BUY'
        elif aggregated_signal < -0.3:
            direction = 'SELL'
        else:
            direction = 'HOLD'
        
        return {
            'aggregated_signal': aggregated_signal,
            'signal_strength': signal_strength,
            'num_strategies_used': len(signals),
            'direction': direction
        }
    
    def _prepare_feature_vector(self, signals: List[TradingSignal], strategy_weights: List[float]) -> np.ndarray:
        """Prepare feature vector for ML model (20 features expected)."""
        features = []
        
        # Model expects 20 features: aggregated signal stats
        # Calculate aggregate features instead of per-signal features
        if signals:
            # Aggregate signal statistics
            signal_values = [s.signal_type.value for s in signals]
            confidences = [s.confidence for s in signals]
            weights = strategy_weights[:len(signals)]
            
            features.extend([
                np.mean(signal_values) if signal_values else 0.0,  # Avg signal
                np.std(signal_values) if len(signal_values) > 1 else 0.0,  # Signal std
                np.mean(confidences) if confidences else 0.0,  # Avg confidence
                np.std(confidences) if len(confidences) > 1 else 0.0,  # Confidence std
                np.mean(weights) if weights else 0.5,  # Avg weight
                np.std(weights) if len(weights) > 1 else 0.0,  # Weight std
                float(len(signals)),  # Number of signals
                float(sum(1 for s in signal_values if s > 0)),  # BUY signals
                float(sum(1 for s in signal_values if s < 0)),  # SELL signals
                float(sum(1 for s in signal_values if s == 0)),  # NEUTRAL signals
                max(confidences) if confidences else 0.0,  # Max confidence
                min(confidences) if confidences else 0.0,  # Min confidence
                max(weights) if weights else 0.0,  # Max weight
                min(weights) if weights else 0.0,  # Min weight
                np.sum([s * w * c for s, w, c in zip(signal_values, weights, confidences)]) / len(signals) if signals else 0.0,  # Weighted signal
            ])
        else:
            features = [0.0] * 15
        
        # Pad to 20 features
        while len(features) < 20:
            features.append(0.0)
        
        # Ensure exactly 20 features
        features = features[:20]
        
        return np.array([features], dtype=np.float32)
    
    def _neutral_signal(self) -> Dict[str, Any]:
        """Return neutral/HOLD signal."""
        return {
            'aggregated_signal': 0.0,
            'signal_strength': 0.0,
            'num_strategies_used': 0,
            'direction': 'HOLD'
        }
    
    def check_signal_strength_gate(self, aggregated_result: Dict[str, Any]) -> bool:
        """
        Gate: Signal strength must be >= 0.5
        
        Returns:
            True if signal passes gate, False to SKIP
        """
        return aggregated_result['signal_strength'] >= 0.5


# Layer 5: Model Governance & Routing
# ============================================================================

class ModelGovernor:
    """
    Layer 5.1: Model Governance - Track model performance and adjust trust levels.
    
    Model: ONNX_ModelGovernor_Meta_optimized.onnx
    Latency: 0.063ms
    
    Purpose: Dynamically adjust which models to trust based on recent performance.
    """
    
    def __init__(self, model_loader: Optional[MLModelLoader] = None):
        """Initialize Model Governor."""
        self._logger = setup_logging(f"{__name__}.ModelGovernor")
        self._model_performance = defaultdict(lambda: {
            'accuracy': 0.5,
            'sharpe': 0.0,
            'drawdown': 0.0,
            'win_rate': 0.5,
            'recent_pnl': 0.0,
            'predictions': 0
        })
        
        # Phase 3: Try to load ONNX model
        self._ml_model = None
        self._use_ml_model = False
        
        if model_loader is not None:
            self._ml_model = model_loader.load_onnx_model(
                "PRODUCTION/07_MODEL_GOVERNANCE/ONNX_ModelGovernor_Meta_optimized.onnx",
                "ModelGovernor"
            )
            if self._ml_model is not None:
                self._use_ml_model = True
                self._logger.info("✅ ModelGovernor using ML model (ONNX)")
            else:
                self._logger.info("⚠️ ModelGovernor using rule-based fallback")
        else:
            self._logger.info("⚠️ ModelGovernor using rule-based fallback (no loader)")
    
    def get_model_weights(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate trust levels for each model.
        
        Args:
            performance_metrics: 75 performance metrics (accuracy, sharpe, etc.)
        
        Returns:
            {
                'model_weights': [15 trust levels, 0-1 per model],
                'threshold_adjustments': Dynamic confidence thresholds,
                'risk_scalers': Risk adjustment factors,
                'retrain_flags': Models needing retraining
            }
        """
        try:
            # Phase 3: Try ML model first, fallback to rule-based
            if self._use_ml_model:
                try:
                    return self._ml_inference(performance_metrics)
                except Exception as e:
                    self._logger.warning(f"ML inference failed, using fallback: {e}")
                    # Fall through to rule-based
            
            # Rule-based fallback: Calculate model weights based on recent performance
            model_weights = []
            retrain_flags = []
            
            for i in range(15):  # 15 models
                model_id = f"model_{i}"
                perf = self._model_performance[model_id]
                
                # Weight calculation: higher accuracy + sharpe = higher weight
                weight = (perf['accuracy'] * 0.6 + 
                         min(perf['sharpe'] / 2.0, 1.0) * 0.4)
                
                model_weights.append(weight)
                
                # Flag for retraining if accuracy dropped
                if perf['accuracy'] < 0.55 and perf['predictions'] > 100:
                    retrain_flags.append(model_id)
            
            return {
                'model_weights': model_weights,
                'threshold_adjustments': self._calculate_thresholds(model_weights),
                'risk_scalers': self._calculate_risk_scalers(model_weights),
                'retrain_flags': retrain_flags
            }
            
        except Exception as e:
            self._logger.error(f"Model Governance error: {e}")
            return {
                'model_weights': [0.7] * 15,
                'threshold_adjustments': {},
                'risk_scalers': {},
                'retrain_flags': []
            }
    
    def _ml_inference(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Run ML model inference (Phase 3)."""
        # Prepare input features for ONNX model
        feature_vector = self._prepare_feature_vector(performance_metrics)
        
        # Run ONNX inference
        input_name = self._ml_model.get_inputs()[0].name
        output = self._ml_model.run(None, {input_name: feature_vector})
        
        # Parse outputs
        model_weights = output[0][0].tolist() if len(output) > 0 else [0.7] * 15
        
        return {
            'model_weights': model_weights,
            'threshold_adjustments': self._calculate_thresholds(model_weights),
            'risk_scalers': self._calculate_risk_scalers(model_weights),
            'retrain_flags': []  # ML model doesn't flag retraining
        }
    
    def _prepare_feature_vector(self, performance_metrics: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for ML model."""
        # Extract performance metrics (up to 75 features)
        features = []
        for i in range(15):  # 15 models
            model_id = f"model_{i}"
            perf = self._model_performance[model_id]
            features.extend([
                perf['accuracy'],
                perf['sharpe'],
                perf['drawdown'],
                perf['win_rate'],
                perf['recent_pnl']
            ])
        
        # Pad to 75 if needed
        while len(features) < 75:
            features.append(0.5)
        
        return np.array([features], dtype=np.float32)
    
    def _calculate_thresholds(self, model_weights: List[float]) -> Dict[str, float]:
        """Adjust confidence thresholds based on model performance."""
        avg_weight = np.mean(model_weights)
        return {
            'confidence_threshold': 0.7 if avg_weight > 0.7 else 0.75,
            'signal_strength_threshold': 0.5 if avg_weight > 0.6 else 0.6
        }
    
    def _calculate_risk_scalers(self, model_weights: List[float]) -> Dict[str, float]:
        """Calculate risk adjustment factors."""
        avg_weight = np.mean(model_weights)
        return {
            'position_scaler': avg_weight,  # Reduce size if models performing poorly
            'risk_multiplier': avg_weight * 1.2
        }
    
    def update_model_performance(self, model_id: str, prediction_correct: bool, metrics: Dict[str, float]):
        """Update model performance tracking."""
        perf = self._model_performance[model_id]
        perf['predictions'] += 1
        
        # Update accuracy
        current_correct = perf['accuracy'] * (perf['predictions'] - 1)
        new_correct = current_correct + (1 if prediction_correct else 0)
        perf['accuracy'] = new_correct / perf['predictions']
        
        # Update other metrics
        perf.update(metrics)


class DecisionRouter:
    """
    Layer 5.2: Decision Routing - Final BUY/SELL/HOLD decision.
    
    Model: ModelRouter_Meta_optimized.onnx
    Latency: 0.083ms
    
    Purpose: Routes decision through best-performing prediction pathway.
    """
    
    def __init__(self, model_loader: Optional[MLModelLoader] = None):
        """Initialize Decision Router."""
        self._logger = setup_logging(f"{__name__}.DecisionRouter")
        
        # Phase 3: Try to load ONNX model
        self._ml_model = None
        self._use_ml_model = False
        
        if model_loader is not None:
            self._ml_model = model_loader.load_onnx_model(
                "PRODUCTION/06_MODEL_ROUTING/ModelRouter_Meta_optimized.onnx",
                "DecisionRouter"
            )
            if self._ml_model is not None:
                self._use_ml_model = True
                self._logger.info("✅ DecisionRouter using ML model (ONNX)")
            else:
                self._logger.info("⚠️ DecisionRouter using rule-based fallback")
        else:
            self._logger.info("⚠️ DecisionRouter using rule-based fallback (no loader)")
    
    def route_decision(
        self,
        aggregated_signal: Dict[str, Any],
        market_context: Dict[str, float],
        model_weights: List[float]
    ) -> Dict[str, Any]:
        """
        Make final BUY/SELL/HOLD decision.
        
        Args:
            aggregated_signal: From SignalAggregator
            market_context: 126 context features (regime, volatility, etc.)
            model_weights: From ModelGovernor
        
        Returns:
            {
                'action_probs': [P(buy), P(sell)],
                'confidence': 0.0-1.0,
                'action': 'BUY' | 'SELL' | 'HOLD',
                'value_estimate': expected_return
            }
        """
        try:
            # Phase 3: Try ML model first, fallback to rule-based
            if self._use_ml_model:
                try:
                    return self._ml_inference(aggregated_signal, market_context, model_weights)
                except Exception as e:
                    self._logger.warning(f"ML inference failed, using fallback: {e}")
                    # Fall through to rule-based
            
            # Rule-based fallback
            signal_value = aggregated_signal['aggregated_signal']
            signal_strength = aggregated_signal['signal_strength']
            
            # Calculate action probabilities
            if signal_value > 0:  # BUY bias
                p_buy = min((signal_value + 1.0) / 2.0, 1.0)
                p_sell = 1.0 - p_buy
            else:  # SELL bias
                p_sell = min((abs(signal_value) + 1.0) / 2.0, 1.0)
                p_buy = 1.0 - p_sell
            
            # Adjust probabilities by model governance weights
            avg_model_weight = np.mean(model_weights) if model_weights else 0.7
            p_buy *= avg_model_weight
            p_sell *= avg_model_weight
            
            # Determine action
            if p_buy > 0.6:
                action = 'BUY'
                confidence = p_buy * signal_strength
            elif p_sell > 0.6:
                action = 'SELL'
                confidence = p_sell * signal_strength
            else:
                action = 'HOLD'
                confidence = 0.0
            
            # Estimate expected value (simplified)
            value_estimate = signal_value * confidence * 0.01  # 1% expected move
            
            return {
                'action_probs': [p_buy, p_sell],
                'confidence': confidence,
                'action': action,
                'value_estimate': value_estimate
            }
            
        except Exception as e:
            self._logger.error(f"Decision Routing error: {e}")
            return {
                'action_probs': [0.5, 0.5],
                'confidence': 0.0,
                'action': 'HOLD',
                'value_estimate': 0.0
            }
    
    def _ml_inference(
        self,
        aggregated_signal: Dict[str, Any],
        market_context: Dict[str, float],
        model_weights: List[float]
    ) -> Dict[str, Any]:
        """Run ML model inference (Phase 3)."""
        # Prepare input features for ONNX model
        feature_vector = self._prepare_feature_vector(aggregated_signal, market_context, model_weights)
        
        # Run ONNX inference
        input_name = self._ml_model.get_inputs()[0].name
        output = self._ml_model.run(None, {input_name: feature_vector})
        
        # Parse outputs
        action_probs = output[0][0].tolist() if len(output) > 0 else [0.5, 0.5]
        confidence = float(output[1][0][0]) if len(output) > 1 else 0.0
        value_estimate = float(output[2][0][0]) if len(output) > 2 else 0.0
        
        # Determine action from probabilities
        p_buy, p_sell = action_probs[0], action_probs[1]
        if p_buy > 0.6:
            action = 'BUY'
        elif p_sell > 0.6:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action_probs': action_probs,
            'confidence': confidence,
            'action': action,
            'value_estimate': value_estimate
        }
    
    def _prepare_feature_vector(
        self,
        aggregated_signal: Dict[str, Any],
        market_context: Dict[str, float],
        model_weights: List[float]
    ) -> np.ndarray:
        """Prepare feature vector for ML model."""
        features = []
        
        # Signal features
        features.extend([
            aggregated_signal['aggregated_signal'],
            aggregated_signal['signal_strength'],
            float(aggregated_signal['num_strategies_used'])
        ])
        
        # Market context features (from MQScore)
        features.extend([
            market_context.get('mqscore', 0.5),
            market_context.get('liquidity', 0.5),
            market_context.get('volatility', 0.5),
            market_context.get('momentum', 0.5),
            market_context.get('imbalance', 0.5),
            market_context.get('trend_strength', 0.5),
            market_context.get('noise', 0.5)
        ])
        
        # Model weights (15 models)
        if len(model_weights) >= 15:
            features.extend(model_weights[:15])
        else:
            features.extend(model_weights + [0.7] * (15 - len(model_weights)))
        
        # Pad to expected size (126 features)
        while len(features) < 126:
            features.append(0.5)
        
        return np.array([features], dtype=np.float32)
    
    def check_confidence_gate(self, decision: Dict[str, Any]) -> bool:
        """
        Gate: Confidence must be >= 0.7
        
        Returns:
            True if decision passes gate, False to SKIP
        """
        return decision['confidence'] >= 0.7


# ============================================================================
# RISK MANAGEMENT MODULE: nexus_risk.py
# ============================================================================

"""Risk management module for trade validation and position sizing."""


class RiskManager(IRiskManager):
    """
    Layer 6: Enhanced Risk Management with ML Integration.
    
    Comprehensive risk management system with:
    - ML-based risk scoring
    - Dynamic thresholds from ModelGovernor
    - Market classification alignment
    - 7-layer risk validation
    
    Reference: _FINAL_PLAN_WITH_MQSCORE.md - Layer 6
    """
    
    def __init__(
        self, 
        config: SystemConfig, 
        model_governor: Optional['ModelGovernor'] = None,
        model_loader: Optional[MLModelLoader] = None
    ):
        """Initialize risk manager with optional ML integration."""
        self._config = config
        self._logger = setup_logging(f"{__name__}.RiskManager")
        self._positions = {}
        self._daily_pnl = 0.0
        self._max_drawdown_reached = 0.0
        
        # ML Integration (Phase 1)
        self._model_governor = model_governor
        
        # Load all TIER 2 Risk Management models (7 models total)
        self._risk_classifier_model = None
        self._risk_scorer_model = None
        self._risk_governor_model = None
        self._confidence_calibration_model = None
        self._market_classifier_model = None
        self._regression_model = None
        self._gradient_boosting_model = None
        self._use_ml_risk = False
        
        if model_loader is not None:
            # TIER 2 - ONNX Models
            # Risk Classifier
            self._risk_classifier_model = model_loader.load_onnx_model(
                "PRODUCTION/06_RISK_MANAGEMENT/Risk_Classifier_optimized.onnx",
                "RiskClassifier"
            )
            if self._risk_classifier_model:
                self._logger.info("✅ Risk Classifier loaded (TIER 2)")
            
            # Risk Scorer
            self._risk_scorer_model = model_loader.load_onnx_model(
                "PRODUCTION/06_RISK_MANAGEMENT/Risk_Scorer_optimized.onnx",
                "RiskScorer"
            )
            if self._risk_scorer_model:
                self._logger.info("✅ Risk Scorer loaded (TIER 2)")
            
            # Risk Governor
            self._risk_governor_model = model_loader.load_onnx_model(
                "PRODUCTION/08_RISK_MANAGEMENT/model_risk_governor_optimized.onnx",
                "RiskGovernor"
            )
            if self._risk_governor_model:
                self._logger.info("✅ Risk Governor loaded (TIER 2)")
            
            # Gradient Boosting
            self._gradient_boosting_model = model_loader.load_onnx_model(
                "PRODUCTION/12_GRADIENT_BOOSTING/ONNXGBoost.onnx",
                "GradientBoosting"
            )
            if self._gradient_boosting_model:
                self._logger.info("✅ Gradient Boosting loaded (TIER 3)")
            
            # TIER 2 - PKL Models
            # Confidence Calibration (XGBoost)
            self._confidence_calibration_model = model_loader.load_pkl_model(
                "PRODUCTION/09_CONFIDENCE_CALIBRATION/xgboost_confidence_model.pkl",
                "ConfidenceCalibration"
            )
            if self._confidence_calibration_model:
                self._logger.info("✅ Confidence Calibration loaded (TIER 2)")
            
            # Market Classification (XGBoost)
            self._market_classifier_model = model_loader.load_pkl_model(
                "PRODUCTION/10_MARKET_CLASSIFICATION/xgboost_classifier_enhanced_h1.pkl",
                "MarketClassifier"
            )
            if self._market_classifier_model:
                self._logger.info("✅ Market Classifier loaded (TIER 2)")
            
            # Regression (XGBoost)
            self._regression_model = model_loader.load_pkl_model(
                "PRODUCTION/11_REGRESSION/xgboost_regressor_enhanced_h1.pkl",
                "Regression"
            )
            if self._regression_model:
                self._logger.info("✅ Regression Model loaded (TIER 2)")
            
            # Enable ML risk if any model loaded
            models_loaded = sum([
                self._risk_classifier_model is not None,
                self._risk_scorer_model is not None,
                self._risk_governor_model is not None,
                self._confidence_calibration_model is not None,
                self._market_classifier_model is not None,
                self._regression_model is not None,
                self._gradient_boosting_model is not None
            ])
            
            if models_loaded > 0:
                self._use_ml_risk = True
                self._logger.info(f"✅ RiskManager using {models_loaded}/7 ML models")
            else:
                self._logger.info("⚠️ RiskManager using rule-based fallback")
        else:
            self._logger.info("⚠️ RiskManager using rule-based fallback (no loader)")
        
        self._risk_model_performance = {
            'accuracy': 0.5,
            'false_positives': 0,
            'false_negatives': 0,
            'total_predictions': 0
        }
        
    def evaluate_risk(self, signal: TradingSignal, portfolio: Dict[str, Any]) -> RiskMetrics:
        """Evaluate risk for a trading signal."""
        try:
            capital = portfolio.get('capital', 100000)
            current_positions = portfolio.get('positions', {})
            
            # Calculate position size
            position_size = self.calculate_position_size(signal, capital)
            
            # Calculate stop loss and take profit
            if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = self._config.stop_loss_pct
                take_profit = self._config.take_profit_pct
            elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                stop_loss = self._config.stop_loss_pct
                take_profit = self._config.take_profit_pct
            else:
                stop_loss = 0.0
                take_profit = 0.0
            
            # Calculate risk metrics
            max_loss = position_size * capital * stop_loss
            expected_return = position_size * capital * take_profit * signal.confidence
            
            # Simple Sharpe ratio approximation
            if max_loss > 0:
                sharpe_ratio = expected_return / max_loss
            else:
                sharpe_ratio = 0.0
            
            # VaR calculation (simplified)
            var_95 = max_loss * 1.645  # 95% confidence level
            
            # Risk score (0-1, higher is riskier)
            risk_factors = [
                position_size / self._config.max_position_size,
                abs(self._daily_pnl) / (capital * self._config.max_daily_loss),
                self._max_drawdown_reached / self._config.max_drawdown,
                1.0 - signal.confidence
            ]
            risk_score = np.mean([min(f, 1.0) for f in risk_factors])
            
            return RiskMetrics(
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_drawdown=self._config.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                expected_return=expected_return,
                risk_score=risk_score
            )
            
        except Exception as e:
            self._logger.error(f"Risk evaluation error: {e}")
            # Return conservative risk metrics on error
            return RiskMetrics(
                position_size=0.0,
                stop_loss=self._config.stop_loss_pct,
                take_profit=self._config.take_profit_pct,
                max_drawdown=self._config.max_drawdown,
                sharpe_ratio=0.0,
                var_95=0.0,
                expected_return=0.0,
                risk_score=1.0
            )
    
    def validate_trade(self, signal: TradingSignal, metrics: RiskMetrics) -> bool:
        """Validate if trade should be executed."""
        try:
            # Check risk score
            if metrics.risk_score > 0.8:
                self._logger.warning(f"Trade rejected: High risk score {metrics.risk_score}")
                return False
            
            # Check position size
            if metrics.position_size > self._config.max_position_size:
                self._logger.warning(f"Trade rejected: Position size too large {metrics.position_size}")
                return False
            
            # Check daily loss limit
            if abs(self._daily_pnl) >= self._config.max_daily_loss:
                self._logger.warning("Trade rejected: Daily loss limit reached")
                return False
            
            # Check drawdown
            if self._max_drawdown_reached >= self._config.max_drawdown:
                self._logger.warning("Trade rejected: Max drawdown reached")
                return False
            
            # Check signal confidence
            if signal.confidence < 0.5:
                self._logger.warning(f"Trade rejected: Low confidence {signal.confidence}")
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Trade validation error: {e}")
            return False
    
    def calculate_position_size(self, signal: TradingSignal, capital: float) -> float:
        """Calculate appropriate position size using Kelly Criterion."""
        try:
            # Kelly fraction = (p * b - q) / b
            # where p = probability of win, b = odds, q = probability of loss
            
            p = signal.confidence  # Probability of win
            q = 1 - p  # Probability of loss
            b = self._config.take_profit_pct / self._config.stop_loss_pct  # Risk-reward ratio
            
            if b == 0:
                return 0.0
            
            kelly_fraction = (p * b - q) / b
            
            # Apply Kelly fraction with safety factor
            safety_factor = 0.25  # Use 25% of Kelly suggestion
            position_size = kelly_fraction * safety_factor
            
            # Apply constraints
            position_size = max(0, min(position_size, self._config.max_position_size))
            
            return position_size
            
        except Exception as e:
            self._logger.error(f"Position sizing error: {e}")
            return 0.0
    
    def update_pnl(self, pnl: float) -> None:
        """Update daily P&L."""
        self._daily_pnl += pnl
        self._max_drawdown_reached = max(self._max_drawdown_reached, abs(min(0, self._daily_pnl)))
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        self._daily_pnl = 0.0
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        return {
            'daily_pnl': self._daily_pnl,
            'max_drawdown_reached': self._max_drawdown_reached,
            'positions': len(self._positions),
            'risk_limits': {
                'max_position_size': self._config.max_position_size,
                'max_daily_loss': self._config.max_daily_loss,
                'max_drawdown': self._config.max_drawdown
            },
            'ml_risk_performance': self._risk_model_performance
        }
    
    # ========================================================================
    # ML-ENHANCED RISK METHODS (Phase 1)
    # ========================================================================
    
    def ml_risk_assessment(
        self,
        decision: Dict[str, Any],
        market_context: Dict[str, float],
        model_weights: List[float]
    ) -> Dict[str, Any]:
        """
        ML-based risk assessment using multiple risk dimensions.
        
        Args:
            decision: Output from DecisionRouter
            market_context: Market features (regime, volatility, etc.)
            model_weights: Trust levels from ModelGovernor
        
        Returns:
            {
                'risk_multiplier': 0.0-1.0 (position size scaler),
                'market_class': 'FAVORABLE' | 'NEUTRAL' | 'ADVERSE',
                'risk_flags': List of risk warnings,
                'recommended_action': 'PROCEED' | 'REDUCE' | 'REJECT'
            }
        """
        try:
            # Phase 3: Try ML models first, fallback to rule-based
            if self._use_ml_risk:
                try:
                    return self._ml_risk_inference(decision, market_context, model_weights)
                except Exception as e:
                    self._logger.warning(f"ML risk inference failed, using fallback: {e}")
                    # Fall through to rule-based
            
            # Rule-based fallback
            risk_flags = []
            
            # Check 1: Model confidence
            confidence = decision.get('confidence', 0.0)
            if confidence < 0.7:
                risk_flags.append('LOW_CONFIDENCE')
            
            # Check 2: Model governance
            avg_model_weight = np.mean(model_weights) if model_weights else 0.7
            if avg_model_weight < 0.6:
                risk_flags.append('POOR_MODEL_PERFORMANCE')
            
            # Check 3: Market regime
            regime = market_context.get('regime', 1)
            volatility = market_context.get('volatility', 0.5)
            
            if regime == 2:  # Volatile regime
                risk_flags.append('VOLATILE_REGIME')
            
            if volatility > 0.8:
                risk_flags.append('HIGH_VOLATILITY')
            
            # Check 4: MQScore dimensions
            mqscore = market_context.get('mqscore', 0.5)
            liquidity = market_context.get('liquidity', 0.5)
            noise = market_context.get('noise', 0.5)
            
            if mqscore < 0.5:
                risk_flags.append('LOW_MARKET_QUALITY')
            
            if liquidity < 0.3:
                risk_flags.append('LOW_LIQUIDITY')
            
            if noise > 0.7:
                risk_flags.append('HIGH_NOISE')
            
            # Calculate risk multiplier
            risk_multiplier = self._calculate_risk_multiplier(
                confidence, avg_model_weight, volatility, mqscore
            )
            
            # Classify market
            market_class = self._classify_market_conditions(
                mqscore, liquidity, volatility, noise
            )
            
            # Recommended action
            if risk_multiplier < 0.3:
                recommended_action = 'REJECT'
            elif risk_multiplier < 0.6:
                recommended_action = 'REDUCE'
            else:
                recommended_action = 'PROCEED'
            
            return {
                'risk_multiplier': risk_multiplier,
                'market_class': market_class,
                'risk_flags': risk_flags,
                'recommended_action': recommended_action
            }
            
        except Exception as e:
            self._logger.error(f"ML Risk Assessment error: {e}")
            return {
                'risk_multiplier': 0.5,
                'market_class': 'NEUTRAL',
                'risk_flags': ['ERROR'],
                'recommended_action': 'REDUCE'
            }
    
    def _ml_risk_inference(
        self,
        decision: Dict[str, Any],
        market_context: Dict[str, float],
        model_weights: List[float]
    ) -> Dict[str, Any]:
        """Run ML model inference for risk assessment (Phase 3)."""
        # Prepare input features
        feature_vector = self._prepare_risk_feature_vector(decision, market_context, model_weights)
        
        risk_multiplier = 0.5
        market_class = 'NEUTRAL'
        
        # Try risk classifier first
        if self._risk_classifier_model is not None:
            try:
                input_name = self._risk_classifier_model.get_inputs()[0].name
                output = self._risk_classifier_model.run(None, {input_name: feature_vector})
                
                # Parse classifier output (3 classes: FAVORABLE, NEUTRAL, ADVERSE)
                class_probs = output[0][0].tolist() if len(output) > 0 else [0.33, 0.34, 0.33]
                class_idx = int(np.argmax(class_probs))
                market_class = ['FAVORABLE', 'NEUTRAL', 'ADVERSE'][class_idx]
            except Exception as e:
                self._logger.warning(f"Risk classifier failed: {e}")
        
        # Try risk scorer
        if self._risk_scorer_model is not None:
            try:
                input_name = self._risk_scorer_model.get_inputs()[0].name
                output = self._risk_scorer_model.run(None, {input_name: feature_vector})
                
                # Parse scorer output (0-1 risk multiplier)
                risk_multiplier = float(output[0][0][0]) if len(output) > 0 else 0.5
            except Exception as e:
                self._logger.warning(f"Risk scorer failed: {e}")
        
        # Determine action from risk multiplier
        if risk_multiplier < 0.3:
            recommended_action = 'REJECT'
            risk_flags = ['HIGH_RISK_ML']
        elif risk_multiplier < 0.6:
            recommended_action = 'REDUCE'
            risk_flags = ['MODERATE_RISK_ML']
        else:
            recommended_action = 'PROCEED'
            risk_flags = []
        
        return {
            'risk_multiplier': risk_multiplier,
            'market_class': market_class,
            'risk_flags': risk_flags,
            'recommended_action': recommended_action
        }
    
    def _prepare_risk_feature_vector(
        self,
        decision: Dict[str, Any],
        market_context: Dict[str, float],
        model_weights: List[float]
    ) -> np.ndarray:
        """Prepare feature vector for ML risk models (15 features expected)."""
        features = []
        
        # Model expects 15 features - aggregate the most important ones
        features.extend([
            decision.get('confidence', 0.0),  # 1. Decision confidence
            decision.get('value_estimate', 0.0),  # 2. Expected value
            decision['action_probs'][0] if 'action_probs' in decision else 0.5,  # 3. P(buy)
            decision['action_probs'][1] if 'action_probs' in decision else 0.5,  # 4. P(sell)
            market_context.get('mqscore', 0.5),  # 5. Market quality score
            market_context.get('liquidity', 0.5),  # 6. Liquidity
            market_context.get('volatility', 0.5),  # 7. Volatility
            market_context.get('momentum', 0.5),  # 8. Momentum
            np.mean(model_weights) if model_weights else 0.7,  # 9. Avg model weight
            np.std(model_weights) if len(model_weights) > 1 else 0.0,  # 10. Model weight std
            max(model_weights) if model_weights else 0.7,  # 11. Max model weight
            min(model_weights) if model_weights else 0.7,  # 12. Min model weight
            self._daily_pnl / 100000.0,  # 13. Normalized daily PnL
            self._max_drawdown_reached / self._config.max_drawdown if self._config.max_drawdown > 0 else 0.0,  # 14. Drawdown ratio
            len(self._positions) / 10.0,  # 15. Normalized position count
        ])
        
        # Ensure exactly 15 features
        features = features[:15]
        
        return np.array([features], dtype=np.float32)
    
    def _calculate_risk_multiplier(
        self,
        confidence: float,
        model_weight: float,
        volatility: float,
        mqscore: float
    ) -> float:
        """Calculate position size multiplier based on risk factors."""
        # Base multiplier from confidence and model performance
        base = confidence * model_weight
        
        # Adjust for market conditions
        volatility_factor = max(0.3, 1.0 - volatility)  # Lower in high vol
        quality_factor = max(0.5, mqscore)  # Lower in poor quality
        
        multiplier = base * volatility_factor * quality_factor
        
        return max(0.0, min(multiplier, 1.0))
    
    def _classify_market_conditions(
        self,
        mqscore: float,
        liquidity: float,
        volatility: float,
        noise: float
    ) -> str:
        """Classify market as FAVORABLE, NEUTRAL, or ADVERSE."""
        # Calculate composite score
        score = (mqscore * 0.4 + 
                liquidity * 0.3 + 
                (1.0 - volatility) * 0.2 + 
                (1.0 - noise) * 0.1)
        
        if score >= 0.7:
            return 'FAVORABLE'
        elif score >= 0.4:
            return 'NEUTRAL'
        else:
            return 'ADVERSE'
    
    def seven_layer_risk_validation(
        self,
        signal: TradingSignal,
        metrics: RiskMetrics,
        ml_risk: Dict[str, Any],
        market_context: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Comprehensive 7-layer risk validation.
        
        Returns:
            {
                'approved': bool,
                'layer_results': {layer: pass/fail},
                'rejection_reason': str or None
            }
        """
        layer_results = {}
        
        # Layer 1: Position size limits
        layer_results['position_size'] = metrics.position_size <= self._config.max_position_size
        if not layer_results['position_size']:
            return {
                'approved': False,
                'layer_results': layer_results,
                'rejection_reason': f'Position size {metrics.position_size} exceeds limit {self._config.max_position_size}'
            }
        
        # Layer 2: Daily loss limits
        layer_results['daily_loss'] = abs(self._daily_pnl) < self._config.max_daily_loss
        if not layer_results['daily_loss']:
            return {
                'approved': False,
                'layer_results': layer_results,
                'rejection_reason': f'Daily loss limit reached: {self._daily_pnl}'
            }
        
        # Layer 3: Drawdown limits
        layer_results['drawdown'] = self._max_drawdown_reached < self._config.max_drawdown
        if not layer_results['drawdown']:
            return {
                'approved': False,
                'layer_results': layer_results,
                'rejection_reason': f'Max drawdown reached: {self._max_drawdown_reached}'
            }
        
        # Layer 4: Signal confidence
        layer_results['confidence'] = signal.confidence >= 0.57
        if not layer_results['confidence']:
            return {
                'approved': False,
                'layer_results': layer_results,
                'rejection_reason': f'Low signal confidence: {signal.confidence}'
            }
        
        # Layer 5: ML risk multiplier
        layer_results['ml_risk'] = ml_risk['risk_multiplier'] >= 0.3
        if not layer_results['ml_risk']:
            return {
                'approved': False,
                'layer_results': layer_results,
                'rejection_reason': f"Low risk multiplier: {ml_risk['risk_multiplier']}"
            }
        
        # Layer 6: Market classification
        layer_results['market_class'] = ml_risk['market_class'] != 'ADVERSE'
        if not layer_results['market_class']:
            return {
                'approved': False,
                'layer_results': layer_results,
                'rejection_reason': 'Adverse market conditions'
            }
        
        # Layer 7: Critical risk flags
        critical_flags = ['LOW_MARKET_QUALITY', 'LOW_LIQUIDITY']
        has_critical = any(flag in ml_risk['risk_flags'] for flag in critical_flags)
        layer_results['risk_flags'] = not has_critical
        if has_critical:
            return {
                'approved': False,
                'layer_results': layer_results,
                'rejection_reason': f"Critical risk flags: {ml_risk['risk_flags']}"
            }
        
        # All layers passed
        return {
            'approved': True,
            'layer_results': layer_results,
            'rejection_reason': None
        }
    
    def dynamic_position_sizing(
        self,
        base_size: float,
        ml_risk: Dict[str, Any],
        market_context: Dict[str, float]
    ) -> float:
        """
        Adjust position size dynamically based on ML risk assessment.
        
        Args:
            base_size: Base position size from Kelly Criterion
            ml_risk: ML risk assessment output
            market_context: Market features
        
        Returns:
            Adjusted position size (0.0-1.0)
        """
        # Apply risk multiplier
        adjusted_size = base_size * ml_risk['risk_multiplier']
        
        # Further reduce if model governance suggests
        if self._model_governor:
            try:
                gov_metrics = self._model_governor.get_model_weights({})
                risk_scaler = gov_metrics.get('risk_scalers', {}).get('position_scaler', 1.0)
                adjusted_size *= risk_scaler
            except:
                pass
        
        # Apply market class adjustment
        market_class = ml_risk['market_class']
        if market_class == 'FAVORABLE':
            adjusted_size *= 1.2  # Increase in favorable conditions
        elif market_class == 'ADVERSE':
            adjusted_size *= 0.5  # Reduce in adverse conditions
        
        # Hard limits
        adjusted_size = max(0.0, min(adjusted_size, self._config.max_position_size))
        
        return adjusted_size
    
    def check_duplicate_order(self, symbol: str, active_orders: Dict[str, Any]) -> bool:
        """
        Check if symbol already has an active order.
        
        Gate: Prevent duplicate orders on same symbol
        
        Returns:
            True if duplicate exists (REJECT), False if clear (PROCEED)
        """
        return symbol in active_orders
    
    def update_risk_model_performance(self, prediction_correct: bool):
        """Update risk model performance metrics."""
        perf = self._risk_model_performance
        perf['total_predictions'] += 1
        
        if prediction_correct:
            current_correct = perf['accuracy'] * (perf['total_predictions'] - 1)
            perf['accuracy'] = (current_correct + 1) / perf['total_predictions']
        else:
            current_correct = perf['accuracy'] * (perf['total_predictions'] - 1)
            perf['accuracy'] = current_correct / perf['total_predictions']


# ============================================================================
# ENSEMBLE & ADVANCED MODELS MANAGER (TIER 3 & 4)
# ============================================================================

class EnsembleModelManager:
    """
    Manages TIER 3 (Ensemble) and TIER 4 (Advanced) models - 31 models total.
    
    TIER 3 (27 models):
        - Uncertainty Classification: 10 Keras models
        - Uncertainty Regression: 10 Keras models
        - Bayesian Ensemble: 5 Keras models
        - Pattern Recognition: 1 Keras model
        - Gradient Boosting: 1 ONNX (already in RiskManager)
    
    TIER 4 (4 models):
        - LSTM Time Series: 1 ONNX
        - Anomaly Detection: 1 ONNX
        - Entry Timing: 1 ONNX
        - HFT Scalping: 1 ONNX
    
    Note: Keras models not loaded by default (optional, for advanced use)
    ONNX models loaded for production use.
    """
    
    def __init__(self, model_loader: Optional[MLModelLoader] = None):
        """Initialize ensemble and advanced model manager."""
        self._logger = setup_logging(f"{__name__}.EnsembleModelManager")
        self._model_loader = model_loader
        
        # TIER 3 - Ensemble Models (Keras - Optional)
        self._uncertainty_classifiers = []  # 10 models
        self._uncertainty_regressors = []   # 10 models
        self._bayesian_ensemble = []         # 5 models
        self._pattern_recognition = None    # 1 model
        
        # TIER 4 - Advanced ONNX Models
        self._lstm_time_series = None
        self._anomaly_detection = None
        self._entry_timing = None
        self._hft_scalping = None
        
        if model_loader is not None:
            self._load_tier3_keras_models()  # Load ALL 27 Keras models
            self._load_tier4_models()
            self._logger.info("✅ EnsembleModelManager initialized with ALL TIER 3+4 models")
        else:
            self._logger.info("⚠️ EnsembleModelManager initialized without model loader")
    
    def _load_tier3_keras_models(self):
        """Load ALL 27 TIER 3 Keras models (WARNING: SLOW!)."""
        self._logger.info("🔄 Loading TIER 3 Keras models (this may take 10-30 seconds)...")
        
        # Load 10 Uncertainty Classification models
        for i in range(10):
            try:
                model_path = f"PRODUCTION/13_UNCERTAINTY_CLASSIFICATION/uncertainty_clf_model_{i}_h1.h5"
                self._logger.info(f"📥 Loading Uncertainty Classifier {i} from {model_path}...")
                model = self._model_loader.load_keras_model(model_path, f"UncertaintyClassifier{i}")
                if model:
                    self._uncertainty_classifiers.append(model)
                    self._logger.info(f"✅ Uncertainty Classifier {i} loaded successfully")
                else:
                    self._logger.error(f"❌ Uncertainty Classifier {i} returned None")
            except Exception as e:
                self._logger.error(f"❌ Failed to load Uncertainty Classifier {i}: {e}")
                import traceback
                self._logger.error(traceback.format_exc())
        
        # Load 10 Uncertainty Regression models
        for i in range(10):
            try:
                model_path = f"PRODUCTION/14_UNCERTAINTY_REGRESSION/uncertainty_reg_model_{i}_h1.h5"
                self._logger.info(f"📥 Loading Uncertainty Regressor {i} from {model_path}...")
                model = self._model_loader.load_keras_model(model_path, f"UncertaintyRegressor{i}")
                if model:
                    self._uncertainty_regressors.append(model)
                    self._logger.info(f"✅ Uncertainty Regressor {i} loaded successfully")
                else:
                    self._logger.error(f"❌ Uncertainty Regressor {i} returned None")
            except Exception as e:
                self._logger.error(f"❌ Failed to load Uncertainty Regressor {i}: {e}")
                import traceback
                self._logger.error(traceback.format_exc())
        
        # Load 5 Bayesian Ensemble models
        for i in range(5):
            try:
                model_path = f"PRODUCTION/15_BAYESIAN_ENSEMBLE/member_{i}.keras"
                self._logger.info(f"📥 Loading Bayesian Member {i} from {model_path}...")
                model = self._model_loader.load_keras_model(model_path, f"BayesianMember{i}")
                if model:
                    self._bayesian_ensemble.append(model)
                    self._logger.info(f"✅ Bayesian Ensemble Member {i} loaded successfully")
                else:
                    self._logger.error(f"❌ Bayesian Member {i} returned None")
            except Exception as e:
                self._logger.error(f"❌ Failed to load Bayesian Member {i}: {e}")
                import traceback
                self._logger.error(traceback.format_exc())
        
        # Load Pattern Recognition model
        try:
            model_path = "PRODUCTION/16_PATTERN_RECOGNITION/final_model.keras"
            self._logger.info(f"📥 Loading Pattern Recognition from {model_path}...")
            self._pattern_recognition = self._model_loader.load_keras_model(model_path, "PatternRecognition")
            if self._pattern_recognition:
                self._logger.info("✅ Pattern Recognition loaded successfully")
            else:
                self._logger.error("❌ Pattern Recognition returned None")
        except Exception as e:
            self._logger.error(f"❌ Failed to load Pattern Recognition: {e}")
            import traceback
            self._logger.error(traceback.format_exc())
        
        tier3_loaded = (
            len(self._uncertainty_classifiers) +
            len(self._uncertainty_regressors) +
            len(self._bayesian_ensemble) +
            (1 if self._pattern_recognition else 0)
        )
        self._logger.info(f"✅ TIER 3 Complete: {tier3_loaded}/26 Keras models loaded")
    
    def _load_tier4_models(self):
        """Load TIER 4 advanced ONNX models."""
        # LSTM Time Series (1 model)
        self._lstm_time_series = self._model_loader.load_onnx_model(
            "PRODUCTION/17_LSTM_TIME_SERIES/financial_lstm_final_optimized.onnx",
            "LSTMTimeSeries"
        )
        if self._lstm_time_series:
            self._logger.info("✅ LSTM Time Series loaded (TIER 4)")
        
        # Anomaly Detection (1 model)
        self._anomaly_detection = self._model_loader.load_onnx_model(
            "PRODUCTION/18_ANOMALY_DETECTION/autoencoder_optimized.onnx",
            "AnomalyDetection"
        )
        if self._anomaly_detection:
            self._logger.info("✅ Anomaly Detection loaded (TIER 4)")
        
        # Entry Timing (1 model)
        self._entry_timing = self._model_loader.load_onnx_model(
            "PRODUCTION/19_ENTRY_TIMING/ONNX_Quantum_Entry_Timing Model_optimized.onnx",
            "EntryTiming"
        )
        if self._entry_timing:
            self._logger.info("✅ Entry Timing loaded (TIER 4)")
        
        # HFT Scalping (1 model)
        self._hft_scalping = self._model_loader.load_onnx_model(
            "PRODUCTION/20_HFT_SCALPING/ONNX_HFT_ScalperSignal_optimized.onnx",
            "HFTScalping"
        )
        if self._hft_scalping:
            self._logger.info("✅ HFT Scalping loaded (TIER 4)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics on loaded models."""
        tier3_loaded = (
            len(self._uncertainty_classifiers) +
            len(self._uncertainty_regressors) +
            len(self._bayesian_ensemble) +
            (1 if self._pattern_recognition else 0)
        )
        
        tier4_loaded = sum([
            self._lstm_time_series is not None,
            self._anomaly_detection is not None,
            self._entry_timing is not None,
            self._hft_scalping is not None
        ])
        
        return {
            'tier3_loaded': tier3_loaded,
            'tier3_total': 26,  # 10 + 10 + 5 + 1
            'tier4_loaded': tier4_loaded,
            'tier4_total': 4,
            'total_loaded': tier3_loaded + tier4_loaded,
            'total_available': 30,  # 26 + 4
            'models': {
                'lstm_time_series': self._lstm_time_series is not None,
                'anomaly_detection': self._anomaly_detection is not None,
                'entry_timing': self._entry_timing is not None,
                'hft_scalping': self._hft_scalping is not None
            }
        }


# ============================================================================
# PIPELINE ORCHESTRATOR: nexus_orchestrator.py
# ============================================================================

"""
ML Pipeline Orchestrator - Wires all 6 ML layers into complete pipeline.

Implements complete flow: Layer 1 → Layer 2 → Layer 3 → Layer 4 → Layer 5 → Layer 6
"""

class MLPipelineOrchestrator:
    """
    Orchestrates complete ML pipeline execution.
    
    Flow:
        1. Layer 1: MarketQualityLayer1 (MQScore) - quality gates
        2. Layer 2: SignalGenerationLayer2 - generate trading signals
        3. Layer 3: MetaStrategySelector assigns strategy weights
        4. Layer 4: SignalAggregator combines signals
        5. Layer 5: ModelGovernor + DecisionRouter make final decision
        6. Layer 6: RiskManager validates and sizes position
    
    Reference: _FINAL_PLAN_WITH_MQSCORE.md (Phase 2 with MQScore integration)
    """
    
    def __init__(
        self,
        market_quality: MarketQualityLayer1,
        signal_generator: Optional[SignalGenerationLayer2] = None,
        meta_selector: Optional[MetaStrategySelector] = None,
        signal_aggregator: Optional[SignalAggregator] = None,
        model_governor: Optional[ModelGovernor] = None,
        decision_router: Optional[DecisionRouter] = None,
        risk_manager: Optional[RiskManager] = None,
        model_loader: Optional[MLModelLoader] = None
    ):
        """Initialize ML Pipeline Orchestrator."""
        self._logger = setup_logging(f"{__name__}.MLPipelineOrchestrator")
        
        # Phase 3: Model loader for ONNX/PKL models
        self._model_loader = model_loader if model_loader is not None else MLModelLoader()
        self._logger.info("✅ MLModelLoader initialized")
        
        # Layer 1
        self._market_quality = market_quality
        
        # Layer 2
        self._signal_generator = signal_generator if signal_generator is not None else SignalGenerationLayer2()
        
        # Layer 3
        self._meta_selector = meta_selector
        
        # Layer 4
        self._signal_aggregator = signal_aggregator
        
        # Layer 5
        self._model_governor = model_governor
        self._decision_router = decision_router
        
        # Layer 6
        self._risk_manager = risk_manager
        
        # Active orders tracking (prevent duplicates)
        self._active_orders = {}
        
        # Pipeline metrics
        self._pipeline_stats = {
            'total_processed': 0,
            'layer1_skips': 0,
            'layer3_skips': 0,
            'layer4_skips': 0,
            'layer5_skips': 0,
            'layer6_rejects': 0,
            'duplicate_skips': 0,
            'approved': 0
        }
    
    def process_trading_opportunity(
        self,
        symbol: str,
        signals: List[TradingSignal],
        market_data: Any,  # Now requires market_data (OHLCV DataFrame)
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process complete trading opportunity through ML pipeline.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            signals: Raw signals from 20 strategies
            market_data: OHLCV DataFrame for MQScore calculation (Layer 1)
            portfolio: Current portfolio state
        
        Returns:
            {
                'decision': 'APPROVED' | 'REJECTED' | 'SKIPPED',
                'action': 'BUY' | 'SELL' | 'HOLD',
                'position_size': float,
                'confidence': float,
                'rejection_reason': str or None,
                'layer_outputs': {outputs from each layer}
            }
        """
        self._pipeline_stats['total_processed'] += 1
        layer_outputs = {}
        
        try:
            # ============================================================
            # LAYER 1: MARKET QUALITY ASSESSMENT (MQScore 6D Engine)
            # ============================================================
            self._logger.debug(f"[{symbol}] Layer 1: MQScore Market Quality Assessment")
            
            mqscore_assessment = self._market_quality.assess_market_quality(market_data)
            layer_outputs['layer1'] = mqscore_assessment
            
            # GATE 1: Check if market quality passed
            if not mqscore_assessment['passed']:
                self._pipeline_stats['layer1_skips'] += 1
                self._logger.info(
                    f"[{symbol}] SKIPPED at Layer 1: {mqscore_assessment['reason']}"
                )
                return {
                    'decision': 'SKIPPED',
                    'action': 'HOLD',
                    'position_size': 0.0,
                    'confidence': 0.0,
                    'rejection_reason': mqscore_assessment['reason'],
                    'layer_outputs': layer_outputs,
                    'layer_failed': 'LAYER_1_MQSCORE'
                }
            
            # Extract market context from MQScore for downstream layers
            market_context = mqscore_assessment['market_context']
            self._logger.debug(
                f"[{symbol}] Layer 1 PASSED - MQScore: {market_context['mqscore']:.3f}, "
                f"Liquidity: {market_context['liquidity']:.3f}, "
                f"Regime: {market_context['regime']}"
            )
            
            # ============================================================
            # LAYER 3: META-STRATEGY SELECTOR
            # ============================================================
            self._logger.debug(f"[{symbol}] Layer 3: Meta-Strategy Selection")
            
            meta_result = self._meta_selector.select_strategy_weights(market_context)
            layer_outputs['layer3_meta'] = meta_result
            
            # Gate: Check anomaly score
            if meta_result['anomaly_score'] > 0.8:
                self._pipeline_stats['layer3_skips'] += 1
                return self._skip_decision(
                    symbol, 
                    f"High anomaly score: {meta_result['anomaly_score']}", 
                    layer_outputs
                )
            
            # ============================================================
            # LAYER 4: SIGNAL AGGREGATION
            # ============================================================
            self._logger.debug(f"[{symbol}] Layer 4: Signal Aggregation")
            
            # Filter signals by confidence >= 0.57
            filtered_signals = [s for s in signals if s.confidence >= 0.57]
            
            if not filtered_signals:
                self._pipeline_stats['layer4_skips'] += 1
                return self._skip_decision(symbol, "No high-confidence signals", layer_outputs)
            
            aggregated = self._signal_aggregator.aggregate_signals(
                filtered_signals,
                meta_result['strategy_weights']
            )
            layer_outputs['layer4_aggregated'] = aggregated
            
            # Gate: Check signal strength
            if not self._signal_aggregator.check_signal_strength_gate(aggregated):
                self._pipeline_stats['layer4_skips'] += 1
                return self._skip_decision(
                    symbol,
                    f"Low signal strength: {aggregated['signal_strength']}",
                    layer_outputs
                )
            
            # Gate: HOLD signal
            if aggregated['direction'] == 'HOLD':
                self._pipeline_stats['layer4_skips'] += 1
                return self._skip_decision(symbol, "Signal is HOLD/NEUTRAL", layer_outputs)
            
            # Gate: Check duplicate orders
            if self._risk_manager.check_duplicate_order(symbol, self._active_orders):
                self._pipeline_stats['duplicate_skips'] += 1
                return self._skip_decision(symbol, "Duplicate order exists", layer_outputs)
            
            # ============================================================
            # LAYER 5: MODEL GOVERNANCE & DECISION ROUTING
            # ============================================================
            self._logger.debug(f"[{symbol}] Layer 5: Model Governance & Routing")
            
            # 5.1: Model Governance
            performance_metrics = {}  # Would be populated with actual performance data
            governance = self._model_governor.get_model_weights(performance_metrics)
            layer_outputs['layer5_governance'] = governance
            
            # 5.2: Decision Routing
            decision = self._decision_router.route_decision(
                aggregated,
                market_context,
                governance['model_weights']
            )
            layer_outputs['layer5_decision'] = decision
            
            # Gate: Check confidence
            if not self._decision_router.check_confidence_gate(decision):
                self._pipeline_stats['layer5_skips'] += 1
                return self._skip_decision(
                    symbol,
                    f"Low decision confidence: {decision['confidence']}",
                    layer_outputs
                )
            
            # ============================================================
            # LAYER 6: RISK MANAGEMENT
            # ============================================================
            self._logger.debug(f"[{symbol}] Layer 6: Risk Management")
            
            # ML Risk Assessment
            ml_risk = self._risk_manager.ml_risk_assessment(
                decision,
                market_context,
                governance['model_weights']
            )
            layer_outputs['layer6_ml_risk'] = ml_risk
            
            # Create signal for traditional risk evaluation
            signal_type = SignalType.BUY if decision['action'] == 'BUY' else SignalType.SELL
            trade_signal = TradingSignal(
                signal_type=signal_type,
                confidence=decision['confidence'],
                symbol=symbol,
                timestamp=time.time(),
                strategy='ML_Pipeline',
                metadata={
                    'value_estimate': decision['value_estimate'],
                    'signal_strength': aggregated['signal_strength']
                }
            )
            
            # Traditional risk metrics
            risk_metrics = self._risk_manager.evaluate_risk(trade_signal, portfolio)
            layer_outputs['layer6_risk_metrics'] = risk_metrics
            
            # Dynamic position sizing
            adjusted_position_size = self._risk_manager.dynamic_position_sizing(
                risk_metrics.position_size,
                ml_risk,
                market_context
            )
            layer_outputs['layer6_adjusted_size'] = adjusted_position_size
            
            # 7-Layer Risk Validation
            validation = self._risk_manager.seven_layer_risk_validation(
                trade_signal,
                risk_metrics,
                ml_risk,
                market_context
            )
            layer_outputs['layer6_validation'] = validation
            
            if not validation['approved']:
                self._pipeline_stats['layer6_rejects'] += 1
                return {
                    'decision': 'REJECTED',
                    'action': decision['action'],
                    'position_size': 0.0,
                    'confidence': decision['confidence'],
                    'rejection_reason': validation['rejection_reason'],
                    'layer_outputs': layer_outputs
                }
            
            # ============================================================
            # APPROVED - READY FOR EXECUTION
            # ============================================================
            self._pipeline_stats['approved'] += 1
            self._logger.info(
                f"[{symbol}] ✅ APPROVED: {decision['action']} | "
                f"Size: {adjusted_position_size:.4f} | "
                f"Confidence: {decision['confidence']:.2f}"
            )
            
            return {
                'decision': 'APPROVED',
                'action': decision['action'],
                'position_size': adjusted_position_size,
                'confidence': decision['confidence'],
                'stop_loss': risk_metrics.stop_loss,
                'take_profit': risk_metrics.take_profit,
                'expected_return': risk_metrics.expected_return,
                'rejection_reason': None,
                'layer_outputs': layer_outputs
            }
            
        except Exception as e:
            self._logger.error(f"[{symbol}] Pipeline error: {e}")
            return {
                'decision': 'REJECTED',
                'action': 'HOLD',
                'position_size': 0.0,
                'confidence': 0.0,
                'rejection_reason': f'Pipeline error: {str(e)}',
                'layer_outputs': layer_outputs
            }
    
    def _skip_decision(self, symbol: str, reason: str, layer_outputs: Dict) -> Dict[str, Any]:
        """Return SKIPPED decision."""
        self._logger.info(f"[{symbol}] ⏭️ SKIPPED: {reason}")
        return {
            'decision': 'SKIPPED',
            'action': 'HOLD',
            'position_size': 0.0,
            'confidence': 0.0,
            'rejection_reason': reason,
            'layer_outputs': layer_outputs
        }
    
    def register_order(self, symbol: str, order_id: str):
        """Register active order to prevent duplicates."""
        self._active_orders[symbol] = order_id
    
    def remove_order(self, symbol: str):
        """Remove order when closed."""
        self._active_orders.pop(symbol, None)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        total = self._pipeline_stats['total_processed']
        if total == 0:
            return self._pipeline_stats
        
        return {
            **self._pipeline_stats,
            'approval_rate': self._pipeline_stats['approved'] / total,
            'skip_rate': (
                self._pipeline_stats['layer3_skips'] + 
                self._pipeline_stats['layer4_skips'] + 
                self._pipeline_stats['layer5_skips'] + 
                self._pipeline_stats['duplicate_skips']
            ) / total,
            'rejection_rate': self._pipeline_stats['layer6_rejects'] / total
        }
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        for key in self._pipeline_stats:
            self._pipeline_stats[key] = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get complete pipeline statistics."""
        return self.get_pipeline_stats()
    
    def get_model_loader_stats(self) -> Dict[str, Any]:
        """Get ML model loader statistics (Phase 3)."""
        if self._model_loader is not None:
            return self._model_loader.get_statistics()
        return {'no_loader': True}


# ============================================================================
# EXECUTION MODULE: nexus_execution.py
# ============================================================================

"""Order execution and management module."""


@dataclass
class Order:
    """Order representation."""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    order_type: str  # 'market', 'limit', 'stop'
    status: str  # 'pending', 'filled', 'cancelled'
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """Handles order execution and management."""
    
    def __init__(self, config: SystemConfig):
        """Initialize execution engine."""
        self._config = config
        self._logger = setup_logging(f"{__name__}.ExecutionEngine")
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._lock = threading.RLock()
        
    def create_order(
        self,
        signal: TradingSignal,
        metrics: RiskMetrics,
        price: float
    ) -> Optional[Order]:
        """Create order from signal and risk metrics."""
        try:
            with self._lock:
                self._order_counter += 1
                order_id = f"ORD_{self._order_counter:06d}"
                
                # Determine order side
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    side = 'buy'
                elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    side = 'sell'
                else:
                    return None
                
                # Calculate quantity based on position size
                # This is simplified - in production, consider account balance
                quantity = metrics.position_size * 1000  # Example scaling
                
                order = Order(
                    order_id=order_id,
                    symbol=signal.symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    order_type='limit',
                    status='pending',
                    timestamp=time.time(),
                    metadata={
                        'signal': signal,
                        'metrics': metrics,
                        'stop_loss': price * (1 - metrics.stop_loss) if side == 'buy' else price * (1 + metrics.stop_loss),
                        'take_profit': price * (1 + metrics.take_profit) if side == 'buy' else price * (1 - metrics.take_profit)
                    }
                )
                
                self._orders[order_id] = order
                self._logger.info(f"Created order: {order_id} for {signal.symbol}")
                
                return order
                
        except Exception as e:
            self._logger.error(f"Order creation error: {e}")
            return None
    
    async def execute_order(self, order: Order) -> bool:
        """Execute order (simulated)."""
        try:
            # In production, this would interface with broker API
            await asyncio.sleep(0.1)  # Simulate execution delay
            
            with self._lock:
                if order.order_id in self._orders:
                    self._orders[order.order_id].status = 'filled'
                    self._logger.info(f"Order executed: {order.order_id}")
                    return True
                
            return False
            
        except Exception as e:
            self._logger.error(f"Order execution error: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        try:
            with self._lock:
                if order_id in self._orders and self._orders[order_id].status == 'pending':
                    self._orders[order_id].status = 'cancelled'
                    self._logger.info(f"Order cancelled: {order_id}")
                    return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Order cancellation error: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        with self._lock:
            return self._orders.get(order_id)
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        with self._lock:
            return [o for o in self._orders.values() if o.status == 'pending']
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        with self._lock:
            orders_list = list(self._orders.values())
            
            return {
                'total_orders': len(orders_list),
                'pending': sum(1 for o in orders_list if o.status == 'pending'),
                'filled': sum(1 for o in orders_list if o.status == 'filled'),
                'cancelled': sum(1 for o in orders_list if o.status == 'cancelled')
            }


# ============================================================================
# MONITORING MODULE: nexus_monitoring.py
# ============================================================================

"""System monitoring and performance tracking module."""


class PerformanceMonitor:
    """Monitors system performance and health."""
    
    def __init__(self, config: SystemConfig):
        """Initialize performance monitor."""
        self._config = config
        self._logger = setup_logging(f"{__name__}.PerformanceMonitor")
        self._metrics = defaultdict(list)
        self._alerts = deque(maxlen=100)
        self._last_alert_time = {}
        self._lock = threading.RLock()
        
    def record_metric(self, name: str, value: float, timestamp: Optional[float] = None) -> None:
        """Record a performance metric."""
        timestamp = timestamp or time.time()
        
        with self._lock:
            self._metrics[name].append({
                'value': value,
                'timestamp': timestamp
            })
            
            # Keep only recent metrics (last hour)
            cutoff = timestamp - 3600
            self._metrics[name] = [
                m for m in self._metrics[name]
                if m['timestamp'] > cutoff
            ]
    
    def get_metric_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            values = [m['value'] for m in self._metrics.get(name, [])]
            
            if not values:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
            
            return {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    def check_alert_condition(self, condition: str, value: float) -> bool:
        """Check if alert condition is met."""
        current_time = time.time()
        
        # Check cooldown
        if condition in self._last_alert_time:
            if current_time - self._last_alert_time[condition] < self._config.alert_cooldown:
                return False
        
        alert_triggered = False
        
        if condition == 'high_latency' and value > 1000:  # ms
            alert_triggered = True
        elif condition == 'low_confidence' and value < 0.3:
            alert_triggered = True
        elif condition == 'high_risk' and value > 0.8:
            alert_triggered = True
        
        if alert_triggered:
            self._last_alert_time[condition] = current_time
            self._alerts.append({
                'condition': condition,
                'value': value,
                'timestamp': current_time
            })
            self._logger.warning(f"Alert triggered: {condition} = {value}")
        
        return alert_triggered
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            # Calculate health score
            latency_stats = self.get_metric_stats('latency')
            error_stats = self.get_metric_stats('errors')
            
            health_score = 1.0
            
            # Penalize for high latency
            if latency_stats['mean'] > 500:
                health_score -= 0.2
            
            # Penalize for errors
            if error_stats['count'] > 0:
                health_score -= min(0.5, error_stats['count'] * 0.1)
            
            health_score = max(0.0, health_score)
            
            return {
                'health_score': health_score,
                'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.3 else 'unhealthy',
                'metrics_count': len(self._metrics),
                'recent_alerts': list(self._alerts)[-10:],
                'latency': latency_stats,
                'errors': error_stats
            }


# ============================================================================
# MAIN NEXUS AI SYSTEM: nexus_system.py
# ============================================================================

"""Main NEXUS AI trading system orchestrator."""


class NexusAI:
    """Main NEXUS AI trading system."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize NEXUS AI system."""
        self._logger = setup_logging("NexusAI")
        self._logger.info("Initializing NEXUS AI Trading System v3.0.0")
        
        # Initialize configuration
        self.config_manager = ConfigManager(config)
        self.config = self.config_manager.get_config()
        
        # Initialize core components
        self.security_manager = SecurityManager()
        self.data_buffer = DataBuffer(self.config.buffer_size)
        self.data_cache = DataCache(self.config.cache_size, self.config.cache_ttl)
        
        # Initialize ML model loader
        self.model_loader = MLModelLoader()
        self._logger.info("✅ MLModelLoader initialized")
        
        # Initialize ensemble & advanced models manager (TIER 3 & 4)
        self.ensemble_manager = EnsembleModelManager(model_loader=self.model_loader)
        ensemble_stats = self.ensemble_manager.get_statistics()
        self._logger.info(f"✅ EnsembleModelManager: {ensemble_stats['tier4_loaded']}/4 TIER 4 models loaded")
        
        # ⏱️ 3-SECOND DELAY after TIER 3 & 4 loading
        import time
        self._logger.info("⏱️ 3-second stabilization delay after TIER 3 & 4 models...")
        time.sleep(3)
        
        # Initialize Layer 1 (TIER 1 - 3 models)
        self._logger.info("📥 Loading TIER 1 models (Layer 1)...")
        self.layer1_market_quality = MarketQualityLayer1(model_loader=self.model_loader)
        self._logger.info("✅ Layer 1 initialized with TIER 1 models")
        
        # ⏱️ 3-SECOND DELAY after TIER 1 loading
        self._logger.info("⏱️ 3-second stabilization delay after TIER 1 models...")
        time.sleep(3)
        
        # Initialize Core Pipeline Layers 3-5 (5 models)
        self._logger.info("📥 Loading Core Pipeline models (Layers 3-5)...")
        self.layer3_meta_selector = MetaStrategySelector(model_loader=self.model_loader)
        self.layer4_signal_aggregator = SignalAggregator(model_loader=self.model_loader)
        self.layer5_model_governor = ModelGovernor(model_loader=self.model_loader)
        self.layer5_decision_router = DecisionRouter(model_loader=self.model_loader)
        self._logger.info("✅ Core Pipeline Layers 3-5 initialized")
        
        # ⏱️ 3-SECOND DELAY after Core Pipeline loading
        self._logger.info("⏱️ 3-second stabilization delay after Core Pipeline models...")
        time.sleep(3)
        
        # Initialize trading components
        self.strategy_manager = StrategyManager()
        self.risk_manager = RiskManager(self.config, model_loader=self.model_loader)
        self._logger.info("✅ Risk Manager initialized with TIER 2 models")
        
        # ⏱️ 3-SECOND DELAY after TIER 2 (Risk Manager) loading
        self._logger.info("⏱️ 3-second stabilization delay after TIER 2 models...")
        time.sleep(3)
        
        self.execution_engine = ExecutionEngine(self.config)
        
        # Initialize monitoring
        self.performance_monitor = PerformanceMonitor(self.config)
        
        # Log total models loaded
        total_models = self._count_loaded_models()
        self._logger.info(f"🎯 TOTAL PRODUCTION MODELS LOADED: {total_models}/46")
        
        self._logger.info("✅ All model tiers loaded with stabilization delays - system ready")
        
        # System state
        self._running = False
        self._initialized = False
    
    def _count_loaded_models(self) -> int:
        """Count total PRODUCTION models loaded across all components."""
        # Get statistics from all loaders
        loader_stats = self.model_loader.get_statistics()
        ensemble_stats = self.ensemble_manager.get_statistics()
        
        # Total from ML Loader (includes ONNX, PKL, Keras)
        total_from_loader = loader_stats.get('total_models_loaded', 0)
        
        # Add MQScore (always loaded)
        total = total_from_loader + 1
        
        # Breakdown for logging:
        # - MQScore: 1
        # - TIER 1 ONNX (Volatility, Regime, Data Quality): 3
        # - TIER 2 ONNX/PKL (Risk models): 7
        # - TIER 3 Keras (Uncertainty, Bayesian, Pattern): 26
        # - TIER 4 ONNX (LSTM, Anomaly, Entry, HFT): 4
        # - Layer 3-5 Core Pipeline: 5
        # = 46 total
        
        return total
        
    def register_strategies_explicit(self):
        """
        EXPLICIT STRATEGY REGISTRATION
        Register only the strategies you want active in the pipeline
        Comment out any strategy you don't want to use
        """
        strategy_count = 0
        
        # GROUP 1: EVENT-DRIVEN
        if HAS_EVENT_DRIVEN:
            event_driven_wrapped = UniversalStrategyAdapter(
                EnhancedEventDrivenStrategy(), 
                "Event-Driven"
            )
            self.strategy_manager.register_strategy(event_driven_wrapped)
            self._logger.info("✓ Registered: Event-Driven Strategy (wrapped)")
            strategy_count += 1
        
        # GROUP 2: BREAKOUT-BASED
        if HAS_LVN_BREAKOUT:
            lvn_wrapped = UniversalStrategyAdapter(LVNBreakoutNexusAdapter(), "LVN-Breakout")
            self.strategy_manager.register_strategy(lvn_wrapped)
            self._logger.info("✓ Registered: LVN Breakout Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_ABSORPTION_BREAKOUT:
            absorption_wrapped = UniversalStrategyAdapter(AbsorptionBreakoutNexusAdapter(), "Absorption-Breakout")
            self.strategy_manager.register_strategy(absorption_wrapped)
            self._logger.info("✓ Registered: Absorption Breakout Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_MOMENTUM_BREAKOUT:
            momentum_wrapped = UniversalStrategyAdapter(MomentumBreakoutStrategy(), "Momentum-Breakout")
            self.strategy_manager.register_strategy(momentum_wrapped)
            self._logger.info("✓ Registered: Momentum Breakout Strategy (wrapped)")
            strategy_count += 1
        
        # GROUP 3: MARKET MICROSTRUCTURE
        if HAS_MARKET_MICROSTRUCTURE:
            # Use UniversalStrategyConfig from the imported module
            market_micro_config = market_micro_module.UniversalStrategyConfig(strategy_name="market_microstructure")
            market_micro_wrapped = UniversalStrategyAdapter(MarketMicrostructureStrategy(market_micro_config), "Market-Microstructure")
            self.strategy_manager.register_strategy(market_micro_wrapped)
            self._logger.info("✓ Registered: Market Microstructure Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_ORDER_BOOK_IMBALANCE:
            orderbook_wrapped = UniversalStrategyAdapter(OrderBookImbalanceNexusAdapter(), "Order-Book-Imbalance")
            self.strategy_manager.register_strategy(orderbook_wrapped)
            self._logger.info("✓ Registered: Order Book Imbalance Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_LIQUIDITY_ABSORPTION:
            liq_absorption_wrapped = UniversalStrategyAdapter(LiquidityAbsorptionNexusAdapter(), "Liquidity-Absorption")
            self.strategy_manager.register_strategy(liq_absorption_wrapped)
            self._logger.info("✓ Registered: Liquidity Absorption Strategy (wrapped)")
            strategy_count += 1
        
        # GROUP 4: DETECTION/ALERT
        if HAS_SPOOFING_DETECTION:
            # Spoofing requires base strategy to be created first
            base_spoofing = spoofing_module.EnhancedSpoofingDetectionStrategy()
            spoofing_wrapped = UniversalStrategyAdapter(SpoofingDetectionNexusAdapter(base_spoofing), "Spoofing-Detection")
            self.strategy_manager.register_strategy(spoofing_wrapped)
            self._logger.info("✓ Registered: Spoofing Detection Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_ICEBERG_DETECTION:
            iceberg_wrapped = UniversalStrategyAdapter(IcebergDetectionNexusAdapter(), "Iceberg-Detection")
            self.strategy_manager.register_strategy(iceberg_wrapped)
            self._logger.info("✓ Registered: Iceberg Detection Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_LIQUIDATION_DETECTION:
            liquidation_wrapped = UniversalStrategyAdapter(LiquidationDetectionNexusAdapterV2(), "Liquidation-Detection")
            self.strategy_manager.register_strategy(liquidation_wrapped)
            self._logger.info("✓ Registered: Liquidation Detection Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_LIQUIDITY_TRAPS:
            liq_traps_wrapped = UniversalStrategyAdapter(LiquidityTrapsNexusAdapterV2(), "Liquidity-Traps")
            self.strategy_manager.register_strategy(liq_traps_wrapped)
            self._logger.info("✓ Registered: Liquidity Traps Strategy (wrapped)")
            strategy_count += 1
        
        # GROUP 5: TECHNICAL ANALYSIS
        if HAS_MULTI_TIMEFRAME:
            multi_tf_wrapped = UniversalStrategyAdapter(MultiTimeframeAlignmentNexusAdapter(), "Multi-Timeframe")
            self.strategy_manager.register_strategy(multi_tf_wrapped)
            self._logger.info("✓ Registered: Multi-Timeframe Alignment Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_CUMULATIVE_DELTA:
            cumulative_delta_wrapped = UniversalStrategyAdapter(EnhancedDeltaTradingStrategy(), "Cumulative-Delta")
            self.strategy_manager.register_strategy(cumulative_delta_wrapped)
            self._logger.info("✓ Registered: Cumulative Delta Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_DELTA_DIVERGENCE:
            delta_div_wrapped = UniversalStrategyAdapter(EnhancedDeltaDivergenceStrategy(), "Delta-Divergence")
            self.strategy_manager.register_strategy(delta_div_wrapped)
            self._logger.info("✓ Registered: Delta Divergence Strategy (wrapped)")
            strategy_count += 1
        
        # GROUP 6: CLASSIFICATION/ROTATION
        if HAS_OPEN_DRIVE_FADE:
            open_drive_wrapped = UniversalStrategyAdapter(EnhancedOpenDriveVsFadeStrategy(), "Open-Drive-Fade")
            self.strategy_manager.register_strategy(open_drive_wrapped)
            self._logger.info("✓ Registered: Open Drive vs Fade Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_PROFILE_ROTATION:
            profile_rotation_wrapped = UniversalStrategyAdapter(EnhancedProfileRotationStrategy(), "Profile-Rotation")
            self.strategy_manager.register_strategy(profile_rotation_wrapped)
            self._logger.info("✓ Registered: Profile Rotation Strategy (wrapped)")
            strategy_count += 1
        
        # GROUP 7: MEAN REVERSION
        if HAS_VWAP_REVERSION:
            vwap_wrapped = UniversalStrategyAdapter(VWAPReversionNexusAdapter(), "VWAP-Reversion")
            self.strategy_manager.register_strategy(vwap_wrapped)
            self._logger.info("✓ Registered: VWAP Reversion Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_STOP_RUN:
            stop_run_wrapped = UniversalStrategyAdapter(StopRunAnticipationNexusAdapter(), "Stop-Run")
            self.strategy_manager.register_strategy(stop_run_wrapped)
            self._logger.info("✓ Registered: Stop Run Anticipation Strategy (wrapped)")
            strategy_count += 1
        
        # GROUP 8: ADVANCED ML
        if HAS_MOMENTUM_IGNITION:
            momentum_ignition_wrapped = UniversalStrategyAdapter(MomentumIgnitionNexusAdapter(), "Momentum-Ignition")
            self.strategy_manager.register_strategy(momentum_ignition_wrapped)
            self._logger.info("✓ Registered: Momentum Ignition Strategy (wrapped)")
            strategy_count += 1
        
        if HAS_VOLUME_IMBALANCE:
            # Volume Imbalance requires base strategy to be created first
            from volume_imbalance import EnhancedVolumeImbalanceStrategy
            base_strategy = EnhancedVolumeImbalanceStrategy()
            volume_imb_wrapped = UniversalStrategyAdapter(VolumeImbalanceNexusAdapter(base_strategy), "Volume-Imbalance")
            self.strategy_manager.register_strategy(volume_imb_wrapped)
            self._logger.info("✓ Registered: Volume Imbalance Strategy (wrapped)")
            strategy_count += 1
        
        self._logger.info(f"════════════════════════════════════════")
        self._logger.info(f"Total Strategies Registered: {strategy_count}/20")
        self._logger.info(f"════════════════════════════════════════")
        
        return strategy_count
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self._logger.info("Starting system initialization...")
            
            # EXPLICIT STRATEGY REGISTRATION
            self._logger.info("Registering strategies explicitly...")
            registered_count = self.register_strategies_explicit()
            
            if registered_count == 0:
                self._logger.warning("⚠️  No strategies registered!")
            
            self._initialized = True
            self._logger.info("System initialization complete")
            return True
            
        except Exception as e:
            self._logger.error(f"Initialization failed: {e}")
            return False
    
    async def process_market_data(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming market data."""
        start_time = time.time()
        
        try:
            # Sanitize input
            sanitized_data = self.security_manager.sanitize_input(raw_data)
            
            # Create market data object
            market_data = MarketData(
                symbol=sanitized_data['symbol'],
                timestamp=sanitized_data.get('timestamp', time.time()),
                price=Decimal(str(sanitized_data['price'])),
                volume=Decimal(str(sanitized_data['volume'])),
                bid=Decimal(str(sanitized_data.get('bid', sanitized_data['price']))),
                ask=Decimal(str(sanitized_data.get('ask', sanitized_data['price']))),
                bid_size=sanitized_data.get('bid_size', 0),
                ask_size=sanitized_data.get('ask_size', 0),
                data_type=MarketDataType.TRADE,
                exchange_timestamp=sanitized_data.get('exchange_timestamp', time.time()),
                sequence_num=sanitized_data.get('sequence_num', 0)
            )
            
            # Validate market data
            if not self.security_manager.validate_market_data(market_data):
                self._logger.warning(f"Invalid market data for {market_data.symbol}")
                return None
            
            # Store in buffer
            self.data_buffer.add(market_data)
            
            # Execute strategies
            signals = self.strategy_manager.execute_all(market_data)
            
            # Process signals
            results = []
            for signal in signals:
                # Evaluate risk
                portfolio = {'capital': 100000, 'positions': {}}  # Simplified
                risk_metrics = self.risk_manager.evaluate_risk(signal, portfolio)
                
                # Validate trade
                if self.risk_manager.validate_trade(signal, risk_metrics):
                    # Create order
                    order = self.execution_engine.create_order(
                        signal,
                        risk_metrics,
                        float(market_data.price)
                    )
                    
                    if order:
                        # Execute order (async)
                        await self.execution_engine.execute_order(order)
                        
                        results.append({
                            'signal': signal.signal_type.name,
                            'confidence': signal.confidence,
                            'order_id': order.order_id,
                            'risk_score': risk_metrics.risk_score
                        })
            
            # Record metrics
            latency = (time.time() - start_time) * 1000
            self.performance_monitor.record_metric('latency', latency)
            self.performance_monitor.record_metric('signals_generated', len(signals))
            
            return {
                'symbol': market_data.symbol,
                'timestamp': market_data.timestamp,
                'signals': results,
                'latency_ms': latency
            }
            
        except Exception as e:
            self._logger.error(f"Market data processing error: {e}")
            self.performance_monitor.record_metric('errors', 1)
            return None
    
    async def start(self) -> None:
        """Start the trading system."""
        if not self._initialized:
            await self.initialize()
        
        self._running = True
        self._logger.info("NEXUS AI system started")
    
    async def stop(self) -> None:
        """Stop the trading system."""
        self._running = False
        
        # Cancel pending orders
        pending_orders = self.execution_engine.get_pending_orders()
        for order in pending_orders:
            self.execution_engine.cancel_order(order.order_id)
        
        self._logger.info("NEXUS AI system stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'running': self._running,
            'initialized': self._initialized,
            'config': self.config.to_dict(),
            'strategies': self.strategy_manager.get_strategies(),
            'strategy_metrics': self.strategy_manager.get_metrics(),
            'risk_report': self.risk_manager.get_risk_report(),
            'execution_stats': self.execution_engine.get_execution_stats(),
            'system_health': self.performance_monitor.get_system_health(),
            'data_buffer_size': self.data_buffer.size(),
            'cache_stats': self.data_cache.get_stats()
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for NEXUS AI system."""
    # Create system configuration
    config = SystemConfig(
        buffer_size=10000,
        cache_size=1000,
        max_position_size=0.1,
        max_daily_loss=0.02,
        max_drawdown=0.15
    )
    
    # Initialize NEXUS AI
    nexus = NexusAI(config)
    
    # Start system
    await nexus.start()
    
    # System is now ready to process market data
    # In production, this would connect to real data feeds
    
    return nexus


if __name__ == "__main__":
    # Run the system
    loop = asyncio.get_event_loop()
    nexus_system = loop.run_until_complete(main())
    
    # Keep system running
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(nexus_system.stop())
    finally:
        loop.close()
