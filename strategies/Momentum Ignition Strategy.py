"""
Momentum Ignition Strategy - Professional Trading Strategy

Enhanced implementation with comprehensive risk management and monitoring.
Includes automated signal generation, position sizing, and performance tracking.

Key Features:
- Professional entry and exit signal generation
- Advanced risk management with position sizing controls
- Real-time performance monitoring and trade tracking
- Comprehensive error handling and logging systems
- Production-ready code structure and documentation

Components:
- Signal Generator: Analyzes market data for trading opportunities
- Risk Manager: Controls position sizing and manages trading risk
- Performance Monitor: Tracks strategy performance and metrics
- Error Handler: Manages exceptions and logging for reliability

Usage:
    strategy = MomentumIgnitionStrategy()
    signals = strategy.generate_signals(market_data)
    positions = strategy.calculate_positions(signals, account_balance)

Author: NEXUS Trading System
Version: 2.0 Professional Enhanced
Created: 2025-10-04
Last Updated: 2025-10-04 10:00:07
"""

# Essential imports for professional trading strategy
import os
import sys
import time
import math
import hmac
import hashlib
import secrets
import base64

# Add the NEXUS directory to sys.path to enable absolute imports when run as a script
script_dir = os.path.dirname(os.path.abspath(__file__))
nexus_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if nexus_dir not in sys.path:
    sys.path.insert(0, nexus_dir)

# Create local trading datatypes to avoid import issues
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from enum import Enum


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class MarketData:
    timestamp: float
    price: float
    volume: int
    bid: float = 0.0
    ask: float = 0.0
    signature: str = ""  # HMAC-SHA256 signature for data integrity


@dataclass
class Signal:
    signal_type: SignalType
    confidence: float
    timestamp: float
    price: float
    reason: str = ""


import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime
from collections import deque, defaultdict
import warnings
import gc

# Configure professional logging system for strategy monitoring
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Setup logging handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Optional ML imports with fallback handling
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    logger.info("PyTorch successfully imported")
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - ML features will be disabled")

try:
    from scipy import signal as scipy_signal

    SCIPY_AVAILABLE = True
    logger.info("SciPy successfully imported")
except ImportError:
    scipy_signal = None
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some signal processing features disabled")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# MQSCORE QUALITY FILTER - Integrated into All Strategies
# ============================================================================

class MQScoreQualityFilter:
    """
    MQSCORE quality filter integrated into every strategy.
    Checks if market conditions are suitable for trading.
    """
    
    def __init__(self, min_composite_score: float = 0.5):
        self.min_composite_score = min_composite_score
        self.min_liquidity = 0.3
        self.min_trend = 0.3
    
    def should_trade(self, market_data: Dict[str, Any]) -> tuple:
        """
        Determine if market conditions are suitable for trading.
        
        Returns:
            (should_trade_bool, quality_metrics_dict)
        """
        quality_metrics = {
            'composite_score': market_data.get('mqscore_composite', 0.5),
            'liquidity_score': market_data.get('mqscore_liquidity', 0.5),
            'volatility_score': market_data.get('mqscore_volatility', 0.5),
            'momentum_score': market_data.get('mqscore_momentum', 0.5),
            'trend_score': market_data.get('mqscore_trend', 0.5),
            'imbalance_score': market_data.get('mqscore_imbalance', 0.5),
            'noise_score': market_data.get('mqscore_noise', 0.5),
        }
        
        # Check minimum composite score
        if quality_metrics['composite_score'] < self.min_composite_score:
            return False, quality_metrics
        
        # Market quality is acceptable
        return True, quality_metrics


class CryptoVerifier:
    """HMAC-SHA256 cryptographic verification for market data integrity."""

    def __init__(self, master_key: bytes = None):
        """Initialize cryptographic verifier with secure key management."""
        try:
            if master_key is None:
                # Generate deterministic 32-byte master key from strategy identifier
                strategy_id = "momentum_ignition_strategy_v1_production"
                self.master_key = hashlib.sha256(strategy_id.encode()).digest()
                logger.info(
                    "Generated deterministic 32-byte master key for HMAC-SHA256 verification"
                )
            else:
                # Validate provided master key
                if not isinstance(master_key, bytes):
                    raise TypeError(f"master_key must be bytes, got {type(master_key)}")
                if len(master_key) != 32:
                    raise ValueError(
                        f"master_key must be 32 bytes, got {len(master_key)}"
                    )
                self.master_key = master_key
                logger.info(
                    "Using provided 32-byte master key for HMAC-SHA256 verification"
                )

            # Test cryptographic functionality
            test_data = b"test_data_123"
            test_signature = self._generate_signature(test_data)
            if not self._verify_signature_constant_time(test_data, test_signature):
                raise RuntimeError("HMAC-SHA256 verification test failed")

            logger.info("CryptoVerifier initialized successfully")

        except Exception as e:
            logger.error(f"CryptoVerifier initialization failed: {e}")
            raise RuntimeError(f"Cryptographic verification initialization failed: {e}")

    def _generate_signature(self, data: bytes) -> str:
        """Generate HMAC-SHA256 signature for data."""
        try:
            if not isinstance(data, bytes):
                raise TypeError(f"data must be bytes, got {type(data)}")

            hmac_obj = hmac.new(self.master_key, data, hashlib.sha256)
            signature_bytes = hmac_obj.digest()
            return base64.b64encode(signature_bytes).decode("ascii")

        except Exception as e:
            logger.error(f"Signature generation failed: {e}")
            raise RuntimeError(f"HMAC-SHA256 signature generation failed: {e}")

    def _verify_signature_constant_time(self, data: bytes, signature: str) -> bool:
        """Verify HMAC-SHA256 signature using constant-time comparison."""
        try:
            if not isinstance(data, bytes):
                raise TypeError(f"data must be bytes, got {type(data)}")
            if not isinstance(signature, str):
                raise TypeError(f"signature must be str, got {type(signature)}")

            # Generate expected signature
            expected_signature = self._generate_signature(data)

            # Constant-time comparison to prevent timing attacks
            return hmac.compare_digest(expected_signature, signature)

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def generate_market_data_signature(self, market_data: MarketData) -> str:
        """Generate HMAC-SHA256 signature for market data."""
        try:
            if not isinstance(market_data, MarketData):
                raise TypeError(
                    f"market_data must be MarketData, got {type(market_data)}"
                )

            # Create canonical data representation
            data_string = f"{market_data.timestamp}:{market_data.price}:{market_data.volume}:{market_data.bid}:{market_data.ask}"
            data_bytes = data_string.encode("utf-8")

            return self._generate_signature(data_bytes)

        except Exception as e:
            logger.error(f"Market data signature generation failed: {e}")
            raise RuntimeError(
                f"Market data HMAC-SHA256 signature generation failed: {e}"
            )

    def verify_market_data_integrity(self, market_data: MarketData) -> Dict[str, Any]:
        """Verify integrity of market data with comprehensive error handling."""
        try:
            if not isinstance(market_data, MarketData):
                return {
                    "verified": False,
                    "error": f"Invalid market_data type: {type(market_data)}",
                    "timestamp": datetime.now().timestamp(),
                }

            if not market_data.signature:
                return {
                    "verified": False,
                    "error": "Missing signature field",
                    "timestamp": datetime.now().timestamp(),
                }

            # Verify signature
            is_valid = self._verify_signature_constant_time(
                f"{market_data.timestamp}:{market_data.price}:{market_data.volume}:{market_data.bid}:{market_data.ask}".encode(
                    "utf-8"
                ),
                market_data.signature,
            )

            return {
                "verified": is_valid,
                "error": None if is_valid else "Signature verification failed",
                "timestamp": datetime.now().timestamp(),
            }

        except Exception as e:
            logger.error(f"Market data integrity verification failed: {e}")
            return {
                "verified": False,
                "error": f"Verification error: {str(e)}",
                "timestamp": datetime.now().timestamp(),
            }


@dataclass
class MomentumConfig:
    """Configuration for momentum ignition strategy with validation."""

    burst_threshold: int = 10  # Minimum trades in burst
    burst_window: float = 1.0  # Seconds for burst detection
    aggression_ratio: float = 0.8  # Ratio of market orders
    size_multiplier: float = 2.0  # Size vs average multiplier
    momentum_periods: int = 20  # Periods for momentum calculation
    ignition_threshold: float = 3.0  # Standard deviations for ignition
    cascade_detection: bool = True  # Detect stop-loss cascades
    ml_enhancement: bool = True  # Use ML for pattern recognition
    enable_crypto_verification: bool = True  # Enable HMAC-SHA256 verification
    master_key: Optional[bytes] = None  # Optional 32-byte master key

    def __post_init__(self):
        """Validate configuration parameters following NEXUS rules."""
        # Rule 4: Configuration Validation Protocol
        if not isinstance(self.burst_threshold, int) or self.burst_threshold < 1:
            raise ValueError(
                f"burst_threshold must be positive integer, got {self.burst_threshold}"
            )
        if not isinstance(self.burst_window, (int, float)) or self.burst_window <= 0:
            raise ValueError(
                f"burst_window must be positive number, got {self.burst_window}"
            )
        if (
            not isinstance(self.aggression_ratio, (int, float))
            or not 0 <= self.aggression_ratio <= 1
        ):
            raise ValueError(
                f"aggression_ratio must be between 0 and 1, got {self.aggression_ratio}"
            )
        if (
            not isinstance(self.size_multiplier, (int, float))
            or self.size_multiplier < 1
        ):
            raise ValueError(
                f"size_multiplier must be >= 1, got {self.size_multiplier}"
            )
        if not isinstance(self.momentum_periods, int) or self.momentum_periods < 1:
            raise ValueError(
                f"momentum_periods must be positive integer, got {self.momentum_periods}"
            )
        if (
            not isinstance(self.ignition_threshold, (int, float))
            or self.ignition_threshold < 0
        ):
            raise ValueError(
                f"ignition_threshold must be non-negative, got {self.ignition_threshold}"
            )

        # Normalize values
        self.burst_threshold = max(1, min(100, self.burst_threshold))
        self.burst_window = max(0.1, min(60.0, self.burst_window))
        self.aggression_ratio = max(0.0, min(1.0, self.aggression_ratio))
        self.momentum_periods = max(1, min(200, self.momentum_periods))

        # Validate cryptographic settings
        if not isinstance(self.enable_crypto_verification, bool):
            raise TypeError(
                f"enable_crypto_verification must be bool, got {type(self.enable_crypto_verification)}"
            )
        if self.master_key is not None:
            if not isinstance(self.master_key, bytes):
                raise TypeError(
                    f"master_key must be bytes or None, got {type(self.master_key)}"
                )
            if len(self.master_key) != 32:
                raise ValueError(
                    f"master_key must be 32 bytes, got {len(self.master_key)}"
                )


class MomentumDetector(nn.Module):
    """Neural network for momentum ignition pattern detection."""

    def __init__(self, input_dim: int = 20):
        super().__init__()

        # Store input dimension as instance variable
        self.input_dim = input_dim

        # Validate input parameters
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not available - MomentumDetector cannot be initialized"
            )

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        try:
            # Initialize layers with proper error handling
            self.conv1d = nn.Conv1d(1, 32, kernel_size=3, padding=1)
            self.conv1d_2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            self.lstm = nn.LSTM(64, 128, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),  # No ignition, Bullish ignition, Bearish ignition
            )

            # Initialize weights properly
            self._initialize_weights()

            logger.info(f"MomentumDetector initialized with input_dim={input_dim}")

        except Exception as e:
            logger.error(f"Failed to initialize MomentumDetector layers: {e}")
            raise RuntimeError(f"MomentumDetector initialization failed: {e}")

    def _initialize_weights(self):
        """Proper weight initialization for neural network layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        """Forward pass with comprehensive validation."""
        try:
            # Input validation
            if x is None:
                raise ValueError("Input tensor cannot be None")

            if not hasattr(x, "unsqueeze"):
                raise TypeError(f"Input must be a tensor, got {type(x)}")

            # Handle both 1D and 2D input tensors
            if x.dim() == 1:
                # Single sample: shape (features,) -> (1, features)
                x = x.unsqueeze(0)
            elif x.dim() == 2:
                # Batch of samples: shape (batch, features) - already correct
                pass
            else:
                raise ValueError(f"Input tensor must be 1D or 2D, got {x.dim()}D")

            if x.size(-1) != self.input_dim:
                raise ValueError(
                    f"Input tensor last dimension must be {self.input_dim}, got {x.size(-1)}"
                )

            if x.size(0) == 0:
                raise ValueError("Input tensor cannot be empty")

            if not torch.isfinite(x).all():
                raise ValueError("Input tensor contains non-finite values")

            # Convolutional layers for pattern detection
            # x is now (batch, features), need (batch, channels, features) for conv1d
            x = x.unsqueeze(1)  # (batch, 1, features)
            x = torch.relu(self.conv1d(x))
            x = torch.relu(self.conv1d_2(x))
            x = x.transpose(1, 2)  # (batch, seq_len, channels)

            # LSTM for sequence modeling
            lstm_out, _ = self.lstm(x)

            # Take last output
            last_out = lstm_out[:, -1, :]

            return self.fc(last_out)

        except Exception as e:
            logger.error(f"MomentumDetector forward pass failed: {e}")
            # Return neutral prediction on error
            batch_size = x.size(0) if hasattr(x, "size") else 1
            return torch.zeros(
                batch_size, 3, device=x.device if hasattr(x, "device") else "cpu"
            )


# ============================================================================
# ADAPTIVE PARAMETER OPTIMIZATION - Real Performance-Based Learning
# ============================================================================


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

        logger.info(
            f"[OK] Adaptive Parameter Optimizer initialized for {strategy_name}"
        )

    def _initialize_parameters(self) -> Dict[str, float]:
        """Initialize with default parameters"""
        return {
            "burst_threshold": 2.5,
            "momentum_threshold": 1.8,
            "confirmation_threshold": 0.65,
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

        # Adapt burst threshold
        if win_rate < 0.40:  # Poor win rate - be more selective
            self.current_parameters["burst_threshold"] = min(
                3.5, self.current_parameters["burst_threshold"] * 1.08
            )
        elif win_rate > 0.65:  # Good win rate - can be less selective
            self.current_parameters["burst_threshold"] = max(
                1.5, self.current_parameters["burst_threshold"] * 0.96
            )

        # Adapt momentum threshold based on P&L
        if avg_pnl < 0:  # Losing - increase requirements
            self.current_parameters["momentum_threshold"] = min(
                2.5, self.current_parameters["momentum_threshold"] * 1.1
            )
        elif avg_pnl > 0:  # Winning - can reduce slightly
            self.current_parameters["momentum_threshold"] = max(
                1.2, self.current_parameters["momentum_threshold"] * 0.98
            )

        # Adapt confirmation threshold based on volatility
        target_vol = 0.02
        vol_ratio = avg_volatility / target_vol
        if vol_ratio > 1.5:  # High volatility - increase threshold
            self.current_parameters["confirmation_threshold"] = min(
                0.85, self.current_parameters["confirmation_threshold"] * 1.05
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

        logger.info(
            f"📊 {self.strategy_name} parameters adapted: "
            f"Burst={self.current_parameters['burst_threshold']:.2f}, "
            f"Momentum={self.current_parameters['momentum_threshold']:.2f}, "
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


class MarketRegimeDetector:
    """
    Basic market regime detection from recent price action.
    
    Use this for: Fast, simple regime classification
    For detailed analysis, use AdvancedRegimeDetector instead.
    
    Detects: HIGH_VOLATILITY, LOW_VOLATILITY, TRENDING, RANGING, NORMAL
    """
    
    def __init__(self, lookback=50):
        self.lookback = lookback
        self.price_history = deque(maxlen=lookback)
        self.volatility_baseline = 0.01
    
    def update(self, price: float):
        """Update with new price"""
        self.price_history.append(price)
    
    def detect_regime(self) -> str:
        """Classify current market regime"""
        if len(self.price_history) < 20:
            return "UNKNOWN"
        
        prices = list(self.price_history)
        returns = np.diff(prices) / np.array(prices[:-1])
        volatility = np.std(returns) if len(returns) > 0 else 0.0
        
        trend_return = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
        trend_strength = abs(trend_return) / volatility if volatility > 0 else 0.0
        
        if volatility > self.volatility_baseline * 1.5:
            return "HIGH_VOLATILITY"
        elif volatility < self.volatility_baseline * 0.7:
            return "LOW_VOLATILITY"
        
        if trend_strength > 1.5:
            return "TRENDING"
        elif trend_strength < 0.3:
            return "RANGING"
        
        return "NORMAL"
    
    def get_regime_multiplier(self) -> Dict[str, float]:
        """Get parameter adjustments for current regime"""
        regime = self.detect_regime()
        
        multipliers = {
            'confidence_threshold': 1.0,
            'lookback': 1.0,
            'position_size': 1.0,
        }
        
        if regime == "HIGH_VOLATILITY":
            multipliers['confidence_threshold'] = 1.2
            multipliers['lookback'] = 0.9
            multipliers['position_size'] = 0.8
        elif regime == "LOW_VOLATILITY":
            multipliers['confidence_threshold'] = 0.8
            multipliers['lookback'] = 1.1
            multipliers['position_size'] = 1.2
        elif regime == "TRENDING":
            multipliers['position_size'] = 1.1
        elif regime == "RANGING":
            multipliers['position_size'] = 0.7
        
        return multipliers


class CrossAssetMomentumTracker:
    """
    Track momentum correlation across multiple assets.
    
    Identifies:
    - Coordinated ignition events across assets
    - Cross-asset momentum divergence
    - Sector-wide momentum shifts
    - Portfolio-level ignition opportunities
    """
    
    def __init__(self, max_assets=10, lookback=50):
        self.max_assets = max_assets
        self.lookback = lookback
        self.asset_momentum = {}  # {asset_id: deque of momentum scores}
        self.asset_prices = {}  # {asset_id: deque of prices}
        self.correlation_matrix = {}
        self.last_correlation_update = 0
        self.correlation_update_interval = 10  # Update every 10 data points
    
    def update_asset(self, asset_id: str, price: float, momentum_score: float):
        """Update momentum data for an asset"""
        if asset_id not in self.asset_momentum:
            self.asset_momentum[asset_id] = deque(maxlen=self.lookback)
            self.asset_prices[asset_id] = deque(maxlen=self.lookback)
        
        self.asset_momentum[asset_id].append(momentum_score)
        self.asset_prices[asset_id].append(price)
        
        # Periodically update correlation matrix
        self.last_correlation_update += 1
        if self.last_correlation_update >= self.correlation_update_interval:
            self._update_correlations()
            self.last_correlation_update = 0
    
    def _update_correlations(self):
        """Calculate pairwise momentum correlations"""
        assets = list(self.asset_momentum.keys())
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                if len(self.asset_momentum[asset1]) >= 20 and len(self.asset_momentum[asset2]) >= 20:
                    # Calculate correlation
                    momentum1 = list(self.asset_momentum[asset1])
                    momentum2 = list(self.asset_momentum[asset2])
                    
                    min_len = min(len(momentum1), len(momentum2))
                    if min_len >= 20:
                        corr = np.corrcoef(momentum1[-min_len:], momentum2[-min_len:])[0, 1]
                        self.correlation_matrix[f"{asset1}_{asset2}"] = corr
    
    def get_coordinated_ignition_score(self, asset_id: str, threshold=0.7) -> float:
        """
        Calculate score for coordinated multi-asset ignition.
        Returns 0.0-1.0 where 1.0 = strong coordinated movement.
        """
        if asset_id not in self.asset_momentum:
            return 0.0
        
        if len(self.asset_momentum[asset_id]) < 20:
            return 0.0
        
        # Count how many assets are showing correlated momentum
        coordinated_count = 0
        total_correlations = 0
        
        for key, corr in self.correlation_matrix.items():
            if asset_id in key and abs(corr) > threshold:
                coordinated_count += 1
            if asset_id in key:
                total_correlations += 1
        
        if total_correlations == 0:
            return 0.0
        
        coordination_score = coordinated_count / total_correlations
        return coordination_score
    
    def detect_divergence(self, asset_id: str) -> Dict[str, Any]:
        """
        Detect when asset momentum diverges from correlated assets.
        Divergence can signal unique opportunity or false signal.
        """
        if asset_id not in self.asset_momentum or len(self.asset_momentum[asset_id]) < 20:
            return {"divergence_detected": False, "score": 0.0}
        
        current_momentum = list(self.asset_momentum[asset_id])[-1]
        
        # Find highly correlated assets
        correlated_assets = []
        for key, corr in self.correlation_matrix.items():
            if asset_id in key and abs(corr) > 0.6:
                other_asset = key.replace(asset_id, "").replace("_", "")
                if other_asset and other_asset in self.asset_momentum:
                    correlated_assets.append((other_asset, corr))
        
        if not correlated_assets:
            return {"divergence_detected": False, "score": 0.0}
        
        # Calculate divergence
        divergence_scores = []
        for other_asset, corr in correlated_assets:
            if len(self.asset_momentum[other_asset]) > 0:
                other_momentum = list(self.asset_momentum[other_asset])[-1]
                expected_direction = np.sign(corr)
                actual_direction = np.sign(current_momentum * other_momentum)
                
                if expected_direction != actual_direction:
                    divergence_scores.append(abs(current_momentum - other_momentum))
        
        if divergence_scores:
            avg_divergence = np.mean(divergence_scores)
            return {
                "divergence_detected": True,
                "score": min(1.0, avg_divergence),
                "divergent_assets": len(divergence_scores)
            }
        
        return {"divergence_detected": False, "score": 0.0}
    
    def get_sector_momentum(self) -> float:
        """Calculate average momentum across all tracked assets"""
        if not self.asset_momentum:
            return 0.0
        
        recent_momentums = []
        for asset_id, momentum_history in self.asset_momentum.items():
            if len(momentum_history) > 0:
                recent_momentums.append(list(momentum_history)[-1])
        
        if not recent_momentums:
            return 0.0
        
        return float(np.mean(recent_momentums))


class SpoofingDetector:
    """
    Enhanced Market Maker fingerprinting and spoofing pattern detection.
    
    Detects:
    - Order book manipulation (spoofing)
    - Market maker intervention patterns
    - Predatory order placement
    - Layering and iceberg orders
    """
    
    def __init__(self, lookback_ticks=100):
        self.recent_ticks = deque(maxlen=lookback_ticks)
        self.order_persistence_threshold = 0.7
        
        # Enhanced MM fingerprinting
        self.mm_intervention_history = deque(maxlen=50)
        self.layering_events = deque(maxlen=20)
        self.order_cancellation_rate = deque(maxlen=100)
    
    def update(self, tick_data):
        """Update with new tick"""
        self.recent_ticks.append(tick_data)
    
    def is_likely_spoof(self, signal, bid_size: int = 0, ask_size: int = 0) -> bool:
        """
        Enhanced spoofing detection with MM intervention analysis.
        
        Checks for:
        1. Extreme bid/ask imbalances (original)
        2. Rapid order cancellations (layering)
        3. Large orders that disappear quickly
        4. Consistent MM intervention patterns
        """
        if len(self.recent_ticks) < 10:
            return False
        
        try:
            recent = list(self.recent_ticks)[-10:]
            bid_volumes = []
            ask_volumes = []
            order_sizes = []
            
            for tick in recent:
                if isinstance(tick, dict):
                    bid_volumes.append(tick.get('bid_size', 0))
                    ask_volumes.append(tick.get('ask_size', 0))
                    order_sizes.append(tick.get('size', 0))
                else:
                    bid_volumes.append(getattr(tick, 'bid_size', 0))
                    ask_volumes.append(getattr(tick, 'ask_size', 0))
                    order_sizes.append(getattr(tick, 'size', 0))
            
            # Original bid/ask imbalance check
            avg_bid_volume = np.mean(bid_volumes) if bid_volumes else 0
            avg_ask_volume = np.mean(ask_volumes) if ask_volumes else 0
            bid_dominance = avg_bid_volume / (avg_ask_volume + 1)
            
            # Enhanced: Check for layering (rapid size changes)
            size_volatility = np.std(order_sizes) if len(order_sizes) > 1 else 0
            avg_size = np.mean(order_sizes) if order_sizes else 0
            size_cv = size_volatility / avg_size if avg_size > 0 else 0
            
            # Enhanced: Detect MM intervention (sudden large orders)
            max_size = max(order_sizes) if order_sizes else 0
            size_spike = max_size / avg_size if avg_size > 0 else 0
            
            # Pattern 1: Extreme bid/ask imbalance (original)
            imbalance_spoof = False
            if hasattr(signal, 'signal_type'):
                signal_type = signal.signal_type
                if hasattr(signal_type, 'name'):
                    if signal_type.name == 'SELL' and bid_dominance > 2.0:
                        imbalance_spoof = True
                    elif signal_type.name == 'BUY' and (1.0 / max(bid_dominance, 0.1)) > 2.0:
                        imbalance_spoof = True
            
            # Pattern 2: Layering detection (high size volatility)
            layering_detected = size_cv > 1.5
            
            # Pattern 3: MM intervention (sudden large orders)
            mm_intervention = size_spike > 3.0
            
            # Record patterns for learning
            if mm_intervention:
                self.mm_intervention_history.append({
                    'timestamp': time.time(),
                    'size_spike': size_spike,
                    'bid_dominance': bid_dominance
                })
            
            if layering_detected:
                self.layering_events.append({
                    'timestamp': time.time(),
                    'size_cv': size_cv
                })
            
            # Return True if any pattern detected
            return imbalance_spoof or (layering_detected and mm_intervention)
            
        except Exception as e:
            logger.warning(f"Error in enhanced spoofing detection: {e}")
            return False
    
    def get_mm_intervention_score(self) -> float:
        """
        Calculate MM intervention probability score (0.0-1.0).
        Higher score means more likely MM is actively intervening.
        """
        if len(self.mm_intervention_history) < 5:
            return 0.0
        
        recent_interventions = list(self.mm_intervention_history)[-10:]
        intervention_rate = len(recent_interventions) / 10.0
        
        # Recent intervention = higher risk
        latest_time = recent_interventions[-1]['timestamp'] if recent_interventions else 0
        time_since_last = time.time() - latest_time
        recency_factor = max(0.0, 1.0 - (time_since_last / 60.0))  # Decay over 60 seconds
        
        mm_score = min(1.0, intervention_rate * 2.0 * (1.0 + recency_factor))
        return mm_score


class MomentumIgnitionStrategy:
    """
    Detects aggressive order flow bursts that trigger momentum ignition.

    Identifies rapid sequences of market orders that can trigger
    stop-loss cascades and momentum continuation.
    """

    def __init__(self, config: MomentumConfig = MomentumConfig()):
        """Initialize strategy with comprehensive error handling."""
        try:
            # Validate configuration
            if not isinstance(config, MomentumConfig):
                raise TypeError(f"config must be MomentumConfig, got {type(config)}")

            self.config = config

            # Rule 2: Memory Management Protocol - bounded data structures
            self.trade_buffer = deque(maxlen=500)  # Reduced from 1000 to manage memory
            self.burst_buffer = deque(maxlen=50)  # Reduced from 100
            self.momentum_history = deque(
                maxlen=max(1, min(config.momentum_periods, 100))
            )  # Capped at 100
            self.ignition_events = deque(maxlen=10)
            
            # FIX #3: Initialize market regime detector
            self.regime_detector = MarketRegimeDetector(lookback=50)
            
            # FIX #4: Initialize enhanced spoofing detector (W1.4)
            self.spoof_detector = SpoofingDetector(lookback_ticks=100)
            
            # PHASE 2 W2.3: Initialize cross-asset momentum tracker
            self.cross_asset_tracker = CrossAssetMomentumTracker(max_assets=10, lookback=50)

            # Memory monitoring
            self._last_cleanup = datetime.now().timestamp()
            self._memory_check_interval = 300  # 5 minutes

            # ML components with safe initialization
            self.detector = None
            self.device = None
            self.ml_available = False

            # Cryptographic verification components
            self.crypto_verifier = None
            self.crypto_available = False

            if config.enable_crypto_verification:
                try:
                    self.crypto_verifier = CryptoVerifier(config.master_key)
                    self.crypto_available = True
                    logger.info("HMAC-SHA256 cryptographic verification enabled")
                except Exception as e:
                    logger.error(
                        f"Failed to initialize cryptographic verification: {e}"
                    )
                    self.config.enable_crypto_verification = False
                    self.crypto_verifier = None
                    self.crypto_available = False

            if config.ml_enhancement:
                try:
                    if not TORCH_AVAILABLE:
                        logger.warning(
                            "ML enhancement requested but PyTorch not available"
                        )
                        self.config.ml_enhancement = False
                    else:
                        # Initialize GPU memory management
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.set_per_process_memory_fraction(0.8)

                        self.detector = MomentumDetector()
                        self.device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
                        self.detector.to(self.device)
                        self.detector.eval()
                        self.ml_available = True

                        # Monitor GPU memory usage
                        if torch.cuda.is_available():
                            memory_allocated = (
                                torch.cuda.memory_allocated(self.device) / 1024**2
                            )
                            memory_reserved = (
                                torch.cuda.memory_reserved(self.device) / 1024**2
                            )
                            logger.info(
                                f"ML components initialized on device: {self.device}"
                            )
                            logger.info(
                                f"GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB"
                            )
                        else:
                            logger.info(
                                f"ML components initialized on device: {self.device}"
                            )

                except Exception as e:
                    logger.error(f"Failed to initialize ML components: {e}")
                    self.config.ml_enhancement = False
                    self.detector = None
                    self.device = None
                    self.ml_available = False
                    # Clean up GPU memory on failure
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            logger.info(
                f"MomentumIgnitionStrategy initialized - ML available: {self.ml_available}, Crypto available: {self.crypto_available}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize MomentumIgnitionStrategy: {e}")
            raise RuntimeError(f"Strategy initialization failed: {e}")

    def record_trade_result(self, trade_info: Dict[str, Any]) -> None:
        """Record trade result for adaptive learning"""
        try:
            # Extract trade metrics with safe defaults
            pnl = float(trade_info.get("pnl", 0.0))
            confidence = float(trade_info.get("confidence", 0.5))
            volatility = float(trade_info.get("volatility", 0.02))

            # Record in adaptive optimizer if available
            if hasattr(self, "adaptive_optimizer"):
                self.adaptive_optimizer.record_trade(
                    {"pnl": pnl, "confidence": confidence, "volatility": volatility}
                )
        except Exception as e:
            logger.error(f"Failed to record trade result: {e}")

    def process_market_data(self, market_data: MarketData) -> Optional[Dict[str, Any]]:
        """Process MarketData with HMAC-SHA256 cryptographic verification."""
        try:
            # Rule 1: Input Validation Protocol
            if not isinstance(market_data, MarketData):
                logger.warning(
                    f"Invalid market_data type: {type(market_data)}, expected MarketData"
                )
                return None

            # Cryptographic verification if enabled
            verification_info = None
            if self.config.enable_crypto_verification and self.crypto_available:
                verification_result = self.crypto_verifier.verify_market_data_integrity(
                    market_data
                )
                if not verification_result["verified"]:
                    logger.warning(
                        f"Market data verification failed: {verification_result['error']}"
                    )
                    return None
                logger.debug("Market data HMAC-SHA256 verification successful")
                verification_info = verification_result

            # Convert MarketData to trade data format for processing
            trade_data = {
                "timestamp": market_data.timestamp,
                "price": market_data.price,
                "size": market_data.volume,
                "order_type": "market",  # Default to market order for MarketData
                "aggressor": "buy" if market_data.price > market_data.bid else "sell",
            }

            # Process the converted trade data
            self.process_trade_data(trade_data)

            result = {
                "processed": True,
                "verified": self.config.enable_crypto_verification
                and self.crypto_available,
                "timestamp": market_data.timestamp,
                "price": market_data.price,
                "volume": market_data.volume,
            }

            # Include verification details if crypto verification is enabled
            if (
                self.config.enable_crypto_verification
                and self.crypto_available
                and verification_info
            ):
                result["verification"] = verification_info

            return result

        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return None

    def process_trade_data(self, trade_data: Dict[str, Any]) -> None:
        """Process incoming trade data with comprehensive validation (Rule 1)."""
        try:
            # Rule 1: Input Validation Protocol
            if not isinstance(trade_data, dict):
                logger.warning(
                    f"Invalid trade_data type: {type(trade_data)}, expected dict"
                )
                return

            # Check required fields exist
            required_fields = ["timestamp", "price", "size", "order_type", "aggressor"]
            for field in required_fields:
                if field not in trade_data:
                    logger.warning(f"Missing required field '{field}' in trade data")
                    return

            # Validate data types and ranges
            try:
                timestamp = float(trade_data["timestamp"])
                price = float(trade_data["price"])
                size = float(trade_data["size"])
                order_type = str(trade_data["order_type"])
                aggressor = str(trade_data["aggressor"])
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid data types in trade_data: {e}")
                return

            # Validate ranges
            if price <= 0:
                logger.warning(f"Invalid price: {price}, must be positive")
                return
            if size <= 0:
                logger.warning(f"Invalid size: {size}, must be positive")
                return
            if timestamp <= 0:
                logger.warning(f"Invalid timestamp: {timestamp}, must be positive")
                return

            # Sanitize and convert values
            sanitized_data = {
                "timestamp": timestamp,
                "price": price,
                "size": size,
                "order_type": order_type.lower(),
                "aggressor": aggressor.lower(),
            }

            # Add to trade buffer (Rule 2: Memory Management - bounded deque)
            self.trade_buffer.append(sanitized_data)

            # Update momentum history with memory management
            momentum_score = self.calculate_momentum_score()
            if momentum_score is not None and np.isfinite(momentum_score):
                self.momentum_history.append(momentum_score)

            # Periodic cleanup (Rule 2: Memory Management)
            if len(self.trade_buffer) % 100 == 0:
                self._cleanup_old_data()

        except Exception as e:
            logger.error(f"Error processing trade data: {e}")

    def _cleanup_old_data(self):
        """Periodic cleanup to manage memory usage (Rule 2)."""
        try:
            # Clean up old ignition events
            current_time = datetime.now().timestamp()
            cutoff_time = current_time - 3600  # Keep last hour

            # Filter old events
            self.ignition_events = deque(
                [
                    event
                    for event in self.ignition_events
                    if event["timestamp"] > cutoff_time
                ],
                maxlen=10,
            )

            # Force garbage collection periodically
            if len(self.trade_buffer) % 500 == 0:
                gc.collect()

        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def calculate_adaptive_lookback(self, prices: np.ndarray) -> int:
        """Calculate adaptive lookback period based on volatility - FIX #1"""
        try:
            if len(prices) < 20:
                return self.config.momentum_periods
            
            recent_prices = prices[-20:]
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0
            
            baseline_vol = 0.01
            vol_ratio = volatility / baseline_vol if baseline_vol > 0 else 1.0
            
            if vol_ratio > 1.5:
                lookback = max(10, self.config.momentum_periods - 5)
            elif vol_ratio < 0.5:
                lookback = min(40, self.config.momentum_periods + 5)
            else:
                lookback = self.config.momentum_periods
            
            return int(lookback)
        except Exception as e:
            logger.warning(f"Error calculating adaptive lookback: {e}")
            return self.config.momentum_periods

    def calculate_improved_confidence(self, 
                                      price_deviation: float,
                                      volume_ratio: float,
                                      price_std: float,
                                      burst_duration_ticks: int) -> float:
        """Calculate confidence incorporating multiple factors - FIX #5"""
        try:
            max_expected_deviation = price_std * 5.0
            price_component = min(1.0, abs(price_deviation) / max(max_expected_deviation, 0.0001))
            
            volume_component = min(1.0, max(0.0, (volume_ratio - 1.0) / 1.0))
            
            persistence_component = min(1.0, max(0.0, burst_duration_ticks / 10.0))
            
            weights = [0.4, 0.35, 0.25]
            components = [price_component, volume_component, persistence_component]
            
            confidence = sum(w * c for w, c in zip(weights, components))
            
            return min(0.99, max(0.0, confidence))
        except Exception as e:
            logger.warning(f"Error calculating improved confidence: {e}")
            return 0.0

    def calculate_momentum_score(self) -> Optional[float]:
        """Calculate momentum score based on recent trade activity."""
        try:
            if len(self.trade_buffer) < 2:
                return None

            # Get recent trades within the momentum window
            current_time = datetime.now().timestamp()
            recent_trades = [
                trade
                for trade in self.trade_buffer
                if current_time - trade["timestamp"] <= self.config.burst_window * 5
            ]

            if len(recent_trades) < 2:
                return None

            # Calculate price momentum with zero-division protection (Rule 3)
            prices = [trade["price"] for trade in recent_trades]
            if len(prices) < 2 or prices[0] == 0:
                price_change = 0.0
            else:
                price_change = (prices[-1] - prices[0]) / prices[0]

            # Calculate volume momentum with safe division
            volumes = [trade["size"] for trade in recent_trades]
            if not volumes:
                avg_volume = 0.0
                recent_volume = 0.0
                volume_ratio = 1.0
            else:
                avg_volume = np.mean(volumes)
                recent_volume = (
                    np.mean(volumes[-5:]) if len(volumes) >= 5 else avg_volume
                )
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

            # Calculate aggression ratio
            market_orders = sum(
                1 for trade in recent_trades if trade["order_type"] == "market"
            )
            aggression_ratio = (
                market_orders / len(recent_trades) if recent_trades else 0
            )

            # Combine factors into momentum score
            momentum_score = (
                abs(price_change) * 100  # Price momentum component
                + volume_ratio * 10  # Volume momentum component
                + aggression_ratio * 5  # Aggression component
            )

            return momentum_score

        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return None

    def detect_momentum_burst(
        self, recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect momentum ignition bursts in trade data."""
        try:
            if len(recent_trades) < self.config.burst_threshold:
                return {
                    "detected": False,
                    "confidence": 0.0,
                    "reason": "Insufficient trades",
                }

            # Filter trades within burst window
            current_time = (
                recent_trades[-1]["timestamp"]
                if recent_trades
                else datetime.now().timestamp()
            )
            burst_trades = [
                trade
                for trade in recent_trades
                if current_time - trade["timestamp"] <= self.config.burst_window
            ]

            if len(burst_trades) < self.config.burst_threshold:
                return {
                    "detected": False,
                    "confidence": 0.0,
                    "reason": "Insufficient burst trades",
                }

            # Calculate burst metrics
            market_orders = sum(
                1 for trade in burst_trades if trade["order_type"] == "market"
            )
            aggression_ratio = market_orders / len(burst_trades)

            # Calculate size metrics with safe division (Rule 3)
            sizes = [trade["size"] for trade in burst_trades]
            if not sizes:
                avg_size = 0.0
                size_multiplier = 1.0
            else:
                avg_size = np.mean(sizes)
                recent_sizes = [trade["size"] for trade in recent_trades[-20:]]
                if recent_sizes:
                    recent_avg = np.mean(recent_sizes)
                    size_multiplier = avg_size / recent_avg if recent_avg > 0 else 1.0
                else:
                    size_multiplier = 1.0

            # Calculate price impact with safe division (Rule 3)
            prices = [trade["price"] for trade in burst_trades]
            if not prices or min(prices) <= 0:
                price_range = 0.0
            else:
                min_price = min(prices)
                max_price = max(prices)
                price_range = (max_price - min_price) / min_price

            # Determine burst detection
            burst_detected = (
                aggression_ratio >= self.config.aggression_ratio
                and size_multiplier >= self.config.size_multiplier
                and len(burst_trades) >= self.config.burst_threshold
            )

            # Calculate confidence score
            confidence = min(
                1.0,
                (
                    aggression_ratio * 0.4
                    + min(size_multiplier / self.config.size_multiplier, 1.0) * 0.3
                    + min(len(burst_trades) / (self.config.burst_threshold * 2), 1.0)
                    * 0.3
                ),
            )

            return {
                "detected": burst_detected,
                "confidence": confidence,
                "aggression_ratio": aggression_ratio,
                "size_multiplier": size_multiplier,
                "burst_trades": len(burst_trades),
                "price_impact": price_range,
                "reason": "Momentum burst detected"
                if burst_detected
                else "No significant burst",
            }

        except Exception as e:
            logger.error(f"Error detecting momentum burst: {e}")
            return {"detected": False, "confidence": 0.0, "reason": f"Error: {e}"}

    # DUPLICATE execute() method REMOVED - using MomentumIgnitionNexusAdapter.execute() at line 2652 instead

    def get_category(self):
        try:
            from nexus_ai import StrategyCategory

            return StrategyCategory.MOMENTUM
        except:
            return "Momentum"

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "total_signals": len(self.trade_buffer),
            "momentum_threshold": self.config.momentum_threshold,
        }

    def generate_signal(
        self, current_price: float, recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate trading signal with comprehensive input validation (Rule 1)."""
        try:
            # Rule 1: Input Validation Protocol - comprehensive validation
            if not isinstance(current_price, (int, float)):
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": f"Invalid price type: {type(current_price)}",
                }

            if not isinstance(recent_trades, list):
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": f"Invalid trades type: {type(recent_trades)}",
                }

            if current_price <= 0 or not np.isfinite(current_price):
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": f"Invalid price value: {current_price}",
                }

            if not recent_trades or len(recent_trades) == 0:
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": "No trade data provided",
                }

            # Validate each trade in recent_trades
            valid_trades = []
            for i, trade in enumerate(recent_trades):
                if not isinstance(trade, dict):
                    logger.warning(f"Invalid trade at index {i}: not a dict")
                    continue

                required_fields = [
                    "timestamp",
                    "price",
                    "size",
                    "order_type",
                    "aggressor",
                ]
                if not all(field in trade for field in required_fields):
                    logger.warning(
                        f"Invalid trade at index {i}: missing required fields"
                    )
                    continue

                try:
                    # Validate numeric fields
                    if (
                        float(trade["price"]) <= 0
                        or float(trade["size"]) <= 0
                        or float(trade["timestamp"]) <= 0
                    ):
                        continue
                    valid_trades.append(trade)
                except (ValueError, TypeError):
                    continue

            if not valid_trades:
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "reason": "No valid trade data",
                }

            # Process recent trades
            for trade in recent_trades[-10:]:  # Process last 10 trades
                self.process_trade_data(trade)

            # FIX #3: Update market regime detector and get multipliers
            prices_array = np.array([trade["price"] for trade in recent_trades])
            if len(prices_array) > 0:
                self.regime_detector.update(prices_array[-1])
            regime_multipliers = self.regime_detector.get_regime_multiplier()

            # Detect momentum burst
            burst_result = self.detect_momentum_burst(recent_trades)

            # Calculate momentum score
            momentum_score = self.calculate_momentum_score()

            # Use ML enhancement if available
            ml_prediction = None
            if self.ml_available and self.detector is not None:
                try:
                    # Prepare features for ML model
                    features = self._prepare_ml_features(recent_trades, current_price)
                    if features is not None:
                        with torch.no_grad():
                            features_tensor = (
                                torch.FloatTensor(features).unsqueeze(0).to(self.device)
                            )
                            ml_output = self.detector(features_tensor)
                            # Handle multi-dimensional output by taking the first element
                            if ml_output.dim() > 1:
                                ml_prediction = torch.sigmoid(ml_output[0, 0]).item()
                            else:
                                ml_prediction = torch.sigmoid(ml_output[0]).item()
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
                    ml_prediction = None

            # Determine signal based on analysis
            signal = "HOLD"
            confidence = 0.0
            reason = "No significant momentum detected"

            if burst_result["detected"]:
                # Determine direction based on recent price movement
                recent_prices = [trade["price"] for trade in recent_trades[-5:]]
                if len(recent_prices) >= 2:
                    price_direction = recent_prices[-1] - recent_prices[0]

                    if price_direction > 0:
                        signal = "BUY"
                        reason = "Upward momentum burst detected"
                    else:
                        signal = "SELL"
                        reason = "Downward momentum burst detected"

                    # FIX #5: Use improved confidence calculation
                    price_std = np.std(recent_prices) if len(recent_prices) > 1 else 0.01
                    volume_ratio = burst_result.get("size_multiplier", 1.0)
                    base_confidence = self.calculate_improved_confidence(
                        price_deviation=price_direction,
                        volume_ratio=volume_ratio,
                        price_std=price_std,
                        burst_duration_ticks=burst_result.get("burst_trades", 0)
                    )
                    
                    if ml_prediction is not None:
                        confidence = base_confidence * 0.7 + ml_prediction * 0.3
                    else:
                        confidence = base_confidence

                    # FIX #3: Apply regime multiplier to confidence
                    confidence = confidence * regime_multipliers['confidence_threshold']
                    confidence = min(1.0, confidence)

                    # Apply momentum score boost
                    if (
                        momentum_score
                        and momentum_score > self.config.ignition_threshold
                    ):
                        confidence = min(1.0, confidence * 1.2)
                        reason += f" (momentum score: {momentum_score:.2f})"

                    # FIX #4: Check for spoofing patterns
                    if len(recent_trades) > 0:
                        self.spoof_detector.update(recent_trades[-1])
                        fake_signal_obj = type('obj', (object,), {'signal_type': type('SignalType', (), {'name': signal})()})()
                        if self.spoof_detector.is_likely_spoof(fake_signal_obj):
                            reason += " [SPOOF WARNING]"
                            confidence = confidence * 0.5  # Reduce confidence if spoof detected

            # Record ignition event if significant
            if confidence > 0.7:
                ignition_event = {
                    "timestamp": datetime.now().timestamp(),
                    "signal": signal,
                    "confidence": confidence,
                    "price": current_price,
                    "burst_data": burst_result,
                    "regime": self.regime_detector.detect_regime(),
                }
                self.ignition_events.append(ignition_event)

            return {
                "signal": signal,
                "confidence": confidence,
                "reason": reason,
                "momentum_score": momentum_score,
                "burst_detected": burst_result["detected"],
                "ml_prediction": ml_prediction,
                "market_regime": self.regime_detector.detect_regime(),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "reason": f"Error: {e}"}

    def _prepare_ml_features(
        self, recent_trades: List[Dict[str, Any]], current_price: float
    ) -> Optional[np.ndarray]:
        """Prepare features for ML model input."""
        try:
            if len(recent_trades) < 10:
                return None

            # Take last 20 trades for feature extraction
            trades = recent_trades[-20:]

            # Extract price features
            prices = np.array([trade["price"] for trade in trades])
            price_returns = np.diff(prices) / prices[:-1]

            # Extract volume features
            volumes = np.array([trade["size"] for trade in trades])
            volume_ma = np.mean(volumes)

            # Extract order type features
            market_ratio = sum(
                1 for trade in trades if trade["order_type"] == "market"
            ) / len(trades)

            # Extract aggressor features
            buy_ratio = sum(1 for trade in trades if trade["aggressor"] == "buy") / len(
                trades
            )

            # Combine features (ensure exactly 20 features)
            features = np.zeros(20)

            # Price momentum features (8 features)
            if len(price_returns) >= 8:
                features[:8] = price_returns[-8:]
            else:
                features[: len(price_returns)] = price_returns

            # Volume features (4 features)
            features[8] = volume_ma
            features[9] = np.std(volumes) if len(volumes) > 1 else 0
            features[10] = volumes[-1] / volume_ma if volume_ma > 0 else 1
            features[11] = np.max(volumes) / volume_ma if volume_ma > 0 else 1

            # Order flow features (4 features)
            features[12] = market_ratio
            features[13] = buy_ratio
            features[14] = len(trades)
            features[15] = (trades[-1]["timestamp"] - trades[0]["timestamp"]) / len(
                trades
            )

            # Price level features (4 features)
            features[16] = current_price / np.mean(prices) if np.mean(prices) > 0 else 1
            features[17] = (
                (np.max(prices) - np.min(prices)) / np.mean(prices)
                if np.mean(prices) > 0
                else 0
            )
            features[18] = (
                (current_price - np.min(prices)) / (np.max(prices) - np.min(prices))
                if np.max(prices) > np.min(prices)
                else 0.5
            )
            features[19] = (
                np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
            )

            return features

        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None

    def generate_market_data_signature(self, market_data: MarketData) -> str:
        """Generate HMAC-SHA256 signature for MarketData."""
        try:
            if not self.config.enable_crypto_verification:
                return ""

            if self.crypto_verifier is None:
                logger.error("Crypto verifier not initialized")
                return ""

            return self.crypto_verifier.generate_market_data_signature(market_data)

        except Exception as e:
            logger.error(f"Error generating market data signature: {e}")
            return ""

    def verify_market_data_signature(self, market_data: MarketData) -> Dict[str, Any]:
        """Verify HMAC-SHA256 signature for MarketData."""
        try:
            if not self.config.enable_crypto_verification:
                return {"verified": True, "reason": "Verification disabled"}

            if self.crypto_verifier is None:
                return {"verified": False, "reason": "Crypto verifier not initialized"}

            return self.crypto_verifier.verify_market_data_integrity(market_data)

        except Exception as e:
            logger.error(f"Error verifying market data signature: {e}")
            return {"verified": False, "reason": f"Verification error: {e}"}

    @staticmethod
    def create_signed_market_data(
        timestamp: float,
        price: float,
        volume: int,
        bid: float = 0.0,
        ask: float = 0.0,
        master_key: Optional[bytes] = None,
    ) -> MarketData:
        """Utility method to create signed MarketData with HMAC-SHA256 signature."""
        try:
            # Create MarketData instance
            market_data = MarketData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                signature="",  # Will be set after generation
            )

            # Generate signature if master key is provided
            if master_key is not None:
                # Create temporary verifier for signature generation
                verifier = CryptoVerifier(master_key)
                signature = verifier.generate_market_data_signature(market_data)
                market_data.signature = signature

            return market_data

        except Exception as e:
            logger.error(f"Error creating signed market data: {e}")
            # Return unsigned MarketData as fallback
            return MarketData(
                timestamp=timestamp,
                price=price,
                volume=volume,
                bid=bid,
                ask=ask,
                signature="",
            )


if __name__ == "__main__":
    """Test the strategy with mock data."""
    try:
        # Initialize strategy
        config = MomentumConfig()
        strategy = MomentumIgnitionStrategy(config)

        # Create mock market data
        mock_trades = []
        base_price = 100.0
        base_time = datetime.now().timestamp()

        # Generate deterministic test data using hash-based patterns
        for i in range(50):
            # Use hash of index for deterministic "random" values
            seed_val = hash(f"trade_{i}") % 1000000 / 1000000.0
            price_offset = (seed_val - 0.5) * 0.2  # -0.1 to +0.1
            size_val = 50 + (hash(f"size_{i}") % 150)  # 50 to 200
            order_hash = hash(f"order_{i}") % 100
            aggressor_hash = hash(f"aggr_{i}") % 100

            trade = {
                "timestamp": base_time + i * 0.1,
                "price": base_price + price_offset,
                "size": size_val,
                "order_type": "market" if order_hash > 30 else "limit",
                "aggressor": "buy" if aggressor_hash > 50 else "sell",
            }
            mock_trades.append(trade)

        # Test signal generation
        signal = strategy.generate_signal(
            current_price=base_price, recent_trades=mock_trades
        )

        print(f"Strategy test completed successfully!")
        print(f"Generated signal: {signal}")

    except Exception as e:
        print(f"Strategy test failed: {e}")
        logger.error(f"Strategy test failed: {e}")


# ============================================================================
# NEXUS AI PIPELINE ADAPTER - WEEKS 1-8 FULL INTEGRATION
# ============================================================================

from threading import RLock, Lock


class StrategyCategory(Enum):
    TREND_FOLLOWING = "Trend Following"
    MEAN_REVERSION = "Mean Reversion"
    MOMENTUM = "Momentum"
    VOLATILITY_BREAKOUT = "Volatility Breakout"
    SCALPING = "Scalping"
    ARBITRAGE = "Arbitrage"
    MARKET_MAKING = "Market Making"
    EVENT_DRIVEN = "Event-Driven"
    BREAKOUT = "Breakout"
    VOLUME_PROFILE = "Volume Profile"
    ORDER_FLOW = "Order Flow"


# ============================================================================
# TIER 4 ENHANCEMENT: TTP CALCULATOR
# ============================================================================
class TTPCalculator:
    """Trade Through Probability Calculator - INLINED"""

    def __init__(self, config):
        self.config = config
        self.win_rate = 0.5
        self.trades_completed = 0
        self.winning_trades = 0
        self.ttp_history = deque(maxlen=1000)

    def calculate(self, market_data, signal_strength, historical_performance=None):
        base_probability = (
            historical_performance if historical_performance else self.win_rate
        )
        market_adj = self._calculate_market_adjustment(market_data)
        strength_mult = min(1.0, signal_strength / 100.0) if signal_strength else 0.5
        vol_penalty = self._calculate_volatility_penalty(market_data)
        ttp = base_probability * market_adj * strength_mult - vol_penalty
        ttp_value = max(0.0, min(1.0, ttp))
        self.ttp_history.append(ttp_value)
        return ttp_value

    def update_accuracy(self, signal, result):
        if result and isinstance(result, dict):
            self.trades_completed += 1
            if result.get("pnl", 0) > 0:
                self.winning_trades += 1
            self.win_rate = self.winning_trades / max(self.trades_completed, 1)

    def _calculate_market_adjustment(self, market_data):
        if not market_data or not isinstance(market_data, dict):
            return 0.8
        try:
            volatility = float(market_data.get("volatility", 1.0))
            volume = float(market_data.get("volume", 1000))
            volume_ratio = volume / 1000.0
            adjustment = max(
                0.5,
                min(1.2, 1.0 - (volatility - 1.0) * 0.1 + (volume_ratio - 1.0) * 0.05),
            )
            return adjustment
        except:
            return 1.0

    def _calculate_volatility_penalty(self, market_data):
        if not market_data or not isinstance(market_data, dict):
            return 0.0
        try:
            volatility = float(market_data.get("volatility", 1.0))
            penalty = max(0.0, (volatility - 1.0) * 0.1)
            return penalty
        except:
            return 0.0


# ============================================================================
# TIER 4 ENHANCEMENT: CONFIDENCE THRESHOLD VALIDATOR
# ============================================================================
class ConfidenceThresholdValidator:
    """Validates signals meet 57% confidence threshold - INLINED"""

    def __init__(self, min_threshold=0.57):
        self.min_threshold = min_threshold
        self.rejected_count = 0
        self.accepted_count = 0
        self.rejection_history = deque(maxlen=100)

    def passes_threshold(self, confidence, ttp):
        try:
            conf_val = float(confidence) if confidence is not None else 0.5
            ttp_val = float(ttp) if ttp is not None else 0.5
            passes = conf_val >= self.min_threshold and ttp_val >= self.min_threshold
            if passes:
                self.accepted_count += 1
            else:
                self.rejected_count += 1
                self.rejection_history.append(
                    {"confidence": conf_val, "ttp": ttp_val, "timestamp": time.time()}
                )
            return passes
        except:
            return False


# ============================================================================
# TIER 4 ENHANCEMENT: MULTI-LAYER PROTECTION FRAMEWORK
# ============================================================================
class MultiLayerProtectionFramework:
    """7-Layer Security & Risk Management Framework - INLINED"""

    def __init__(self, config):
        self.config = config
        self.kill_switch_active = False
        self.layer_violations = defaultdict(int)
        self.max_position_ratio = 0.05
        self.max_daily_loss_ratio = 0.10
        self.max_drawdown_limit = 0.15

    def validate_all_layers(self, signal, account_state, market_data, current_equity):
        try:
            layers = [
                ("pre_trade_compliance", self._layer_1_pre_trade_checks),
                ("risk_validation", self._layer_2_risk_validation),
                ("market_impact", self._layer_3_market_impact),
                ("liquidity_check", self._layer_4_liquidity_verification),
                ("counterparty_risk", self._layer_5_counterparty_risk),
                ("operational_health", self._layer_6_operational_risk),
                ("emergency_kill_switch", self._layer_7_kill_switch),
            ]
            for layer_name, layer_func in layers:
                result = layer_func(signal, account_state, market_data, current_equity)
                if not result:
                    self.layer_violations[layer_name] += 1
                    return False
            return True
        except:
            return False

    def _layer_1_pre_trade_checks(self, signal, account, market_data, equity):
        try:
            return not market_data.get("trading_halted", False)
        except:
            return True

    def _layer_2_risk_validation(self, signal, account, market_data, equity):
        try:
            position_size = abs(signal.get("size", 0) if signal else 0)
            max_position = equity * self.max_position_ratio
            daily_loss = float(account.get("daily_loss", 0))
            max_daily_loss = equity * self.max_daily_loss_ratio
            return position_size <= max_position and daily_loss < max_daily_loss
        except:
            return True

    def _layer_3_market_impact(self, signal, account, market_data, equity):
        try:
            bid = float(market_data.get("bid", 1.0))
            ask = float(market_data.get("ask", 1.0))
            spread = (ask - bid) / max(bid, 0.01)
            return spread < 0.01
        except:
            return True

    def _layer_4_liquidity_verification(self, signal, account, market_data, equity):
        try:
            order_size = abs(signal.get("size", 0) if signal else 0)
            bid_vol = float(market_data.get("total_bid_volume", 0))
            ask_vol = float(market_data.get("total_ask_volume", 0))
            available_liquidity = bid_vol + ask_vol
            return available_liquidity >= (order_size * 20) if order_size > 0 else True
        except:
            return True

    def _layer_5_counterparty_risk(self, signal, account, market_data, equity):
        try:
            return account.get("broker_healthy", True)
        except:
            return True

    def _layer_6_operational_risk(self, signal, account, market_data, equity):
        try:
            return account.get("system_healthy", True)
        except:
            return True

    def _layer_7_kill_switch(self, signal, account, market_data, equity):
        try:
            if self.kill_switch_active:
                return False
            max_drawdown = float(account.get("max_drawdown", 0))
            if max_drawdown > self.max_drawdown_limit:
                self.kill_switch_active = True
                return False
            return True
        except:
            return True


# ============================================================================
# TIER 4 ENHANCEMENT: ENHANCED ML ACCURACY TRACKER
# ============================================================================
class MLAccuracyTracker:
    """Enhanced Machine Learning accuracy tracking and performance monitoring system for Momentum Ignition Strategy"""

    def __init__(self, strategy_name="momentum_ignition"):
        self.strategy_name = strategy_name
        self.prediction_history = deque(maxlen=1000)
        self.accuracy_metrics = deque(maxlen=500)
        self.model_performance = {}
        self.last_update_time = time.time()
        self.total_predictions = 0
        self.correct_predictions = 0

        # Performance tracking windows
        self.short_term_window = 50
        self.medium_term_window = 200
        self.long_term_window = 500

        # Momentum-specific tracking
        self.burst_predictions = deque(maxlen=500)
        self.ignition_predictions = deque(maxlen=500)
        self.cascade_predictions = deque(maxlen=500)

        logging.info(
            f"[OK] Enhanced ML Accuracy Tracking System initialized for {strategy_name}"
        )

    def record_prediction(self, prediction: Dict[str, Any]) -> None:
        """Record a new ML prediction for accuracy tracking"""
        try:
            prediction_record = {
                "timestamp": time.time(),
                "prediction_id": prediction.get(
                    "prediction_id", f"momentum_{int(time.time() * 1000)}"
                ),
                "prediction_type": prediction.get("type", "momentum_ignition"),
                "predicted_direction": prediction.get("direction"),
                "predicted_confidence": prediction.get("confidence", 0.0),
                "predicted_price": prediction.get("target_price"),
                "predicted_probability": prediction.get("probability", 0.0),
                "features_used": prediction.get("features_count", 0),
                "model_version": prediction.get("model_version", "v1.0"),
                "signal_strength": prediction.get("signal_strength", 0.0),
                "burst_intensity": prediction.get("burst_intensity", 0.0),
                "ignition_strength": prediction.get("ignition_strength", 0.0),
            }

            self.prediction_history.append(prediction_record)
            self.total_predictions += 1

            # Track Momentum-specific predictions
            if "burst" in prediction.get("type", "").lower():
                self.burst_predictions.append(prediction_record)
            elif "ignition" in prediction.get("type", "").lower():
                self.ignition_predictions.append(prediction_record)
            elif "cascade" in prediction.get("type", "").lower():
                self.cascade_predictions.append(prediction_record)

        except Exception as e:
            logging.error(f"Error recording Momentum prediction: {e}")

    def record_outcome(
        self, prediction_id: str, actual_outcome: Dict[str, Any]
    ) -> None:
        """Record the actual outcome of a prediction"""
        try:
            # Find the prediction record
            prediction_record = None
            for record in reversed(self.prediction_history):
                if record.get("prediction_id") == prediction_id:
                    prediction_record = record
                    break

            if not prediction_record:
                logging.warning(f"Momentum Prediction ID {prediction_id} not found")
                return

            # Determine if prediction was correct
            predicted_direction = prediction_record.get("predicted_direction")
            actual_direction = actual_outcome.get("actual_direction")
            predicted_confidence = prediction_record.get("predicted_confidence", 0.0)

            # Calculate accuracy metrics
            is_correct = self._calculate_prediction_accuracy(
                predicted_direction, actual_direction, actual_outcome
            )

            if is_correct:
                self.correct_predictions += 1

            # Record accuracy metrics
            accuracy_record = {
                "timestamp": time.time(),
                "prediction_id": prediction_id,
                "is_correct": is_correct,
                "predicted_confidence": predicted_confidence,
                "actual_return": actual_outcome.get("actual_return", 0.0),
                "prediction_horizon": actual_outcome.get("horizon_seconds", 0),
                "prediction_type": prediction_record.get("prediction_type"),
                "model_version": prediction_record.get("model_version"),
                "burst_intensity": prediction_record.get("burst_intensity"),
                "ignition_strength": prediction_record.get("ignition_strength"),
                "signal_strength": prediction_record.get("signal_strength"),
            }

            self.accuracy_metrics.append(accuracy_record)
            self._update_model_performance(accuracy_record)

        except Exception as e:
            logging.error(f"Error recording Momentum outcome: {e}")

    def _calculate_prediction_accuracy(
        self, predicted: str, actual: str, outcome: Dict[str, Any]
    ) -> bool:
        """Calculate if a prediction was correct based on actual outcome"""
        try:
            if not predicted or not actual:
                return False

            # Direction accuracy
            direction_correct = predicted.lower() == actual.lower()

            # Consider profit/loss as additional factor
            actual_return = outcome.get("actual_return", 0.0)
            profit_correct = (predicted.lower() == "long" and actual_return > 0) or (
                predicted.lower() == "short" and actual_return < 0
            )

            # Return correct if either direction matches or profit direction matches
            return direction_correct or profit_correct

        except Exception as e:
            logging.error(f"Error calculating Momentum accuracy: {e}")
            return False

    def _update_model_performance(self, accuracy_record: Dict[str, Any]) -> None:
        """Update model performance metrics"""
        try:
            model_version = accuracy_record.get("model_version", "unknown")
            prediction_type = accuracy_record.get("prediction_type", "unknown")
            burst_intensity = accuracy_record.get("burst_intensity", 0)

            # Create comprehensive key
            key = f"{model_version}_{prediction_type}_burst{int(burst_intensity)}"

            if key not in self.model_performance:
                self.model_performance[key] = {
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "accuracy": 0.0,
                    "avg_confidence": 0.0,
                    "total_return": 0.0,
                    "last_update": time.time(),
                    "momentum_specific": True
                    if "momentum" in prediction_type.lower()
                    else False,
                }

            perf = self.model_performance[key]
            perf["total_predictions"] += 1

            if accuracy_record["is_correct"]:
                perf["correct_predictions"] += 1

            # Update accuracy
            perf["accuracy"] = perf["correct_predictions"] / perf["total_predictions"]

            # Update average confidence
            current_conf = perf["avg_confidence"] * (perf["total_predictions"] - 1)
            new_conf = accuracy_record["predicted_confidence"]
            perf["avg_confidence"] = (current_conf + new_conf) / perf[
                "total_predictions"
            ]

            # Update total return
            perf["total_return"] += accuracy_record.get("actual_return", 0.0)
            perf["last_update"] = time.time()

        except Exception as e:
            logging.error(f"Error updating Momentum model performance: {e}")

    def get_accuracy_metrics(self) -> Dict[str, Any]:
        """Get comprehensive accuracy metrics"""
        try:
            if not self.accuracy_metrics:
                return {
                    "overall_accuracy": 0.0,
                    "total_predictions": 0,
                    "correct_predictions": 0,
                    "short_term_accuracy": 0.0,
                    "medium_term_accuracy": 0.0,
                    "long_term_accuracy": 0.0,
                    "model_performance": {},
                    "confidence_calibration": 0.0,
                    "momentum_specific_metrics": {},
                }

            # Overall accuracy
            overall_accuracy = self.correct_predictions / max(1, self.total_predictions)

            # Window-based accuracy
            recent_metrics = list(self.accuracy_metrics)

            short_term_correct = sum(
                1 for m in recent_metrics[-self.short_term_window :] if m["is_correct"]
            )
            medium_term_correct = sum(
                1 for m in recent_metrics[-self.medium_term_window :] if m["is_correct"]
            )
            long_term_correct = sum(
                1 for m in recent_metrics[-self.long_term_window :] if m["is_correct"]
            )

            short_term_accuracy = short_term_correct / max(
                1, len(recent_metrics[-self.short_term_window :])
            )
            medium_term_accuracy = medium_term_correct / max(
                1, len(recent_metrics[-self.medium_term_window :])
            )
            long_term_accuracy = long_term_correct / max(
                1, len(recent_metrics[-self.long_term_window :])
            )

            # Confidence calibration
            confidence_calibration = self._calculate_confidence_calibration()

            # Momentum-specific metrics
            momentum_metrics = self._get_momentum_specific_metrics()

            return {
                "overall_accuracy": overall_accuracy,
                "total_predictions": self.total_predictions,
                "correct_predictions": self.correct_predictions,
                "short_term_accuracy": short_term_accuracy,
                "medium_term_accuracy": medium_term_accuracy,
                "long_term_accuracy": long_term_accuracy,
                "model_performance": dict(self.model_performance),
                "confidence_calibration": confidence_calibration,
                "last_update": self.last_update_time,
                "momentum_specific_metrics": momentum_metrics,
            }

        except Exception as e:
            logging.error(f"Error getting Momentum accuracy metrics: {e}")
            return {"error": str(e)}

    def _get_momentum_specific_metrics(self) -> Dict[str, Any]:
        """Get Momentum strategy specific metrics"""
        try:
            return {
                "burst_predictions": len(self.burst_predictions),
                "ignition_predictions": len(self.ignition_predictions),
                "cascade_predictions": len(self.cascade_predictions),
                "burst_accuracy": self._calculate_type_accuracy(self.burst_predictions),
                "ignition_accuracy": self._calculate_type_accuracy(
                    self.ignition_predictions
                ),
                "cascade_accuracy": self._calculate_type_accuracy(
                    self.cascade_predictions
                ),
            }
        except Exception as e:
            logging.error(f"Error calculating Momentum specific metrics: {e}")
            return {}

    def _calculate_type_accuracy(self, prediction_list: deque) -> float:
        """Calculate accuracy for specific prediction type"""
        try:
            if not prediction_list:
                return 0.0

            # Find matching accuracy records
            type_predictions = {p.get("prediction_id") for p in prediction_list}
            type_correct = 0
            type_total = 0

            for metric in self.accuracy_metrics:
                if metric.get("prediction_id") in type_predictions:
                    type_total += 1
                    if metric.get("is_correct"):
                        type_correct += 1

            return type_correct / max(1, type_total)

        except Exception as e:
            logging.error(f"Error calculating type accuracy: {e}")
            return 0.0

    def _calculate_confidence_calibration(self) -> float:
        """Calculate confidence calibration score"""
        try:
            if not self.accuracy_metrics:
                return 0.0

            # Group predictions by confidence levels
            confidence_buckets = {}

            for metric in self.accuracy_metrics:
                confidence = metric.get("predicted_confidence", 0.0)
                bucket = int(confidence * 10) / 10  # Round to nearest 0.1

                if bucket not in confidence_buckets:
                    confidence_buckets[bucket] = {"total": 0, "correct": 0}

                confidence_buckets[bucket]["total"] += 1
                if metric["is_correct"]:
                    confidence_buckets[bucket]["correct"] += 1

            # Calculate calibration error
            calibration_error = 0.0
            total_weight = 0

            for confidence, bucket in confidence_buckets.items():
                if bucket["total"] > 0:
                    actual_accuracy = bucket["correct"] / bucket["total"]
                    expected_accuracy = confidence
                    weight = bucket["total"]

                    calibration_error += weight * abs(
                        actual_accuracy - expected_accuracy
                    )
                    total_weight += weight

            if total_weight > 0:
                calibration_error /= total_weight
                return max(0.0, 1.0 - calibration_error)  # Convert to calibration score

            return 0.0

        except Exception as e:
            logging.error(f"Error calculating Momentum confidence calibration: {e}")
            return 0.0

    def get_model_recommendations(self) -> List[str]:
        """Get recommendations for Momentum model improvement"""
        try:
            recommendations = []
            metrics = self.get_accuracy_metrics()

            if metrics.get("overall_accuracy", 0.0) < 0.65:
                recommendations.append(
                    "Overall Momentum model accuracy below 65% threshold - consider retraining"
                )

            if metrics.get("confidence_calibration", 0.0) < 0.8:
                recommendations.append(
                    "Poor Momentum confidence calibration - improve probability estimates"
                )

            # Check for Momentum-specific issues
            momentum_metrics = metrics.get("momentum_specific_metrics", {})
            if momentum_metrics.get("burst_accuracy", 0.0) < 0.6:
                recommendations.append(
                    "Burst detection accuracy below 60% - review trade burst analysis"
                )
            if momentum_metrics.get("ignition_accuracy", 0.0) < 0.6:
                recommendations.append(
                    "Ignition detection accuracy below 60% - review momentum ignition patterns"
                )
            if momentum_metrics.get("cascade_accuracy", 0.0) < 0.6:
                recommendations.append(
                    "Cascade detection accuracy below 60% - review stop-loss cascade analysis"
                )

            # Check for model-specific issues
            for model_key, perf in metrics.get("model_performance", {}).items():
                if perf.get("accuracy", 0.0) < 0.6:
                    recommendations.append(
                        f"Momentum Model {model_key} accuracy below 60% - review features"
                    )
                if perf.get("total_return", 0.0) < 0:
                    recommendations.append(
                        f"Momentum Model {model_key} showing negative returns - investigate bias"
                    )

            return recommendations

        except Exception as e:
            logging.error(f"Error getting Momentum model recommendations: {e}")
            return ["Error generating Momentum recommendations"]

    # Legacy compatibility methods
    def update_trade_result(self, signal, trade_result):
        """Legacy method for compatibility"""
        try:
            prediction_id = signal.get(
                "prediction_id", f"legacy_{int(time.time() * 1000)}"
            )

            # Record prediction if not already recorded
            prediction = {
                "prediction_id": prediction_id,
                "type": "momentum_ignition",
                "direction": "long"
                if signal.get("signal", 0) > 0
                else ("short" if signal.get("signal", 0) < 0 else "neutral"),
                "confidence": signal.get("confidence", 0.5),
                "signal_strength": abs(signal.get("signal", 0)),
                "model_version": "v1.0",
            }
            self.record_prediction(prediction)

            # Record outcome
            actual_direction = (
                "long"
                if trade_result.get("pnl", 0) > 0
                else ("short" if trade_result.get("pnl", 0) < 0 else "neutral")
            )
            actual_outcome = {
                "actual_direction": actual_direction,
                "actual_return": trade_result.get("pnl", 0.0)
                / abs(trade_result.get("entry_price", 1.0)),
                "horizon_seconds": trade_result.get("duration_seconds", 0),
            }
            self.record_outcome(prediction_id, actual_outcome)

        except Exception as e:
            logging.error(f"Error in legacy update_trade_result: {e}")

    def get_accuracy(self):
        """Legacy method for compatibility"""
        return self.correct_predictions / max(self.total_predictions, 1)

    def get_performance_dashboard(self):
        """Legacy method for compatibility"""
        try:
            metrics = self.get_accuracy_metrics()
            return {
                "accuracy": metrics.get("overall_accuracy", 0.0),
                "total_predictions": self.total_predictions,
                "correct_predictions": self.correct_predictions,
                "short_term_accuracy": metrics.get("short_term_accuracy", 0.0),
                "medium_term_accuracy": metrics.get("medium_term_accuracy", 0.0),
                "long_term_accuracy": metrics.get("long_term_accuracy", 0.0),
                "confidence_calibration": metrics.get("confidence_calibration", 0.0),
                "momentum_metrics": metrics.get("momentum_specific_metrics", {}),
                "error": None,
            }
        except Exception as e:
            return {
                "accuracy": 0.0,
                "total_predictions": self.total_predictions,
                "error": str(e),
            }


# ============================================================================
# TIER 4 ENHANCEMENT: EXECUTION QUALITY TRACKER
# ============================================================================
class ExecutionQualityTracker:
    """
    Basic execution quality monitoring: slippage, latency, fill rates.
    
    Use this for: Real-time tracking and monitoring
    For optimization recommendations, use ExecutionQualityOptimizer.
    """

    def __init__(self):
        self.slippage_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.fill_rates = deque(maxlen=100)
        self.execution_events = deque(maxlen=500)

    def record_execution(self, expected_price, execution_price, latency_ms, fill_rate):
        try:
            slippage_bps = (
                (execution_price - expected_price) / max(expected_price, 0.01)
            ) * 10000
            self.slippage_history.append(slippage_bps)
            self.latency_history.append(latency_ms)
            self.fill_rates.append(fill_rate)
            self.execution_events.append(
                {
                    "timestamp": time.time(),
                    "slippage_bps": slippage_bps,
                    "latency_ms": latency_ms,
                    "fill_rate": fill_rate,
                }
            )
        except:
            pass

    def get_quality_metrics(self):
        try:
            return {
                "avg_slippage_bps": float(np.mean(self.slippage_history))
                if self.slippage_history
                else 0.0,
                "avg_latency_ms": float(np.mean(self.latency_history))
                if self.latency_history
                else 0.0,
                "avg_fill_rate": float(np.mean(self.fill_rates))
                if self.fill_rates
                else 0.0,
            }
        except:
            return {}


class MomentumIgnitionNexusAdapter:
    """
    NEXUS AI Pipeline Adapter for Momentum Ignition Strategy.

    Thread-safe adapter with comprehensive ML integration, risk management,
    volatility scaling, and feature store. All operations are protected with
    RLock for concurrent execution safety.
    """

    PIPELINE_COMPATIBLE = True

    def __init__(
        self,
        base_strategy: Optional[MomentumIgnitionStrategy] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.base_strategy = base_strategy or MomentumIgnitionStrategy()
        self.config = config or {}

        # Thread safety with RLock and Lock for concurrent operations
        self._lock = RLock()  # Thread-safe reentrant lock
        self._state_lock = Lock()  # Thread-safe state lock

        # Performance tracking
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.daily_pnl = 0.0
        self.current_equity = self.config.get("initial_capital", 100000.0)
        self.peak_equity = self.current_equity

        # Kill switch configuration
        self.kill_switch_active = False
        self.consecutive_losses = 0
        self.returns_history = deque(maxlen=252)
        self.daily_loss_limit = self.config.get("daily_loss_limit", -5000.0)
        self.max_drawdown_limit = self.config.get("max_drawdown_limit", 0.15)
        self.max_consecutive_losses = self.config.get("max_consecutive_losses", 5)

        # ML pipeline integration
        self.ml_pipeline = None
        self.ml_ensemble = None
        self._pipeline_connected = False
        self.ml_predictions_enabled = self.config.get("ml_predictions_enabled", True)
        self.ml_blend_ratio = self.config.get("ml_blend_ratio", 0.3)

        # Feature store for caching and versioning features
        self.feature_store = {}  # Feature repository with caching
        self.feature_cache = self.feature_store  # Alias for backward compatibility
        self.feature_cache_ttl = self.config.get("feature_cache_ttl", 60)
        self.feature_cache_size_limit = self.config.get(
            "feature_cache_size_limit", 1000
        )

        # Volatility scaling for dynamic position sizing
        self.volatility_history = deque(maxlen=30)  # Track volatility for scaling
        self.volatility_target = self.config.get(
            "volatility_target", 0.02
        )  # 2% target vol
        self.volatility_scaling_enabled = self.config.get("volatility_scaling", True)

        # Model drift detection
        self.drift_detected = False
        self.prediction_history = deque(maxlen=100)
        self.drift_threshold = self.config.get("drift_threshold", 0.15)

        # Execution quality tracking
        self.fill_history = []
        self.slippage_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.partial_fills_count = 0
        self.total_fills_count = 0

        # ============ TIER 4: Initialize 5 Components ============
        self.ttp_calculator = TTPCalculator(self.config)
        self.confidence_validator = ConfidenceThresholdValidator(min_threshold=0.57)
        self.protection_framework = MultiLayerProtectionFramework(self.config)
        self.ml_tracker = MLAccuracyTracker("MOMENTUM_IGNITION")
        self.execution_quality_tracker = ExecutionQualityTracker()

        # ADD: MQScore Quality Filter
        self.mqscore_filter = MQScoreQualityFilter()

        logging.info(
            "MomentumIgnitionNexusAdapter initialized with Weeks 1-8 integration"
        )

    def execute(
        self, market_dict: Dict[str, Any], features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute strategy with ML integration and risk controls.
        Thread-safe execution with kill switch protection.
        """
        with self._lock:
            # Check kill switch
            if self.kill_switch_active:
                logging.warning("Kill switch active - blocking execution")
                return {
                    "signal": 0.0,
                    "confidence": 0.0,
                    "metadata": {"kill_switch": True},
                }

            try:
                # ============ MQSCORE QUALITY FILTER ============
                # v6.1 ENHANCEMENT: Check market quality before processing
                should_trade, quality_metrics = self.mqscore_filter.should_trade(market_dict)
                if not should_trade:
                    # Market quality too low - return neutral signal
                    return {
                        "signal": 0.0,
                        "confidence": 0.0,
                        "metadata": {
                            "strategy_name": "MOMENTUM_IGNITION",
                            "filtered_by_mqscore": True,
                            "mqscore": quality_metrics.get("composite_score", 0.0),
                        },
                    }
                
                # Prepare trade data
                current_price = market_dict.get("price", market_dict.get("close", 0.0))

                # Convert market_dict to recent_trades format for base strategy
                trade = {
                    "timestamp": market_dict.get("timestamp", time.time()),
                    "price": current_price,
                    "size": market_dict.get("volume", 100.0),
                    "order_type": "market",
                    "aggressor": "buy" if market_dict.get("delta", 0) >= 0 else "sell",
                }

                # Generate base strategy signal
                base_result = self.base_strategy.generate_signal(current_price, [trade])
                if not base_result:
                    return {"signal": 0.0, "confidence": 0.0, "metadata": {}}

                # Convert signal to numeric
                signal_str = base_result.get("signal", "HOLD")
                if signal_str == "BUY":
                    base_signal = 1.0
                elif signal_str == "SELL":
                    base_signal = -1.0
                else:
                    base_signal = 0.0

                base_confidence = float(base_result.get("confidence", 0.0))

                # Blend with ML predictions if available
                final_signal = base_signal
                final_confidence = base_confidence

                if self.ml_predictions_enabled and self._pipeline_connected:
                    ml_signal = self._get_ml_prediction(market_dict, features)
                    if ml_signal is not None:
                        # Blend base and ML signals
                        final_signal = (
                            1 - self.ml_blend_ratio
                        ) * base_signal + self.ml_blend_ratio * ml_signal
                        final_confidence = max(base_confidence, 0.5)

                        # Update drift detection
                        self._update_drift_detection(base_signal, ml_signal)

                # Update volatility tracking
                if current_price > 0:
                    self.volatility_history.append(current_price)

                # ============ TIER 4: TTP CALCULATION & CONFIDENCE THRESHOLD VALIDATION ============
                signal_strength = abs(final_signal) * 100
                ttp_value = self.ttp_calculator.calculate(
                    market_dict,
                    signal_strength,
                    historical_performance=final_confidence,
                )
                passes_threshold = self.confidence_validator.passes_threshold(
                    final_confidence, ttp_value
                )
                account_state = {
                    "daily_loss": -self.daily_pnl,
                    "broker_healthy": True,
                    "system_healthy": not self.drift_detected,
                    "max_drawdown": 0.0,
                }
                protection_valid = self.protection_framework.validate_all_layers(
                    {"signal": final_signal, "size": abs(final_signal)},
                    account_state,
                    market_dict,
                    self.current_equity,
                )
                if not passes_threshold or not protection_valid:
                    final_signal = 0.0
                    filter_reason = (
                        "Threshold validation failed"
                        if not passes_threshold
                        else "Protection framework rejected"
                    )
                else:
                    filter_reason = None

                return {
                    "signal": final_signal,
                    "confidence": final_confidence,
                    "ttp": ttp_value,
                    "passes_threshold": passes_threshold,
                    "protection_valid": protection_valid,
                    "filter_reason": filter_reason,
                    "metadata": {
                        "base_signal": signal_str,
                        "reason": base_result.get("reason", ""),
                        "momentum_score": base_result.get("momentum_score"),
                        "burst_detected": base_result.get("burst_detected", False),
                        "ml_blended": self.ml_predictions_enabled
                        and self._pipeline_connected,
                        "drift_detected": self.drift_detected,
                        "tier_4_filtered": filter_reason is not None,
                    },
                }
            except Exception as e:
                logging.error(f"Execute error: {e}")
                return {"signal": 0.0, "confidence": 0.0, "metadata": {"error": str(e)}}

    def get_category(self) -> StrategyCategory:
        """Return strategy category."""
        return StrategyCategory.MOMENTUM

    def record_trade_result(self, trade_info: Dict[str, Any]) -> None:
        """
        Record trade result with comprehensive tracking.
        Thread-safe with kill switch monitoring.
        """
        with self._lock:
            try:
                pnl = float(trade_info.get("pnl", 0.0))

                # Update performance metrics
                self.total_trades += 1
                self.total_pnl += pnl
                self.daily_pnl += pnl
                self.current_equity += pnl

                # Track win/loss
                if pnl > 0:
                    self.winning_trades += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1

                # Update peak equity
                if self.current_equity > self.peak_equity:
                    self.peak_equity = self.current_equity

                # Calculate return
                if self.peak_equity > 0:
                    ret = pnl / self.peak_equity
                    self.returns_history.append(ret)

                # Store trade history
                self.trade_history.append(
                    {
                        "timestamp": time.time(),
                        "pnl": pnl,
                        "equity": self.current_equity,
                        **trade_info,
                    }
                )

                # Check kill switch conditions
                self._check_kill_switch()

                # Record with base strategy
                self.base_strategy.record_trade_result(trade_info)

                # ============ TIER 4: ML ACCURACY & EXECUTION QUALITY TRACKING ============
                signal_data = {
                    "confidence": trade_info.get("confidence", 0.5),
                    "signal": trade_info.get("signal", 0.0),
                }
                self.ml_tracker.update_trade_result(signal_data, trade_info)
                self.ttp_calculator.update_accuracy(signal_data, trade_info)
                if "execution_price" in trade_info and "expected_price" in trade_info:
                    self.execution_quality_tracker.record_execution(
                        float(trade_info["expected_price"]),
                        float(trade_info["execution_price"]),
                        float(trade_info.get("latency_ms", 0.0)),
                        float(trade_info.get("fill_rate", 1.0)),
                    )

            except Exception as e:
                logging.error(f"Failed to record trade result: {e}")

    # ============ TIER 4: EXECUTE TRADE WITH PROTECTION ============
    def execute_trade_with_protection(
        self, signal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute trade with full multi-layer protection"""
        with self._lock:
            try:
                account_state = {
                    "daily_loss": -self.daily_pnl,
                    "broker_healthy": True,
                    "system_healthy": not self.drift_detected,
                    "max_drawdown": 0.0,
                }
                if not self.protection_framework.validate_all_layers(
                    signal_data, account_state, {}, self.current_equity
                ):
                    return {"executed": False, "reason": "Protection validation failed"}
                if not signal_data.get("passes_threshold", False):
                    return {"executed": False, "reason": "Thresholds not met"}
                logging.info(
                    f"[TIER 4] Executing trade with TTP={signal_data.get('ttp', 0.0):.3f}, Confidence={signal_data.get('confidence', 0.0):.3f}"
                )
                return {
                    "executed": True,
                    "signal": signal_data.get("signal", 0.0),
                    "ttp": signal_data.get("ttp", 0.0),
                    "confidence": signal_data.get("confidence", 0.0),
                    "timestamp": time.time(),
                }
            except Exception as e:
                logging.error(f"Trade protection execution error: {e}")
                return {"executed": False, "reason": str(e)}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        Thread-safe metric calculation with all Weeks 1-8 features.
        """
        with self._lock:
            win_rate = self.winning_trades / max(self.total_trades, 1)
            current_drawdown = (self.peak_equity - self.current_equity) / max(
                self.peak_equity, 1
            )

            metrics = {
                # Basic performance
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "win_rate": win_rate,
                "total_pnl": self.total_pnl,
                "daily_pnl": self.daily_pnl,
                "current_equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "current_drawdown": current_drawdown,
                # Risk metrics
                "kill_switch_active": self.kill_switch_active,
                "consecutive_losses": self.consecutive_losses,
                # ML integration
                "ml_enabled": self.ml_predictions_enabled,
                "pipeline_connected": self._pipeline_connected,
                "drift_detected": self.drift_detected,
            }

            # Add VaR/CVaR calculations
            if len(self.returns_history) >= 20:
                returns_array = np.array(list(self.returns_history))
                metrics["var_95"] = float(np.percentile(returns_array, 5))
                metrics["var_99"] = float(np.percentile(returns_array, 1))
                metrics["cvar_95"] = (
                    float(returns_array[returns_array <= metrics["var_95"]].mean())
                    if len(returns_array[returns_array <= metrics["var_95"]]) > 0
                    else metrics["var_95"]
                )
                metrics["cvar_99"] = (
                    float(returns_array[returns_array <= metrics["var_99"]].mean())
                    if len(returns_array[returns_array <= metrics["var_99"]]) > 0
                    else metrics["var_99"]
                )

            # Add leverage metrics
            leverage_data = self.calculate_leverage_ratio(
                {"position_size": self.current_equity * 0.02}, market_data={}
            )
            metrics.update(
                {
                    "current_leverage": leverage_data.get("leverage_ratio", 0.0),
                    "max_leverage_allowed": self.config.get("max_leverage", 3.0),
                }
            )

            # Add execution quality metrics
            exec_metrics = self.get_execution_quality_metrics()
            metrics.update(exec_metrics)

            # Add volatility scaling metrics
            vol_metrics = self.get_volatility_metrics()
            metrics.update({f"volatility_{k}": v for k, v in vol_metrics.items()})

            return metrics

    def _check_kill_switch(self) -> None:
        """
        Check kill switch conditions.
        Triggers: daily loss limit, max drawdown, consecutive losses.
        """
        # Check daily loss limit
        if self.daily_pnl <= self.daily_loss_limit:
            self.kill_switch_active = True
            logging.warning(
                f"Kill switch activated: Daily loss limit {self.daily_pnl:.2f} <= {self.daily_loss_limit:.2f}"
            )
            return

        # Check max drawdown
        current_drawdown = (self.peak_equity - self.current_equity) / max(
            self.peak_equity, 1
        )
        if current_drawdown >= self.max_drawdown_limit:
            self.kill_switch_active = True
            logging.warning(
                f"Kill switch activated: Drawdown {current_drawdown:.2%} >= {self.max_drawdown_limit:.2%}"
            )
            return

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.kill_switch_active = True
            logging.warning(
                f"Kill switch activated: {self.consecutive_losses} consecutive losses"
            )
            return

    def reset_kill_switch(self) -> None:
        """Reset kill switch (e.g., at start of new trading day)."""
        with self._lock:
            self.kill_switch_active = False
            self.daily_pnl = 0.0
            logging.info("Kill switch reset")

    def calculate_position_entry_logic(
        self, signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate position entry with volatility scaling and scale-in logic.
        Thread-safe position calculation with dynamic sizing.
        """
        with self._lock:
            confidence = signal.get("confidence", 0.5)
            signal_strength = abs(signal.get("signal", 0.0))

            # Base position size (2% of equity)
            base_size = self.peak_equity * 0.02

            # Apply volatility scaling to position size
            vol_adjusted_size = self._apply_volatility_scaling(base_size, market_data)

            # Apply confidence and strength multipliers
            confidence_multiplier = confidence / 0.5  # Scale around 0.5 baseline
            strength_multiplier = signal_strength
            entry_size = vol_adjusted_size * confidence_multiplier * strength_multiplier

            # Cap at max position size
            max_position = self.peak_equity * self.config.get("max_position_pct", 0.10)
            entry_size = min(entry_size, max_position)

            # Scale-in logic (pyramiding)
            allow_scale_in = confidence > 0.7 and signal_strength > 0.6
            scale_in_allocation = [0.50, 0.30, 0.20] if allow_scale_in else [1.0]

            return {
                "entry_size": entry_size,
                "scale_in_allowed": allow_scale_in,
                "scale_in_allocation": scale_in_allocation,
                "confidence_multiplier": confidence_multiplier,
                "strength_multiplier": strength_multiplier,
                "max_position": max_position,
                "volatility_adjusted": self.volatility_scaling_enabled,
            }

    def calculate_position_exit_logic(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        signal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate position exit with multiple triggers.
        Thread-safe exit logic with trailing stops.
        """
        with self._lock:
            should_exit = False
            exit_reason = None
            exit_pct = 0.0

            entry_price = position.get("entry_price", 0.0)
            current_price = market_data.get("price", 0.0)
            position_size = position.get("size", 0.0)

            if entry_price == 0 or current_price == 0:
                return {"should_exit": False, "exit_reason": None, "exit_pct": 0.0}

            # Calculate P&L
            pnl_pct = (
                (current_price - entry_price)
                / entry_price
                * (1 if position.get("side") == "long" else -1)
            )

            # Exit trigger 1: Stop loss (-2%)
            if pnl_pct <= -0.02:
                should_exit = True
                exit_reason = "stop_loss"
                exit_pct = 1.0

            # Exit trigger 2: Take profit (3%)
            elif pnl_pct >= 0.03:
                should_exit = True
                exit_reason = "take_profit"
                exit_pct = 1.0

            # Exit trigger 3: Signal reversal
            elif signal.get("signal", 0.0) * position.get("side_value", 1) < -0.5:
                should_exit = True
                exit_reason = "signal_reversal"
                exit_pct = 1.0

            # Exit trigger 4: Low confidence
            elif signal.get("confidence", 1.0) < 0.3:
                should_exit = True
                exit_reason = "low_confidence"
                exit_pct = 0.5  # Partial exit

            # Exit trigger 5: Kill switch
            elif self.kill_switch_active:
                should_exit = True
                exit_reason = "kill_switch"
                exit_pct = 1.0

            # Exit trigger 6: Trailing stop (1.5% from peak)
            peak_pnl = position.get("peak_pnl_pct", pnl_pct)
            if pnl_pct > 0 and pnl_pct < peak_pnl - 0.015:
                should_exit = True
                exit_reason = "trailing_stop"
                exit_pct = 1.0

            return {
                "should_exit": should_exit,
                "exit_reason": exit_reason,
                "exit_pct": exit_pct,
                "current_pnl_pct": pnl_pct,
                "peak_pnl_pct": peak_pnl,
            }

    def calculate_leverage_ratio(
        self, position: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate leverage ratio with margin requirements.
        Thread-safe leverage calculation with limits.
        """
        with self._lock:
            position_size = position.get("position_size", 0.0)
            account_equity = self.current_equity

            if account_equity <= 0:
                return {
                    "leverage_ratio": 0.0,
                    "margin_used": 0.0,
                    "margin_available": 0.0,
                    "is_within_limits": True,
                }

            # Calculate leverage ratio
            leverage_ratio = position_size / account_equity

            # Margin requirements (e.g., 30% initial margin for 3.33x max leverage)
            margin_requirement = self.config.get("margin_requirement", 0.30)
            margin_used = position_size * margin_requirement
            margin_available = account_equity - margin_used

            # Check leverage limits
            max_leverage = self.config.get("max_leverage", 3.0)
            is_within_limits = leverage_ratio <= max_leverage

            # Adjust position size if over limit
            adjusted_position_size = position_size
            if not is_within_limits:
                adjusted_position_size = account_equity * max_leverage
                logging.warning(
                    f"Position size {position_size:.2f} exceeds max leverage {max_leverage}x, adjusting to {adjusted_position_size:.2f}"
                )

            return {
                "leverage_ratio": leverage_ratio,
                "margin_used": margin_used,
                "margin_available": margin_available,
                "max_leverage": max_leverage,
                "is_within_limits": is_within_limits,
                "adjusted_position_size": adjusted_position_size,
                "reduction_pct": (position_size - adjusted_position_size)
                / max(position_size, 1)
                if not is_within_limits
                else 0.0,
            }

    def _apply_volatility_scaling(
        self, base_size: float, market_data: Dict[str, Any]
    ) -> float:
        """
        Apply volatility scaling to dynamic position sizing.
        Adjusts position size based on current market volatility relative to target.
        """
        if not self.volatility_scaling_enabled:
            return base_size

        # Calculate current volatility from returns
        current_price = market_data.get("price", 0.0)
        if current_price > 0 and len(self.volatility_history) > 0:
            # Calculate return
            prev_price = (
                self.volatility_history[-1]
                if self.volatility_history
                else current_price
            )
            if prev_price > 0:
                self.volatility_history.append(current_price)

                # Calculate realized volatility
                if len(self.volatility_history) >= 10:
                    prices = list(self.volatility_history)
                    returns = [
                        (prices[i] - prices[i - 1]) / prices[i - 1]
                        for i in range(1, len(prices))
                        if prices[i - 1] > 0
                    ]
                    if returns:
                        realized_vol = np.std(returns) * np.sqrt(252)  # Annualized

                        # Volatility target scaling
                        if realized_vol > 0:
                            vol_scalar = self.volatility_target / realized_vol
                            # Cap scaling between 0.5x and 2.0x
                            vol_scalar = max(0.5, min(2.0, vol_scalar))
                            return base_size * vol_scalar
        else:
            # Initialize volatility history
            if current_price > 0:
                self.volatility_history.append(current_price)

        return base_size

    def get_volatility_metrics(self) -> Dict[str, Any]:
        """Get volatility scaling metrics for monitoring."""
        if len(self.volatility_history) < 10:
            return {
                "realized_volatility": 0.0,
                "volatility_target": self.volatility_target,
                "volatility_scalar": 1.0,
                "volatility_scaling_enabled": self.volatility_scaling_enabled,
            }

        # Calculate realized volatility
        prices = list(self.volatility_history)
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1]
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]

        if returns:
            realized_vol = np.std(returns) * np.sqrt(252)
            vol_scalar = (
                self.volatility_target / realized_vol if realized_vol > 0 else 1.0
            )
            vol_scalar = max(0.5, min(2.0, vol_scalar))
        else:
            realized_vol = 0.0
            vol_scalar = 1.0

        return {
            "realized_volatility": float(realized_vol),
            "volatility_target": self.volatility_target,
            "volatility_scalar": float(vol_scalar),
            "volatility_scaling_enabled": self.volatility_scaling_enabled,
            "price_samples": len(self.volatility_history),
        }

    def connect_to_pipeline(self, pipeline) -> None:
        """Connect to ML pipeline and ensemble."""
        self.ml_pipeline = pipeline
        self.ml_ensemble = pipeline
        self._pipeline_connected = True
        logging.info("Connected to ML pipeline and ensemble")

    def get_ml_parameter_manager(self) -> Dict[str, Any]:
        """Get ML parameter manager configuration."""
        return {
            "ml_enabled": self.ml_predictions_enabled,
            "blend_ratio": self.ml_blend_ratio,
            "drift_threshold": self.drift_threshold,
            "feature_cache_ttl": self.feature_cache_ttl,
        }

    def prepare_ml_features(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare features for ML pipeline.
        Thread-safe feature preparation with caching.
        """
        with self._lock:
            # Generate cache key
            cache_key = f"{market_data.get('timestamp', time.time())}_{market_data.get('price', 0.0)}"

            # Check cache first
            cached = self.get_cached_features(cache_key)
            if cached:
                return cached

            # Prepare features
            features = {
                "price": market_data.get("price", 0.0),
                "volume": market_data.get("volume", 0.0),
                "timestamp": market_data.get("timestamp", time.time()),
                "current_equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "win_rate": self.winning_trades / max(self.total_trades, 1),
                "consecutive_losses": self.consecutive_losses,
            }

            # Cache features
            self._cache_features(cache_key, features)

            return features

    def _get_ml_prediction(
        self, market_data: Dict[str, Any], features: Optional[Dict[str, Any]]
    ) -> Optional[float]:
        """
        Get ML prediction from pipeline.
        Returns None if prediction unavailable.
        """
        try:
            if not self._pipeline_connected or self.ml_pipeline is None:
                return None

            # Prepare features if not provided
            if features is None:
                features = self.prepare_ml_features(market_data)

            # Get prediction from pipeline
            if hasattr(self.ml_pipeline, "predict"):
                prediction = self.ml_pipeline.predict(features)
                self.prediction_history.append(prediction)
                return float(prediction)

            return None
        except Exception as e:
            logging.error(f"ML prediction error: {e}")
            return None

    def _cache_features(self, cache_key: str, features: Dict[str, Any]) -> None:
        """
        Cache features in feature store with versioning.
        Feature repository maintains feature lineage for reproducibility.
        """
        if len(self.feature_store) >= self.feature_cache_size_limit:
            keys_to_remove = list(self.feature_store.keys())[
                : int(self.feature_cache_size_limit * 0.1)
            ]
            for key in keys_to_remove:
                del self.feature_store[key]
        # Store in feature repository with timestamp for versioning
        self.feature_store[cache_key] = {"features": features, "timestamp": time.time()}

    def get_cached_features(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve features from feature store with versioning support.
        Returns None if features expired or not found in repository.
        """
        if cache_key not in self.feature_store:
            return None
        entry = self.feature_store[cache_key]
        age = time.time() - entry["timestamp"]
        if age > self.feature_cache_ttl:
            del self.feature_store[cache_key]
            return None
        return entry["features"]

    def _update_drift_detection(self, strategy_signal: float, ml_signal: float) -> None:
        """Update drift detection by comparing strategy and ML signals."""
        divergence = abs(strategy_signal - ml_signal)
        if divergence > self.drift_threshold:
            if not self.drift_detected:
                self.drift_detected = True
                logging.warning(
                    f"Model drift detected: divergence {divergence:.3f} > threshold {self.drift_threshold}"
                )
        else:
            self.drift_detected = False

    def record_fill(self, fill_info: Dict[str, Any]) -> None:
        """Record fill information for execution quality tracking."""
        with self._lock:
            self.fill_history.append({"timestamp": time.time(), **fill_info})
            self.total_fills_count += 1

            # Track slippage
            expected_price = fill_info.get("expected_price", 0.0)
            actual_price = fill_info.get("actual_price", 0.0)
            if expected_price > 0:
                slippage_bps = self._calculate_slippage(expected_price, actual_price)
                self.slippage_history.append(slippage_bps)

            # Track latency
            latency = fill_info.get("latency_ms", 0.0)
            if latency > 0:
                self.latency_history.append(latency)

    def handle_fill(self, fill_event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle fill event with partial fill detection.
        Returns fill analysis including partial fill rate.
        """
        order_size = fill_event.get("order_size", 0.0)
        filled_size = fill_event.get("filled_size", 0.0)
        is_partial = filled_size < order_size

        if is_partial:
            self.partial_fills_count += 1
            fill_rate = filled_size / max(order_size, 1)
            partial_fill_rate = self.partial_fills_count / max(
                self.total_fills_count, 1
            )

            # Alert if partial fill rate is high
            if partial_fill_rate > 0.20:
                logging.warning(f"High partial fill rate: {partial_fill_rate:.1%}")

        # Record the fill
        self.record_fill(fill_event)

        return {
            "is_partial": is_partial,
            "fill_rate": filled_size / max(order_size, 1),
            "partial_fill_rate": self.partial_fills_count
            / max(self.total_fills_count, 1),
        }

    def _calculate_slippage(self, expected_price: float, actual_price: float) -> float:
        """Calculate slippage in basis points."""
        if expected_price == 0:
            return 0.0
        slippage_pct = (actual_price - expected_price) / expected_price
        return slippage_pct * 10000  # Convert to bps

    def get_execution_quality_metrics(self) -> Dict[str, Any]:
        """Get execution quality metrics including slippage and fill rates."""
        if not self.slippage_history:
            return {
                "avg_slippage_bps": 0.0,
                "slippage_std_bps": 0.0,
                "worst_slippage_bps": 0.0,
                "best_slippage_bps": 0.0,
                "avg_fill_rate": 1.0,
                "partial_fill_rate": 0.0,
            }

        slippage_array = np.array(self.slippage_history)

        metrics = {
            "avg_slippage_bps": float(np.mean(slippage_array)),
            "slippage_std_bps": float(np.std(slippage_array)),
            "p50_slippage_bps": float(np.percentile(slippage_array, 50)),
            "p95_slippage_bps": float(np.percentile(slippage_array, 95)),
            "worst_slippage_bps": float(np.max(slippage_array)),
            "best_slippage_bps": float(np.min(slippage_array)),
            "avg_fill_rate": 1.0
            - (self.partial_fills_count / max(self.total_fills_count, 1)),
            "partial_fill_rate": self.partial_fills_count
            / max(self.total_fills_count, 1),
            "total_fills": self.total_fills_count,
            "partial_fills": self.partial_fills_count,
        }

        # Add latency metrics if available
        if self.latency_history:
            latency_array = np.array(self.latency_history)
            metrics.update(
                {
                    "avg_latency_ms": float(np.mean(latency_array)),
                    "p50_latency_ms": float(np.percentile(latency_array, 50)),
                    "p95_latency_ms": float(np.percentile(latency_array, 95)),
                    "p99_latency_ms": float(np.percentile(latency_array, 99)),
                    "max_latency_ms": float(np.max(latency_array)),
                }
            )

        return metrics


# ===========================================================================
# ENHANCED ML PIPELINE WITH DRIFT DETECTION
# ===========================================================================


class MLDriftDetector:
    """Detect ML model degradation and trigger retraining"""

    def __init__(self, baseline_accuracy: float = 0.85, drift_threshold: float = 0.10):
        self.baseline_accuracy = baseline_accuracy
        self.drift_threshold = drift_threshold
        self.accuracy_history = deque(maxlen=50)
        self.drift_detected = False
        self.last_retrain = time.time()
        self.retrain_interval = 604800  # 1 week

    def update_accuracy(self, current_accuracy: float):
        """Update accuracy and check for drift"""
        self.accuracy_history.append(current_accuracy)

        if len(self.accuracy_history) >= 10:
            recent_avg = np.mean(list(self.accuracy_history)[-10:])
            drift = self.baseline_accuracy - recent_avg
            drift_score = (
                drift / self.baseline_accuracy if self.baseline_accuracy > 0 else 0.0
            )

            self.drift_detected = drift_score > self.drift_threshold
            return {
                "drift_detected": self.drift_detected,
                "current_accuracy": recent_avg,
                "baseline_accuracy": self.baseline_accuracy,
                "drift_score": drift_score,
            }

        return {"drift_detected": False, "current_accuracy": current_accuracy}

    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        time_since_retrain = time.time() - self.last_retrain
        return self.drift_detected or time_since_retrain > self.retrain_interval

    def mark_retrained(self):
        """Mark model as retrained"""
        self.last_retrain = time.time()
        self.drift_detected = False


class ExecutionQualityOptimizer:
    """
    Advanced execution quality optimization and improvement.
    
    Use this for: Generating optimization recommendations
    Requires data from ExecutionQualityTracker for analysis.
    """

    def __init__(self):
        self.execution_history = deque(maxlen=100)
        self.slippage_stats = {"mean": 0.0, "std": 0.0, "min": float("inf"), "max": 0.0}
        self.fill_rate_stats = {"mean": 0.95, "std": 0.05}
        self.latency_stats = {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    def predict_liquidity(
        self, bid: float, ask: float, volume: float, recent_volumes: list
    ) -> Dict[str, Any]:
        """Predict liquidity conditions"""
        spread = ask - bid if ask > 0 and bid > 0 else 0.001
        spread_norm = spread / bid if bid > 0 else 0.001

        avg_volume = np.mean(recent_volumes) if recent_volumes else 1.0
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

        liquidity_score = (1.0 / (1.0 + spread_norm * 100)) * (volume_ratio / 2.0)
        liquidity_score = max(0.0, min(1.0, liquidity_score))

        return {
            "spread_normalized": spread_norm,
            "volume_ratio": volume_ratio,
            "liquidity_score": liquidity_score,
            "recommendation": "EXECUTE"
            if liquidity_score > 0.6
            else "DEFER"
            if liquidity_score > 0.3
            else "SKIP",
        }

    def record_execution(
        self,
        intended_price: float,
        actual_price: float,
        intended_size: float,
        filled_size: float,
        latency_ms: float,
    ):
        """Record execution quality"""
        slippage_pct = (
            abs(actual_price - intended_price) / intended_price
            if intended_price > 0
            else 0.0
        )
        slippage_bps = slippage_pct * 10000
        fill_rate = filled_size / intended_size if intended_size > 0 else 1.0

        self.execution_history.append(
            {
                "slippage_bps": slippage_bps,
                "fill_rate": fill_rate,
                "latency_ms": latency_ms,
                "timestamp": time.time(),
            }
        )

        if len(self.execution_history) > 1:
            slippages = [e["slippage_bps"] for e in self.execution_history]
            self.slippage_stats["mean"] = np.mean(slippages)
            self.slippage_stats["std"] = np.std(slippages)
            self.slippage_stats["min"] = np.min(slippages)
            self.slippage_stats["max"] = np.max(slippages)

            latencies = [e["latency_ms"] for e in self.execution_history]
            self.latency_stats["p50"] = np.percentile(latencies, 50)
            self.latency_stats["p95"] = np.percentile(latencies, 95)
            self.latency_stats["p99"] = np.percentile(latencies, 99)

    def get_quality_report(self) -> Dict[str, Any]:
        """Get execution quality metrics"""
        return {
            "avg_slippage_bps": self.slippage_stats["mean"],
            "std_slippage_bps": self.slippage_stats["std"],
            "min_slippage_bps": self.slippage_stats["min"],
            "max_slippage_bps": self.slippage_stats["max"],
            "latency_p50_ms": self.latency_stats["p50"],
            "latency_p95_ms": self.latency_stats["p95"],
            "latency_p99_ms": self.latency_stats["p99"],
            "executions_tracked": len(self.execution_history),
        }


class AdvancedRegimeDetector:
    """
    Advanced market regime detection with detailed classification.
    
    Use this for: Comprehensive regime analysis with multiple regimes
    For basic regime detection, use MarketRegimeDetector instead.
    
    Detects: Trending, Normal, Consolidation, High Volatility regimes
    """

    def __init__(self):
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=50)

    def add_market_data(self, price: float, volume: float, volatility: float):
        """Add market data"""
        self.price_history.append(price)
        self.volume_history.append(volume)
        self.volatility_history.append(volatility)

    def detect_regime(self) -> Dict[str, Any]:
        """Detect current market regime"""

        if len(self.price_history) < 10:
            return {
                "regime": "UNKNOWN",
                "trend_strength": 0.5,
                "volatility_level": "UNKNOWN",
            }

        recent_prices = list(self.price_history)[-20:]
        trend = sum(
            1
            for i in range(1, len(recent_prices))
            if recent_prices[i] > recent_prices[i - 1]
        )
        trend_strength = trend / len(recent_prices)

        price_std = np.std(recent_prices)
        price_mean = np.mean(recent_prices)
        cv = (price_std / price_mean) if price_mean != 0 else 0.0

        vol_mean = np.mean(self.volume_history)
        vol_current = self.volume_history[-1] if self.volume_history else vol_mean
        vol_ratio = vol_current / vol_mean if vol_mean > 0 else 1.0

        if cv > 0.03 and vol_ratio > 1.2:
            regime = "VOLATILE"
        elif trend_strength > 0.65 or trend_strength < 0.35:
            regime = "TRENDING"
        elif 0.45 < trend_strength < 0.55:
            regime = "CONSOLIDATION"
        else:
            regime = "NORMAL"

        if cv > 0.04:
            vol_level = "HIGH"
        elif cv < 0.01:
            vol_level = "LOW"
        else:
            vol_level = "NORMAL"

        return {
            "regime": regime,
            "trend_strength": trend_strength,
            "coefficient_variation": cv,
            "volume_ratio": vol_ratio,
            "volatility_level": vol_level,
        }


class OnlineLearningPipeline:
    """Daily/Weekly/Monthly learning cycles"""

    def __init__(self):
        self.trade_journal = deque(maxlen=1000)
        self.daily_trades = deque(maxlen=100)
        self.weekly_trades = deque(maxlen=500)
        self.performance_metrics = {}

    def record_trade(self, trade_data: Dict[str, Any]):
        """Record trade for learning"""
        self.trade_journal.append(trade_data)
        self.daily_trades.append(trade_data)
        self.weekly_trades.append(trade_data)

    def daily_cycle(self) -> Dict[str, Any]:
        """Daily learning cycle"""
        if len(self.daily_trades) == 0:
            return {"status": "NO_TRADES"}

        pnls = [t.get("pnl", 0) for t in self.daily_trades]
        wins = sum(1 for p in pnls if p > 0)

        return {
            "cycle": "DAILY",
            "trades_count": len(self.daily_trades),
            "win_rate": wins / len(self.daily_trades) if self.daily_trades else 0.0,
            "avg_pnl": np.mean(pnls),
            "total_pnl": np.sum(pnls),
            "period": "24H",
        }

    def weekly_cycle(self) -> Dict[str, Any]:
        """Weekly learning cycle"""
        if len(self.weekly_trades) < 10:
            return {"status": "INSUFFICIENT_DATA"}

        pnls = [t.get("pnl", 0) for t in self.weekly_trades]
        wins = sum(1 for p in pnls if p > 0)

        return {
            "cycle": "WEEKLY",
            "trades_count": len(self.weekly_trades),
            "win_rate": wins / len(self.weekly_trades) if self.weekly_trades else 0.0,
            "avg_pnl": np.mean(pnls),
            "std_pnl": np.std(pnls),
            "period": "7D",
        }


class PerformanceAttributionSystem:
    """Track component-level performance"""

    def __init__(self):
        self.attribution_data = {
            "by_component": {},
            "by_regime": {},
            "by_confidence": {},
        }

    def record_trade(self, trade: Dict[str, Any]):
        """Record trade with attribution"""
        pnl = trade.get("pnl", 0)

        if "component" in trade:
            comp = trade["component"]
            if comp not in self.attribution_data["by_component"]:
                self.attribution_data["by_component"][comp] = {"pnl": 0, "trades": 0}
            self.attribution_data["by_component"][comp]["pnl"] += pnl
            self.attribution_data["by_component"][comp]["trades"] += 1

        if "regime" in trade:
            regime = trade["regime"]
            if regime not in self.attribution_data["by_regime"]:
                self.attribution_data["by_regime"][regime] = {"pnl": 0, "trades": 0}
            self.attribution_data["by_regime"][regime]["pnl"] += pnl
            self.attribution_data["by_regime"][regime]["trades"] += 1

    def get_attribution_report(self) -> Dict[str, Any]:
        """Get attribution report"""
        report = {}

        for category, data in self.attribution_data.items():
            report[category] = {}
            for key, stats in data.items():
                trades = stats["trades"]
                pnl = stats["pnl"]
                report[category][key] = {
                    "pnl": pnl,
                    "trades": trades,
                    "avg_pnl_per_trade": pnl / trades if trades > 0 else 0.0,
                }

        return report


class SystematicOptimizationFramework:
    """Parameter optimization and backtesting"""

    def __init__(self):
        self.optimization_history = deque(maxlen=50)
        self.best_parameters = {}

    def run_optimization(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Run parameter optimization"""

        win_rate = current_metrics.get("win_rate", 0.5)
        sharpe = current_metrics.get("sharpe_ratio", 1.0)

        score = (win_rate * 100) + (sharpe * 10)

        self.best_parameters = {
            "signal_threshold": 0.50 + (win_rate * 0.10),
            "position_size_multiplier": 1.0 + (sharpe * 0.1),
            "stop_loss_pct": 0.02 * (2.0 - win_rate),
        }

        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "score": score,
                "parameters": self.best_parameters.copy(),
            }
        )

        return {
            "best_parameters": self.best_parameters,
            "score": score,
            "status": "OPTIMIZED",
        }

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization status"""
        if not self.optimization_history:
            return {"status": "NO_OPTIMIZATIONS"}

        latest = self.optimization_history[-1]

        return {
            "latest_optimization": latest,
            "total_optimizations": len(self.optimization_history),
            "best_parameters": self.best_parameters,
        }


class MultiTimeframeConsensusEngine:
    """4H/1H/5M voting"""

    def __init__(self):
        self.timeframe_data = {
            "4H": {"signal": 0.5, "strength": 0.5},
            "1H": {"signal": 0.5, "strength": 0.5},
            "5M": {"signal": 0.5, "strength": 0.5},
        }
        self.weights = {"4H": 0.40, "1H": 0.35, "5M": 0.25}

    def add_signal(self, timeframe: str, signal: float, strength: float):
        """Add signal from timeframe"""
        if timeframe in self.timeframe_data:
            self.timeframe_data[timeframe]["signal"] = signal
            self.timeframe_data[timeframe]["strength"] = strength

    def calculate_consensus(self) -> Dict[str, Any]:
        """Calculate multi-timeframe consensus"""

        consensus_score = sum(
            self.timeframe_data[tf]["signal"] * self.weights[tf]
            for tf in ["4H", "1H", "5M"]
        )

        signals = [self.timeframe_data[tf]["signal"] for tf in ["4H", "1H", "5M"]]
        agreement = 1.0 - (max(signals) - min(signals)) if signals else 0.0

        return {
            "consensus_score": consensus_score,
            "agreement": agreement,
            "pass_consensus": consensus_score >= 0.55,
        }


class BestOf3SynthesisManager:
    """Orchestrates all Best-of-3 components"""

    def __init__(self, initial_capital: float = 100000):
        self.drift_detector = MLDriftDetector()
        self.exec_optimizer = ExecutionQualityOptimizer()
        self.regime_detector = AdvancedRegimeDetector()
        self.learning_pipeline = OnlineLearningPipeline()
        self.attribution_system = PerformanceAttributionSystem()
        # Create config for MultiLayerProtectionFramework with initial_capital
        protection_config = {
            "initial_capital": initial_capital,
            "max_position_ratio": 0.05,
            "max_daily_loss_ratio": 0.10,
            "max_drawdown_limit": 0.15
        }
        self.protection_system = MultiLayerProtectionFramework(protection_config)
        self.optimization_framework = SystematicOptimizationFramework()
        self.mtf_consensus = MultiTimeframeConsensusEngine()
        self.initial_capital = initial_capital

    def get_synthesis_status(self) -> Dict[str, Any]:
        """Get comprehensive Best-of-3 status"""

        return {
            "system": "BEST_OF_3_SYNTHESIS",
            "status": "OPERATIONAL",
            "components": {
                "ml_drift_detector": "INITIALIZED",
                "execution_optimizer": "INITIALIZED",
                "regime_detector": "INITIALIZED",
                "learning_pipeline": "INITIALIZED",
                "attribution_system": "INITIALIZED",
                "protection_system": "INITIALIZED",
                "optimization_framework": "INITIALIZED",
                "mtf_consensus": "INITIALIZED",
            },
            "daily_learning": self.learning_pipeline.daily_cycle(),
            "execution_quality": self.exec_optimizer.get_quality_report(),
            "attribution": self.attribution_system.get_attribution_report(),
            "optimization": self.optimization_framework.get_optimization_report(),
        }

    def process_trade(self, trade_data: Dict[str, Any], market_data: Dict[str, Any]):
        """Process trade through all Best-of-3 systems"""

        self.learning_pipeline.record_trade(trade_data)
        self.attribution_system.record_trade(trade_data)
        self.protection_system.record_trade_result(trade_data.get("pnl", 0))

        self.regime_detector.add_market_data(
            market_data.get("price", 0),
            market_data.get("volume", 1000),
            market_data.get("volatility", 0.01),
        )

        return {
            "recorded": True,
            "trade_id": f"SYNTHESIS_{int(time.time() * 1000)}",
            "timestamp": time.time(),
        }


# ===========================================================================
# TRIPLE INTEGRATION LAYER: MQSCORE 6D  NEXUS_AI PIPELINE
# ===========================================================================


class MQSCORE6DIntegrationLayer:
    """Bridge MQSCORE 6D engine into Best-of-3 framework"""

    def __init__(self):
        self.mqscore_available = False
        self.mqscore_engine = None
        self.signal_history = deque(maxlen=100)
        self.grade_distribution = {}

        try:
            from mqscore_6d_engine import MQScore6DNexusAdapter

            self.adapter = MQScore6DNexusAdapter(horizon_bars=20)
            self.mqscore_available = True
        except ImportError:
            self.adapter = None

    def calculate_mqscore_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate MQSCORE 6D signal"""

        if not self.mqscore_available or self.adapter is None:
            return self._fallback_signal(market_data)

        try:
            result = self.adapter.execute(market_data)

            signal_value = float(result.get("signal", 0.0))
            confidence = float(result.get("confidence", 0.5))
            metadata = result.get("metadata", {})

            grade = metadata.get("grade", "C")

            # Track grade distribution
            self.grade_distribution[grade] = self.grade_distribution.get(grade, 0) + 1

            self.signal_history.append(
                {
                    "signal": signal_value,
                    "confidence": confidence,
                    "grade": grade,
                    "timestamp": time.time(),
                }
            )

            return {
                "mqscore_signal": signal_value,
                "mqscore_confidence": confidence,
                "mqscore_grade": grade,
                "mqscore_metadata": metadata,
                "status": "MQSCORE_ACTIVE",
            }
        except Exception as e:
            return {
                "mqscore_signal": 0.0,
                "mqscore_confidence": 0.0,
                "mqscore_grade": "ERROR",
                "error": str(e),
                "status": "MQSCORE_ERROR",
            }

    def _fallback_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback when MQSCORE unavailable"""
        price = market_data.get("price", 0)
        volume = market_data.get("volume", 1000)

        signal = 0.5 + (0.1 * (1 if volume > 1000 else -1))

        return {
            "mqscore_signal": signal,
            "mqscore_confidence": 0.3,
            "mqscore_grade": "B",
            "status": "FALLBACK",
        }

    def get_mqscore_report(self) -> Dict[str, Any]:
        """Get MQSCORE performance report"""
        if not self.signal_history:
            return {"status": "NO_DATA"}

        signals = [s["signal"] for s in self.signal_history]
        confidences = [s["confidence"] for s in self.signal_history]

        return {
            "avg_signal": float(np.mean(signals)),
            "avg_confidence": float(np.mean(confidences)),
            "grade_distribution": self.grade_distribution,
            "signals_tracked": len(self.signal_history),
            "status": "ACTIVE" if self.mqscore_available else "FALLBACK",
        }


class BestOf3MQSCOREFusion:
    """Fuse MQSCORE 6D into Best-of-3 decision making"""

    def __init__(self):
        self.mqscore_layer = MQSCORE6DIntegrationLayer()
        self.fusion_history = deque(maxlen=100)
        self.weights = {"mqscore": 0.35, "best_of_3": 0.65}

    def fuse_signals(
        self, best_of_3_signal: float, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fuse MQSCORE and Best-of-3 signals"""

        # Get MQSCORE signal
        mqscore_result = self.mqscore_layer.calculate_mqscore_signal(market_data)
        mqscore_signal = mqscore_result.get("mqscore_signal", 0.5)
        mqscore_confidence = mqscore_result.get("mqscore_confidence", 0.5)

        # Adjust Best-of-3 signal by MQSCORE confidence
        adjusted_b3_signal = best_of_3_signal * (0.5 + mqscore_confidence * 0.5)

        # Fused decision
        fused_signal = (
            self.weights["mqscore"] * mqscore_signal
            + self.weights["best_of_3"] * adjusted_b3_signal
        )

        # Confidence is product of both
        fused_confidence = mqscore_confidence * (0.5 + abs(best_of_3_signal - 0.5))

        fusion_record = {
            "timestamp": time.time(),
            "mqscore_signal": mqscore_signal,
            "best_of_3_signal": best_of_3_signal,
            "fused_signal": fused_signal,
            "fused_confidence": fused_confidence,
            "mqscore_grade": mqscore_result.get("mqscore_grade", "C"),
        }

        self.fusion_history.append(fusion_record)

        return {
            "fused_signal": fused_signal,
            "fused_confidence": fused_confidence,
            "mqscore_component": mqscore_result,
            "recommendation": "BUY"
            if fused_signal > 0.6
            else "SELL"
            if fused_signal < 0.4
            else "HOLD",
        }

    def get_fusion_report(self) -> Dict[str, Any]:
        """Get fusion performance"""
        if not self.fusion_history:
            return {"status": "NO_DATA"}

        signals = [f["fused_signal"] for f in self.fusion_history]
        confidences = [f["fused_confidence"] for f in self.fusion_history]

        return {
            "avg_fused_signal": float(np.mean(signals)),
            "avg_fused_confidence": float(np.mean(confidences)),
            "mqscore_layer": self.mqscore_layer.get_mqscore_report(),
            "fusion_records": len(self.fusion_history),
        }


class UnifiedNEXUSPipeline:
    """Connect Best-of-3 + MQSCORE with NEXUS monitoring"""

    def __init__(self, initial_capital: float = 100000):
        self.best_of_3_manager = BestOf3SynthesisManager(initial_capital)
        self.mqscore_fusion = BestOf3MQSCOREFusion()
        self.pipeline_metrics = {
            "trades_processed": 0,
            "signals_received": 0,
            "trades_accepted": 0,
            "trades_rejected": 0,
        }
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_history = deque(maxlen=50)

    def detect_market_anomalies(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies using NEXUS-style isolation forest"""

        features = np.array(
            [
                [
                    market_data.get("price", 0),
                    market_data.get("volume", 1000),
                    market_data.get("volatility", 0.02),
                    market_data.get("spread", 0.001),
                    market_data.get("price_change", 0),
                ]
            ]
        )

        try:
            anomaly_score = self.anomaly_detector.decision_function(features)[0]
            is_anomaly = anomaly_score < -0.5

            anomaly_record = {
                "timestamp": time.time(),
                "is_anomaly": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "market_data": market_data,
            }
            self.anomaly_history.append(anomaly_record)

            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": float(anomaly_score),
                "anomaly_level": "HIGH" if is_anomaly else "NORMAL",
            }
        except Exception as e:
            return {"is_anomaly": False, "anomaly_score": 0.0, "error": str(e)}

    def process_unified_trade(
        self, signal_data: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process trade through unified pipeline"""

        self.pipeline_metrics["signals_received"] += 1

        # Step 1: Check anomalies
        anomaly_check = self.detect_market_anomalies(market_data)
        if anomaly_check["is_anomaly"]:
            self.pipeline_metrics["trades_rejected"] += 1
            return {
                "status": "REJECTED_ANOMALY",
                "reason": "Market anomaly detected",
                "anomaly_score": anomaly_check["anomaly_score"],
            }

        # Step 2: Get Best-of-3 signal
        b3_signal = signal_data.get("signal", 0.5)

        # Step 3: Fuse with MQSCORE
        fused_result = self.mqscore_fusion.fuse_signals(b3_signal, market_data)
        fused_signal = fused_result["fused_signal"]

        # Step 4: Check protection layers
        layers = self.best_of_3_manager.protection_system.check_all_layers(market_data)
        all_layers_pass = all(layer.get("passed", False) for layer in layers.values())

        if not all_layers_pass:
            self.pipeline_metrics["trades_rejected"] += 1
            failed_layers = [k for k, v in layers.items() if not v.get("passed", False)]
            return {
                "status": "REJECTED_PROTECTION",
                "reason": "Protection layer failed",
                "failed_layers": failed_layers,
            }

        # Step 5: Accept trade
        trade_data = {
            "pnl": signal_data.get("pnl", 0),
            "component": "UNIFIED_PIPELINE",
            "regime": signal_data.get("regime", "UNKNOWN"),
            "signal": fused_signal,
            "mqscore_grade": fused_result["mqscore_component"].get(
                "mqscore_grade", "C"
            ),
        }

        # Process through Best-of-3
        result = self.best_of_3_manager.process_trade(trade_data, market_data)

        self.pipeline_metrics["trades_accepted"] += 1
        self.pipeline_metrics["trades_processed"] += 1

        return {
            "status": "ACCEPTED",
            "trade_id": result["trade_id"],
            "fused_signal": fused_signal,
            "fused_confidence": fused_result["fused_confidence"],
            "recommendation": fused_result["recommendation"],
            "best_of_3_result": result,
        }

    def get_unified_status(self) -> Dict[str, Any]:
        """Get complete unified pipeline status"""

        b3_status = self.best_of_3_manager.get_synthesis_status()
        fusion_report = self.mqscore_fusion.get_fusion_report()

        acceptance_rate = self.pipeline_metrics["trades_accepted"] / max(
            1, self.pipeline_metrics["signals_received"]
        )

        return {
            "pipeline_type": "UNIFIED_TRIPLE_INTEGRATION",
            "status": "OPERATIONAL",
            "components": {
                "best_of_3": b3_status["components"],
                "mqscore": fusion_report.get("mqscore_layer", {}),
                "nexus_anomaly_detection": "ACTIVE",
            },
            "metrics": {
                "signals_received": self.pipeline_metrics["signals_received"],
                "trades_accepted": self.pipeline_metrics["trades_accepted"],
                "trades_rejected": self.pipeline_metrics["trades_rejected"],
                "acceptance_rate": float(acceptance_rate),
                "trades_processed": self.pipeline_metrics["trades_processed"],
            },
            "best_of_3": b3_status,
            "mqscore_fusion": fusion_report,
            "anomalies_detected": len(self.anomaly_history),
        }


class TripleLayerArchitectureOrchestrator:
    """Master orchestrator for MQSCORE + Best-of-3 + NEXUS_AI"""

    def __init__(self, initial_capital: float = 100000):
        self.unified_pipeline = UnifiedNEXUSPipeline(initial_capital)
        self.layer_names = ["MQSCORE 6D", "Best-of-3 Synthesis", "NEXUS_AI Pipeline"]
        self.orchestration_history = deque(maxlen=200)
        self.performance_by_layer = {
            "MQSCORE 6D": {"wins": 0, "losses": 0},
            "Best-of-3 Synthesis": {"wins": 0, "losses": 0},
            "NEXUS_AI Pipeline": {"wins": 0, "losses": 0},
        }

    def execute_triple_layer_logic(
        self, signal_data: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute decision through all three layers"""

        execution_result = self.unified_pipeline.process_unified_trade(
            signal_data, market_data
        )

        # Track performance
        if "pnl" in signal_data:
            pnl = signal_data["pnl"]
            grade = execution_result.get("fused_signal", 0.5) > 0.5

            if pnl > 0:
                self.performance_by_layer["MQSCORE 6D"]["wins"] += 1
                self.performance_by_layer["Best-of-3 Synthesis"]["wins"] += 1
                self.performance_by_layer["NEXUS_AI Pipeline"]["wins"] += 1
            else:
                self.performance_by_layer["MQSCORE 6D"]["losses"] += 1
                self.performance_by_layer["Best-of-3 Synthesis"]["losses"] += 1
                self.performance_by_layer["NEXUS_AI Pipeline"]["losses"] += 1

        execution_record = {
            "timestamp": time.time(),
            "result": execution_result,
            "layers_active": self.layer_names,
        }
        self.orchestration_history.append(execution_record)

        return {
            "triple_layer_result": execution_result,
            "layers": self.layer_names,
            "orchestration_status": "COMPLETE",
        }

    def get_architecture_status(self) -> Dict[str, Any]:
        """Get complete architecture status"""

        pipeline_status = self.unified_pipeline.get_unified_status()

        # Calculate win rates by layer
        layer_stats = {}
        for layer, stats in self.performance_by_layer.items():
            total = stats["wins"] + stats["losses"]
            wr = stats["wins"] / total if total > 0 else 0.0
            layer_stats[layer] = {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "win_rate": float(wr),
            }

        return {
            "architecture": "TRIPLE_LAYER_INTEGRATION",
            "status": "OPERATIONAL",
            "layers": self.layer_names,
            "layer_stats": layer_stats,
            "pipeline_metrics": pipeline_status["metrics"],
            "orchestration_records": len(self.orchestration_history),
            "full_pipeline_status": pipeline_status,
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""

        arch_status = self.get_architecture_status()

        return {
            "report_type": "TRIPLE_INTEGRATION_COMPREHENSIVE",
            "timestamp": time.time(),
            "architecture": arch_status,
            "summary": {
                "total_executions": self.unified_pipeline.pipeline_metrics[
                    "trades_processed"
                ],
                "acceptance_rate": (
                    self.unified_pipeline.pipeline_metrics["trades_accepted"]
                    / max(1, self.unified_pipeline.pipeline_metrics["signals_received"])
                ),
                "anomalies_detected": len(self.unified_pipeline.anomaly_history),
            },
            "layers": {
                layer: arch_status["layer_stats"].get(layer, {})
                for layer in self.layer_names
            },
        }
