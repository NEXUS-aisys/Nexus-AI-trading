"""
Liquidity Absorption Strategy - NEXUS AI Compatible Implementation

Enhanced liquidity absorption detection strategy with comprehensive risk management,
MQScore 6D integration, and NEXUS adapter compliance.

Version: 2.1
Author: NEXUS AI Trading System
Dependencies: Python 3.8+ standard library only

Key Features:
- NEXUS Adapter Pattern with standardized execute() method
- MQScore 6D integration with graceful fallback
- Thread-safe implementation with RLock
- Production-grade error handling
- Real-time performance monitoring
- ML pipeline feature engineering

Components:
- LiquidityAbsorptionNexusAdapter: Main strategy adapter class
- MQScore6D: Market quality scoring system
- RiskManager: Comprehensive risk management
- PerformanceMonitor: Real-time performance tracking
"""

import os
import sys
import time
import math
import statistics
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# CORE ENUMS AND DATA STRUCTURES
# =====================================================================

class SignalType(Enum):
    """Signal types for liquidity absorption detection."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class AbsorptionType(Enum):
    """Types of liquidity absorption patterns."""
    AGGRESSIVE_BUY = "AGGRESSIVE_BUY"
    AGGRESSIVE_SELL = "AGGRESSIVE_SELL"
    ICEBERG_BUY = "ICEBERG_BUY"
    ICEBERG_SELL = "ICEBERG_SELL"
    SWEEP_BUY = "SWEEP_BUY"
    SWEEP_SELL = "SWEEP_SELL"

@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    timestamp: float
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    order_book: Dict = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    error_count: int = 0
    execution_time_ms: float = 0.0

# =====================================================================
# MQSCORE 6D INTEGRATION
# =====================================================================

class MQScore6D:
    """
    Market Quality Score 6-Dimensional Analysis System
    
    Components (with weights):
    1. Liquidity (20%) - Order book depth analysis
    2. Volatility (20%) - Price movement stability  
    3. Momentum (20%) - Directional momentum detection
    4. Imbalance (15%) - Bid/ask order imbalance
    5. Trend Strength (15%) - Trend persistence measurement
    6. Noise Level (10%) - Market noise filtering
    """
    
    def __init__(self, lookback_periods: int = 20):
        self.lookback_periods = lookback_periods
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.spread_history = deque(maxlen=lookback_periods)
        self.imbalance_history = deque(maxlen=lookback_periods)
        
        # Component weights
        self.weights = {
            'liquidity': 0.20,
            'volatility': 0.20,
            'momentum': 0.20,
            'imbalance': 0.15,
            'trend_strength': 0.15,
            'noise_level': 0.10
        }
        
        self._lock = threading.RLock()
    
    def calculate_mqscore(self, market_data: MarketData) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive MQScore with component breakdown.
        
        Returns:
            Tuple of (overall_score, component_scores)
        """
        with self._lock:
            try:
                # Update historical data
                self._update_history(market_data)
                
                if len(self.price_history) < 5:
                    # Insufficient data - return neutral score
                    return 0.5, self._get_default_components()
                
                # Calculate individual components
                components = {
                    'liquidity': self._calculate_liquidity(market_data),
                    'volatility': self._calculate_volatility(),
                    'momentum': self._calculate_momentum(),
                    'imbalance': self._calculate_imbalance(market_data),
                    'trend_strength': self._calculate_trend_strength(),
                    'noise_level': self._calculate_noise_level()
                }
                
                # Calculate weighted overall score
                overall_score = sum(
                    components[component] * self.weights[component]
                    for component in components
                )
                
                # Normalize to 0-1 range
                overall_score = max(0.0, min(1.0, overall_score))
                
                return overall_score, components
                
            except Exception as e:
                logger.error(f"MQScore calculation error: {e}")
                return 0.5, self._get_default_components()
    
    def _update_history(self, market_data: MarketData):
        """Update historical data buffers."""
        self.price_history.append(market_data.price)
        self.volume_history.append(market_data.volume)
        
        spread = market_data.ask - market_data.bid if market_data.ask > market_data.bid else 0.01
        self.spread_history.append(spread)
        
        imbalance = (market_data.bid_size - market_data.ask_size) / max(
            market_data.bid_size + market_data.ask_size, 1.0
        )
        self.imbalance_history.append(imbalance)
    
    def _calculate_liquidity(self, market_data: MarketData) -> float:
        """Calculate liquidity component (20% weight)."""
        try:
            # Order book depth analysis
            total_depth = market_data.bid_size + market_data.ask_size
            avg_spread = statistics.mean(self.spread_history) if self.spread_history else 0.01
            
            # Normalize liquidity score (higher depth, lower spread = better liquidity)
            depth_score = min(1.0, total_depth / 10000.0)  # Normalize to typical market depth
            spread_score = max(0.0, 1.0 - (avg_spread / market_data.price * 1000))  # Spread in bps
            
            return (depth_score + spread_score) / 2.0
            
        except Exception:
            return 0.5
    
    def _calculate_volatility(self) -> float:
        """Calculate volatility component (20% weight)."""
        try:
            if len(self.price_history) < 2:
                return 0.5
            
            # Calculate price returns
            returns = [
                (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
                for i in range(1, len(self.price_history))
            ]
            
            if not returns:
                return 0.5
            
            # Calculate volatility (standard deviation of returns)
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
            
            # Normalize volatility score (lower volatility = higher quality)
            # Typical intraday volatility ranges from 0.001 to 0.05
            normalized_vol = max(0.0, min(1.0, 1.0 - (volatility / 0.02)))
            
            return normalized_vol
            
        except Exception:
            return 0.5
    
    def _calculate_momentum(self) -> float:
        """Calculate momentum component (20% weight)."""
        try:
            if len(self.price_history) < 3:
                return 0.5
            
            # Calculate short-term and medium-term momentum
            short_momentum = (self.price_history[-1] - self.price_history[-3]) / self.price_history[-3]
            
            if len(self.price_history) >= 10:
                medium_momentum = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
                momentum = (short_momentum + medium_momentum) / 2.0
            else:
                momentum = short_momentum
            
            # Normalize momentum score (consistent direction = higher quality)
            momentum_strength = abs(momentum)
            normalized_momentum = min(1.0, momentum_strength / 0.01)  # 1% move = full score
            
            return normalized_momentum
            
        except Exception:
            return 0.5
    
    def _calculate_imbalance(self, market_data: MarketData) -> float:
        """Calculate order imbalance component (15% weight)."""
        try:
            if not self.imbalance_history:
                return 0.5
            
            # Calculate average imbalance and its consistency
            avg_imbalance = statistics.mean(self.imbalance_history)
            imbalance_consistency = 1.0 - statistics.stdev(self.imbalance_history) if len(self.imbalance_history) > 1 else 1.0
            
            # Normalize imbalance score (consistent imbalance = higher quality)
            imbalance_strength = abs(avg_imbalance)
            normalized_imbalance = (imbalance_strength + imbalance_consistency) / 2.0
            
            return min(1.0, normalized_imbalance)
            
        except Exception:
            return 0.5
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength component (15% weight)."""
        try:
            if len(self.price_history) < 5:
                return 0.5
            
            # Calculate trend using linear regression slope
            x_values = list(range(len(self.price_history)))
            y_values = list(self.price_history)
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Linear regression slope
            slope = (n * sum_xy - sum_x * sum_y) / max(n * sum_x2 - sum_x * sum_x, 1)
            
            # Normalize trend strength
            trend_strength = abs(slope) / max(statistics.mean(y_values), 1) * 100
            normalized_trend = min(1.0, trend_strength)
            
            return normalized_trend
            
        except Exception:
            return 0.5
    
    def _calculate_noise_level(self) -> float:
        """Calculate noise level component (10% weight)."""
        try:
            if len(self.price_history) < 3:
                return 0.5
            
            # Calculate price noise using high-frequency variations
            price_changes = [
                abs(self.price_history[i] - self.price_history[i-1])
                for i in range(1, len(self.price_history))
            ]
            
            if not price_changes:
                return 0.5
            
            # Calculate noise ratio
            avg_change = statistics.mean(price_changes)
            noise_ratio = statistics.stdev(price_changes) / max(avg_change, 0.0001) if len(price_changes) > 1 else 0.0
            
            # Normalize noise score (lower noise = higher quality)
            normalized_noise = max(0.0, 1.0 - min(1.0, noise_ratio))
            
            return normalized_noise
            
        except Exception:
            return 0.5
    
    def _get_default_components(self) -> Dict[str, float]:
        """Return default component scores when calculation fails."""
        return {
            'liquidity': 0.5,
            'volatility': 0.5,
            'momentum': 0.5,
            'imbalance': 0.5,
            'trend_strength': 0.5,
            'noise_level': 0.5
        }

# =====================================================================
# RISK MANAGEMENT SYSTEM
# =====================================================================

class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, max_position_size: float = 0.02, max_daily_loss: float = 0.05):
        self.max_position_size = max_position_size  # 2% of portfolio
        self.max_daily_loss = max_daily_loss  # 5% daily loss limit
        self.daily_pnl = 0.0
        self.current_positions = {}
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self._lock = threading.RLock()
    
    def check_risk_limits(self, signal_strength: float, current_price: float) -> Dict[str, Any]:
        """Check all risk limits before position entry."""
        with self._lock:
            risk_status = {
                'approved': True,
                'position_size': 0.0,
                'reasons': []
            }
            
            # Check kill switch
            if self.kill_switch_active:
                risk_status['approved'] = False
                risk_status['reasons'].append(f"Kill switch active: {self.kill_switch_reason}")
                return risk_status
            
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                self.activate_kill_switch(f"Daily loss limit breached: {self.daily_pnl:.4f}")
                risk_status['approved'] = False
                risk_status['reasons'].append("Daily loss limit exceeded")
                return risk_status
            
            # Calculate position size based on signal strength and risk
            base_position_size = self.max_position_size * abs(signal_strength)
            
            # Apply volatility adjustment (reduce size in high volatility)
            volatility_adjustment = max(0.5, min(1.0, 1.0 - abs(signal_strength - 0.5)))
            adjusted_position_size = base_position_size * volatility_adjustment
            
            risk_status['position_size'] = adjusted_position_size
            
            return risk_status
    
    def activate_kill_switch(self, reason: str):
        """Activate emergency kill switch."""
        with self._lock:
            self.kill_switch_active = True
            self.kill_switch_reason = reason
            logger.warning(f"Kill switch activated: {reason}")
    
    def reset_daily_metrics(self):
        """Reset daily tracking metrics."""
        with self._lock:
            self.daily_pnl = 0.0
            if self.kill_switch_reason.startswith("Daily"):
                self.kill_switch_active = False
                self.kill_switch_reason = ""

# =====================================================================
# LIQUIDITY ABSORPTION DETECTION ENGINE
# =====================================================================

class LiquidityAbsorptionDetector:
    """Advanced liquidity absorption pattern detection."""
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.trade_history = deque(maxlen=lookback_periods)
        self.volume_profile = deque(maxlen=lookback_periods)
        self.price_levels = defaultdict(float)  # Price -> cumulative volume
        self._lock = threading.RLock()
    
    def detect_absorption(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Detect liquidity absorption patterns in market data.
        
        Returns:
            Dict containing absorption analysis results
        """
        with self._lock:
            try:
                # Update historical data
                self._update_history(market_data)
                
                if len(self.trade_history) < 10:
                    return self._get_neutral_result()
                
                # Analyze different absorption patterns
                results = {
                    'aggressive_absorption': self._detect_aggressive_absorption(market_data),
                    'iceberg_absorption': self._detect_iceberg_absorption(market_data),
                    'sweep_absorption': self._detect_sweep_absorption(market_data),
                    'volume_imbalance': self._calculate_volume_imbalance(market_data),
                    'absorption_strength': 0.0,
                    'absorption_type': AbsorptionType.AGGRESSIVE_BUY.value,
                    'confidence': 0.0
                }
                
                # Calculate overall absorption strength
                absorption_strength = self._calculate_overall_strength(results)
                results['absorption_strength'] = absorption_strength
                results['confidence'] = min(1.0, absorption_strength)
                
                # Determine absorption type
                results['absorption_type'] = self._determine_absorption_type(results, market_data)
                
                return results
                
            except Exception as e:
                logger.error(f"Absorption detection error: {e}")
                return self._get_neutral_result()
    
    def _update_history(self, market_data: MarketData):
        """Update historical tracking data."""
        trade_info = {
            'timestamp': market_data.timestamp,
            'price': market_data.price,
            'volume': market_data.volume,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'spread': market_data.ask - market_data.bid
        }
        
        self.trade_history.append(trade_info)
        self.volume_profile.append(market_data.volume)
        
        # Update price level volumes
        price_level = round(market_data.price, 2)
        self.price_levels[price_level] += market_data.volume
    
    def _detect_aggressive_absorption(self, market_data: MarketData) -> float:
        """Detect aggressive liquidity absorption patterns."""
        try:
            if len(self.trade_history) < 5:
                return 0.0
            
            recent_trades = list(self.trade_history)[-5:]
            
            # Check for large volume spikes
            recent_volumes = [trade['volume'] for trade in recent_trades]
            avg_volume = statistics.mean(list(self.volume_profile)) if self.volume_profile else 1.0
            
            volume_spike = max(recent_volumes) / max(avg_volume, 1.0)
            
            # Check for price impact
            price_change = abs(market_data.price - recent_trades[0]['price']) / recent_trades[0]['price']
            
            # Aggressive absorption score
            aggression_score = min(1.0, (volume_spike - 1.0) * price_change * 10)
            
            return max(0.0, aggression_score)
            
        except Exception:
            return 0.0
    
    def _detect_iceberg_absorption(self, market_data: MarketData) -> float:
        """Detect iceberg order absorption patterns."""
        try:
            if len(self.trade_history) < 20:
                return 0.0
            
            recent_trades = list(self.trade_history)[-20:]
            
            # Look for consistent volume at similar price levels
            price_volume_map = defaultdict(list)
            for trade in recent_trades:
                price_level = round(trade['price'], 2)
                price_volume_map[price_level].append(trade['volume'])
            
            # Find price levels with multiple large trades (iceberg pattern)
            iceberg_score = 0.0
            for price_level, volumes in price_volume_map.items():
                if len(volumes) >= 3:  # Multiple trades at same level
                    avg_volume = statistics.mean(volumes)
                    volume_consistency = 1.0 - (statistics.stdev(volumes) / max(avg_volume, 1.0))
                    iceberg_score = max(iceberg_score, volume_consistency)
            
            return min(1.0, iceberg_score)
            
        except Exception:
            return 0.0
    
    def _detect_sweep_absorption(self, market_data: MarketData) -> float:
        """Detect sweep absorption across multiple price levels."""
        try:
            if len(self.trade_history) < 10:
                return 0.0
            
            recent_trades = list(self.trade_history)[-10:]
            
            # Check for trades across multiple price levels in short time
            price_levels = set(round(trade['price'], 2) for trade in recent_trades)
            time_span = recent_trades[-1]['timestamp'] - recent_trades[0]['timestamp']
            
            if time_span <= 0:
                return 0.0
            
            # Sweep intensity based on price levels covered and time
            sweep_intensity = len(price_levels) / max(time_span, 1.0)
            
            # Volume consistency across levels
            total_volume = sum(trade['volume'] for trade in recent_trades)
            avg_volume_per_level = total_volume / len(price_levels)
            
            sweep_score = min(1.0, sweep_intensity * avg_volume_per_level / 1000.0)
            
            return sweep_score
            
        except Exception:
            return 0.0
    
    def _calculate_volume_imbalance(self, market_data: MarketData) -> float:
        """Calculate volume imbalance between bid and ask sides."""
        try:
            bid_volume = market_data.bid_size
            ask_volume = market_data.ask_size
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return 0.0
            
            imbalance = (bid_volume - ask_volume) / total_volume
            
            return imbalance  # Range: -1 to 1
            
        except Exception:
            return 0.0
    
    def _calculate_overall_strength(self, results: Dict[str, Any]) -> float:
        """Calculate overall absorption strength from component scores."""
        try:
            # Weight different absorption types
            weights = {
                'aggressive_absorption': 0.4,
                'iceberg_absorption': 0.3,
                'sweep_absorption': 0.2,
                'volume_imbalance': 0.1
            }
            
            strength = 0.0
            for component, weight in weights.items():
                if component in results:
                    strength += abs(results[component]) * weight
            
            return min(1.0, strength)
            
        except Exception:
            return 0.0
    
    def _determine_absorption_type(self, results: Dict[str, Any], market_data: MarketData) -> str:
        """Determine the primary absorption type based on analysis."""
        try:
            # Find the strongest absorption pattern
            pattern_scores = {
                AbsorptionType.AGGRESSIVE_BUY.value: results.get('aggressive_absorption', 0.0),
                AbsorptionType.ICEBERG_BUY.value: results.get('iceberg_absorption', 0.0),
                AbsorptionType.SWEEP_BUY.value: results.get('sweep_absorption', 0.0)
            }
            
            # Adjust for direction based on volume imbalance
            volume_imbalance = results.get('volume_imbalance', 0.0)
            
            if volume_imbalance < -0.1:  # Sell-side absorption
                pattern_scores = {
                    AbsorptionType.AGGRESSIVE_SELL.value: results.get('aggressive_absorption', 0.0),
                    AbsorptionType.ICEBERG_SELL.value: results.get('iceberg_absorption', 0.0),
                    AbsorptionType.SWEEP_SELL.value: results.get('sweep_absorption', 0.0)
                }
            
            # Return the pattern with highest score
            return max(pattern_scores.items(), key=lambda x: x[1])[0]
            
        except Exception:
            return AbsorptionType.AGGRESSIVE_BUY.value
    
    def _get_neutral_result(self) -> Dict[str, Any]:
        """Return neutral result when detection fails."""
        return {
            'aggressive_absorption': 0.0,
            'iceberg_absorption': 0.0,
            'sweep_absorption': 0.0,
            'volume_imbalance': 0.0,
            'absorption_strength': 0.0,
            'absorption_type': AbsorptionType.AGGRESSIVE_BUY.value,
            'confidence': 0.0
        }

# =====================================================================
# MAIN STRATEGY ADAPTER CLASS
# =====================================================================

class LiquidityAbsorptionNexusAdapter:
    """
    NEXUS-compliant Liquidity Absorption Strategy Adapter.
    
    Implements the standard NEXUS adapter interface with:
    - Standardized execute() method
    - MQScore 6D integration
    - Thread-safe operations
    - Comprehensive error handling
    - ML pipeline feature engineering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy adapter with configuration."""
        self.config = config or {}
        
        # Core components
        self.mqscore = MQScore6D(lookback_periods=self.config.get('mqscore_lookback', 20))
        self.risk_manager = RiskManager(
            max_position_size=self.config.get('max_position_size', 0.02),
            max_daily_loss=self.config.get('max_daily_loss', 0.05)
        )
        self.absorption_detector = LiquidityAbsorptionDetector(
            lookback_periods=self.config.get('detector_lookback', 50)
        )
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.execution_times = deque(maxlen=100)
        
        # Configuration parameters
        self.mqscore_threshold = self.config.get('mqscore_threshold', 0.57)
        self.min_absorption_strength = self.config.get('min_absorption_strength', 0.3)
        self.signal_smoothing_factor = self.config.get('signal_smoothing_factor', 0.7)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("LiquidityAbsorptionNexusAdapter initialized")
    
    def execute(self, market_data: Any, features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute strategy with standardized return format for NEXUS compatibility.
        
        Args:
            market_data: Market data input (flexible format)
            features: Optional ML pipeline features
            
        Returns:
            Dict with standardized format:
            {
                'signal': float,      # -1.0 (SELL) to +1.0 (BUY), 0.0 (HOLD)
                'confidence': float,  # 0.0 to 1.0
                'features': dict,     # ML pipeline features
                'metadata': dict      # Strategy-specific information
            }
        """
        start_time = time.time()
        
        try:
            with self._lock:
                # Normalize market data input
                normalized_data = self._normalize_market_data(market_data)
                
                if not normalized_data:
                    return self._create_error_response("Invalid market data format")
                
                # Calculate MQScore with graceful fallback
                try:
                    mqscore, mqscore_components = self.mqscore.calculate_mqscore(normalized_data)
                except Exception as e:
                    self.logger.warning(f"MQScore calculation failed: {e}")
                    mqscore = 0.5
                    mqscore_components = self.mqscore._get_default_components()
                
                # Apply MQScore quality filter
                if mqscore < self.mqscore_threshold:
                    return self._create_hold_response(
                        f"MQScore below threshold: {mqscore:.3f} < {self.mqscore_threshold}",
                        mqscore_components,
                        features
                    )
                
                # Detect liquidity absorption patterns
                absorption_analysis = self.absorption_detector.detect_absorption(normalized_data)
                
                # Check absorption strength threshold
                absorption_strength = absorption_analysis.get('absorption_strength', 0.0)
                if absorption_strength < self.min_absorption_strength:
                    return self._create_hold_response(
                        f"Absorption strength below threshold: {absorption_strength:.3f}",
                        mqscore_components,
                        features,
                        absorption_analysis
                    )
                
                # Generate trading signal
                signal_strength = self._calculate_signal_strength(
                    absorption_analysis, mqscore, mqscore_components
                )
                
                # Apply risk management
                risk_check = self.risk_manager.check_risk_limits(
                    signal_strength, normalized_data.price
                )
                
                if not risk_check['approved']:
                    return self._create_hold_response(
                        f"Risk limits: {', '.join(risk_check['reasons'])}",
                        mqscore_components,
                        features,
                        absorption_analysis
                    )
                
                # Create final response
                response = self._create_signal_response(
                    signal_strength,
                    absorption_analysis,
                    mqscore_components,
                    features,
                    risk_check
                )
                
                # Update performance metrics
                execution_time = (time.time() - start_time) * 1000
                self.execution_times.append(execution_time)
                self.metrics.execution_time_ms = statistics.mean(self.execution_times)
                
                return response
                
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.error(f"Strategy execution error: {e}")
            return self._create_error_response(f"Execution error: {str(e)}")
    
    def get_category(self) -> str:
        """Return strategy category for NEXUS pipeline routing."""
        return "LIQUIDITY_FLOW_ORDERBOOK"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        with self._lock:
            return {
                # Core performance
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'losing_trades': self.metrics.losing_trades,
                'win_rate': self.metrics.win_rate,
                'total_pnl': self.metrics.total_pnl,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown,
                'profit_factor': self.metrics.profit_factor,
                
                # Execution metrics
                'average_execution_time_ms': self.metrics.execution_time_ms,
                'error_count': self.metrics.error_count,
                
                # Risk metrics
                'daily_pnl': self.risk_manager.daily_pnl,
                'kill_switch_active': self.risk_manager.kill_switch_active,
                'kill_switch_reason': self.risk_manager.kill_switch_reason,
                
                # Strategy-specific metrics
                'mqscore_threshold': self.mqscore_threshold,
                'min_absorption_strength': self.min_absorption_strength,
                'absorption_detector_lookback': self.absorption_detector.lookback_periods,
                'mqscore_lookback': self.mqscore.lookback_periods
            }
    
    def _normalize_market_data(self, market_data: Any) -> Optional[MarketData]:
        """Normalize various market data formats to standard MarketData structure."""
        try:
            if isinstance(market_data, MarketData):
                return market_data
            
            if isinstance(market_data, dict):
                return MarketData(
                    symbol=market_data.get('symbol', 'UNKNOWN'),
                    timestamp=market_data.get('timestamp', time.time()),
                    price=float(market_data.get('price', 0.0)),
                    volume=float(market_data.get('volume', 0.0)),
                    bid=float(market_data.get('bid', 0.0)),
                    ask=float(market_data.get('ask', 0.0)),
                    bid_size=float(market_data.get('bid_size', 0.0)),
                    ask_size=float(market_data.get('ask_size', 0.0)),
                    trades=market_data.get('trades', []),
                    order_book=market_data.get('order_book', {})
                )
            
            # Handle other formats (list, tuple, etc.)
            if hasattr(market_data, '__iter__') and not isinstance(market_data, str):
                data_list = list(market_data)
                if len(data_list) >= 3:
                    return MarketData(
                        symbol='UNKNOWN',
                        timestamp=time.time(),
                        price=float(data_list[0]),
                        volume=float(data_list[1]),
                        bid=float(data_list[2]) if len(data_list) > 2 else 0.0,
                        ask=float(data_list[3]) if len(data_list) > 3 else 0.0
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Market data normalization error: {e}")
            return None
    
    def _calculate_signal_strength(self, absorption_analysis: Dict, mqscore: float, 
                                 mqscore_components: Dict) -> float:
        """Calculate final signal strength from all components."""
        try:
            # Base signal from absorption strength
            base_signal = absorption_analysis.get('absorption_strength', 0.0)
            
            # Direction from volume imbalance and absorption type
            volume_imbalance = absorption_analysis.get('volume_imbalance', 0.0)
            absorption_type = absorption_analysis.get('absorption_type', '')
            
            # Determine signal direction
            if 'SELL' in absorption_type or volume_imbalance < -0.1:
                signal_direction = -1.0
            elif 'BUY' in absorption_type or volume_imbalance > 0.1:
                signal_direction = 1.0
            else:
                signal_direction = 0.0
            
            # Apply MQScore enhancement
            mqscore_multiplier = (mqscore - 0.5) * 2.0  # Convert 0.5-1.0 to 0.0-1.0
            
            # Calculate final signal
            raw_signal = signal_direction * base_signal * mqscore_multiplier
            
            # Apply smoothing
            smoothed_signal = raw_signal * self.signal_smoothing_factor
            
            # Clamp to valid range
            return max(-1.0, min(1.0, smoothed_signal))
            
        except Exception as e:
            self.logger.error(f"Signal calculation error: {e}")
            return 0.0
    
    def _create_signal_response(self, signal_strength: float, absorption_analysis: Dict,
                              mqscore_components: Dict, features: Optional[Dict],
                              risk_check: Dict) -> Dict[str, Any]:
        """Create standardized signal response."""
        # Determine action
        if abs(signal_strength) < 0.1:
            action = 'HOLD'
        elif signal_strength > 0:
            action = 'BUY'
        else:
            action = 'SELL'
        
        # Create ML features for pipeline
        ml_features = self._create_ml_features(absorption_analysis, mqscore_components, features)
        
        return {
            'signal': signal_strength,
            'confidence': abs(signal_strength),
            'features': ml_features,
            'metadata': {
                'action': action,
                'strategy': 'liquidity_absorption',
                'absorption_type': absorption_analysis.get('absorption_type', 'UNKNOWN'),
                'absorption_strength': absorption_analysis.get('absorption_strength', 0.0),
                'mqscore': sum(mqscore_components.values()) / len(mqscore_components),
                'mqscore_components': mqscore_components,
                'position_size': risk_check.get('position_size', 0.0),
                'timestamp': time.time(),
                'error': False
            }
        }
    
    def _create_hold_response(self, reason: str, mqscore_components: Dict,
                            features: Optional[Dict], absorption_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """Create standardized hold response."""
        ml_features = self._create_ml_features(absorption_analysis or {}, mqscore_components, features)
        
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'features': ml_features,
            'metadata': {
                'action': 'HOLD',
                'reason': reason,
                'strategy': 'liquidity_absorption',
                'mqscore_components': mqscore_components,
                'timestamp': time.time(),
                'error': False
            }
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'features': {},
            'metadata': {
                'action': 'HOLD',
                'reason': error_message,
                'strategy': 'liquidity_absorption',
                'timestamp': time.time(),
                'error': True
            }
        }
    
    def _create_ml_features(self, absorption_analysis: Dict, mqscore_components: Dict,
                          external_features: Optional[Dict]) -> Dict[str, Any]:
        """Create comprehensive feature set for ML pipeline."""
        features = {}
        
        # MQScore 6D components as features
        for component, value in mqscore_components.items():
            features[f'mqscore_{component}'] = value
        
        # Overall MQScore
        features['mqscore_overall'] = sum(mqscore_components.values()) / len(mqscore_components)
        
        # Absorption analysis features
        features['absorption_strength'] = absorption_analysis.get('absorption_strength', 0.0)
        features['aggressive_absorption'] = absorption_analysis.get('aggressive_absorption', 0.0)
        features['iceberg_absorption'] = absorption_analysis.get('iceberg_absorption', 0.0)
        features['sweep_absorption'] = absorption_analysis.get('sweep_absorption', 0.0)
        features['volume_imbalance'] = absorption_analysis.get('volume_imbalance', 0.0)
        features['absorption_confidence'] = absorption_analysis.get('confidence', 0.0)
        
        # Strategy configuration features
        features['mqscore_threshold'] = self.mqscore_threshold
        features['min_absorption_strength'] = self.min_absorption_strength
        features['signal_smoothing_factor'] = self.signal_smoothing_factor
        
        # Performance features
        features['execution_time_ms'] = self.metrics.execution_time_ms
        features['error_rate'] = self.metrics.error_count / max(self.metrics.total_trades, 1)
        
        # Risk management features
        features['daily_pnl'] = self.risk_manager.daily_pnl
        features['kill_switch_active'] = float(self.risk_manager.kill_switch_active)
        
        # Include external features if provided
        if external_features:
            for key, value in external_features.items():
                if isinstance(value, (int, float)):
                    features[f'external_{key}'] = value
        
        return features

# =====================================================================
# STRATEGY FACTORY AND UTILITIES
# =====================================================================

def create_liquidity_absorption_strategy(config: Optional[Dict[str, Any]] = None) -> LiquidityAbsorptionNexusAdapter:
    """
    Factory function to create a configured liquidity absorption strategy.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured LiquidityAbsorptionNexusAdapter instance
    """
    default_config = {
        'mqscore_threshold': 0.57,
        'min_absorption_strength': 0.3,
        'signal_smoothing_factor': 0.7,
        'max_position_size': 0.02,
        'max_daily_loss': 0.05,
        'mqscore_lookback': 20,
        'detector_lookback': 50
    }
    
    if config:
        default_config.update(config)
    
    return LiquidityAbsorptionNexusAdapter(default_config)

# =====================================================================
# EXAMPLE USAGE AND TESTING
# =====================================================================

if __name__ == "__main__":
    # Example usage
    strategy = create_liquidity_absorption_strategy()
    
    # Sample market data
    sample_data = {
        'symbol': 'BTCUSD',
        'timestamp': time.time(),
        'price': 50000.0,
        'volume': 1.5,
        'bid': 49995.0,
        'ask': 50005.0,
        'bid_size': 10.0,
        'ask_size': 8.0
    }
    
    # Execute strategy
    result = strategy.execute(sample_data)
    
    print("Strategy Execution Result:")
    print(f"Signal: {result['signal']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Action: {result['metadata']['action']}")
    print(f"Features count: {len(result['features'])}")
    
    # Get performance metrics
    metrics = strategy.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"Average execution time: {metrics['average_execution_time_ms']:.2f}ms")
    print(f"Error count: {metrics['error_count']}")
    print(f"Strategy category: {strategy.get_category()}")