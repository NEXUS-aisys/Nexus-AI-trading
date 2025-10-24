"""
LVN (Low Volume Node) Breakout Strategy - NEXUS AI Compatible Implementation

Enhanced LVN breakout detection strategy with comprehensive risk management,
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
- Volume profile analysis for LVN/HVN detection

Components:
- LVNBreakoutNexusAdapter: Main strategy adapter class
- VolumeProfileAnalyzer: Volume profile and node detection
- BreakoutDetector: LVN breakout pattern recognition
- MQScore6D: Market quality scoring system
- RiskManager: Comprehensive risk management
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
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =====================================================================
# CORE ENUMS AND DATA STRUCTURES
# =====================================================================

class SignalType(Enum):
    """Signal types for LVN breakout detection."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class VolumeNodeType(Enum):
    """Types of volume nodes in profile analysis."""
    LVN = "LOW_VOLUME_NODE"    # Low Volume Node - breakout target
    HVN = "HIGH_VOLUME_NODE"   # High Volume Node - support/resistance
    NEUTRAL = "NEUTRAL"        # Normal volume area

class BreakoutType(Enum):
    """Types of LVN breakout patterns."""
    BULLISH_BREAKOUT = "BULLISH_BREAKOUT"
    BEARISH_BREAKOUT = "BEARISH_BREAKOUT"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    PENDING_BREAKOUT = "PENDING_BREAKOUT"

@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    timestamp: float
    price: float
    volume: float
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    close: float = 0.0
    bid: float = 0.0
    ask: float = 0.0

@dataclass
class VolumeNode:
    """Volume profile node data structure."""
    price_level: float
    volume: float
    node_type: VolumeNodeType
    strength: float  # 0.0 to 1.0
    trades_count: int = 0

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
        self.high_history = deque(maxlen=lookback_periods)
        self.low_history = deque(maxlen=lookback_periods)
        
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
        self.high_history.append(market_data.high or market_data.price)
        self.low_history.append(market_data.low or market_data.price)
    
    def _calculate_liquidity(self, market_data: MarketData) -> float:
        """Calculate liquidity component (20% weight)."""
        try:
            # Use volume as proxy for liquidity
            if not self.volume_history:
                return 0.5
            
            current_volume = market_data.volume
            avg_volume = statistics.mean(self.volume_history)
            
            # Normalize liquidity score (higher volume = better liquidity)
            if avg_volume > 0:
                liquidity_ratio = current_volume / avg_volume
                liquidity_score = min(1.0, liquidity_ratio / 2.0)  # Cap at 2x average
            else:
                liquidity_score = 0.5
            
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception:
            return 0.5
    
    def _calculate_volatility(self) -> float:
        """Calculate volatility component (20% weight)."""
        try:
            if len(self.high_history) < 2 or len(self.low_history) < 2:
                return 0.5
            
            # Calculate true range volatility
            true_ranges = []
            for i in range(1, len(self.high_history)):
                high = self.high_history[i]
                low = self.low_history[i]
                prev_close = self.price_history[i-1]
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)
            
            if not true_ranges:
                return 0.5
            
            # Average True Range
            atr = statistics.mean(true_ranges)
            current_price = self.price_history[-1]
            
            # Normalize volatility (lower volatility = higher quality for breakouts)
            if current_price > 0:
                volatility_pct = atr / current_price
                # Invert so lower volatility = higher score
                volatility_score = max(0.0, 1.0 - min(1.0, volatility_pct * 20))
            else:
                volatility_score = 0.5
            
            return volatility_score
            
        except Exception:
            return 0.5
    
    def _calculate_momentum(self) -> float:
        """Calculate momentum component (20% weight)."""
        try:
            if len(self.price_history) < 3:
                return 0.5
            
            # Calculate short-term momentum
            short_momentum = (self.price_history[-1] - self.price_history[-3]) / self.price_history[-3]
            
            # Calculate medium-term momentum if enough data
            if len(self.price_history) >= 10:
                medium_momentum = (self.price_history[-1] - self.price_history[-10]) / self.price_history[-10]
                momentum = (short_momentum + medium_momentum) / 2.0
            else:
                momentum = short_momentum
            
            # Normalize momentum score (consistent direction = higher quality)
            momentum_strength = abs(momentum)
            normalized_momentum = min(1.0, momentum_strength / 0.02)  # 2% move = full score
            
            return normalized_momentum
            
        except Exception:
            return 0.5
    
    def _calculate_imbalance(self, market_data: MarketData) -> float:
        """Calculate order imbalance component (15% weight)."""
        try:
            # Use bid/ask if available, otherwise use volume trend
            if market_data.bid > 0 and market_data.ask > 0:
                spread = market_data.ask - market_data.bid
                mid_price = (market_data.bid + market_data.ask) / 2.0
                
                if mid_price > 0:
                    spread_pct = spread / mid_price
                    # Lower spread = higher quality
                    imbalance_score = max(0.0, 1.0 - min(1.0, spread_pct * 100))
                else:
                    imbalance_score = 0.5
            else:
                # Use volume momentum as proxy
                if len(self.volume_history) >= 3:
                    recent_vol_trend = statistics.mean(list(self.volume_history)[-3:])
                    older_vol_trend = statistics.mean(list(self.volume_history)[:-3]) if len(self.volume_history) > 3 else recent_vol_trend
                    
                    if older_vol_trend > 0:
                        vol_momentum = recent_vol_trend / older_vol_trend
                        imbalance_score = min(1.0, vol_momentum / 2.0)
                    else:
                        imbalance_score = 0.5
                else:
                    imbalance_score = 0.5
            
            return max(0.0, min(1.0, imbalance_score))
            
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
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Normalize trend strength
                avg_price = statistics.mean(y_values)
                if avg_price > 0:
                    trend_strength = abs(slope) / avg_price * 100
                    normalized_trend = min(1.0, trend_strength)
                else:
                    normalized_trend = 0.5
            else:
                normalized_trend = 0.5
            
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
            if len(price_changes) > 1:
                noise_ratio = statistics.stdev(price_changes) / max(avg_change, 0.0001)
            else:
                noise_ratio = 0.0
            
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
# VOLUME PROFILE ANALYZER
# =====================================================================

class VolumeProfileAnalyzer:
    """Advanced volume profile analysis for LVN/HVN detection."""
    
    def __init__(self, lookback_periods: int = 100, price_bins: int = 50):
        self.lookback_periods = lookback_periods
        self.price_bins = price_bins
        self.price_history = deque(maxlen=lookback_periods)
        self.volume_history = deque(maxlen=lookback_periods)
        self.high_history = deque(maxlen=lookback_periods)
        self.low_history = deque(maxlen=lookback_periods)
        
        # Volume profile data
        self.volume_profile = defaultdict(float)  # price_level -> total_volume
        self.volume_nodes = []  # List of VolumeNode objects
        
        self._lock = threading.RLock()
    
    def update_profile(self, market_data: MarketData):
        """Update volume profile with new market data."""
        with self._lock:
            # Update historical data
            self.price_history.append(market_data.price)
            self.volume_history.append(market_data.volume)
            self.high_history.append(market_data.high or market_data.price)
            self.low_history.append(market_data.low or market_data.price)
            
            # Update volume profile
            price_level = round(market_data.price, 2)  # Round to 2 decimal places
            self.volume_profile[price_level] += market_data.volume
            
            # Recalculate nodes if we have enough data
            if len(self.price_history) >= 20:
                self._calculate_volume_nodes()
    
    def _calculate_volume_nodes(self):
        """Calculate LVN and HVN nodes from volume profile."""
        try:
            if not self.volume_profile:
                return
            
            # Get price range
            min_price = min(self.price_history)
            max_price = max(self.price_history)
            price_range = max_price - min_price
            
            if price_range <= 0:
                return
            
            # Create price bins
            bin_size = price_range / self.price_bins
            binned_volumes = defaultdict(float)
            
            # Aggregate volume into bins
            for price, volume in self.volume_profile.items():
                if min_price <= price <= max_price:
                    bin_index = int((price - min_price) / bin_size)
                    bin_price = min_price + (bin_index * bin_size)
                    binned_volumes[bin_price] += volume
            
            if not binned_volumes:
                return
            
            # Calculate statistics
            volumes = list(binned_volumes.values())
            mean_volume = statistics.mean(volumes)
            std_volume = statistics.stdev(volumes) if len(volumes) > 1 else 0
            
            # Identify nodes
            self.volume_nodes = []
            
            for price, volume in binned_volumes.items():
                # Calculate z-score
                if std_volume > 0:
                    z_score = (volume - mean_volume) / std_volume
                else:
                    z_score = 0
                
                # Classify node type
                if z_score > 1.5:  # High volume node
                    node_type = VolumeNodeType.HVN
                    strength = min(1.0, (z_score - 1.5) / 2.0)  # Normalize strength
                elif z_score < -1.0:  # Low volume node
                    node_type = VolumeNodeType.LVN
                    strength = min(1.0, abs(z_score + 1.0) / 2.0)  # Normalize strength
                else:
                    node_type = VolumeNodeType.NEUTRAL
                    strength = 0.5
                
                node = VolumeNode(
                    price_level=price,
                    volume=volume,
                    node_type=node_type,
                    strength=strength
                )
                
                self.volume_nodes.append(node)
            
            # Sort nodes by price
            self.volume_nodes.sort(key=lambda x: x.price_level)
            
        except Exception as e:
            logger.error(f"Volume node calculation error: {e}")
    
    def get_nearest_lvn(self, current_price: float) -> Optional[VolumeNode]:
        """Find the nearest LVN to current price."""
        try:
            lvn_nodes = [node for node in self.volume_nodes if node.node_type == VolumeNodeType.LVN]
            
            if not lvn_nodes:
                return None
            
            # Find closest LVN
            closest_lvn = min(lvn_nodes, key=lambda x: abs(x.price_level - current_price))
            
            return closest_lvn
            
        except Exception as e:
            logger.error(f"Nearest LVN search error: {e}")
            return None
    
    def get_support_resistance_levels(self, current_price: float) -> Dict[str, List[float]]:
        """Get support and resistance levels from HVN nodes."""
        try:
            hvn_nodes = [node for node in self.volume_nodes if node.node_type == VolumeNodeType.HVN]
            
            support_levels = [node.price_level for node in hvn_nodes if node.price_level < current_price]
            resistance_levels = [node.price_level for node in hvn_nodes if node.price_level > current_price]
            
            # Sort by proximity to current price
            support_levels.sort(reverse=True)  # Closest support first
            resistance_levels.sort()  # Closest resistance first
            
            return {
                'support': support_levels[:5],  # Top 5 support levels
                'resistance': resistance_levels[:5]  # Top 5 resistance levels
            }
            
        except Exception as e:
            logger.error(f"Support/resistance calculation error: {e}")
            return {'support': [], 'resistance': []}

# =====================================================================
# LVN BREAKOUT DETECTOR
# =====================================================================

class LVNBreakoutDetector:
    """Advanced LVN breakout pattern detection."""
    
    def __init__(self, volume_analyzer: VolumeProfileAnalyzer):
        self.volume_analyzer = volume_analyzer
        self.breakout_threshold = 0.02  # 2% price movement threshold
        self.volume_confirmation_multiplier = 1.5  # Volume must be 1.5x average
        self.lookback_periods = 20
        
        # Historical tracking
        self.price_history = deque(maxlen=self.lookback_periods)
        self.volume_history = deque(maxlen=self.lookback_periods)
        self.breakout_history = deque(maxlen=50)  # Track recent breakouts
        
        self._lock = threading.RLock()
    
    def detect_breakout(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Detect LVN breakout patterns in market data.
        
        Returns:
            Dict containing breakout analysis results
        """
        with self._lock:
            try:
                # Update historical data
                self.price_history.append(market_data.price)
                self.volume_history.append(market_data.volume)
                
                if len(self.price_history) < 10:
                    return self._get_neutral_result()
                
                # Find nearest LVN
                nearest_lvn = self.volume_analyzer.get_nearest_lvn(market_data.price)
                
                if not nearest_lvn:
                    return self._get_neutral_result()
                
                # Calculate distance to LVN
                lvn_distance = abs(market_data.price - nearest_lvn.price_level) / nearest_lvn.price_level
                
                # Check if we're approaching or at an LVN
                if lvn_distance > 0.05:  # More than 5% away from LVN
                    return self._get_neutral_result()
                
                # Analyze breakout potential
                breakout_analysis = self._analyze_breakout_potential(market_data, nearest_lvn)
                
                # Check volume confirmation
                volume_confirmation = self._check_volume_confirmation(market_data.volume)
                
                # Determine breakout type and strength
                breakout_type, breakout_strength = self._determine_breakout_type(
                    market_data, nearest_lvn, breakout_analysis, volume_confirmation
                )
                
                # Calculate confidence
                confidence = self._calculate_breakout_confidence(
                    breakout_analysis, volume_confirmation, nearest_lvn.strength
                )
                
                result = {
                    'breakout_detected': breakout_strength > 0.3,
                    'breakout_type': breakout_type.value,
                    'breakout_strength': breakout_strength,
                    'confidence': confidence,
                    'lvn_price': nearest_lvn.price_level,
                    'lvn_distance': lvn_distance,
                    'volume_confirmation': volume_confirmation,
                    'support_resistance': self.volume_analyzer.get_support_resistance_levels(market_data.price)
                }
                
                # Record breakout if significant
                if breakout_strength > 0.5:
                    self.breakout_history.append({
                        'timestamp': market_data.timestamp,
                        'price': market_data.price,
                        'type': breakout_type.value,
                        'strength': breakout_strength
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"Breakout detection error: {e}")
                return self._get_neutral_result()
    
    def _analyze_breakout_potential(self, market_data: MarketData, lvn_node: VolumeNode) -> Dict[str, float]:
        """Analyze the potential for breakout from LVN."""
        try:
            analysis = {
                'price_momentum': 0.0,
                'volume_momentum': 0.0,
                'trend_alignment': 0.0,
                'volatility_expansion': 0.0
            }
            
            if len(self.price_history) < 5:
                return analysis
            
            # Price momentum
            recent_prices = list(self.price_history)[-5:]
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            analysis['price_momentum'] = min(1.0, abs(price_change) / 0.02)  # Normalize to 2%
            
            # Volume momentum
            if len(self.volume_history) >= 5:
                recent_volumes = list(self.volume_history)[-5:]
                avg_recent_volume = statistics.mean(recent_volumes)
                avg_historical_volume = statistics.mean(list(self.volume_history)[:-5]) if len(self.volume_history) > 5 else avg_recent_volume
                
                if avg_historical_volume > 0:
                    volume_ratio = avg_recent_volume / avg_historical_volume
                    analysis['volume_momentum'] = min(1.0, volume_ratio / 2.0)  # Normalize to 2x
            
            # Trend alignment (is breakout in direction of trend?)
            if len(self.price_history) >= 10:
                trend_slope = (recent_prices[-1] - list(self.price_history)[-10]) / list(self.price_history)[-10]
                breakout_direction = 1 if market_data.price > lvn_node.price_level else -1
                trend_direction = 1 if trend_slope > 0 else -1
                
                analysis['trend_alignment'] = 1.0 if trend_direction == breakout_direction else 0.0
            
            # Volatility expansion
            if len(self.price_history) >= 10:
                recent_volatility = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
                historical_volatility = statistics.stdev(list(self.price_history)[:-5]) if len(self.price_history) > 5 else recent_volatility
                
                if historical_volatility > 0:
                    volatility_ratio = recent_volatility / historical_volatility
                    analysis['volatility_expansion'] = min(1.0, volatility_ratio / 2.0)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Breakout analysis error: {e}")
            return {'price_momentum': 0.0, 'volume_momentum': 0.0, 'trend_alignment': 0.0, 'volatility_expansion': 0.0}
    
    def _check_volume_confirmation(self, current_volume: float) -> float:
        """Check if volume confirms the breakout."""
        try:
            if len(self.volume_history) < 5:
                return 0.5
            
            avg_volume = statistics.mean(self.volume_history)
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                # Volume confirmation score (1.5x average = full confirmation)
                confirmation = min(1.0, volume_ratio / self.volume_confirmation_multiplier)
            else:
                confirmation = 0.5
            
            return confirmation
            
        except Exception:
            return 0.5
    
    def _determine_breakout_type(self, market_data: MarketData, lvn_node: VolumeNode, 
                               analysis: Dict[str, float], volume_confirmation: float) -> Tuple[BreakoutType, float]:
        """Determine the type and strength of breakout."""
        try:
            # Determine direction
            if market_data.price > lvn_node.price_level:
                base_type = BreakoutType.BULLISH_BREAKOUT
            else:
                base_type = BreakoutType.BEARISH_BREAKOUT
            
            # Calculate overall strength
            strength_components = [
                analysis.get('price_momentum', 0.0) * 0.3,
                analysis.get('volume_momentum', 0.0) * 0.25,
                analysis.get('trend_alignment', 0.0) * 0.25,
                analysis.get('volatility_expansion', 0.0) * 0.1,
                volume_confirmation * 0.1
            ]
            
            overall_strength = sum(strength_components)
            
            # Determine final type based on strength
            if overall_strength > 0.7:
                breakout_type = base_type
            elif overall_strength > 0.3:
                breakout_type = BreakoutType.PENDING_BREAKOUT
            else:
                breakout_type = BreakoutType.FALSE_BREAKOUT
            
            return breakout_type, overall_strength
            
        except Exception as e:
            logger.error(f"Breakout type determination error: {e}")
            return BreakoutType.FALSE_BREAKOUT, 0.0
    
    def _calculate_breakout_confidence(self, analysis: Dict[str, float], 
                                     volume_confirmation: float, lvn_strength: float) -> float:
        """Calculate confidence in breakout signal."""
        try:
            # Weight different factors
            confidence_factors = [
                analysis.get('price_momentum', 0.0) * 0.25,
                analysis.get('volume_momentum', 0.0) * 0.20,
                analysis.get('trend_alignment', 0.0) * 0.20,
                volume_confirmation * 0.20,
                lvn_strength * 0.15  # Stronger LVN = more reliable breakout
            ]
            
            confidence = sum(confidence_factors)
            
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.0
    
    def _get_neutral_result(self) -> Dict[str, Any]:
        """Return neutral result when detection fails."""
        return {
            'breakout_detected': False,
            'breakout_type': BreakoutType.FALSE_BREAKOUT.value,
            'breakout_strength': 0.0,
            'confidence': 0.0,
            'lvn_price': 0.0,
            'lvn_distance': 1.0,
            'volume_confirmation': 0.0,
            'support_resistance': {'support': [], 'resistance': []}
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
# MAIN STRATEGY ADAPTER CLASS
# =====================================================================

class LVNBreakoutNexusAdapter:
    """
    NEXUS-compliant LVN Breakout Strategy Adapter.
    
    Implements the standard NEXUS adapter interface with:
    - Standardized execute() method
    - MQScore 6D integration
    - Thread-safe operations
    - Comprehensive error handling
    - ML pipeline feature engineering
    - Volume profile analysis for LVN breakout detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the strategy adapter with configuration."""
        self.config = config or {}
        
        # Core components
        self.mqscore = MQScore6D(lookback_periods=self.config.get('mqscore_lookback', 20))
        self.volume_analyzer = VolumeProfileAnalyzer(
            lookback_periods=self.config.get('volume_lookback', 100),
            price_bins=self.config.get('price_bins', 50)
        )
        self.breakout_detector = LVNBreakoutDetector(self.volume_analyzer)
        self.risk_manager = RiskManager(
            max_position_size=self.config.get('max_position_size', 0.02),
            max_daily_loss=self.config.get('max_daily_loss', 0.05)
        )
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.execution_times = deque(maxlen=100)
        
        # Configuration parameters
        self.mqscore_threshold = self.config.get('mqscore_threshold', 0.57)
        self.min_breakout_strength = self.config.get('min_breakout_strength', 0.4)
        self.signal_smoothing_factor = self.config.get('signal_smoothing_factor', 0.8)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("LVNBreakoutNexusAdapter initialized")
    
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
                
                # Update volume profile
                self.volume_analyzer.update_profile(normalized_data)
                
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
                
                # Detect LVN breakout patterns
                breakout_analysis = self.breakout_detector.detect_breakout(normalized_data)
                
                # Check breakout strength threshold
                breakout_strength = breakout_analysis.get('breakout_strength', 0.0)
                if breakout_strength < self.min_breakout_strength:
                    return self._create_hold_response(
                        f"Breakout strength below threshold: {breakout_strength:.3f}",
                        mqscore_components,
                        features,
                        breakout_analysis
                    )
                
                # Generate trading signal
                signal_strength = self._calculate_signal_strength(
                    breakout_analysis, mqscore, mqscore_components
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
                        breakout_analysis
                    )
                
                # Create final response
                response = self._create_signal_response(
                    signal_strength,
                    breakout_analysis,
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
        return "VOLUME_PROFILE_BREAKOUT"
    
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
                'min_breakout_strength': self.min_breakout_strength,
                'volume_nodes_count': len(self.volume_analyzer.volume_nodes),
                'lvn_count': len([n for n in self.volume_analyzer.volume_nodes if n.node_type == VolumeNodeType.LVN]),
                'hvn_count': len([n for n in self.volume_analyzer.volume_nodes if n.node_type == VolumeNodeType.HVN]),
                'recent_breakouts': len(self.breakout_detector.breakout_history)
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
                    price=float(market_data.get('price', market_data.get('close', 0.0))),
                    volume=float(market_data.get('volume', 0.0)),
                    high=float(market_data.get('high', market_data.get('price', 0.0))),
                    low=float(market_data.get('low', market_data.get('price', 0.0))),
                    open=float(market_data.get('open', market_data.get('price', 0.0))),
                    close=float(market_data.get('close', market_data.get('price', 0.0))),
                    bid=float(market_data.get('bid', 0.0)),
                    ask=float(market_data.get('ask', 0.0))
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
                        high=float(data_list[2]) if len(data_list) > 2 else float(data_list[0]),
                        low=float(data_list[3]) if len(data_list) > 3 else float(data_list[0])
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Market data normalization error: {e}")
            return None
    
    def _calculate_signal_strength(self, breakout_analysis: Dict, mqscore: float, 
                                 mqscore_components: Dict) -> float:
        """Calculate final signal strength from all components."""
        try:
            # Base signal from breakout strength
            base_signal = breakout_analysis.get('breakout_strength', 0.0)
            
            # Direction from breakout type
            breakout_type = breakout_analysis.get('breakout_type', '')
            
            # Determine signal direction
            if 'BEARISH' in breakout_type:
                signal_direction = -1.0
            elif 'BULLISH' in breakout_type:
                signal_direction = 1.0
            else:
                signal_direction = 0.0
            
            # Apply MQScore enhancement
            mqscore_multiplier = (mqscore - 0.5) * 2.0  # Convert 0.5-1.0 to 0.0-1.0
            
            # Apply confidence weighting
            confidence_weight = breakout_analysis.get('confidence', 0.5)
            
            # Calculate final signal
            raw_signal = signal_direction * base_signal * mqscore_multiplier * confidence_weight
            
            # Apply smoothing
            smoothed_signal = raw_signal * self.signal_smoothing_factor
            
            # Clamp to valid range
            return max(-1.0, min(1.0, smoothed_signal))
            
        except Exception as e:
            self.logger.error(f"Signal calculation error: {e}")
            return 0.0
    
    def _create_signal_response(self, signal_strength: float, breakout_analysis: Dict,
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
        ml_features = self._create_ml_features(breakout_analysis, mqscore_components, features)
        
        return {
            'signal': signal_strength,
            'confidence': abs(signal_strength),
            'features': ml_features,
            'metadata': {
                'action': action,
                'strategy': 'lvn_breakout',
                'breakout_type': breakout_analysis.get('breakout_type', 'UNKNOWN'),
                'breakout_strength': breakout_analysis.get('breakout_strength', 0.0),
                'lvn_price': breakout_analysis.get('lvn_price', 0.0),
                'mqscore': sum(mqscore_components.values()) / len(mqscore_components),
                'mqscore_components': mqscore_components,
                'position_size': risk_check.get('position_size', 0.0),
                'support_resistance': breakout_analysis.get('support_resistance', {}),
                'timestamp': time.time(),
                'error': False
            }
        }
    
    def _create_hold_response(self, reason: str, mqscore_components: Dict,
                            features: Optional[Dict], breakout_analysis: Optional[Dict] = None) -> Dict[str, Any]:
        """Create standardized hold response."""
        ml_features = self._create_ml_features(breakout_analysis or {}, mqscore_components, features)
        
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'features': ml_features,
            'metadata': {
                'action': 'HOLD',
                'reason': reason,
                'strategy': 'lvn_breakout',
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
                'strategy': 'lvn_breakout',
                'timestamp': time.time(),
                'error': True
            }
        }
    
    def _create_ml_features(self, breakout_analysis: Dict, mqscore_components: Dict,
                          external_features: Optional[Dict]) -> Dict[str, Any]:
        """Create comprehensive feature set for ML pipeline."""
        features = {}
        
        # MQScore 6D components as features
        for component, value in mqscore_components.items():
            features[f'mqscore_{component}'] = value
        
        # Overall MQScore
        features['mqscore_overall'] = sum(mqscore_components.values()) / len(mqscore_components)
        
        # Breakout analysis features
        features['breakout_strength'] = breakout_analysis.get('breakout_strength', 0.0)
        features['breakout_confidence'] = breakout_analysis.get('confidence', 0.0)
        features['lvn_distance'] = breakout_analysis.get('lvn_distance', 1.0)
        features['volume_confirmation'] = breakout_analysis.get('volume_confirmation', 0.0)
        
        # Volume profile features
        features['volume_nodes_count'] = len(self.volume_analyzer.volume_nodes)
        features['lvn_count'] = len([n for n in self.volume_analyzer.volume_nodes if n.node_type == VolumeNodeType.LVN])
        features['hvn_count'] = len([n for n in self.volume_analyzer.volume_nodes if n.node_type == VolumeNodeType.HVN])
        
        # Support/resistance features
        sr_levels = breakout_analysis.get('support_resistance', {})
        features['support_levels_count'] = len(sr_levels.get('support', []))
        features['resistance_levels_count'] = len(sr_levels.get('resistance', []))
        
        # Strategy configuration features
        features['mqscore_threshold'] = self.mqscore_threshold
        features['min_breakout_strength'] = self.min_breakout_strength
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

def create_lvn_breakout_strategy(config: Optional[Dict[str, Any]] = None) -> LVNBreakoutNexusAdapter:
    """
    Factory function to create a configured LVN breakout strategy.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured LVNBreakoutNexusAdapter instance
    """
    default_config = {
        'mqscore_threshold': 0.57,
        'min_breakout_strength': 0.4,
        'signal_smoothing_factor': 0.8,
        'max_position_size': 0.02,
        'max_daily_loss': 0.05,
        'mqscore_lookback': 20,
        'volume_lookback': 100,
        'price_bins': 50
    }
    
    if config:
        default_config.update(config)
    
    return LVNBreakoutNexusAdapter(default_config)

# =====================================================================
# EXAMPLE USAGE AND TESTING
# =====================================================================

if __name__ == "__main__":
    # Example usage
    strategy = create_lvn_breakout_strategy()
    
    # Sample market data with OHLCV
    sample_data = {
        'symbol': 'BTCUSD',
        'timestamp': time.time(),
        'price': 50000.0,
        'volume': 2.5,
        'high': 50100.0,
        'low': 49900.0,
        'open': 49950.0,
        'close': 50000.0,
        'bid': 49995.0,
        'ask': 50005.0
    }
    
    # Feed some historical data first
    print("Building volume profile with sample data...")
    for i in range(100):
        # Generate sample historical data
        price_variation = (i - 50) * 10  # Price range around base
        volume_variation = np.random.uniform(0.5, 3.0)  # Random volume
        
        historical_data = {
            'symbol': 'BTCUSD',
            'timestamp': time.time() - (100 - i) * 60,  # 1 minute intervals
            'price': 50000.0 + price_variation + np.random.uniform(-50, 50),
            'volume': volume_variation,
            'high': 50000.0 + price_variation + 25,
            'low': 50000.0 + price_variation - 25,
            'open': 50000.0 + price_variation,
            'close': 50000.0 + price_variation
        }
        
        # Execute strategy (will build profile)
        result = strategy.execute(historical_data)
    
    # Now execute with current data
    result = strategy.execute(sample_data)
    
    print("Strategy Execution Result:")
    print(f"Signal: {result['signal']:.4f}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Action: {result['metadata']['action']}")
    print(f"Features count: {len(result['features'])}")
    print(f"Breakout type: {result['metadata'].get('breakout_type', 'N/A')}")
    
    # Get performance metrics
    metrics = strategy.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"Average execution time: {metrics['average_execution_time_ms']:.2f}ms")
    print(f"Error count: {metrics['error_count']}")
    print(f"Volume nodes: {metrics['volume_nodes_count']} (LVN: {metrics['lvn_count']}, HVN: {metrics['hvn_count']})")
    print(f"Strategy category: {strategy.get_category()}")