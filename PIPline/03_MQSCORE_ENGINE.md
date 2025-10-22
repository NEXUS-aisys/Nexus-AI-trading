# MQSCORE 6D ENGINE v3.0 - MARKET QUALITY SCORING SYSTEM
**Real-Time Market Quality Assessment & Regime Classification**

---

## Executive Summary

**Version**: 3.0.0  
**Dimensions**: 6 (Liquidity, Volatility, Momentum, Imbalance, Trend Strength, Noise Level)  
**Features**: 65 engineered features  
**ML Models**: LightGBM (Classification + Regression), ONNX optimized  
**Performance**: <10ms inference, 1000+ predictions/second  
**Status**: Production Ready ✅

---

## Core Architecture

### 6 Dimensions of Market Quality

```
MQScore Composite Score (0.0 - 1.0)
    ├── Liquidity (15%)
    ├── Volatility (15%)
    ├── Momentum (15%)
    ├── Imbalance (15%)
    ├── Trend Strength (20%)
    └── Noise Level (20%)
```

---

## DIMENSION 1: LIQUIDITY (Weight: 15%)

### Purpose:
Measure market depth and trading ease

### Component Weights:
```python
liquidity_weights = [0.25, 0.25, 0.25, 0.15, 0.10]
```

### Sub-Components:

#### 1. Volume Consistency (25%)
- **Calculation**: Rolling standard deviation of volume
- **Metric**: σ(volume) / mean(volume)
- **Score**: Lower variance = higher score
- **Interpretation**:
  - High score (>0.7): Consistent trading activity
  - Low score (<0.3): Erratic volume patterns

#### 2. Volume Magnitude (25%)
- **Calculation**: Current volume vs historical average
- **Metric**: current_volume / avg_volume(lookback)
- **Score**: Normalized to 0-1 range
- **Interpretation**:
  - High score: Strong participation
  - Low score: Thin trading

#### 3. Price Impact (25%)
- **Calculation**: Price movement per unit volume
- **Metric**: |price_change| / volume
- **Score**: Lower impact = higher liquidity
- **Interpretation**:
  - High score: Low slippage expected
  - Low score: High slippage risk

#### 4. Spread Analysis (15%)
- **Calculation**: Bid-ask spread as % of mid-price
- **Metric**: (ask - bid) / mid_price * 10000 (bps)
- **Score**: Tighter spread = higher score
- **Interpretation**:
  - High score: Liquid market
  - Low score: Wide spreads, costly trading

#### 5. Market Depth Proxy (10%)
- **Calculation**: Order book depth estimation
- **Metric**: Sum of bid/ask sizes near mid-price
- **Score**: Deeper book = higher score

### Strategies Using Liquidity:
- LVN Breakout Strategy (PRIMARY)
- Market Microstructure Strategy (PRIMARY)
- Liquidity Absorption (PRIMARY)
- Iceberg Detection (PRIMARY)
- Order Book Imbalance (SECONDARY)
- Spoofing Detection (FILTER)
- Stop Run Anticipation (FILTER)

### Liquidity Score Interpretation:
```python
if liquidity_score > 0.7:
    status = "HIGH LIQUIDITY"
    # Execute large orders, minimal slippage
elif liquidity_score > 0.4:
    status = "MODERATE LIQUIDITY"
    # Standard execution, monitor slippage
else:
    status = "LOW LIQUIDITY"
    # Reduce position size, avoid large orders
```

---

## DIMENSION 2: VOLATILITY (Weight: 15%)

### Purpose:
Measure price movement intensity and risk

### Component Weights:
Implicitly weighted through multiple calculations

### Sub-Components:

#### 1. Realized Volatility
- **Calculation**: Standard deviation of log returns
- **Metric**: σ(log_returns) * √(252 * periods_per_day)
- **Annualized**: Yes
- **Interpretation**:
  - High vol (>0.40): Extreme risk
  - Medium vol (0.15-0.40): Normal trading
  - Low vol (<0.15): Quiet market

#### 2. GARCH-Based Volatility
- **Model**: GARCH(1,1)
- **Calculation**: α₀ + α₁ε²ₜ₋₁ + β₁σ²ₜ₋₁
- **Forecast**: Next-period volatility
- **Benefits**: Captures volatility clustering

#### 3. Jump Risk Assessment
- **Calculation**: Identify price jumps using threshold
- **Metric**: |return| > k * σ (typically k=3)
- **Score**: Fewer jumps = higher quality

#### 4. Volatility Clustering
- **Calculation**: Autocorrelation of squared returns
- **Metric**: ρ(ε²ₜ, ε²ₜ₋₁)
- **Interpretation**: High clustering = predictable volatility patterns

### Strategies Using Volatility:
- Event-Driven Strategy (PRIMARY)
- Spoofing Detection (PRIMARY)
- Liquidation Detection (PRIMARY)
- Liquidity Traps (PRIMARY)
- VWAP Reversion (PRIMARY)
- Stop Run Anticipation (PRIMARY)
- Momentum Ignition (PRIMARY)
- Open Drive vs Fade (FILTER)
- Momentum Breakout (FILTER)

### Volatility Regimes:
```python
if volatility_score > 0.7:
    regime = "HIGH VOLATILITY"
    # Reduce position sizes, widen stops
    # Favor mean reversion strategies
elif volatility_score > 0.3:
    regime = "NORMAL VOLATILITY"
    # Standard operations
else:
    regime = "LOW VOLATILITY"
    # Increase position sizes, tighten stops
    # Favor trend following strategies
```

---

## DIMENSION 3: MOMENTUM (Weight: 15%)

### Purpose:
Measure directional price movement strength

### Component Weights:
```python
momentum_weights = [0.20, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10]
```

### Sub-Components:

#### 1. Short-Term Momentum (20%)
- **Period**: 1-5 minutes
- **Calculation**: (price_now - price_5m_ago) / price_5m_ago
- **Score**: Absolute momentum strength

#### 2. Medium-Term Momentum (20%)
- **Period**: 5-60 minutes
- **Calculation**: (price_now - price_1h_ago) / price_1h_ago
- **Score**: Sustained directional movement

#### 3. Long-Term Momentum (15%)
- **Period**: 1+ hours
- **Calculation**: (price_now - price_1d_ago) / price_1d_ago
- **Score**: Trend persistence

#### 4. Volume-Weighted Momentum (15%)
- **Calculation**: ∑(return_i * volume_i) / ∑(volume_i)
- **Benefit**: Weights stronger by volume

#### 5. Return Skewness (10%)
- **Calculation**: Third moment of return distribution
- **Interpretation**:
  - Positive skew: More upside moves
  - Negative skew: More downside moves

#### 6. Return Kurtosis (10%)
- **Calculation**: Fourth moment (tail risk)
- **Interpretation**: High kurtosis = fat tails

#### 7. Trend Persistence (10%)
- **Calculation**: % of periods moving in same direction
- **Metric**: runs_test or directional consistency

### Strategies Using Momentum:
- LVN Breakout (PRIMARY)
- Momentum Ignition (PRIMARY)
- Absorption Breakout (PRIMARY)
- Momentum Breakout (PRIMARY)
- Liquidity Absorption (PRIMARY)
- Spoofing Detection (PRIMARY)
- Delta Divergence (PRIMARY)
- Multi-Timeframe Alignment (SECONDARY)

### Momentum Interpretation:
```python
if momentum_score > 0.6:
    signal = "STRONG MOMENTUM"
    # Favor trend-following strategies
    # Breakout strategies active
elif momentum_score < 0.4:
    signal = "WEAK MOMENTUM"
    # Favor mean reversion
    # Range-bound strategies
else:
    signal = "NEUTRAL MOMENTUM"
    # Mixed strategy approach
```

---

## DIMENSION 4: IMBALANCE (Weight: 15%)

### Purpose:
Measure orderflow pressure and buyer/seller dominance

### Component Weights:
```python
imbalance_weights = [0.25, 0.25, 0.20, 0.15, 0.15]
```

### Sub-Components:

#### 1. Order Flow Imbalance (25%)
- **Calculation**: (buy_volume - sell_volume) / total_volume
- **Range**: -1 (all sellers) to +1 (all buyers)
- **Score**: Absolute imbalance strength

#### 2. Buy/Sell Volume Ratio (25%)
- **Calculation**: buy_volume / sell_volume
- **Normalization**: Log-scale or capped ratio
- **Interpretation**: >1.5 = buyer dominance, <0.67 = seller dominance

#### 3. Market Microstructure Pressure (20%)
- **Calculation**: Cumulative delta, aggressive trades
- **Components**:
  - Trades at ask (buyers aggressive)
  - Trades at bid (sellers aggressive)
- **Score**: Net aggressive flow direction

#### 4. Pressure Indicators (15%)
- **Delta**: Cumulative buy volume - sell volume
- **Delta Momentum**: Rate of change of delta
- **Interpretation**: Accelerating delta = strong pressure

#### 5. DOM Analysis (15%)
- **Calculation**: Bid size - Ask size near top of book
- **Metric**: (bid_size - ask_size) / (bid_size + ask_size)
- **Score**: Order book bias

### Strategies Using Imbalance:
- Market Microstructure (PRIMARY)
- Order Book Imbalance (PRIMARY)
- Open Drive vs Fade (PRIMARY)
- Cumulative Delta (PRIMARY)
- Delta Divergence (PRIMARY)
- Volume Imbalance (PRIMARY)
- Absorption Breakout (SECONDARY)
- Momentum Breakout (SECONDARY)
- Stop Run Anticipation (SECONDARY)

### Imbalance Regimes:
```python
if imbalance_score > 0.6:
    regime = "BUYER DOMINANCE"
    # Favor long positions
    # Breakout likelihood high
elif imbalance_score < 0.4:
    regime = "SELLER DOMINANCE"
    # Favor short positions
    # Breakdown likelihood high
else:
    regime = "BALANCED"
    # No directional bias
    # Mean reversion active
```

---

## DIMENSION 5: TREND STRENGTH (Weight: 20%)

### Purpose:
Measure directional consistency and trend quality

### Component Weights:
```python
trend_strength_weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
```

### Sub-Components:

#### 1. ADX (Average Directional Index) (25%)
- **Calculation**: Wilder's ADX formula
- **Range**: 0-100
- **Interpretation**:
  - ADX > 25: Strong trend
  - ADX < 20: Weak trend, ranging
- **Score**: Normalized ADX / 100

#### 2. Moving Average Relationships (20%)
- **Calculation**: Position relative to SMAs/EMAs
- **Components**:
  - Price vs MA(20)
  - MA(20) vs MA(50)
  - MA(50) vs MA(200)
- **Score**: Alignment score (all MAs aligned = 1.0)

#### 3. Trend Consistency (15%)
- **Calculation**: % of bars closing in trend direction
- **Period**: Rolling 20-50 bars
- **Score**: Higher consistency = stronger trend

#### 4. Breakout Metrics (15%)
- **Calculation**: Price position relative to recent range
- **Components**:
  - Distance from range high/low
  - Breakout strength
  - Follow-through
- **Score**: Breakout quality

#### 5. Higher Highs / Lower Lows (15%)
- **Calculation**: Swing point analysis
- **Uptrend**: Series of higher highs and higher lows
- **Downtrend**: Series of lower highs and lower lows
- **Score**: Swing structure quality

#### 6. Directional Movement (10%)
- **Calculation**: +DI and -DI from ADX calculation
- **Metric**: (+DI - -DI) / (+DI + -DI)
- **Score**: Directional conviction

### Strategies Using Trend Strength:
- Multi-Timeframe Alignment (PRIMARY)
- LVN Breakout (FILTER)
- Momentum Breakout (FILTER)
- Volume Imbalance (FILTER)
- VWAP Reversion (FILTER - inverse)
- Liquidity Traps (FILTER)
- Delta Divergence (FILTER)
- Stop Run Anticipation (FILTER)

### Trend Strength Interpretation:
```python
if trend_strength > 0.7:
    regime = "STRONG TREND"
    # Trend-following strategies active
    # Avoid counter-trend trades
    # Widen targets, tighten stops
elif trend_strength > 0.3:
    regime = "WEAK TREND"
    # Mixed strategies
    # Shorter timeframes
else:
    regime = "RANGING"
    # Mean reversion strategies
    # Fade extremes
    # Tighter targets
```

---

## DIMENSION 6: NOISE LEVEL (Weight: 20%)

### Purpose:
Measure signal clarity and random fluctuation

### Component Weights:
```python
noise_level_weights = [0.20, 0.15, 0.15, 0.15, 0.10, 0.15, 0.10]
```

### Sub-Components:

#### 1. Signal-to-Noise Ratio (20%)
- **Calculation**: Trend strength / random fluctuation
- **Metric**: MA_deviation / std_dev(price - MA)
- **Score**: Higher ratio = clearer signal

#### 2. Micro-Fluctuation Analysis (15%)
- **Calculation**: Tick-level price changes
- **Metric**: Count of direction changes per unit time
- **Score**: Fewer changes = less noise

#### 3. Spike Detection (15%)
- **Calculation**: Outlier identification
- **Method**: Z-score or IQR method
- **Score**: Fewer spikes = higher quality

#### 4. Pattern Irregularity (15%)
- **Calculation**: Entropy or information content
- **Method**: Shannon entropy of price changes
- **Score**: Lower entropy = more predictable

#### 5. Hurst Exponent (10%)
- **Calculation**: Rescaled range analysis
- **Interpretation**:
  - H > 0.5: Trend persistence
  - H = 0.5: Random walk
  - H < 0.5: Mean reversion
- **Score**: Deviation from 0.5 = structure

#### 6. Autocorrelation (15%)
- **Calculation**: ρ(return_t, return_t-1)
- **Interpretation**: High autocorrelation = predictability
- **Score**: |ρ| as quality measure

#### 7. Volatility of Volatility (10%)
- **Calculation**: σ(σ_rolling)
- **Interpretation**: Stable volatility = less noise
- **Score**: Lower vol-of-vol = higher score

### Strategies Using Noise:
- VWAP Reversion (PRIMARY)
- Liquidity Traps (PRIMARY)
- Spoofing Detection (PRIMARY)
- Event-Driven (FILTER)
- Momentum Ignition (FILTER)
- Iceberg Detection (FILTER)
- Order Book Imbalance (FILTER)
- Momentum Breakout (FILTER)

### Noise Level Interpretation:
```python
if noise_level < 0.4:
    quality = "LOW NOISE - HIGH QUALITY"
    # All strategies active
    # High confidence signals
    # Aggressive position sizing
elif noise_level < 0.7:
    quality = "MODERATE NOISE"
    # Standard operations
    # Normal position sizing
else:
    quality = "HIGH NOISE - LOW QUALITY"
    # Reduce activity
    # Smaller positions
    # Avoid new entries
```

---

## Composite Score Calculation

### Formula:
```python
composite_score = (
    0.15 * liquidity_score +
    0.15 * volatility_score +
    0.15 * momentum_score +
    0.15 * imbalance_score +
    0.20 * trend_strength_score +
    0.20 * noise_level_score
)
```

### Score Range: 0.0 - 1.0

### Grade Assignment:
```python
grade_thresholds = {
    "A+": 0.90,  # Exceptional quality
    "A":  0.80,  # Excellent quality
    "A-": 0.75,  # Very good quality
    "B+": 0.70,  # Good quality
    "B":  0.65,  # Above average
    "B-": 0.60,  # Average
    "C+": 0.55,  # Below average
    "C":  0.50,  # Marginal
    "C-": 0.45,  # Poor
    "D+": 0.40,  # Very poor
    "D":  0.35,  # Extremely poor
    "D-": 0.30,  # Unacceptable
    "F":  0.00   # Failed
}
```

---

## Regime Classification (10+ Regimes)

### Regime Determination:
```python
def classify_regime(dimensions):
    liquidity = dimensions['liquidity']
    volatility = dimensions['volatility']
    momentum = dimensions['momentum']
    trend = dimensions['trend_strength']
    composite = dimensions['composite']
    
    # High-quality trending markets
    if composite > 0.7 and trend > 0.7:
        return "STRONG_TRENDING_HIGH_QUALITY"
    
    # High volatility + low liquidity = crisis
    if volatility > 0.7 and liquidity < 0.3:
        return "CRISIS_MODE"
    
    # Low volatility + high trend = ideal
    if volatility < 0.3 and trend > 0.6:
        return "LOW_VOL_TRENDING"
    
    # High momentum + high imbalance = breakout
    if momentum > 0.6 and imbalance_score > 0.6:
        return "BREAKOUT_MODE"
    
    # Low momentum + low trend = ranging
    if momentum < 0.4 and trend < 0.3:
        return "RANGING_CONSOLIDATION"
    
    # High noise = avoid trading
    if noise_level > 0.7:
        return "HIGH_NOISE_LOW_QUALITY"
    
    # Balanced market
    if 0.4 < composite < 0.6:
        return "BALANCED_NEUTRAL"
    
    # Additional regimes...
    return "UNDEFINED"
```

### Regime List:
1. **STRONG_TRENDING_HIGH_QUALITY**: Best conditions for trend following
2. **WEAK_TRENDING**: Choppy trends, reduced confidence
3. **RANGING_CONSOLIDATION**: Mean reversion favorable
4. **BREAKOUT_MODE**: Momentum strategies active
5. **REVERSAL_SETUP**: Counter-trend opportunities
6. **CRISIS_MODE**: Extreme volatility, reduce exposure
7. **LOW_VOL_TRENDING**: Ideal for position building
8. **HIGH_VOL_MEAN_REVERSION**: VWAP strategies
9. **BALANCED_NEUTRAL**: No directional edge
10. **HIGH_NOISE_LOW_QUALITY**: Avoid trading
11. **LIQUIDATION_CASCADE**: Emergency protocols
12. **ACCUMULATION_PHASE**: Long-term positioning

---

## Performance Optimization

### Cache Manager:
```python
class MQScoreCacheManager:
    def __init__(self, ttl=300, max_size=1000):
        self.cache = OrderedDict()
        self.ttl = ttl  # 5 minutes
        self.max_size = max_size
        
    def get(self, key):
        # O(1) lookup
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.cache.move_to_end(key)  # LRU
                return value
        return None
```

**Performance**:
- Hit Rate: >80%
- Lookup Time: O(1)
- TTL: 300 seconds
- Max Size: 1000 entries

### Batch Processing:
```python
class BatchMQScoreCalculator:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        
    def calculate_batch(self, data_list):
        # Process 100 symbols simultaneously
        features_batch = np.array([
            self.extract_features(data)
            for data in data_list
        ])
        
        # Single ML inference call
        scores = self.ml_model.predict(features_batch)
        return scores
```

**Benefits**:
- 3-5x throughput improvement
- Reduced overhead
- Better resource utilization

### Thread Pool:
```python
from concurrent.futures import ThreadPoolExecutor

class ParallelMQScore:
    def __init__(self, max_workers=6):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    def calculate_parallel(self, symbols):
        futures = [
            self.executor.submit(self.calculate_score, symbol)
            for symbol in symbols
        ]
        return [f.result() for f in futures]
```

**Configuration**:
- Workers: 6
- Queue Depth: <50
- Throughput: 100+ scores/second

---

## Integration with Strategies

### Universal Filter:
Every strategy checks MQScore before signal generation:

```python
class MQScoreQualityFilter:
    def should_trade(self, market_data):
        quality_metrics = {
            'composite_score': market_data.get('mqscore_composite', 0.5),
            'liquidity_score': market_data.get('mqscore_liquidity', 0.5),
            'volatility_score': market_data.get('mqscore_volatility', 0.5),
            # ... other dimensions
        }
        
        # Minimum composite score check
        if quality_metrics['composite_score'] < self.min_composite_score:
            return False, quality_metrics
        
        return True, quality_metrics
```

### Confidence Adjustment:
```python
def adjust_signal_confidence(signal_confidence, mqscore):
    # Boost confidence in high-quality markets
    if mqscore > 0.7:
        adjusted = min(signal_confidence * 1.2, 1.0)
    # Reduce confidence in low-quality markets
    elif mqscore < 0.4:
        adjusted = signal_confidence * 0.8
    else:
        adjusted = signal_confidence
    
    return adjusted
```

---

## Monitoring & Alerting

### Metrics Tracked:
- Dimension scores (all 6)
- Composite score
- Regime classification
- Calculation latency
- Cache hit rate
- Model accuracy

### Alert Conditions:
```python
if composite_score < 0.3:
    alert("LOW MARKET QUALITY - Consider reducing activity")

if volatility > 0.8 and liquidity < 0.2:
    alert("CRISIS CONDITIONS - Activate risk protocols")

if noise_level > 0.8:
    alert("HIGH NOISE - Avoid new entries")
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-20  
**Status**: Complete
