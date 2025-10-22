# NEXUS AI COMPLETE PIPELINE DIAGRAM
**8-Layer Architecture with 70 ML Models**

Last Updated: 2025-10-21

---

## HIGH-LEVEL OVERVIEW

```
MARKET DATA (10-50 symbols)
    ↓
AUTHENTICATION (HMAC-SHA256)
    ↓
LAYER 1: MARKET QUALITY ASSESSMENT (Hybrid, ~10.83ms)
    ├─ MQScore 6D Engine (PRIMARY) ✅ ~10ms
    ├─ Enhanced Volatility (+0.111ms) ⭐ NEW
    └─ Enhanced Regime (+0.719ms) ⭐ NEW
    ↓
LAYER 2: STRATEGY EXECUTION (20 Strategies, Parallel)
    └─ Generate 20 signals per symbol
    ↓
LAYER 3: META-LEARNING (1 ONNX model, 0.108ms) ⭐ NEW
    └─ Dynamic strategy weight assignment
    ↓
LAYER 4: SIGNAL AGGREGATION (1 ONNX model, 0.237ms)
    └─ Weighted combination → Single signal
    ↓
LAYER 5: MODEL GOVERNANCE & ROUTING (2 ONNX models, 0.146ms) ⭐ NEW
    ├─ Model Governance (Trust levels)
    └─ Decision Routing (BUY/SELL/HOLD)
    ↓
LAYER 6: RISK MANAGEMENT & CONFIDENCE (3 models, 1.7ms)
    ├─ Risk Assessment
    ├─ Confidence Calibration
    ├─ Market Classification
    └─ 7-Layer Risk Check
    ↓
LAYER 7: ORDER EXECUTION (Exchange API)
    └─ Submit → Monitor fill → Register
    ↓
LAYER 8: ML ORDER MANAGER & FEEDBACK (Async)
    ├─ Monitor positions (1-5s loop)
    ├─ Dynamic TP/SL adjustment
    ├─ Exit on conditions
    └─ Update strategy weights

TOTAL LATENCY: 13.66ms per symbol (Layers 1-6)
THROUGHPUT: 73 symbols/second (single-threaded)
NOTE: MQScore provides comprehensive 6D market assessment
```

---

## DETAILED PIPELINE FLOW

### LAYER 1: DATA AUTHENTICATION
```
Market Sources (Binance, Coinbase, etc.)
    ├─ WebSocket: Real-time data (10-50 symbols)
    ├─ NexusSecurityLayer: HMAC-SHA256 verification
    ├─ Timestamp validation (max 10s staleness)
    └─ AuthenticatedMarketData object
    
Output: {symbol, price, volume, bid, ask, authenticated: true}
```

### LAYER 1: MARKET QUALITY ASSESSMENT (Hybrid: ~10.83ms)

#### Step 1.1: MQScore 6D Engine (PRIMARY) ✅ PROVEN (~10ms)
```
Model: LightGBM Classifier (existing, production-ready)
Status: Already working and proven in production
Type: Ensemble ML Model
Latency: ~10ms

Input: 65 engineered market features
    ├─ Price features (20)
    ├─ Volume features (15)
    ├─ Order book depth (10)
    ├─ Momentum indicators (10)
    └─ Volatility measures (10)

Output: Comprehensive 6D Assessment
    ├─ Composite MQScore (0-1): Weighted combination of all dimensions
    │
    ├─ 6 Dimensions:
    │   ├─ Liquidity (15%): Volume, spread, depth
    │   ├─ Volatility (15%): Realized, GARCH, jump risk
    │   ├─ Momentum (15%): Short/med/long momentum
    │   ├─ Imbalance (15%): Order flow, buy/sell ratio
    │   ├─ Trend Strength (20%): ADX, MA alignment
    │   └─ Noise Level (20%): Signal-to-noise, spikes
    │
    ├─ Market Grade: A+, A, B+, B, C+, C, D, F
    │
    └─ Regime Classification: 10+ regimes
        (TRENDING, RANGING, VOLATILE, CRISIS, etc.)

Primary Gates (CRITICAL):
    IF MQScore < 0.5:
        └─ SKIP (Market quality too low)
    
    IF Liquidity < 0.3:
        └─ SKIP (Insufficient liquidity)
    
    IF Regime == CRISIS or HIGH_NOISE:
        └─ SKIP (Unsafe market conditions)

Result: 
    - Proven foundation for market assessment
    - Comprehensive 6D view of market quality
    - Multiple safety gates
```

#### Step 1.2: Enhanced Volatility Forecast (ENHANCEMENT) ⭐ NEW (+0.111ms)
```
Model: quantum_volatility_model_final_optimized.onnx
Path: PRODUCTION/02_VOLATILITY_FORECAST/
Type: ONNX (Quantum-enhanced GARCH)
Latency: +0.111ms

Input: Market OHLCV data, historical volatility

Output: quantum_volatility_forecast

Purpose: Enhance MQScore's volatility dimension with advanced quantum model

Usage (Ensemble Approach):
    # Blend MQScore volatility with quantum forecast
    final_volatility = 0.6 × mqscore.volatility + 0.4 × quantum_vol
    
    # Use enhanced volatility for downstream risk models
    risk_assessment_input.volatility = final_volatility

Benefit: 
    - More accurate volatility prediction
    - Quantum model adds advanced forecasting
    - Ensemble reduces prediction error
```

#### Step 1.3: Enhanced Regime Detection (ENHANCEMENT) ⭐ NEW (+0.719ms)
```
Model: ONNX_RegimeClassifier_optimized.onnx
Path: PRODUCTION/03_REGIME_DETECTION/
Type: ONNX (Deep learning classifier)
Latency: +0.719ms

Input: Market features, price patterns

Output: 
    - regime: 0=Trending, 1=Ranging, 2=Volatile
    - confidence: Classification confidence (0-1)

Purpose: Cross-validate and enhance MQScore's regime classification

Usage (Vote Ensemble):
    # Option 1: Majority vote
    final_regime = majority_vote([mqscore.regime, onnx_regime])
    
    # Option 2: Use higher confidence
    if onnx_confidence > mqscore_confidence:
        final_regime = onnx_regime
    else:
        final_regime = mqscore.regime
    
    # Option 3: Require consensus
    if mqscore.regime == onnx_regime:
        final_regime = mqscore.regime  # High confidence (both agree)
    else:
        final_regime = "UNCERTAIN"  # Caution mode

Benefit:
    - Cross-validation reduces false positives
    - Consensus increases confidence
    - Faster ONNX model for backup
```

#### Fallback Strategy (Critical for Reliability):
```
try:
    # Step 1: Run MQScore (must succeed)
    mqscore = mqscore_engine.calculate(market_data)
    
    # Primary gates (CRITICAL)
    if not mqscore.passes_gates():
        return SKIP
    
    # Step 2: Try enhancements (optional but recommended)
    try:
        quantum_vol = volatility_model.forecast(market_data)
        onnx_regime = regime_classifier.predict(market_data)
        
        # Ensemble/blend results
        final_assessment = ensemble_results(mqscore, quantum_vol, onnx_regime)
    except:
        # If enhancements fail, use MQScore only (proven fallback)
        logger.warning("Enhancements failed, using MQScore only")
        final_assessment = mqscore_only_results(mqscore)
    
    return final_assessment

except MQScoreError:
    # If MQScore fails, ABORT (no fallback - MQScore is critical)
    logger.error("MQScore failed - cannot assess market quality")
    return ABORT
```

#### Summary:
    - MQScore: Foundation (proven, comprehensive)
    - Enhancements: Add-ons (advanced, optional)
    - Total: Best of both worlds
    - Fallback: MQScore always works
```

### LAYER 2: STRATEGY EXECUTION (20 STRATEGIES, Parallel)
```
StrategyOrchestrator (per symbol)
ThreadPoolExecutor(max_workers=20)
    
PARALLEL EXECUTION:
    ├─ Event-Driven Strategy → signal₁, conf₁
    ├─ LVN Breakout Strategy → signal₂, conf₂
    ├─ Absorption Breakout → signal₃, conf₃
    ├─ Momentum Breakout → signal₄, conf₄
    ├─ Market Microstructure → signal₅, conf₅
    ├─ Order Book Imbalance → signal₆, conf₆
    ├─ Liquidity Absorption → signal₇, conf₇
    ├─ Spoofing Detection → signal₈, conf₈
    ├─ Iceberg Detection → signal₉, conf₉
    ├─ Liquidation Detection → signal₁₀, conf₁₀
    ├─ Liquidity Traps → signal₁₁, conf₁₁
    ├─ Multi-Timeframe Alignment → signal₁₂, conf₁₂
    ├─ Cumulative Delta → signal₁₃, conf₁₃
    ├─ Delta Divergence → signal₁₄, conf₁₄
    ├─ Open Drive vs Fade → signal₁₅, conf₁₅
    ├─ Profile Rotation → signal₁₆, conf₁₆
    ├─ VWAP Reversion → signal₁₇, conf₁₇
    ├─ Stop Run Anticipation → signal₁₈, conf₁₈
    ├─ Momentum Ignition → signal₁₉, conf₁₉
    └─ Volume Imbalance → signal₂₀, conf₂₀

Each strategy outputs:
    {
        "signal": +1 (BUY) / -1 (SELL) / 0 (NEUTRAL),
        "confidence": 0.0 to 1.0,
        "metadata": {...}
    }

CONFIDENCE FILTER (65% threshold):
    FOR each signal:
        IF confidence < 0.65:
            └─ Discard signal
        ELSE:
            └─ Keep for aggregation

Confidence Gate:
    IF no signals with confidence >= 0.65:
        └─ SKIP symbol (No high-confidence signals)
    ELSE:
        └─ CONTINUE to Meta-Learning (Layer 3)

Output: List of high-confidence signals (typically 10-18 signals)
```

### LAYER 3: META-LEARNING (STRATEGY SELECTION, 0.108ms) ⭐ NEW
```
Model: Quantum Meta-Strategy Selector.onnx
Path: PRODUCTION/04_STRATEGY_SELECTION/
Type: ONNX (Meta-learning model)
Latency: 0.108ms

Input: 44 market context features
    ├─ Price momentum (5 features)
    ├─ Volume trends (5 features)
    ├─ Regime (from Layer 1)
    ├─ Volatility (from Layer 1)
    ├─ Recent strategy performance (20 features)
    └─ Market microstructure (13 features)

Process:
    Analyzes current market conditions and determines
    which strategies to TRUST based on:
    - Market regime (trending/ranging/volatile)
    - Recent strategy performance
    - Market microstructure patterns

Output: 3 arrays
    1. strategy_weights [19 values]
       - One weight per strategy (0.0 to 1.0)
       - High weight = TRUST this strategy now
       - Low weight = IGNORE this strategy now
    
    2. anomaly_score [1 value]
       - Market anomaly detection (0.0 to 1.0)
       - High score = Abnormal market behavior
    
    3. regime_confidence [3 values]
       - [P(trending), P(ranging), P(volatile)]

Decision Logic:
    IF anomaly_score > 0.8:
        └─ Reduce position size by 50% (Caution mode)
    
Example Scenario:
    Market: Strong trending bull market
    14 strategies say SELL, 6 say BUY
    
    Meta-Selector OUTPUT:
        strategy_weights for BUY strategies: [0.92, 0.88, 0.95, 0.91, 0.87, 0.90]
        strategy_weights for SELL strategies: [0.12, 0.08, 0.15, 0.10, ...]
    
    → BUY strategies TRUSTED (high weights)
    → SELL strategies IGNORED (low weights)
    → Final signal will be BUY despite fewer votes!

Output: 
    {
        "strategy_weights": [19 weights],
        "anomaly_score": 0.15,
        "regime_confidence": [0.85, 0.10, 0.05]
    }
```

### LAYER 4: SIGNAL AGGREGATION (0.237ms)
```
Model: ONNX Signal Aggregator.onnx
Path: PRODUCTION/05_SIGNAL_AGGREGATION/
Type: ONNX (Ensemble aggregator)
Latency: 0.237ms

Input:
    - Filtered signals from Layer 2 (10-18 signals)
    - Strategy weights from Layer 3 (19 weights)
    - Confidence scores per signal

Algorithm:
    weighted_sum = 0
    total_weight = 0
    
    FOR each signal_i with confidence_i:
        weight_i = strategy_weights[i]  # From Meta-Selector
        
        weighted_sum += signal_i × weight_i × confidence_i
        total_weight += weight_i × confidence_i
    
    aggregated_signal = weighted_sum / total_weight

Output:
    {
        "aggregated_signal": -1.0 to +1.0,
        "signal_strength": mean(confidences),
        "num_strategies_used": count
    }

Decision Logic:
    IF signal_strength < 0.5:
        └─ SKIP (Weak conviction)
    
    IF abs(aggregated_signal) < 0.2:
        └─ SKIP (Too neutral, no clear direction)
    
    IF aggregated_signal > 0.3:
        direction = "BUY"
    ELIF aggregated_signal < -0.3:
        direction = "SELL"
    ELSE:
        direction = "HOLD" → SKIP

Duplicate Check:
    IF active_orders[symbol] exists:
        └─ SKIP (Prevent duplicate order)
    ELSE:
        └─ CONTINUE to routing (Layer 5)
```

### LAYER 5: MODEL GOVERNANCE & ROUTING (0.146ms) ⭐ NEW
```
Step 5.1: Model Governance (0.063ms)
──────────────────────────────────────
Model: ONNX_ModelGovernor_Meta_optimized.onnx
Path: PRODUCTION/07_MODEL_GOVERNANCE/
Type: ONNX (Governance model)

Input: 75 performance metrics
    ├─ Recent model accuracy (30-day rolling)
    ├─ Sharpe ratios per model
    ├─ Drawdowns per model
    ├─ Win rates per model
    └─ Recent profit/loss

Process:
    Analyzes which models are performing well and
    adjusts trust levels dynamically

Output:
    - model_weights [15]: Trust level per model (0-1)
    - threshold_adjustments: Dynamic confidence thresholds
    - risk_scalers: Risk adjustment factors
    - retrain_flags: Models needing retraining

Function: Ensure we trust well-performing models more

Step 5.2: Decision Routing (0.083ms)
────────────────────────────────────
Model: ModelRouter_Meta_optimized.onnx
Path: PRODUCTION/06_MODEL_ROUTING/
Type: ONNX (Router model)

Input: 126 context features
    ├─ Aggregated signal (from Layer 4)
    ├─ Market regime (from Layer 1)
    ├─ Volatility (from Layer 1)
    ├─ Strategy weights (from Layer 3)
    ├─ Recent performance metrics
    ├─ Portfolio state
    └─ Model governance weights

Process:
    Makes final BUY/SELL/HOLD decision by routing
    through best-performing prediction pathway

Output:
    {
        "action_probs": [P(buy), P(sell)],
        "confidence": 0.0-1.0,
        "value_estimate": expected_return
    }

Decision Logic:
    IF action_probs[0] > 0.6:
        action = "BUY"
    ELIF action_probs[1] > 0.6:
        action = "SELL"
    ELSE:
        action = "HOLD" → SKIP

    IF confidence < 0.7:
        └─ SKIP (Low confidence decision)
```

### LAYER 6: RISK MANAGEMENT & CONFIDENCE (1.7ms)
```
Step 6.1: Risk Assessment (0.492ms)
───────────────────────────────────
Model: model_risk_governor_optimized.onnx
Path: PRODUCTION/08_RISK_MANAGEMENT/
Type: ONNX (Risk model)

Input: 15 risk features
    ├─ Proposed position size
    ├─ Current portfolio exposure
    ├─ Market volatility (from Layer 1)
    ├─ Recent losses
    ├─ Correlation with existing positions
    └─ Account drawdown

Output:
    - risk_multiplier (0-1): Position size adjustment
    - ensemble_pred: Risk ensemble output
    - uncertainty: Prediction uncertainty

Risk Gate:
    IF risk_multiplier < 0.3:
        └─ REJECT (Too risky)

Step 6.2: Confidence Calibration (0.503ms)
──────────────────────────────────────────
Model: xgboost_confidence_model.pkl
Path: PRODUCTION/09_CONFIDENCE_CALIBRATION/
Type: PKL (XGBoost)

Input:
    - Raw confidence (from Layer 5)
    - Uncertainty (from Step 6.1)
    - Recent model accuracy
    - Market conditions

Output: calibrated_confidence (0-1)

Function: Prevent overconfidence, ensure well-calibrated predictions

Step 6.3: Market Classification (1.339ms)
─────────────────────────────────────────
Model: xgboost_classifier_enhanced_h1.pkl
Path: PRODUCTION/10_MARKET_CLASSIFICATION/
Type: PKL (XGBoost)

Input: Market features
Output: market_class (0=Bullish, 1=Bearish, 2=Neutral)

Alignment Check:
    IF action="BUY" AND market_class=1 (Bearish):
        └─ REJECT (Conflicting signal)
    IF action="SELL" AND market_class=0 (Bullish):
        └─ REJECT (Conflicting signal)

Position Sizing Calculation:
───────────────────────────
    base_size = account_size × 0.02  # 2% base risk
    adjusted_size = base_size × risk_multiplier × calibrated_confidence
    
    IF adjusted_size < minimum_size:
        └─ REJECT (Position too small)

7-Layer Risk Validation:
────────────────────────
    Layer 1: Margin check (sufficient funds?)
    Layer 2: VaR check (Value at Risk acceptable?)
    Layer 3: Position size limit (< 10% equity?)
    Layer 4: Daily loss limit (not exceeded?)
    Layer 5: Weekly loss limit (not exceeded?)
    Layer 6: Drawdown check (< 15%?)
    Layer 7: Final approval (all gates passed?)
    
    IF ANY layer fails:
        └─ REJECT trade
    
Output: APPROVED TRADING DECISION
    {
        symbol, action, entry, quantity, 
        stop_loss, take_profit, 
        confidence, risk_score, expected_value
    }
```

### LAYER 7: ORDER EXECUTION
```
Order Creation:
    ├─ Generate unique ID: NEXUS_{symbol}_{timestamp}_{counter}
    ├─ Example: NEXUS_BTCUSDT_1729432567_000001
    ├─ Create order object with TP/SL from Layer 6
    └─ Status: PENDING

Exchange Submission:
    ├─ API: exchange.create_order(symbol, type, side, amount, price)
    ├─ Response: exchange_order_id
    └─ Status: SUBMITTED

Fill Monitoring (poll every 500ms, timeout 30s):
    ├─ Check status: OPEN → PARTIALLY_FILLED → FILLED
    ├─ ON FILLED:
    │   ├─ Record fill_price, fill_time
    │   ├─ Calculate slippage
    │   └─ Continue to registration
    └─ ON TIMEOUT:
        ├─ Cancel order
        ├─ Log timeout error
        └─ ABORT (do not register)

Position Registration:
    ├─ active_orders[symbol] = order_id (prevent future duplicates!)
    └─ active_positions[order_id] = {
        entry_price, quantity, TP, SL,
        confidence, expected_value, timestamp
    }

Output: EXECUTED & REGISTERED ORDER
    {
        order_id, symbol, side, quantity, 
        fill_price, TP, SL, 
        status: FILLED, registered: true
    }
```

### LAYER 8: ML ORDER MANAGER & FEEDBACK (Async)
```
Position Tracking (by Order ID):
    active_positions[order_id] = {
        entry_price: 50000,
        current_price: 50500,
        unrealized_pnl: +100,
        take_profit: 52000,
        stop_loss: 49000,
        last_update: timestamp
    }

Real-Time Monitoring (every 1-5 seconds):
    ├─ Get current price from data stream
    ├─ Calculate P&L: (current - entry) × quantity
    ├─ Update unrealized_pnl
    └─ Check exit conditions

Dynamic TP/SL Adjustment:
    ├─ Trailing stop: Move SL up if price moves favorably
    ├─ Breakeven: Move SL to entry after +2% profit
    ├─ Scale out: Close 50% at +3%, rest at TP
    └─ Volatility adjustment: Widen SL if volatility spikes

Exit Decision Logic:
    IF current_price >= take_profit → CLOSE (target hit)
    IF current_price <= stop_loss → CLOSE (stop hit)
    IF ML predicts reversal → CLOSE (ML exit signal)
    IF MQScore drops < 0.3 → CLOSE (quality deteriorated)
    IF drawdown > 10% → CLOSE (risk limit)

Order Close Execution:
    ├─ Submit close order to exchange
    ├─ Wait for fill confirmation
    ├─ Remove from active_positions[order_id]
    ├─ Delete active_orders[symbol]
    └─ Record trade outcome for feedback

Output: CLOSED POSITION
    {order_id, entry, exit, pnl, duration, outcome: WIN/LOSS}
```

Feedback Loop & Learning:
──────────────────────────

Trade Outcome Recording:
    ├─ Record: order_id, symbol, strategies used, outcome, pnl
    └─ Store in trade_history database

Strategy Performance Update (Per Trade):
    FOR each strategy that contributed to signal:
        ├─ Update win_count / loss_count
        ├─ Update accuracy: correct_predictions / total
        ├─ Update Sharpe ratio: returns / std_dev
        ├─ Update avg_pnl
        └─ Calculate new performance score

Weight Recalculation (Daily):
    ├─ Calculate new weights based on 30-day rolling window
    ├─ weight_i = 0.5×accuracy + 0.3×sharpe + 0.2×avg_pnl
    ├─ EMA smoothing: new = 0.3×new + 0.7×old
    └─ Update Meta-Strategy Selector training data

ML Model Performance Monitoring:
    ├─ Track prediction accuracy per model
    ├─ Detect concept drift (PSI > 0.3)
    ├─ Monitor calibration error
    ├─ IF accuracy drops > 5% → Flag for retraining
    └─ IF drift detected → Trigger model update pipeline

Continuous Learning Pipeline:
    ├─ Collect last 90 days of outcomes
    ├─ Retrain models monthly (or on drift detection)
    ├─ A/B test new model version:
    │   ├─ Deploy to 10% of traffic
    │   ├─ Run for 7 days
    │   └─ Compare performance metrics
    └─ IF new model better → Full deployment

Model Governance Updates:
    ├─ Update trust levels based on recent performance
    ├─ Adjust confidence thresholds dynamically
    └─ Flag underperforming models

Output: UPDATED & LEARNING SYSTEM
    {
        strategy_weights_updated,
        model_accuracy_tracked,
        drift_status_monitored,
        retrain_flags_set,
        system_improving
    }
```

---

## MULTI-SYMBOL ORCHESTRATION

```
Main Event Loop (runs continuously):

FOR EACH symbol in [BTCUSDT, ETHUSDT, ... 50 symbols]:
    
    LAYER 0: Authentication
        └─ HMAC verification
    
    LAYER 1: Market Quality Assessment (10.83ms)
        ├─ MQScore 6D Engine (~10ms) ✅ PRIMARY
        ├─ Enhanced volatility (+0.111ms)
        └─ Enhanced regime (+0.719ms)
        └─ Gates: MQScore >= 0.5, liquidity >= 0.3, safe regime
    
    LAYER 2: Strategy Execution (Parallel)
        ├─ Run 20 strategies simultaneously
        ├─ Filter: confidence >= 0.65
        └─ Gate: Has high-confidence signals?
    
    LAYER 3: Meta-Learning (0.108ms) ⭐
        ├─ Assign dynamic strategy weights
        ├─ Detect market anomalies
        └─ Gate: anomaly_score < 0.8
    
    LAYER 4: Signal Aggregation (0.237ms)
        ├─ Weighted combination
        ├─ Final signal: -1.0 to +1.0
        └─ Gates: strength >= 0.5, no duplicate
    
    LAYER 5: Governance & Routing (0.146ms) ⭐
        ├─ Model governance: Trust levels
        ├─ Decision routing: BUY/SELL/HOLD
        └─ Gate: confidence >= 0.7
    
    LAYER 6: Risk Management (1.7ms)
        ├─ Risk assessment (0.492ms)
        ├─ Confidence calibration (0.503ms)
        ├─ Market classification (1.339ms)
        └─ Gates: risk_multiplier >= 0.3, 7-layer check, alignment
    
    LAYER 7: Execution
        ├─ Submit order to exchange
        ├─ Monitor fill (30s timeout)
        └─ Register position if filled
    
    LAYER 8: Monitoring & Feedback (Async)
        ├─ Monitor every 1-5 seconds
        ├─ Check TP/SL/ML exit
        ├─ Close on conditions
        └─ Update weights & learn

Result: 
    - Total latency: ~13.66ms per symbol (Layers 1-6)
    - Throughput: 73 symbols/second per thread (single-threaded)
    - MQScore provides comprehensive 6D market assessment
    - Up to 50 concurrent positions (1 per symbol max)
    - Each order tracked by unique ID
    - 70 models working together (MQScore + 69 enhancements)
    - Continuous learning and adaptation
```

---

## KEY SYNCHRONIZATION POINTS

### 1. Duplicate Prevention
```
BEFORE creating order:
    IF symbol in active_orders:
        LOG("Order exists for {symbol}: {order_id}")
        RETURN None  # Skip this signal
```

### 2. Order ID Tracking
```
Order ID = NEXUS_{symbol}_{timestamp}_{counter}

active_orders = {
    "BTCUSDT": "NEXUS_BTCUSDT_1729432567_000001",
    "ETHUSDT": "NEXUS_ETHUSDT_1729432568_000002",
    ...
}

active_positions = {
    "NEXUS_BTCUSDT_1729432567_000001": {position_details},
    "NEXUS_ETHUSDT_1729432568_000002": {position_details},
    ...
}
```

### 3. Position Cleanup
```
ON order close:
    order_id = active_orders[symbol]
    DELETE active_positions[order_id]
    DELETE active_orders[symbol]
    
    # Now symbol is free for new signal
```

---

## PERFORMANCE METRICS

**Latency Breakdown (8-Layer Architecture with MQScore):**
- Layer 0 (Auth): <1ms
- Layer 1 (Market Quality): 10.83ms (MQScore + enhancements)
  - MQScore 6D Engine: ~10ms (PRIMARY)
  - Quantum Volatility: +0.111ms (enhancement)
  - ONNX Regime: +0.719ms (enhancement)
- Layer 2 (20 Strategies): Variable (parallel execution)
- Layer 3 (Meta-Learning): 0.108ms (1 ONNX model)
- Layer 4 (Signal Aggregation): 0.237ms (1 ONNX model)
- Layer 5 (Governance & Routing): 0.146ms (2 ONNX models)
- Layer 6 (Risk & Confidence): 2.334ms (3 models)
- Layer 7 (Order Submit): <100ms (network I/O)
- Layer 8 (Monitoring): Async (1-5s loop)
- **TOTAL (Layers 1-6)**: 13.66ms per symbol
- **TOTAL (with network)**: <115ms (data → order submitted)

**Throughput Capacity:**
- Per thread: 73 symbols/second (13.66ms each)
- 50 symbols concurrent: ~1.5 updates/symbol/second
- Strategy execution: 100+ parallel signals/second
- ML inference: 1000+ predictions/second (for ONNX models)
- MQScore: Provides comprehensive 6D assessment (worth the latency)

**Model Usage:**
- ONNX models: 24 (ultra-fast, < 20ms combined)
- PKL models: 11 (fast, < 3ms combined)
- Keras models: 35 (moderate, for ensemble/async)
- Total active: 12 models in critical path (Tier 1+2)
- Total ecosystem: 70 working models

**Resource Usage:**
- CPU: <30% per symbol (ONNX optimized)
- Memory: <2GB total (model cache)
- GPU: Optional (for Keras models, async)
- Network: <1Mbps per symbol stream
- Disk I/O: Minimal (models cached in RAM)

**Scaling:**
- Single thread: 73 symbols/second
- 4 threads: 292 symbols/second
- 10 threads: 730 symbols/second
- Bottleneck: MQScore calculation (~10ms per symbol)
- Note: MQScore provides comprehensive 6D assessment (worth the tradeoff)

---

## MODEL SUMMARY (71 TOTAL: MQScore + 70 NEW)

**Core Foundation:**
- **MQScore 6D Engine**: Existing, proven, production-ready ✅

**Production Models from 70-model ecosystem (46):**
- Tier 1 (Critical + Enhancements): 7 ONNX models
  - 2 enhancements for MQScore (volatility, regime)
  - 5 meta-learning and routing models
- Tier 2 (Risk): 5 models (2.33ms)
- Tier 3 (Uncertainty): 25 Keras models (async/ensemble)
- Tier 4 (Advanced): 9 models (optional features)

**Backup Models (15):**
- 2nd/3rd choice alternatives for failover

**Model Organization:**
```
EXISTING:
└── MQScore 6D Engine (LightGBM, already integrated)

PRODUCTION/ (NEW 70-model ecosystem):
├── 01: Data Quality (legacy, can skip - MQScore handles this)
├── 02: Volatility (ENHANCEMENT for MQScore)
├── 03: Regime (ENHANCEMENT for MQScore)
├── 04-07: Meta-learning and routing models
├── 08-12: Risk and confidence models
├── 13-15: Uncertainty ensembles (25 models)
└── 16-20: Advanced features (9 models)
```

---

**Document Version**: 2.1  
**Last Updated**: 2025-10-21  
**Status**: ✅ Production Ready - MQScore + 70 Models (71 Total)  
**Architecture**: 8-Layer Pipeline with MQScore Foundation + Meta-Learning
**Key Update**: MQScore 6D Engine remains as PRIMARY Layer 1 model
