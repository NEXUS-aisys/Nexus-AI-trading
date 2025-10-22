# NEXUS AI - UPDATED PIPELINE WITH 70 MODELS
**Complete 8-Layer Integration Flow**

Generated: 2025-10-21

---

## 📊 EXECUTIVE SUMMARY

**Models**: 71 total (MQScore + 70 new: 24 ONNX, 11 PKL, 35 Keras)
**Categories**: 20 functional categories  
**Pipeline Latency**: 13.66ms per symbol (Tier 1+2)
**Throughput**: 73 symbols/second (single-threaded)
**Foundation**: MQScore 6D Engine (PRIMARY, proven)

---

## 🔄 8-LAYER PIPELINE FLOW

### **LAYER 0: DATA INGESTION**
```
Market Data → HMAC Authentication → Continue
```

---

### **LAYER 1: MARKET QUALITY ASSESSMENT (Hybrid: ~10.89ms)**

**Step 1.1: MQScore 6D Engine (PRIMARY) ✅ PROVEN**
- Model: `LightGBM Classifier` (existing, already working)
- Status: Production-ready, proven system
- Latency: ~10ms
- Input: 65 engineered features
- Output: 
  - Composite MQScore (0-1)
  - 6 Dimensions:
    - Liquidity (15%)
    - Volatility (15%)
    - Momentum (15%)
    - Imbalance (15%)
    - Trend Strength (20%)
    - Noise Level (20%)
  - Market Grade (A+ to F)
  - Regime Classification (10+ regimes)
- **Gate 1: IF MQScore < 0.5 → SKIP**
- **Gate 2: IF Liquidity < 0.3 → SKIP**
- **Gate 3: IF Regime == CRISIS → SKIP**

**Step 1.2: Enhanced Volatility Forecast (ENHANCEMENT) ⭐ NEW**
- Model: `quantum_volatility_model_final_optimized.onnx`
- Path: `PRODUCTION/02_VOLATILITY_FORECAST/`
- Latency: +0.111ms
- Input: Market OHLCV
- Output: quantum_volatility_forecast
- **Purpose:** Enhance MQScore's volatility dimension with quantum model
- **Usage:** 
  ```python
  final_volatility = 0.6 * mqscore.volatility + 0.4 * quantum_vol
  ```

**Step 1.3: Enhanced Regime Detection (ENHANCEMENT) ⭐ NEW**
- Model: `ONNX_RegimeClassifier_optimized.onnx`
- Path: `PRODUCTION/03_REGIME_DETECTION/`
- Latency: +0.719ms
- Input: Market features
- Output: regime (0=Trending, 1=Ranging, 2=Volatile)
- **Purpose:** Cross-validate and enhance MQScore's regime classification
- **Usage:**
  ```python
  # Vote ensemble
  final_regime = majority_vote([mqscore.regime, onnx_regime])
  # Or use ONNX if higher confidence
  if onnx_regime_confidence > mqscore_regime_confidence:
      final_regime = onnx_regime
  ```

**Total Layer 1 Latency:** 10ms (MQScore) + 0.83ms (enhancements) = **10.83ms**

**Fallback Strategy:**
```python
try:
    mqscore = mqscore_engine.calculate()  # Primary
    quantum_vol = volatility_model.forecast()  # Enhancement
    onnx_regime = regime_model.classify()  # Enhancement
except:
    # If enhancements fail, use MQScore only (proven fallback)
    use_mqscore_only()
```

---

### **LAYER 2: STRATEGY EXECUTION (Parallel)**

- Execute 20 strategies simultaneously
- Each outputs: signal (-1/0/+1), confidence (0-1)
- Filter: Keep only confidence >= 0.65
- **Gate: IF no high-confidence signals → SKIP**

---

### **LAYER 3: META-LEARNING (0.108ms)**

**Step 3.1: Dynamic Strategy Weighting**
- Model: `Quantum Meta-Strategy Selector.onnx`
- Path: `PRODUCTION/04_STRATEGY_SELECTION/`
- Input: 44 market features (regime, volatility, momentum)
- Output:
  - strategy_weights[19] - Weight per strategy (0-1)
  - anomaly_score - Market anomaly detection (0-1)
  - regime_confidence[3] - Regime probabilities

**How It Works:**
```python
# Example: 14 SELL signals, 6 BUY signals
# But Meta-Selector says: "Trust BUY strategies in this trending market"
strategy_weights = [0.95, 0.12, 0.88, ...]  # BUY strategies get 0.85-0.95
                                             # SELL strategies get 0.05-0.15

# Result: BUY signal wins despite fewer votes!
```

**Gate: IF anomaly_score > 0.8 → Reduce position 50%**

---

### **LAYER 4: SIGNAL AGGREGATION (0.237ms)**

**Step 4.1: Weighted Combination**
- Model: `ONNX Signal Aggregator.onnx`
- Path: `PRODUCTION/05_SIGNAL_AGGREGATION/`
- Algorithm:
```python
weighted_signal = Σ(signal_i × weight_i × confidence_i) / Σ(weight_i × confidence_i)
# Range: -1.0 (SELL) to +1.0 (BUY)
```

**Decision Logic:**
- IF aggregated_signal > 0.3 → BUY
- IF aggregated_signal < -0.3 → SELL
- ELSE → HOLD (SKIP)

**Gate: IF signal_strength < 0.5 → SKIP**
**Gate: IF active_orders[symbol] exists → SKIP (Duplicate Prevention)**

---

### **LAYER 5: MODEL GOVERNANCE & ROUTING (0.146ms)**

**Step 5.1: Model Governance (0.063ms)**
- Model: `ONNX_ModelGovernor_Meta_optimized.onnx`
- Path: `PRODUCTION/07_MODEL_GOVERNANCE/`
- Input: 75 performance metrics (accuracy, Sharpe, drawdown)
- Output: model_weights[15], threshold_adjustments

**Step 5.2: Decision Routing (0.083ms)**
- Model: `ModelRouter_Meta_optimized.onnx`
- Path: `PRODUCTION/06_MODEL_ROUTING/`
- Input: 126 context features
- Output: action_probs[2], confidence, value_estimate

---

### **LAYER 6: RISK MANAGEMENT (1.7ms)**

**Step 6.1: Risk Assessment (0.492ms)**
- Model: `model_risk_governor_optimized.onnx`
- Path: `PRODUCTION/08_RISK_MANAGEMENT/`
- Output: risk_multiplier (0-1)
- **Gate: IF risk_multiplier < 0.3 → REJECT**

**Step 6.2: Confidence Calibration (0.503ms)**
- Model: `xgboost_confidence_model.pkl`
- Path: `PRODUCTION/09_CONFIDENCE_CALIBRATION/`
- Output: calibrated_confidence

**Step 6.3: Market Classification (1.339ms)**
- Model: `xgboost_classifier_enhanced_h1.pkl`
- Path: `PRODUCTION/10_MARKET_CLASSIFICATION/`
- Output: market_class (0=Bull, 1=Bear, 2=Neutral)
- **Gate: IF action conflicts with market_class → REJECT**

**Final Position Sizing:**
```python
adjusted_size = base_size × risk_multiplier × calibrated_confidence
```

**7-Layer Risk Check:**
1. Margin check
2. VaR check
3. Position size limit
4. Daily loss limit
5. Weekly loss limit
6. Drawdown limit
7. Final approval

**Gate: IF ANY risk layer fails → REJECT**

---

### **LAYER 7: EXECUTION**

1. Create order with TP/SL
2. Submit to exchange
3. Wait for fill (poll 500ms, max 30s)
4. IF FILLED → Register in ML Manager
5. IF TIMEOUT → Cancel order

---

### **LAYER 8: MONITORING & FEEDBACK (Async)**

**ML Order Manager Loop (1-5s interval):**
- Update current price
- Calculate P&L
- Check exit conditions:
  - TP hit?
  - SL hit?
  - ML exit signal?
  - Quality drop < 0.3?

**On Exit:**
1. Close position
2. Record outcome (Win/Loss)
3. Update strategy performance
4. Recalculate weights
5. Free symbol for new signals

---

## 🎯 COMPLETE DECISION GATES SUMMARY

| Layer | Gate | Action if Fails |
|-------|------|-----------------|
| 1.1 | quality_score < 0.65 | SKIP |
| 1.3 | regime == CRISIS | SKIP |
| 2 | No high-conf signals | SKIP |
| 3 | anomaly_score > 0.8 | Reduce size 50% |
| 4 | signal_strength < 0.5 | SKIP |
| 4 | Duplicate order | SKIP |
| 6.1 | risk_multiplier < 0.3 | REJECT |
| 6.3 | Market class conflicts | REJECT |
| 6 | Any risk layer fails | REJECT |

---

## 📊 LATENCY BREAKDOWN

| Layer | Time | Cumulative |
|-------|------|------------|
| Layer 1 | 10.830ms (MQScore + enhancements) | 10.830ms |
| Layer 2 | Parallel (built into strategies) | - |
| Layer 3 | 0.108ms | 10.938ms |
| Layer 4 | 0.237ms | 11.175ms |
| Layer 5 | 0.146ms | 11.321ms |
| Layer 6 | 2.334ms | 13.655ms |
| **TOTAL** | **~13.66ms** | per symbol |

**Processing Capacity**: 73 symbols/second per thread (single-threaded)
**Note:** MQScore is the primary bottleneck but provides comprehensive 6D assessment

---

## 🔧 OPTIONAL/ADVANCED MODELS

### **Uncertainty Quantification (Tier 3)**
- 10 classification models: `PRODUCTION/13_UNCERTAINTY_CLASSIFICATION/`
- 10 regression models: `PRODUCTION/14_UNCERTAINTY_REGRESSION/`
- Usage: Confidence intervals (run async/parallel)

### **Bayesian Ensemble (Tier 3)**
- 5 models: `PRODUCTION/15_BAYESIAN_ENSEMBLE/`
- Usage: Uncertainty estimation (run offline)

### **Advanced Features (Tier 4)**
- Pattern Recognition: `PRODUCTION/16_PATTERN_RECOGNITION/final_model.keras`
- LSTM: `PRODUCTION/17_LSTM_TIME_SERIES/financial_lstm_final_optimized.onnx`
- Anomaly: `PRODUCTION/18_ANOMALY_DETECTION/autoencoder_optimized.onnx`
- Entry Timing: `PRODUCTION/19_ENTRY_TIMING/` (3.756ms)
- HFT Scalping: `PRODUCTION/20_HFT_SCALPING/` (9.140ms)

---

## 🔄 PARALLEL PROCESSING

### **Multi-Symbol Execution**
```
50 symbols × Independent pipelines = 50 concurrent positions (max 1/symbol)
Shared resources: WeightCalculator, MLDecisionEngine (thread-safe)
```

### **Strategy Parallelization**
```
20 strategies → ThreadPoolExecutor(max_workers=20) → Collect results
```

---

## 🎯 EXAMPLE EXECUTION TRACE

**Symbol: BTCUSDT**

```
✓ L1.1: MQScore 6D Engine (PRIMARY)
  - Composite Score: 0.72 (PASS >= 0.5)
  - Liquidity: 0.85 (PASS >= 0.3)
  - Volatility: 0.62 (15% dimension)
  - Momentum: 0.78 (15% dimension)
  - Imbalance: 0.55 (15% dimension)
  - Trend Strength: 0.81 (20% dimension)
  - Noise Level: 0.68 (20% dimension)
  - Grade: B+
  - Regime: TRENDING (safe, not CRISIS)

✓ L1.2: Quantum Volatility (ENHANCEMENT)
  - quantum_vol_forecast = 1.2%
  - Blended: 0.6 * 0.62 + 0.4 * 0.012 = 0.377 (enhanced!)

✓ L1.3: ONNX Regime (ENHANCEMENT)
  - onnx_regime = 0 (Trending)
  - Matches MQScore regime ✓ (consensus)

✓ L2: 20 strategies → 16 filtered (conf >= 0.65)
  - 10 say SELL (conf avg: 0.72)
  - 6 say BUY (conf avg: 0.89)

✓ L3: Meta-Selector weights:
  - BUY strategies: [0.92, 0.88, 0.95, 0.91, 0.87, 0.90]
  - SELL strategies: [0.12, 0.08, 0.15, 0.10, 0.09, ...]
  - anomaly_score = 0.18 (Normal)

✓ L4: Aggregated signal = +0.67 (BUY)
  - signal_strength = 0.84 (STRONG)

✓ No duplicate order (PASS)

✓ L5: ModelRouter → action = BUY, confidence = 0.91

✓ L6.1: risk_multiplier = 0.85 (PASS)
✓ L6.2: calibrated_confidence = 0.88 (PASS)
✓ L6.3: market_class = 0 (Bullish) ✓ ALIGNED

Position size = $1000 × 0.85 × 0.88 = $748

✓ All 7 risk layers PASS

→ EXECUTE: BUY 748 BTCUSDT @ $43,250
→ TP: $43,680 (+1%), SL: $42,820 (-1%)
```

---

## 📋 MODEL PATHS REFERENCE

```python
from pathlib import Path

BASE = Path("BEST_UNIQUE_MODELS/PRODUCTION")

# Tier 1 (Critical)
DATA_QUALITY = BASE / "01_DATA_QUALITY/Regressor_lightgbm_optimized.onnx"
VOLATILITY = BASE / "02_VOLATILITY_FORECAST/quantum_volatility_model_final_optimized.onnx"
REGIME = BASE / "03_REGIME_DETECTION/ONNX_RegimeClassifier_optimized.onnx"
STRATEGY_SELECT = BASE / "04_STRATEGY_SELECTION/Quantum Meta-Strategy Selector.onnx"
SIGNAL_AGG = BASE / "05_SIGNAL_AGGREGATION/ONNX Signal Aggregator.onnx"
ROUTER = BASE / "06_MODEL_ROUTING/ModelRouter_Meta_optimized.onnx"
GOVERNOR = BASE / "07_MODEL_GOVERNANCE/ONNX_ModelGovernor_Meta_optimized.onnx"

# Tier 2 (Risk)
RISK = BASE / "08_RISK_MANAGEMENT/model_risk_governor_optimized.onnx"
CONFIDENCE = BASE / "09_CONFIDENCE_CALIBRATION/xgboost_confidence_model.pkl"
CLASSIFIER = BASE / "10_MARKET_CLASSIFICATION/xgboost_classifier_enhanced_h1.pkl"
```

---

**STATUS: ✅ READY FOR PHASE 1 INTEGRATION**

---

**Document Version**: 2.1  
**Last Updated**: 2025-10-21  
**Models Integrated**: 71 total (MQScore + 70 new models)
**Key Update**: MQScore 6D Engine as PRIMARY Layer 1 foundation
