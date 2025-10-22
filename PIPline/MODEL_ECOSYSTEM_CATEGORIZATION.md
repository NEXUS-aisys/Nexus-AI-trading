# NEXUS AI - MODEL ECOSYSTEM CATEGORIZATION
**Organized by Function/Purpose with Primary & Backup Selections**

Generated: 2025-10-20

---

## 📊 ECOSYSTEM OVERVIEW

**Total Categories:** 28
**Total Working Models:** 70
- ONNX: 24 models
- PKL: 11 models  
- Keras: 35 models
- PyTorch: 6 checkpoints (architecture needed)

---

## 🎯 CATEGORY 1: DATA QUALITY SCORING

**Purpose:** Filter low-quality market data before processing

| Priority | Model | Format | Latency | Features | Status |
|----------|-------|--------|---------|----------|--------|
| **1st CHOICE** | Regressor_lightgbm_optimized | ONNX | 0.064ms | 250 | ⭐ PRODUCTION |
| **2nd CHOICE** | reg.pkl (LightGBM) | PKL | 0.887ms | 250 | ✅ BACKUP |

---

## 🎯 CATEGORY 2: MARKET CLASSIFICATION

**Purpose:** Classify market conditions (bullish/bearish/neutral)

| Priority | Model | Format | Latency | Features | Status |
|----------|-------|--------|---------|----------|--------|
| **1st CHOICE** | xgboost_classifier_enhanced | PKL | 1.339ms | Auto | ⭐ PRODUCTION |
| **2nd CHOICE** | final_classifier (XGBoost) | PKL | 0.488ms | Auto | ✅ BACKUP |
| **3rd CHOICE** | cls.pkl (Calibrated) | PKL | 1.481ms | 250 | ✅ BACKUP |

---

## 🎯 CATEGORY 3: REGIME DETECTION

**Purpose:** Detect market regime (trending/ranging/volatile)

| Priority | Model | Format | Latency | Features | Status |
|----------|-------|--------|---------|----------|--------|
| **1st CHOICE** | ONNX_RegimeClassifier_optimized | ONNX | 0.719ms | Tabular | ⭐ PRODUCTION |
| **2nd CHOICE** | lstm_regime_classifier_h1.h5 | Keras | 279ms | Sequence | ✅ BACKUP (LSTM-based) |

---

## 🎯 CATEGORY 4: VOLATILITY FORECASTING

**Purpose:** Predict future volatility for risk management

| Priority | Model | Format | Latency | Features | Status |
|----------|-------|--------|---------|----------|--------|
| **1st CHOICE** | quantum_volatility_model_final | ONNX | 0.111ms | Market data | ⭐ PRODUCTION |
| **2nd CHOICE** | best_model.h5 (volatility hybrid) | Keras | FAILED | - | ❌ FAILED |
| **3rd CHOICE** | ultimate_volatility_model.pkl | PKL | FAILED | - | ❌ FAILED |

**Note:** Only 1 working volatility model

---

## 🎯 CATEGORY 5: STRATEGY SELECTION (META-LEARNING)

**Purpose:** Select which of 20 strategies to use per market condition

| Priority | Model | Format | Latency | Input | Output | Status |
|----------|-------|--------|---------|-------|--------|--------|
| **1st CHOICE** | Quantum Meta-Strategy Selector | ONNX | 0.108ms | 44 features | 19 strategy weights | ⭐ PRODUCTION |
| **2nd CHOICE** | Quantum Meta-Strategy Selector_optimized | ONNX | 0.116ms | 44 features | 19 strategy weights | ✅ BACKUP |

---

## 🎯 CATEGORY 6: SIGNAL AGGREGATION

**Purpose:** Combine multiple strategy signals into 1 signal per symbol

| Priority | Model | Format | Latency | Purpose | Status |
|----------|-------|--------|---------|---------|--------|
| **1st CHOICE** | ONNX Signal Aggregator | ONNX | 0.237ms | Fast aggregation | ⭐ PRODUCTION |
| **2nd CHOICE** | quantum_signal_aggregator_meta_q2025 | ONNX | 0.260ms | Advanced aggregation | ✅ BACKUP |

---

## 🎯 CATEGORY 7: MODEL ROUTING

**Purpose:** Route to best decision model based on context

| Priority | Model | Format | Latency | Input | Output | Status |
|----------|-------|--------|---------|-------|--------|--------|
| **1st CHOICE** | ModelRouter_Meta_optimized | ONNX | 0.083ms | 126 context features | Action probs, confidence | ⭐ PRODUCTION |
| **2nd CHOICE** | ModelRouter_Meta_optimized (duplicate) | ONNX | 0.089ms | 126 context features | Action probs | ✅ BACKUP |

---

## 🎯 CATEGORY 8: MODEL GOVERNANCE

**Purpose:** Decide which models to use dynamically

| Priority | Model | Format | Latency | Input | Output | Status |
|----------|-------|--------|---------|-------|--------|--------|
| **1st CHOICE** | ONNX_ModelGovernor_Meta_optimized | ONNX | 0.063ms | 75 perf metrics | 15 model weights, thresholds | ⭐ PRODUCTION |

**Note:** Only 1 model in this category

---

## 🎯 CATEGORY 9: RISK MANAGEMENT

**Purpose:** Risk-adjusted position sizing and risk scoring

| Priority | Model | Format | Latency | Input | Output | Status |
|----------|-------|--------|---------|-------|--------|--------|
| **1st CHOICE** | model_risk_governor_optimized | ONNX | 0.492ms | 15 risk features | Multiplier, ensemble_pred, uncertainty | ⭐ PRODUCTION |

**Note:** Only 1 model in this category

---

## 🎯 CATEGORY 10: CONFIDENCE CALIBRATION

**Purpose:** Calibrate prediction confidence scores

| Priority | Model | Format | Latency | Features | Status |
|----------|-------|--------|---------|----------|--------|
| **1st CHOICE** | xgboost_confidence_model | PKL | 0.503ms | Auto | ⭐ PRODUCTION |
| **2nd CHOICE** | catboost_confidence_model | PKL | 1.205ms | 50 | ✅ BACKUP |

---

## 🎯 CATEGORY 11: UNCERTAINTY QUANTIFICATION (CLASSIFICATION)

**Purpose:** Quantify prediction uncertainty for classification tasks

| Priority | Model | Format | Latency | Params | Status |
|----------|-------|--------|---------|--------|--------|
| **1st CHOICE** | uncertainty_clf_model_7 | Keras | 64.191ms | 0.02M | ⭐ PRODUCTION |
| **2nd CHOICE** | uncertainty_clf_model_0 | Keras | 65.042ms | 0.02M | ✅ BACKUP |
| **ENSEMBLE** | 10 models (0-9) | Keras | ~700ms total | 0.02M each | ✅ USE ALL FOR ENSEMBLE |

---

## 🎯 CATEGORY 12: UNCERTAINTY QUANTIFICATION (REGRESSION)

**Purpose:** Quantify prediction uncertainty for regression tasks

| Priority | Model | Format | Latency | Params | Status |
|----------|-------|--------|---------|--------|--------|
| **1st CHOICE** | uncertainty_reg_model_4 | Keras | 64.558ms | 0.02M | ⭐ PRODUCTION |
| **2nd CHOICE** | uncertainty_reg_model_2 | Keras | 65.709ms | 0.02M | ✅ BACKUP |
| **ENSEMBLE** | 10 models (0-9) | Keras | ~700ms total | 0.02M each | ✅ USE ALL FOR ENSEMBLE |

---

## 🎯 CATEGORY 13: BAYESIAN UNCERTAINTY ESTIMATION

**Purpose:** Bayesian approach to uncertainty estimation

| Priority | Model | Format | Latency | Params | Status |
|----------|-------|--------|---------|--------|--------|
| **1st CHOICE** | member_0 | Keras | 645.544ms | 0.15M | ⭐ PRODUCTION |
| **2nd CHOICE** | member_1 | Keras | 641.044ms | 0.15M | ✅ BACKUP |
| **ENSEMBLE** | 5 models (0-4) | Keras | ~3200ms total | 0.15M each | ✅ USE ALL FOR ENSEMBLE |

---

## 🎯 CATEGORY 14: PATTERN RECOGNITION (CNN)

**Purpose:** Detect patterns in price/volume data

| Priority | Model | Format | Latency | Params | Status |
|----------|-------|--------|---------|--------|--------|
| **1st CHOICE** | final_model (CNN1D) | Keras | 78.081ms | 0.01M | ⭐ PRODUCTION |
| **2nd CHOICE** | best_val_model (CNN1D) | Keras | 79.631ms | 0.01M | ✅ BACKUP |
| **3rd CHOICE** | cnn1d_best_model | Keras | 227.419ms | 0.01M | ✅ BACKUP |
| **4th CHOICE** | CNN1d_optimized | ONNX | 19.239ms | Large | ⚠️ SLOW |

---

## 🎯 CATEGORY 15: TIME SERIES PREDICTION (LSTM)

**Purpose:** LSTM-based time series forecasting

| Priority | Model | Format | Latency | Params | Status |
|----------|-------|--------|---------|--------|--------|
| **1st CHOICE** | financial_lstm_final_optimized | ONNX | 0.249ms | Small | ⭐ PRODUCTION |
| **2nd CHOICE** | lstm_trading_model | Keras | 262.069ms | 0.02M | ✅ BACKUP |
| **3rd CHOICE** | quantum_lstm_final_production | ONNX | 17.048ms | 11.2M | ⚠️ SLOW |

---

## 🎯 CATEGORY 16: ANOMALY DETECTION

**Purpose:** Detect anomalies in market data

| Priority | Model | Format | Latency | Purpose | Status |
|----------|-------|--------|---------|---------|--------|
| **1st CHOICE** | autoencoder_optimized | ONNX | 2.154ms | Autoencoder-based | ⭐ PRODUCTION |
| **2nd CHOICE** | autoencoder_optimized (09_ANOMALY) | ONNX | 3.215ms | Duplicate | ✅ BACKUP |

---

## 🎯 CATEGORY 17: ENTRY TIMING

**Purpose:** Optimize trade entry timing (microsecond precision)

| Priority | Model | Format | Latency | Input | Status |
|----------|-------|--------|---------|-------|--------|
| **1st CHOICE** | ONNX_Quantum_Entry_Timing Model (05) | ONNX | 3.756ms | 100x128 | ⭐ PRODUCTION |
| **2nd CHOICE** | ONNX_Quantum_Entry_Timing Model (11) | ONNX | 4.049ms | 100x128 | ✅ BACKUP |

---

## 🎯 CATEGORY 18: HIGH-FREQUENCY SCALPING

**Purpose:** Generate HFT scalping signals

| Priority | Model | Format | Latency | Input | Status |
|----------|-------|--------|---------|-------|--------|
| **1st CHOICE** | ONNX_HFT_ScalperSignal_optimized | ONNX | 9.140ms | 128x72 | ⚠️ SLOW |

**Note:** Only 1 model, relatively slow for HFT

---

## 🎯 CATEGORY 19: GRADIENT BOOSTING (TRADITIONAL ML)

**Purpose:** Traditional ML ensemble methods

| Priority | Model | Format | Latency | Purpose | Status |
|----------|-------|--------|---------|---------|--------|
| **1st CHOICE** | ONNXGBoost | ONNX | 0.146ms | Gradient boosting | ⭐ PRODUCTION |
| **2nd CHOICE** | xgboost_regressor_enhanced | PKL | 0.442ms | XGBoost regression | ✅ BACKUP |

---

## 🎯 CATEGORY 20: REGRESSION MODELS

**Purpose:** General-purpose regression

| Priority | Model | Format | Latency | Features | Status |
|----------|-------|--------|---------|----------|--------|
| **1st CHOICE** | xgboost_regressor_enhanced | PKL | 0.442ms | Auto | ⭐ PRODUCTION |
| **2nd CHOICE** | reg.pkl (LightGBM) | PKL | 0.887ms | 250 | ✅ BACKUP |
| **3rd CHOICE** | Regressor_lightgbm_optimized | ONNX | 0.064ms | 250 | ✅ BACKUP (Also in Category 1) |

---

## 🎯 CATEGORY 21-28: PYTORCH CHECKPOINTS (Need Architecture)

### CATEGORY 21: CONFIDENCE META-LEARNING
- **athena_prime.pth** (0.29 MB) - ✅ Loaded

### CATEGORY 22: LSTM BEST MODEL
- **best_model.pth** (191.64 MB) - ✅ Loaded

### CATEGORY 23: TRANSFORMER - BEST
- **final_best_transformer.pth** (2.37 MB) - ✅ Loaded

### CATEGORY 24: TRANSFORMER - ULTRA
- **super_ultra_transformer_best.pth** (86.88 MB) - ✅ Loaded

### CATEGORY 25: TRANSFORMER - ORIGINAL
- **final_original_model.pth** (2.36 MB) - ✅ Loaded

### CATEGORY 26: TABNET
- **network.pt** (0.10 MB) - ✅ Loaded

### CATEGORY 27: VARIATIONAL AUTOENCODER
- ❌ UltraVAE models failed (custom class)

### CATEGORY 28: HMM REGIME DETECTION
- ❌ HMM models failed (missing hmmlearn library)

---

## 📋 PRODUCTION DEPLOYMENT PRIORITY

### **TIER 1 - CRITICAL PATH (Ultra Fast <1ms)**
1. **ModelGovernor_Meta** (0.063ms) - Decide which models to use
2. **Regressor_lightgbm** (0.064ms) - Data quality scoring
3. **ModelRouter_Meta** (0.083ms) - Route to best decision
4. **Quantum Meta-Strategy Selector** (0.108ms) - Select strategies
5. **quantum_volatility_model** (0.111ms) - Volatility forecast
6. **ONNXGBoost** (0.146ms) - Traditional ML
7. **ONNX Signal Aggregator** (0.237ms) - Aggregate signals

**Total Latency: ~0.87ms**

### **TIER 2 - CONFIDENCE & RISK (<2ms)**
8. **xgboost_regressor_enhanced** (0.442ms) - Regression
9. **model_risk_governor** (0.492ms) - Risk management
10. **xgboost_confidence_model** (0.503ms) - Confidence calibration
11. **ONNX_RegimeClassifier** (0.719ms) - Regime detection
12. **catboost_confidence_model** (1.205ms) - Backup confidence

**Total Added Latency: +3.36ms**

### **TIER 3 - UNCERTAINTY QUANTIFICATION (Parallel Ensemble)**
13. **Uncertainty Ensemble (10 clf + 10 reg)** (~700ms) - Run in parallel
14. **Bayesian Ensemble (5 models)** (~3200ms) - Run in parallel offline

### **TIER 4 - ADVANCED FEATURES (Optional)**
15. **Pattern Recognition CNN** (78ms) - Pattern detection
16. **LSTM Regime Classifier** (279ms) - Advanced regime
17. **Entry Timing** (3.7ms) - Microsecond optimization

---

## 🎯 RECOMMENDED ECOSYSTEM CONFIGURATION

### **REAL-TIME TRADING (<1ms latency)**
- Use TIER 1 models only (7 models)
- Total latency: ~0.87ms
- All ONNX format

### **STANDARD TRADING (<5ms latency)**
- TIER 1 + TIER 2 (12 models)
- Total latency: ~4.23ms
- Mix of ONNX and PKL

### **FULL ECOSYSTEM (with uncertainty)**
- TIER 1 + TIER 2 + TIER 3
- Core latency: ~4.23ms
- Uncertainty ensemble: parallel/async
- Total models active: 32+

---

## 📊 SUMMARY BY FORMAT

### **ONNX (24 models) - PRODUCTION READY**
- Ultra-fast (<1ms): 7 models
- Fast (<10ms): 13 models
- Slow (>10ms): 4 models

### **PKL (11 models) - BACKUP & SPECIALTY**
- Fast (<3ms): 11 models
- Best for: XGBoost, LightGBM, CatBoost

### **KERAS (35 models) - ENSEMBLE & ADVANCED**
- Fast (<100ms): 12 models
- Moderate (100-300ms): 7 models
- Slow (>600ms): 16 models (Bayesian ensemble)
- Best for: Uncertainty quantification, ensemble methods

### **PYTORCH (6 checkpoints) - FUTURE INTEGRATION**
- All loaded, need architecture for inference
- Potential for advanced transformer/LSTM models

---

## ✅ NEXT STEPS

1. **Integrate TIER 1 models** - Core trading pipeline (<1ms)
2. **Add TIER 2 selectively** - Based on performance needs
3. **Deploy TIER 3 async** - For confidence intervals
4. **Test end-to-end** - Full pipeline validation
5. **A/B test alternatives** - Compare 1st vs 2nd choice models

---

**END OF CATEGORIZATION**
