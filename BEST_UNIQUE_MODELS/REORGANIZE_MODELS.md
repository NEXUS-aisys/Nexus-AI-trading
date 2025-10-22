# NEXUS AI - MODEL REORGANIZATION PLAN
**Clean up and organize models by category before integration**

Generated: 2025-10-20

---

## 🎯 REORGANIZATION STRATEGY

### **GOAL:**
- Create 20 category folders (based on function)
- Move 1st choice models to category folders
- Move 2nd choice models to BACKUP folder
- DELETE unused/failed models
- Clean structure for integration

---

## 📁 NEW FOLDER STRUCTURE

```
BEST_UNIQUE_MODELS/
├── PRODUCTION/                          # Active production models
│   ├── 01_DATA_QUALITY/
│   │   └── Regressor_lightgbm_optimized.onnx (1st choice)
│   ├── 02_VOLATILITY_FORECAST/
│   │   └── quantum_volatility_model_final_optimized.onnx (1st choice)
│   ├── 03_REGIME_DETECTION/
│   │   └── ONNX_RegimeClassifier_optimized.onnx (1st choice)
│   ├── 04_STRATEGY_SELECTION/
│   │   └── Quantum Meta-Strategy Selector.onnx (1st choice)
│   ├── 05_SIGNAL_AGGREGATION/
│   │   └── ONNX Signal Aggregator.onnx (1st choice)
│   ├── 06_MODEL_ROUTING/
│   │   └── ModelRouter_Meta_optimized.onnx (1st choice)
│   ├── 07_MODEL_GOVERNANCE/
│   │   └── ONNX_ModelGovernor_Meta_optimized.onnx (1st choice)
│   ├── 08_RISK_MANAGEMENT/
│   │   └── model_risk_governor_optimized.onnx (1st choice)
│   ├── 09_CONFIDENCE_CALIBRATION/
│   │   └── xgboost_confidence_model.pkl (1st choice)
│   ├── 10_MARKET_CLASSIFICATION/
│   │   └── xgboost_classifier_enhanced.pkl (1st choice)
│   ├── 11_REGRESSION/
│   │   └── xgboost_regressor_enhanced.pkl (1st choice)
│   ├── 12_GRADIENT_BOOSTING/
│   │   └── ONNXGBoost.onnx (1st choice)
│   ├── 13_UNCERTAINTY_CLASSIFICATION/
│   │   ├── uncertainty_clf_model_0_h1.h5
│   │   ├── uncertainty_clf_model_1_h1.h5
│   │   ├── ... (10 models for ensemble)
│   │   └── uncertainty_clf_model_9_h1.h5
│   ├── 14_UNCERTAINTY_REGRESSION/
│   │   ├── uncertainty_reg_model_0_h1.h5
│   │   ├── ... (10 models for ensemble)
│   │   └── uncertainty_reg_model_9_h1.h5
│   ├── 15_BAYESIAN_ENSEMBLE/
│   │   ├── member_0.keras
│   │   ├── ... (5 models)
│   │   └── member_4.keras
│   ├── 16_PATTERN_RECOGNITION/
│   │   └── final_model.keras (CNN1D, 1st choice)
│   ├── 17_LSTM_TIME_SERIES/
│   │   └── financial_lstm_final_optimized.onnx (1st choice)
│   ├── 18_ANOMALY_DETECTION/
│   │   └── autoencoder_optimized.onnx (1st choice)
│   ├── 19_ENTRY_TIMING/
│   │   └── ONNX_Quantum_Entry_Timing Model_optimized.onnx (1st choice)
│   └── 20_HFT_SCALPING/
│       └── ONNX_HFT_ScalperSignal_optimized.onnx (1st choice)
│
├── BACKUP/                              # 2nd choice backup models
│   ├── 01_DATA_QUALITY/
│   │   └── reg.pkl (2nd choice)
│   ├── 02_VOLATILITY_FORECAST/
│   │   └── [no backup - only 1 working]
│   ├── 03_REGIME_DETECTION/
│   │   └── lstm_regime_classifier_h1.h5 (2nd choice)
│   ├── 04_STRATEGY_SELECTION/
│   │   └── Quantum Meta-Strategy Selector_optimized.onnx (2nd choice)
│   ├── 05_SIGNAL_AGGREGATION/
│   │   └── quantum_signal_aggregator_meta_q2025_optimized.onnx (2nd choice)
│   ├── 06_MODEL_ROUTING/
│   │   └── ModelRouter_Meta_optimized_duplicate.onnx (2nd choice)
│   ├── 09_CONFIDENCE_CALIBRATION/
│   │   └── catboost_confidence_model.pkl (2nd choice)
│   ├── 10_MARKET_CLASSIFICATION/
│   │   ├── final_classifier.pkl (2nd choice)
│   │   └── cls.pkl (3rd choice)
│   ├── 11_REGRESSION/
│   │   └── reg.pkl (2nd choice)
│   ├── 16_PATTERN_RECOGNITION/
│   │   ├── best_val_model.keras (2nd choice)
│   │   └── cnn1d_best_model.keras (3rd choice)
│   └── 17_LSTM_TIME_SERIES/
│       ├── lstm_trading_model.keras (2nd choice)
│       └── quantum_lstm_final_production_optimized.onnx (3rd choice)
│
├── ARCHIVE/                             # Working but not selected
│   ├── ALTERNATIVE_VOLATILITY/
│   ├── ALTERNATIVE_LSTM/
│   └── EXTRA_UNCERTAINTY/
│
└── DELETE/                              # Failed models to remove
    ├── CORRUPTED_ONNX/
    │   ├── All REDEY ONNX models (6 files - protobuf errors)
    │   ├── ONNX_GARCHML_VolForecast_optimized.onnx
    │   └── transformer_model_optimized.onnx
    ├── FAILED_PKL/
    │   ├── hmm_regime_detector_h1.pkl (missing hmmlearn)
    │   ├── random_forest_confidence_model.pkl
    │   └── ultimate_volatility_model.pkl
    ├── FAILED_KERAS/
    │   ├── best_ultra_model_CHAMPION_V3.keras (UltraVAE)
    │   ├── best_ultra_model_v3.keras (UltraVAE)
    │   └── best_model.h5 (Cast layer error)
    └── PYTORCH_CHECKPOINTS/
        └── [All .pth/.pt files - need architecture]
```

---

## 🔄 REORGANIZATION STEPS

### **STEP 1: Create New Folder Structure**

Create these folders:
- `PRODUCTION/` with 20 category subfolders
- `BACKUP/` with category subfolders
- `ARCHIVE/` for working but unused models
- `DELETE/` for failed models

### **STEP 2: Move 1st Choice Models to PRODUCTION**

| Source | Destination |
|--------|-------------|
| `07_REGRESSION/Regressor_lightgbm_optimized.onnx` | `PRODUCTION/01_DATA_QUALITY/` |
| `02_VOLATILITY_PREDICTION/quantum_volatility_model_final_optimized.onnx` | `PRODUCTION/02_VOLATILITY_FORECAST/` |
| `01_REGIME_CLASSIFICATION/ONNX_RegimeClassifier_optimized.onnx` | `PRODUCTION/03_REGIME_DETECTION/` |
| `08_META_LEARNING/Quantum Meta-Strategy Selector.onnx` | `PRODUCTION/04_STRATEGY_SELECTION/` |
| `05_SIGNAL_PROCESSING/ONNX Signal Aggregator.onnx` | `PRODUCTION/05_SIGNAL_AGGREGATION/` |
| `08_META_LEARNING/ModelRouter_Meta_optimized.onnx` | `PRODUCTION/06_MODEL_ROUTING/` |
| `08_META_LEARNING/ONNX_ModelGovernor_Meta_optimized.onnx` | `PRODUCTION/07_MODEL_GOVERNANCE/` |
| `01_RISK_MANAGEMENT/model_risk_governor_optimized.onnx` | `PRODUCTION/08_RISK_MANAGEMENT/` |
| `Onnx/REDEY/confidence_meta/xgboost_confidence_model.pkl` | `PRODUCTION/09_CONFIDENCE_CALIBRATION/` |
| `Onnx/REDEY/Xgbosst/xgboost_classifier_enhanced_h1.pkl` | `PRODUCTION/10_MARKET_CLASSIFICATION/` |
| `Onnx/REDEY/Xgbosst/xgboost_regressor_enhanced_h1.pkl` | `PRODUCTION/11_REGRESSION/` |
| `10_TRADITIONAL_ML/ONNXGBoost.onnx` | `PRODUCTION/12_GRADIENT_BOOSTING/` |
| `Onnx/REDEY/uncertainty/uncertainty_clf_model_*_h1.h5` (10 files) | `PRODUCTION/13_UNCERTAINTY_CLASSIFICATION/` |
| `Onnx/REDEY/uncertainty/uncertainty_reg_model_*_h1.h5` (10 files) | `PRODUCTION/14_UNCERTAINTY_REGRESSION/` |
| `Onnx/REDEY/bayesia/member_*.keras` (5 files) | `PRODUCTION/15_BAYESIAN_ENSEMBLE/` |
| `Onnx/REDEY/train_cnn1d/final_model.keras` | `PRODUCTION/16_PATTERN_RECOGNITION/` |
| `04_LSTM_TIME_SERIES/financial_lstm_final_optimized.onnx` | `PRODUCTION/17_LSTM_TIME_SERIES/` |
| `09_ANOMALY_DETECTION/autoencoder_optimized.onnx` | `PRODUCTION/18_ANOMALY_DETECTION/` |
| `05_SIGNAL_PROCESSING/ONNX_Quantum_Entry_Timing Model_optimized.onnx` | `PRODUCTION/19_ENTRY_TIMING/` |
| `05_SIGNAL_PROCESSING/ONNX_HFT_ScalperSignal_optimized.onnx` | `PRODUCTION/20_HFT_SCALPING/` |

### **STEP 3: Move 2nd Choice Models to BACKUP**

| Source | Destination |
|--------|-------------|
| `Onnx/REDEY/LightGBM/reg.pkl` | `BACKUP/01_DATA_QUALITY/` |
| `Onnx/REDEY/Regime Detector/lstm_regime_classifier_h1.h5` | `BACKUP/03_REGIME_DETECTION/` |
| `Onnx/Quantum Meta-Strategy Selector_optimized.onnx` | `BACKUP/04_STRATEGY_SELECTION/` |
| `05_SIGNAL_PROCESSING/quantum_signal_aggregator_meta_q2025_optimized.onnx` | `BACKUP/05_SIGNAL_AGGREGATION/` |
| `Onnx/REDEY/confidence_meta/catboost_confidence_model.pkl` | `BACKUP/09_CONFIDENCE_CALIBRATION/` |
| `Onnx/REDEY/Xgbosst/final_classifier.pkl` | `BACKUP/10_MARKET_CLASSIFICATION/` |
| `Onnx/REDEY/LightGBM/cls.pkl` | `BACKUP/10_MARKET_CLASSIFICATION/` (3rd choice) |
| `Onnx/REDEY/train_cnn1d/best_val_model.keras` | `BACKUP/16_PATTERN_RECOGNITION/` |
| `03_CNN_DEEP_LEARNING/cnn1d_best_model.keras` | `BACKUP/16_PATTERN_RECOGNITION/` (3rd choice) |
| `Onnx/REDEY/LSTM/lstm_trading_model.keras` | `BACKUP/17_LSTM_TIME_SERIES/` |

### **STEP 4: Move to DELETE Folder**

**CORRUPTED ONNX (8 files):**
- `Onnx/REDEY/LSTM/best_model.onnx`
- `Onnx/REDEY/LSTM/cls.onnx`
- `Onnx/REDEY/LSTM/reg.onnx`
- `Onnx/REDEY/LSTM/lstm_trading_model.onnx`
- `Onnx/REDEY/LSTM/final_best_transformer.onnx`
- `Onnx/REDEY/LSTM/super_ultra_transformer_best.onnx`
- `02_VOLATILITY_PREDICTION/ONNX_GARCHML_VolForecast_optimized.onnx`
- `04_LSTM_TIME_SERIES/transformer_model_optimized.onnx`

**FAILED PKL (7 files):**
- `04_LSTM_TIME_SERIES/hmm_regime_detector_h1_1_1.pkl`
- `07_REGRESSION/hmm_regime_detector_h1.pkl`
- `Onnx/REDEY/Regime Detector/hmm_regime_detector_h1.pkl`
- `Onnx/REDEY/Confidence meta/athena_prime_scaler.pkl`
- `Onnx/REDEY/confidence_meta/random_forest_confidence_model.pkl`
- `Onnx/REDEY/volatility hybrid/ultimate_volatility_model.pkl`

**FAILED KERAS (3 files):**
- `Onnx/REDEY/autoencoder/best_ultra_model_CHAMPION_V3.keras`
- `Onnx/REDEY/autoencoder/best_ultra_model_v3.keras`
- `Onnx/REDEY/volatility hybrid/best_model.h5`

**PYTORCH CHECKPOINTS (6 files - optional to keep):**
- All `.pth` and `.pt` files in `Onnx/REDEY/` subfolders

### **STEP 5: Clean Up Empty Folders**

Delete these empty/old folders after moving:
- `01_RISK_MANAGEMENT/` (after moving models)
- `02_VOLATILITY_PREDICTION/` (after moving models)
- `03_CNN_DEEP_LEARNING/` (after moving models)
- `04_LSTM_TIME_SERIES/` (after moving models)
- `05_SIGNAL_PROCESSING/` (after moving models)
- `06_CLASSIFICATION/` (likely empty)
- `07_REGRESSION/` (after moving models)
- `08_META_LEARNING/` (after moving models)
- `09_ANOMALY_DETECTION/` (after moving models)
- `10_TRADITIONAL_ML/` (after moving models)
- `11_ENTRY_TIMING/` (after moving models)
- `9 BAYESIAN/` (old folder)
- `Onnx/` entire folder (after moving all models)

---

## 📊 BEFORE vs AFTER

### **BEFORE (Current Mess):**
```
BEST_UNIQUE_MODELS/
├── 94 files scattered across 15+ folders
├── Duplicates in multiple locations
├── Failed models mixed with working
├── No clear structure
└── Hard to find what you need
```

### **AFTER (Clean Structure):**
```
BEST_UNIQUE_MODELS/
├── PRODUCTION/ → 46 working models in 20 categories
├── BACKUP/ → 15 backup models
├── ARCHIVE/ → 9 alternative models (optional)
└── DELETE/ → 24 failed models (to remove)

Total: 70 working + 24 to delete = 94 files accounted for
```

---

## ✅ VERIFICATION CHECKLIST

After reorganization, verify:
- [ ] All 46 PRODUCTION models in correct category folders
- [ ] All 15 BACKUP models organized by category
- [ ] DELETE folder contains all 24 failed models
- [ ] No duplicates in PRODUCTION
- [ ] Old folders deleted or empty
- [ ] Easy to find any model by category

---

## 🚨 IMPORTANT NOTES

1. **BACKUP BEFORE MOVING**
   - Create a zip backup of entire `BEST_UNIQUE_MODELS/` folder first
   - Just in case we need to restore

2. **TEST PATHS AFTER**
   - Update all file paths in code
   - Test that models load correctly from new locations

3. **DELETE FOLDER**
   - Don't actually delete immediately
   - Move to DELETE folder first
   - After 1 week of successful operation, then permanently delete

---

**READY TO EXECUTE REORGANIZATION?**

I can create a Python script to automate this entire reorganization!
