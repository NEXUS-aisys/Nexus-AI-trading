# NEXUS AI - NON-ONNX MODEL ANALYSIS REPORT

## Summary

- **PKL Models:** 18
- **Keras Models:** 38
- **PyTorch Models:** 6
- **Total:** 62

## 📦 PKL MODELS

- **Working:** 11/18
- **Failed:** 7/18

### ✅ Working PKL Models

| Model | Type | Size (MB) | Latency (ms) | Strategy |
|-------|------|-----------|--------------|----------|
| catboost_confidence_model.pkl | CatBoostClassifier | 0.12 | 1.860 | catboost_50_features |
| cls.pkl | CalibratedClassifierCV | 0.18 | 1275.929 | sklearn_generic_250_features |
| final_classifier.pkl | XGBClassifier | 0.28 | 2.670 | xgboost_batch_1 |
| reg.pkl | LGBMRegressor | 0.07 | 1.083 | lightgbm_correct_features_batch_1 |
| catboost_confidence_model.pkl | CatBoostClassifier | 0.12 | 1.205 | catboost_50_features |
| xgboost_confidence_model.pkl | XGBClassifier | 0.39 | 0.503 | xgboost_batch_1 |
| cls.pkl | CalibratedClassifierCV | 0.18 | 1.481 | sklearn_generic_250_features |
| reg.pkl | LGBMRegressor | 0.07 | 0.887 | lightgbm_correct_features_batch_1 |
| final_classifier.pkl | XGBClassifier | 0.28 | 0.488 | xgboost_batch_1 |
| xgboost_classifier_enhanced_h1.pkl | XGBClassifier | 6.23 | 1.339 | xgboost_batch_1 |
| xgboost_regressor_enhanced_h1.pkl | XGBRegressor | 0.11 | 0.442 | xgboost_batch_1 |

### ❌ Failed PKL Models

| Model | Size (MB) | Error |
|-------|-----------|-------|
| hmm_regime_detector_h1_1_1.pkl | 0.03 | No module named 'hmmlearn' |
| hmm_regime_detector_h1.pkl | 0.03 | No module named 'hmmlearn' |
| athena_prime_scaler.pkl | 0.00 | Unknown error |
| random_forest_confidence_model.pkl | 105.95 | Unknown error |
| hmm_regime_detector_h1.pkl | 0.03 | No module named 'hmmlearn' |
| hmm_regime_detector_h1.pkl | 0.03 | No module named 'hmmlearn' |
| ultimate_volatility_model.pkl | 11.91 | Can't get attribute 'PowerfulEnsemble' on <module '__main__' from 'C:\\Users\\Nexus AI\\Documents\\N |

## 🧠 KERAS MODELS

- **Working:** 35/38
- **Failed:** 3/38

### ✅ Working Keras Models

| Model | Size (MB) | Params (M) | Latency (ms) |
|-------|-----------|------------|--------------|\ n| cnn1d_best_model.keras | 0.17 | 0.01 | 227.419 |
| lstm_trading_model.keras | 0.29 | 0.02 | 286.596 |
| final_model.keras | 0.17 | 0.01 | 79.568 |
| member_0.keras | 1.79 | 0.15 | 662.563 |
| member_1.keras | 1.79 | 0.15 | 641.044 |
| member_0.keras | 1.79 | 0.15 | 645.544 |
| member_1.keras | 1.79 | 0.15 | 657.352 |
| member_2.keras | 1.79 | 0.15 | 648.463 |
| member_3.keras | 1.79 | 0.15 | 891.514 |
| member_4.keras | 1.79 | 0.15 | 653.879 |
| lstm_trading_model.keras | 0.29 | 0.02 | 262.069 |
| best_val_model.keras | 0.17 | 0.01 | 79.631 |
| final_model.keras | 0.17 | 0.01 | 78.081 |
| lstm_regime_classifier_h1.h5 | 0.58 | 0.05 | 282.178 |
| lstm_regime_classifier_h1.h5 | 0.58 | 0.05 | 279.154 |
| uncertainty_clf_model_0_h1.h5 | 0.31 | 0.02 | 65.042 |
| uncertainty_clf_model_1_h1.h5 | 0.31 | 0.02 | 72.822 |
| uncertainty_clf_model_2_h1.h5 | 0.31 | 0.02 | 77.081 |
| uncertainty_clf_model_3_h1.h5 | 0.31 | 0.02 | 69.037 |
| uncertainty_clf_model_4_h1.h5 | 0.31 | 0.02 | 66.397 |
| uncertainty_clf_model_5_h1.h5 | 0.31 | 0.02 | 65.525 |
| uncertainty_clf_model_6_h1.h5 | 0.31 | 0.02 | 69.602 |
| uncertainty_clf_model_7_h1.h5 | 0.31 | 0.02 | 64.191 |
| uncertainty_clf_model_8_h1.h5 | 0.31 | 0.02 | 74.631 |
| uncertainty_clf_model_9_h1.h5 | 0.31 | 0.02 | 75.216 |
| uncertainty_reg_model_0_h1.h5 | 0.31 | 0.02 | 67.728 |
| uncertainty_reg_model_1_h1.h5 | 0.31 | 0.02 | 81.057 |
| uncertainty_reg_model_2_h1.h5 | 0.31 | 0.02 | 65.709 |
| uncertainty_reg_model_3_h1.h5 | 0.31 | 0.02 | 69.484 |
| uncertainty_reg_model_4_h1.h5 | 0.31 | 0.02 | 64.558 |
| uncertainty_reg_model_5_h1.h5 | 0.31 | 0.02 | 67.722 |
| uncertainty_reg_model_6_h1.h5 | 0.31 | 0.02 | 66.196 |
| uncertainty_reg_model_7_h1.h5 | 0.31 | 0.02 | 67.246 |
| uncertainty_reg_model_8_h1.h5 | 0.31 | 0.02 | 68.529 |
| uncertainty_reg_model_9_h1.h5 | 0.31 | 0.02 | 66.244 |

### ❌ Failed Keras Models

| Model | Size (MB) | Error |
|-------|-----------|-------|
| best_ultra_model_CHAMPION_V3.keras | 6.14 | Could not locate class 'UltraVAE'. Make sure custom classes are decorated with `@keras.saving.regist |
| best_ultra_model_v3.keras | 6.14 | Could not locate class 'UltraVAE'. Make sure custom classes are decorated with `@keras.saving.regist |
| best_model.h5 | 99.98 | Unknown layer: 'Cast'. Please ensure you are using a `keras.utils.custom_object_scope` and that this |

## 🔥 PYTORCH MODELS

| Model | Size (MB) | Status | Notes |
|-------|-----------|--------|-------|
| athena_prime.pth | 0.29 | ✅ Loaded | Loaded checkpoint - need model architecture to test inference |
| best_model.pth | 191.64 | ✅ Loaded | Loaded checkpoint - need model architecture to test inference |
| final_best_transformer.pth | 2.37 | ✅ Loaded | Loaded checkpoint - need model architecture to test inference |
| final_original_model.pth | 2.36 | ✅ Loaded | Loaded checkpoint - need model architecture to test inference |
| super_ultra_transformer_best.pth | 86.88 | ✅ Loaded | Loaded checkpoint - need model architecture to test inference |
| network.pt | 0.10 | ✅ Loaded | Loaded checkpoint - need model architecture to test inference |

