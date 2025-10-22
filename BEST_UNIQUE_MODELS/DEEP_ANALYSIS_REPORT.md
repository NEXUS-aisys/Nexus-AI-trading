# NEXUS AI - DEEP ONNX MODEL ANALYSIS REPORT

## Summary

- **Total Models:** 32
- **Successful:** 24
- **Failed:** 8

## ✅ WORKING MODELS

| Model | Size (MB) | Latency (ms) | Strategy |
|-------|-----------|--------------|----------|
| 08_META_LEARNING\ONNX_ModelGovernor_Meta_optimized.onnx | 0.09 | 0.063 | metadata_based |
| Onnx\Regressor_lightgbm_optimized.onnx | 0.14 | 0.064 | metadata_based |
| 08_META_LEARNING\ModelRouter_Meta_optimized.onnx | 0.19 | 0.083 | metadata_based |
| Onnx\ModelRouter_Meta_optimized.onnx | 0.19 | 0.089 | metadata_based |
| 07_REGRESSION\Regressor_lightgbm_optimized.onnx | 0.14 | 0.098 | metadata_based |
| 08_META_LEARNING\Quantum Meta-Strategy Selector.onnx | 1.38 | 0.108 | metadata_based |
| Onnx\quantum_volatility_model_final_optimized.onnx | 1.41 | 0.111 | metadata_based |
| Onnx\Quantum Meta-Strategy Selector_optimized.onnx | 1.38 | 0.116 | metadata_based |
| 02_VOLATILITY_PREDICTION\quantum_volatility_model_final_optimized.onnx | 1.41 | 0.131 | metadata_based |
| 10_TRADITIONAL_ML\ONNXGBoost.onnx | 4.55 | 0.146 | metadata_based |
| 05_SIGNAL_PROCESSING\ONNX Signal Aggregator.onnx | 0.28 | 0.237 | metadata_based |
| 04_LSTM_TIME_SERIES\financial_lstm_final_optimized.onnx | 0.29 | 0.249 | metadata_based |
| 05_SIGNAL_PROCESSING\quantum_signal_aggregator_meta_q2025_optimized.onnx | 0.38 | 0.260 | metadata_based |
| 01_RISK_MANAGEMENT\model_risk_governor_optimized.onnx | 0.12 | 0.492 | batch_2 |
| 01_REGIME_CLASSIFICATION\ONNX_RegimeClassifier_optimized.onnx | 1.66 | 0.719 | metadata_based |
| 09_ANOMALY_DETECTION\autoencoder_optimized.onnx | 1.95 | 2.154 | metadata_based |
| 03_CNN_DEEP_LEARNING\autoencoder_optimized.onnx | 1.95 | 3.215 | metadata_based |
| 05_SIGNAL_PROCESSING\ONNX_Quantum_Entry_Timing Model_optimized.onnx | 16.79 | 3.756 | metadata_based |
| 11_ENTRY_TIMING\ONNX_Quantum_Entry_Timing Model_optimized.onnx | 16.79 | 4.049 | metadata_based |
| Onnx\ONNX_HFT_ScalperSignal_optimized.onnx | 22.87 | 9.140 | metadata_based |
| 05_SIGNAL_PROCESSING\ONNX_HFT_ScalperSignal_optimized.onnx | 22.87 | 10.100 | metadata_based |
| Onnx\quantum_lstm_final_production_optimized.onnx | 44.34 | 17.048 | metadata_based |
| 04_LSTM_TIME_SERIES\quantum_lstm_final_production_optimized.onnx | 44.34 | 17.222 | metadata_based |
| 03_CNN_DEEP_LEARNING\CNN1d_optimized.onnx | 22.27 | 19.239 | metadata_based |

## ❌ FAILED MODELS

| Model | Size (MB) | Error |
|-------|-----------|-------|
| 02_VOLATILITY_PREDICTION\ONNX_GARCHML_VolForecast_optimized.onnx | 0.89 | [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gather node. |
| 04_LSTM_TIME_SERIES\transformer_model_optimized.onnx | 19.26 | [ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned while running TopK node. Name:'/blocks |
| Onnx\REDEY\LSTM\best_model.onnx | 191.64 | ONNX load failed: Error parsing message with type 'onnx.ModelProto' |
| Onnx\REDEY\LSTM\cls.onnx | 0.18 | ONNX load failed: Error parsing message with type 'onnx.ModelProto' |
| Onnx\REDEY\LSTM\final_best_transformer.onnx | 2.37 | ONNX load failed: Error parsing message with type 'onnx.ModelProto' |
| Onnx\REDEY\LSTM\lstm_trading_model.onnx | 0.29 | ONNX load failed: Error parsing message with type 'onnx.ModelProto' |
| Onnx\REDEY\LSTM\reg.onnx | 0.07 | ONNX load failed: Error parsing message with type 'onnx.ModelProto' |
| Onnx\REDEY\LSTM\super_ultra_transformer_best.onnx | 86.88 | ONNX load failed: Error parsing message with type 'onnx.ModelProto' |
