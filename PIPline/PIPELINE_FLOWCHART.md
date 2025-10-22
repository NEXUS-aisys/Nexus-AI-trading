# NEXUS AI PIPELINE FLOWCHART
**8-Layer Visual Flow with 70 ML Models**

Last Updated: 2025-10-21

---

## COMPLETE EXECUTION FLOW

```
┌──────────────────────────────────────────────────────────────┐
│  START: Market Data Received (Symbol: BTCUSDT)               │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ↓
            ┌───────────────────────┐
            │ HMAC-SHA256 Verified? │
            └───────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         NO                    YES
         │                      │
         ↓                      ↓
    ┌────────┐          ┌─────────────┐
    │ REJECT │          │  CONTINUE   │
    └────────┘          └──────┬──────┘
                               │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 1: MARKET QUALITY ASSESSMENT (Hybrid, 10.83ms)        ║
╚═══════════════════════════════════════════════════════════════╝
                               │
                               ↓
                   ┌─────────────────────────────────────────┐
                   │ MQScore 6D Engine (PRIMARY) ✅          │
                   │ Model: LightGBM (existing, proven)      │
                   │ Latency: ~10ms                          │
                   │ ───────────────────────────────────────│
                   │ Input: 65 engineered features           │
                   │                                         │
                   │ Output: 6D Assessment                   │
                   │  - Composite MQScore (0-1)             │
                   │  - Liquidity (15%)                     │
                   │  - Volatility (15%)                    │
                   │  - Momentum (15%)                      │
                   │  - Imbalance (15%)                     │
                   │  - Trend Strength (20%)                │
                   │  - Noise Level (20%)                   │
                   │  - Market Grade (A+ to F)              │
                   │  - Regime Classification               │
                   └──────────┬──────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   MQScore < 0.5?        │
                   └──────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    YES               NO
                    │                 │
                    ↓                 ↓
              ┌──────────┐     ┌──────────────────┐
              │   SKIP   │     │ Check Liquidity  │
              │ (Low     │     │ >= 0.3?          │
              │ Quality) │     └────────┬─────────┘
              └──────────┘              │
                              ┌─────────┴─────────┐
                              YES               NO
                              │                 │
                              ↓                 ↓
                    ┌──────────────┐     ┌──────────┐
                    │ Check Regime │     │   SKIP   │
                    │ Safe?        │     │ (Low Liq)│
                    └──────┬───────┘     └──────────┘
                           │
                ┌──────────┴──────────┐
                CRISIS              SAFE
                │                   │
                ↓                   ↓
          ┌──────────┐     ┌────────────────────────────┐
          │   SKIP   │     │ ENHANCEMENTS (optional):   │
          │ (Unsafe) │     │ ─────────────────────────│
          └──────────┘     │ • Quantum Volatility       │
                           │   (+0.111ms) ⭐            │
                           │ • ONNX Regime              │
                           │   (+0.719ms) ⭐            │
                           │                            │
                           │ Ensemble with MQScore      │
                           └──────────┬─────────────────┘
                                      │
                                      ↓
                            ┌──────────────┐
                            │  CONTINUE    │
                            └──────┬───────┘
                                   │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 2: STRATEGY EXECUTION (20 Strategies, Parallel)       ║
╚═══════════════════════════════════════════════════════════════╝
                                               │
                                               ↓
                              ┌─────────────────────────────────┐
                              │ Execute 20 Strategies           │
                              │ ThreadPoolExecutor(max=20)      │
                              │                                 │
                              │ Each outputs:                   │
                              │  signal: +1/-1/0                │
                              │  confidence: 0.0-1.0            │
                              └──────────┬──────────────────────┘
                                         │
                                         ↓
                              ┌─────────────────────┐
                              │ Filter: Keep only   │
                              │ confidence >= 0.65  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────┴──────────┐
                              Any signals passed?   │
                              └──────────┬──────────┘
                                         │
                               ┌─────────┴─────────┐
                               NO                 YES
                               │                  │
                               ↓                  ↓
                         ┌──────────┐      ┌─────────────┐
                         │   SKIP   │      │  CONTINUE   │
                         │ (No High │      └──────┬──────┘
                         │  Conf)   │             │
                         └──────────┘             │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 3: META-LEARNING (0.108ms, 1 ONNX model) ⭐ NEW       ║
╚═══════════════════════════════════════════════════════════════╝
                                                │
                                                ↓
                    ┌────────────────────────────────────────────┐
                    │ Quantum Meta-Strategy Selector              │
                    │ ──────────────────────────────────────────│
                    │ Input: 44 market features                  │
                    │  - Regime (from Layer 1)                   │
                    │  - Volatility (from Layer 1)               │
                    │  - Recent strategy performance             │
                    │                                            │
                    │ Output:                                    │
                    │  - strategy_weights[19] (0-1 per strategy)│
                    │  - anomaly_score (market anomaly detect)  │
                    │  - regime_confidence[3]                   │
                    │                                            │
                    │ Function: Dynamically assign weights      │
                    │           based on market conditions       │
                    └──────────────┬─────────────────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        anomaly_score > 0.8?  │
                        └──────────┬──────────┘
                                   │
                         ┌─────────┴─────────┐
                         YES               NO
                         │                 │
                         ↓                 ↓
                ┌────────────────┐   ┌─────────────┐
                │ Reduce position│   │  CONTINUE   │
                │ size by 50%    │   └──────┬──────┘
                └────────┬───────┘          │
                         └──────────────────┘
                                   │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 4: SIGNAL AGGREGATION (0.237ms, 1 ONNX model)         ║
╚═══════════════════════════════════════════════════════════════╝
                                   │
                                   ↓
                    ┌───────────────────────────────────────┐
                    │ ONNX Signal Aggregator                │
                    │ ─────────────────────────────────────│
                    │ weighted_sum = Σ(signal_i × weight_i  │
                    │                  × confidence_i)      │
                    │ total_weight = Σ(weight_i ×          │
                    │                  confidence_i)        │
                    │                                       │
                    │ aggregated_signal = weighted_sum /    │
                    │                     total_weight      │
                    │                                       │
                    │ Output: -1.0 (SELL) to +1.0 (BUY)   │
                    └──────────┬────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    signal_strength < 0.5?│
                    └──────────┬──────────┘
                               │
                     ┌─────────┴─────────┐
                     YES               NO
                     │                 │
                     ↓                 ↓
               ┌──────────┐     ┌────────────────┐
               │   SKIP   │     │ Determine      │
               │ (Weak)   │     │ Direction      │
               └──────────┘     │ BUY/SELL/HOLD  │
                                └────────┬───────┘
                                         │
                              ┌──────────┴──────────┐
                              Direction == HOLD?    │
                              └──────────┬──────────┘
                                         │
                               ┌─────────┴─────────┐
                               YES               NO
                               │                 │
                               ↓                 ↓
                         ┌──────────┐     ┌─────────────────┐
                         │   SKIP   │     │ Check Duplicate │
                         └──────────┘     │ active_orders?  │
                                          └────────┬────────┘
                                                   │
                                        ┌──────────┴──────────┐
                                        EXISTS             NEW
                                        │                  │
                                        ↓                  ↓
                                  ┌──────────┐      ┌──────────┐
                                  │   SKIP   │      │ CONTINUE │
                                  │ (Dup)    │      └─────┬────┘
                                  └──────────┘            │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 5: GOVERNANCE & ROUTING (0.146ms, 2 ONNX) ⭐ NEW      ║
╚═══════════════════════════════════════════════════════════════╝
                                                     │
                                                     ↓
                              ┌──────────────────────────────────┐
                              │ Model Governance (0.063ms)       │
                              │ ────────────────────────────────│
                              │ Input: 75 performance metrics    │
                              │ Output: model_weights[15]        │
                              │         trust levels per model   │
                              └────────────┬─────────────────────┘
                                           │
                                           ↓
                              ┌──────────────────────────────────┐
                              │ Decision Router (0.083ms)        │
                              │ ────────────────────────────────│
                              │ Input: 126 context features      │
                              │ Output: action_probs[2]          │
                              │         confidence               │
                              └────────────┬─────────────────────┘
                                           │
                                ┌──────────┴──────────┐
                                confidence < 0.7?     │
                                └──────────┬──────────┘
                                           │
                                 ┌─────────┴─────────┐
                                 YES               NO
                                 │                 │
                                 ↓                 ↓
                           ┌──────────┐     ┌──────────┐
                           │   SKIP   │     │ CONTINUE │
                           └──────────┘     └─────┬────┘
                                                  │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 6: RISK MANAGEMENT (1.7ms, 3 models)                  ║
╚═══════════════════════════════════════════════════════════════╝
                                                  │
                                                  ↓
                              ┌─────────────────────────────────┐
                              │ Risk Assessment (0.492ms)       │
                              │ Model: risk_governor ONNX       │
                              │ Output: risk_multiplier (0-1)   │
                              └────────────┬────────────────────┘
                                           │
                                ┌──────────┴──────────┐
                                risk_multiplier<0.3? │
                                └──────────┬──────────┘
                                           │
                                 ┌─────────┴─────────┐
                                 YES               NO
                                 │                 │
                                 ↓                 ↓
                           ┌──────────┐     ┌──────────────────┐
                           │  REJECT  │     │ Confidence Cal.  │
                           │ (Risky)  │     │ (0.503ms)        │
                           └──────────┘     └────────┬─────────┘
                                                     │
                                                     ↓
                                          ┌──────────────────────┐
                                          │ Market Classification│
                                          │ (1.339ms)            │
                                          │ 0=Bull, 1=Bear       │
                                          └────────┬─────────────┘
                                                   │
                                        ┌──────────┴──────────┐
                                        Signal conflicts with│
                                        market_class?        │
                                        └──────────┬──────────┘
                                                   │
                                         ┌─────────┴─────────┐
                                         YES               NO
                                         │                 │
                                         ↓                 ↓
                                   ┌──────────┐     ┌──────────────┐
                                   │  REJECT  │     │ 7-Layer Risk │
                                   │ (Conflict)│     │ Check        │
                                   └──────────┘     └──────┬───────┘
                                                           │
                                                ┌──────────┴──────────┐
                                                ANY layer fails?      │
                                                └──────────┬──────────┘
                                                           │
                                                 ┌─────────┴─────────┐
                                                 YES               NO
                                                 │                 │
                                                 ↓                 ↓
                                           ┌──────────┐     ┌──────────┐
                                           │  REJECT  │     │ APPROVE  │
                                           └──────────┘     └─────┬────┘
                                                                  │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 7: ORDER EXECUTION                                     ║
╚═══════════════════════════════════════════════════════════════╝
                                                                  │
                                                                  ↓
                                                      ┌──────────────────┐
                                                      │ Create Order     │
                                                      │ (Unique ID)      │
                                                      └────────┬─────────┘
                                                               │
                                                               ↓
                                                      ┌──────────────────┐
                                                      │ Submit to        │
                                                      │ Exchange API     │
                                                      └────────┬─────────┘
                                                               │
                                                               ↓
                                                      ┌──────────────────┐
                                                      │ Wait for Fill    │
                                                      │ (poll 500ms,30s) │
                                                      └────────┬─────────┘
                                                               │
                                                     ┌─────────┴─────────┐
                                                     FILLED            TIMEOUT
                                                     │                 │
                                                     ↓                 ↓
                                            ┌────────────────┐  ┌──────────┐
                                            │ Register       │  │  CANCEL  │
                                            │ Position       │  │  ORDER   │
                                            └────────┬───────┘  └──────────┘
                                                     │
                                                     ↓
                                            ┌────────────────────┐
                                            │ active_orders[sym] │
                                            │ = order_id         │
                                            └────────┬───────────┘
                                                     │
╔═══════════════════════════════════════════════════════════════╗
║ LAYER 8: MONITORING & FEEDBACK (Async, 1-5s loop)           ║
╚═══════════════════════════════════════════════════════════════╝
                                                     │
                                                     ↓
                                            ┌────────────────────┐
                                            │ ML Order Manager   │
                                            │ Monitor Loop       │
                                            │ ─────────────────│
                                            │ Update price       │
                                            │ Calculate P&L      │
                                            │ Check TP/SL        │
                                            │ Check ML exit      │
                                            │ Check quality drop │
                                            └────────┬───────────┘
                                                     │
                                          ┌──────────┴──────────┐
                                          Exit condition met?   │
                                          └──────────┬──────────┘
                                                     │
                                           ┌─────────┴─────────┐
                                           NO                YES
                                           │                 │
                                           ↓                 ↓
                                     ┌──────────┐     ┌───────────────┐
                                     │ CONTINUE │     │ Close Position│
                                     │ MONITOR  │     └───────┬───────┘
                                     │ (Loop)   │             │
                                     └──────────┘             ↓
                                                     ┌────────────────────┐
                                                     │ Delete from:       │
                                                     │ - active_positions │
                                                     │ - active_orders    │
                                                     └────────┬───────────┘
                                                              │
                                                              ↓
                                                     ┌────────────────────┐
                                                     │ Record Outcome     │
                                                     │ (Win/Loss, P&L)    │
                                                     └────────┬───────────┘
                                                              │
                                                              ↓
                                                     ┌────────────────────┐
                                                     │ FEEDBACK LOOP:     │
                                                     │ - Update strategy  │
                                                     │   weights          │
                                                     │ - Update model     │
                                                     │   performance      │
                                                     │ - ML learning      │
                                                     └────────┬───────────┘
                                                              │
                                                              ↓
                                                     ┌────────────────────┐
                                                     │ Symbol freed       │
                                                     │ Ready for new      │
                                                     │ signal             │
                                                     └────────────────────┘
```

---

## DECISION POINTS DETAILED (8-LAYER ARCHITECTURE)

### Decision 1: Data Authentication
```
IF HMAC signature invalid:
    ├─ Log security error
    ├─ Increment security_errors counter
    └─ REJECT data (do not process)

ELSE:
    └─ Continue to Layer 1
```

### Decision 2: MQScore Quality Gate (Layer 1.1) ✅
```
IF MQScore < 0.5:
    └─ SKIP (market quality too low)

ELSE:
    └─ Continue to liquidity check
```

### Decision 3: Liquidity Gate (Layer 1.1) ✅
```
IF Liquidity < 0.3:
    └─ SKIP (insufficient liquidity)

ELSE:
    └─ Continue to regime check
```

### Decision 4: Regime Safety Gate (Layer 1.1) ✅
```
IF Regime == CRISIS or HIGH_NOISE:
    └─ SKIP (unsafe market conditions)

ELSE:
    └─ Try optional enhancements → Continue to strategy execution
```

### Decision 5: Signal Confidence Filter (Layer 2)
```
FOR each strategy signal:
    IF confidence < 0.65:
        └─ Discard signal
    ELSE:
        └─ Keep for aggregation

IF no signals kept:
    └─ SKIP (no high-confidence signals)

ELSE:
    └─ Continue to meta-learning
```

### Decision 6: Anomaly Detection (Layer 3) ⭐ NEW
```
IF anomaly_score > 0.8:
    └─ Reduce position size by 50% (caution mode)

ELSE:
    └─ Continue with normal position sizing
```

### Decision 7: Signal Strength Gate (Layer 4)
```
IF signal_strength < 0.5:
    └─ SKIP (weak conviction)

IF abs(aggregated_signal) < 0.2:
    └─ SKIP (too neutral)

IF aggregated_signal > 0.3:
    direction = "BUY"
ELIF aggregated_signal < -0.3:
    direction = "SELL"
ELSE:
    direction = "HOLD" → SKIP
```

### Decision 8: Duplicate Prevention (Layer 4) **CRITICAL**
```
IF active_orders[symbol] exists:
    ├─ Log: "Order already active for {symbol}"
    ├─ order_id = active_orders[symbol]
    └─ SKIP (prevent duplicate)

ELSE:
    └─ Continue to routing
```

### Decision 9: Confidence Threshold (Layer 5) ⭐ NEW
```
IF confidence < 0.7:
    └─ SKIP (low confidence decision)

ELSE:
    └─ Continue to risk management
```

### Decision 10: Risk Multiplier Gate (Layer 6)
```
IF risk_multiplier < 0.3:
    └─ REJECT (too risky)

ELSE:
    └─ Continue to confidence calibration
```

### Decision 11: Market Alignment Check (Layer 6) ⭐ NEW
```
IF action="BUY" AND market_class=1 (Bearish):
    └─ REJECT (conflicting signal)

IF action="SELL" AND market_class=0 (Bullish):
    └─ REJECT (conflicting signal)

ELSE:
    └─ Continue to 7-layer risk check
```

### Decision 12: 7-Layer Risk Validation (Layer 6)
```
Layer 1: Pre-trade checks
    IF margin insufficient: REJECT
    IF position limit exceeded: REJECT

Layer 2: Risk validation
    IF VaR > threshold: REJECT
    IF correlation > 0.7: REJECT

Layer 3: Position sizing
    IF size > 10% equity: REJECT

Layer 4: Kill switches
    IF daily loss exceeded: REJECT
    IF drawdown > 15%: REJECT

Layer 5: Loss limits
    IF weekly loss exceeded: REJECT

Layer 6: Drawdown check
    IF current drawdown > limit: REJECT

Layer 7: Final approval
    IF all above pass: APPROVE

IF ANY layer fails:
    └─ REJECT trade

ELSE:
    └─ Continue to execution
```

### Decision 13: Order Fill Status (Layer 7)
```
Poll exchange status every 500ms for up to 30 seconds:

IF status == FILLED:
    ├─ Record fill price
    ├─ Calculate slippage
    ├─ Register with ML manager
    └─ Continue to monitoring

IF timeout (30s elapsed) and status != FILLED:
    ├─ Cancel order
    ├─ Log timeout
    └─ ABORT (do not monitor)
```

### Decision 14: Exit Conditions (Layer 8)
```
Check every 1-5 seconds:

IF current_price >= take_profit:
    └─ CLOSE (target hit)

IF current_price <= stop_loss:
    └─ CLOSE (stop hit)

IF ML predicts strong reversal:
    └─ CLOSE (ML exit signal)

IF quality_score drops < 0.3:
    └─ CLOSE (quality deteriorated)

IF unrealized_loss > risk_limit:
    └─ CLOSE (emergency exit)

ELSE:
    └─ CONTINUE monitoring
```

---

## SUMMARY STATISTICS

**Total Decision Gates**: 14 gates across 8 layers
**Critical Gates** (hard stop):
- MQScore < 0.5 ✅
- Liquidity < 0.3 ✅
- Regime == CRISIS ✅
- No high-confidence signals
- Duplicate order exists
- Confidence < 0.7
- Risk multiplier < 0.3
- Market class conflicts
- Any 7-layer risk check fails

**Soft Gates** (modify behavior):
- Anomaly score > 0.8 → Reduce size
- Signal strength < 0.5 → Skip

**Result:**
- MQScore 6D Engine as PRIMARY foundation ✅
- Multiple layers of protection
- Meta-learning optimization
- Dynamic risk adjustment
- Comprehensive 6D market assessment
- Optional enhancements (quantum vol, regime)

---

**Document Version**: 2.1  
**Last Updated**: 2025-10-21  
**Status**: ✅ Updated with MQScore + 70 Models (71 Total)
**Key Changes:**
- MQScore 6D Engine as PRIMARY Layer 1 ✅
- Enhanced volatility and regime as optional add-ons
- Added Layer 3 (Meta-Learning)
- Added Layer 5 (Model Governance & Routing)
- Updated all decision gates (14 total)
- Added anomaly detection
- Added market alignment check

                                           │
                                  ┌────────┴────────┐
                                  YES               NO
                                  │                 │
                                  ↓                 ↓
                            ┌──────────┐    ┌─────────────────────┐
                            │   SKIP   │    │ Check Duplicate:    │
                            └──────────┘    │ active_orders[sym]? │
                                            └──────┬──────────────┘
                                                   │
                                          ┌────────┴────────┐
                                          EXISTS           NEW
                                          │                │
                                          ↓                ↓
                                    ┌──────────┐   ┌────────────────┐
                                    │   SKIP   │   │ ML Decision    │
                                    │ (Prevent │   │ Engine         │
                                    │ Dup)     │   │ + TP/SL Calc   │
                                    └──────────┘   └────────┬───────┘
                                                            │
                                                            ↓
                                                  ┌──────────────────────┐
                                                  │ 7-Layer Risk Check   │
                                                  │ (All layers pass?)   │
                                                  └──────┬───────────────┘
                                                         │
                                                ┌────────┴────────┐
                                                FAIL              PASS
                                                │                 │
                                                ↓                 ↓
                                          ┌──────────┐     ┌────────────────┐
                                          │  REJECT  │     │ Create Order   │
                                          │ (Risk    │     │ (Unique ID)    │
                                          │ Limits)  │     └────────┬───────┘
                                          └──────────┘              │
                                                                    ↓
                                                          ┌──────────────────┐
                                                          │ Submit to        │
                                                          │ Exchange API     │
                                                          └────────┬─────────┘
                                                                   │
                                                                   ↓
                                                         ┌──────────────────────┐
                                                         │ Wait for Fill        │
                                                         │ (Poll 500ms, 30s)    │
                                                         └──────┬───────────────┘
                                                                │
                                                       ┌────────┴────────┐
                                                       FILLED            TIMEOUT
                                                       │                 │
                                                       ↓                 ↓
                                              ┌────────────────┐  ┌──────────┐
                                              │ Register with  │  │  CANCEL  │
                                              │ ML Manager     │  │  ORDER   │
                                              │ (Track by ID)  │  └──────────┘
                                              └────────┬───────┘
                                                       │
                                                       ↓
                                              ┌────────────────────────┐
                                              │ active_orders[symbol]  │
                                              │ = order_id             │
                                              │ (Prevent future dups)  │
                                              └────────┬───────────────┘
                                                       │
                                                       ↓
                                              ┌────────────────────────┐
                                              │ ML ORDER MANAGER       │
                                              │ Monitor Loop (1-5s)    │
                                              └────────┬───────────────┘
                                                       │
                                                       ↓
                                              ┌────────────────────────┐
                                              │ Update Current Price   │
                                              │ Calculate P&L          │
                                              └────────┬───────────────┘
                                                       │
                                                       ↓
                                              ┌────────────────────────┐
                                              │ Check Exit Conditions: │
                                              │ - TP hit?              │
                                              │ - SL hit?              │
                                              │ - ML exit signal?      │
                                              │ - Quality drop?        │
                                              └────────┬───────────────┘
                                                       │
                                              ┌────────┴────────┐
                                              NO EXIT           EXIT
                                              │                 │
                                              ↓                 ↓
                                      ┌────────────┐     ┌────────────────┐
                                      │  CONTINUE  │     │ Close Position │
                                      │  MONITOR   │     │ (Submit close) │
                                      │  (Loop)    │     └────────┬───────┘
                                      └────────────┘              │
                                                                  ↓
                                                         ┌──────────────────┐
                                                         │ Wait for Fill    │
                                                         └────────┬─────────┘
                                                                  │
                                                                  ↓
                                                         ┌──────────────────────┐
                                                         │ Delete from:         │
                                                         │ - active_positions   │
                                                         │ - active_orders[sym] │
                                                         └────────┬─────────────┘
                                                                  │
                                                                  ↓
                                                         ┌──────────────────────┐
                                                         │ Record Trade Outcome │
                                                         │ (Win/Loss, P&L)      │
                                                         └────────┬─────────────┘
                                                                  │
                                                                  ↓
                                                         ┌──────────────────────┐
                                                         │ FEEDBACK LOOP:       │
                                                         │ - Update strategy    │
                                                         │   performance        │
                                                         │ - Recalculate        │
                                                         │   weights            │
                                                         │ - ML learning        │
                                                         └────────┬─────────────┘
                                                                  │
                                                                  ↓
                                                         ┌──────────────────────┐
                                                         │  Symbol freed        │
                                                         │  Ready for new       │
                                                         │  signal              │
                                                         └──────────────────────┘
```

---

## DECISION POINTS DETAILED

### Decision 1: Data Authentication
```
IF HMAC signature invalid:
    ├─ Log security error
    ├─ Increment security_errors counter
    └─ REJECT data (do not process)

ELSE:
    └─ Continue to MQScore
```

### Decision 2: Market Quality Gate
```
IF MQScore < 0.5:
    └─ SKIP (market quality too low)

IF Liquidity < 0.3:
    └─ SKIP (insufficient liquidity)

IF Regime in [CRISIS, HIGH_NOISE]:
    └─ SKIP (unsafe market conditions)

ELSE:
    └─ Continue to strategies
```

### Decision 3: Signal Confidence
```
FOR each strategy signal:
    IF confidence < 0.65:
        └─ Discard signal
    ELSE:
        └─ Keep for aggregation

IF no signals kept:
    └─ SKIP (no high-confidence signals)

ELSE:
    └─ Continue to aggregation
```

### Decision 4: Signal Type
```
IF aggregated_signal == NEUTRAL:
    └─ SKIP (no directional conviction)

IF aggregated_signal in [BUY, SELL]:
    └─ Continue to duplicate check
```

### Decision 5: Duplicate Prevention (CRITICAL)
```
IF active_orders[symbol] exists:
    ├─ Log: "Order already active for {symbol}"
    ├─ order_id = active_orders[symbol]
    └─ SKIP (prevent duplicate)

ELSE:
    └─ Continue to ML decision
```

### Decision 6: Risk Validation (7 Layers)
```
Layer 1: Pre-trade checks
    IF margin insufficient: REJECT
    IF position limit exceeded: REJECT

Layer 2: Risk validation
    IF VaR > threshold: REJECT
    IF correlation > 0.7: REJECT

Layer 3: Position sizing
    IF size > 10% equity: REJECT

Layer 4: Kill switches
    IF daily loss exceeded: REJECT
    IF drawdown > 15%: REJECT

Layer 5: Loss limits
    IF weekly loss exceeded: REJECT

Layer 6: Drawdown check
    IF current drawdown > limit: REJECT

Layer 7: Final approval
    IF all above pass: APPROVE

IF ANY layer fails:
    └─ REJECT trade

ELSE:
    └─ Continue to execution
```

### Decision 7: Order Fill Status
```
Poll exchange status every 500ms for up to 30 seconds:

IF status == FILLED:
    ├─ Record fill price
    ├─ Calculate slippage
    ├─ Register with ML manager
    └─ Continue to monitoring

IF timeout (30s elapsed) and status != FILLED:
    ├─ Cancel order
    ├─ Log timeout
    └─ ABORT (do not monitor)
```

### Decision 8: Exit Conditions (ML Order Manager)
```
Check every 1-5 seconds:

IF current_price >= take_profit:
    └─ CLOSE (target hit)

IF current_price <= stop_loss:
    └─ CLOSE (stop hit)

IF ML predicts strong reversal:
    └─ CLOSE (ML exit signal)

IF MQScore drops < 0.3:
    └─ CLOSE (quality deteriorated)

IF unrealized_loss > risk_limit:
    └─ CLOSE (emergency exit)

ELSE:
    └─ CONTINUE monitoring
```

---

## PARALLEL PROCESSING

### Multi-Symbol Concurrent Flow

```
Symbol Loop (50 symbols running concurrently):

Thread 1: BTCUSDT  ─┐
Thread 2: ETHUSDT  ─┤
Thread 3: BNBUSDT  ─┤
Thread 4: SOLUSDT  ─┼──→ All run same pipeline independently
...                 │
Thread 50: XRPUSDT ─┘

Each thread:
    ├─ Independent data stream
    ├─ Independent MQScore calculation
    ├─ Independent strategy execution
    ├─ Independent ML decision
    └─ Independent order management

Shared resources:
    ├─ WeightCalculator (thread-safe)
    ├─ MLDecisionEngine (thread-safe)
    └─ OrderExecutionManager (with locks)

Result: Up to 50 concurrent positions (max 1 per symbol)
```

### Strategy Execution Parallelization

```
For each symbol, 20 strategies execute in parallel:

Executor Pool (ThreadPoolExecutor, max_workers=20):

Task 1: Event-Driven       ─┐
Task 2: LVN Breakout        ─┤
Task 3: Absorption Breakout ─┤
Task 4: Momentum Breakout   ─┤
...                          ├──→ All execute simultaneously
Task 19: Momentum Ignition  ─┤
Task 20: Volume Imbalance   ─┘

Wait for all tasks to complete (asyncio.gather)
    ↓
Collect results: [signal₁, signal₂, ..., signal₂₀]
    ↓
Filter + Aggregate → Single signal
```

---

## ERROR HANDLING

```
TRY:
    ├─ Data authentication
    ├─ MQScore calculation
    ├─ Strategy execution
    ├─ ML decision
    ├─ Order execution
    └─ Position monitoring

CATCH AuthenticationError:
    ├─ Log security alert
    ├─ Increment error counter
    └─ SKIP this data point

CATCH MQScoreError:
    ├─ Log calculation error
    ├─ Use fallback values
    └─ Continue with reduced confidence

CATCH StrategyError:
    ├─ Log strategy failure
    ├─ Exclude failed strategy from aggregation
    └─ Continue with remaining strategies

CATCH OrderExecutionError:
    ├─ Log execution failure
    ├─ Cancel order if possible
    └─ Do not register position

CATCH PositionMonitoringError:
    ├─ Log monitoring error
    ├─ Attempt emergency close
    └─ Alert operator

FINALLY:
    └─ Record all errors in audit log
```

---

## STATE TRANSITIONS

```
Order Lifecycle States:

CREATED → PENDING → SUBMITTED → OPEN → PARTIALLY_FILLED → FILLED
                                  ↓
                              CANCELLED / REJECTED / TIMEOUT

Position Lifecycle States:

OPENED → MONITORING → (TP_HIT | SL_HIT | ML_EXIT | EMERGENCY) → CLOSING → CLOSED
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-20  
**Status**: Complete Flowchart
