# NEXUS AI - Pipeline Integration & Architecture
## Part 3: System Integration & Data Flow

**Version:** 3.0.0  
**Last Updated:** October 24, 2025

---

## Table of Contents

1. [8-Layer Pipeline Architecture](#1-8-layer-pipeline-architecture)
2. [Component Integration](#2-component-integration)
3. [Message Protocols](#3-message-protocols)
4. [Error Handling Framework](#4-error-handling-framework)
5. [Data Flow Diagrams](#5-data-flow-diagrams)

---

## 1. 8-Layer Pipeline Architecture

### 1.1 Complete Pipeline Overview

**Location:** `nexus_ai.py` lines 4600-4987 (TradingPipeline class)

```
┌────────────────────────────────────────────────────────────────┐
│                    NEXUS AI TRADING PIPELINE                    │
│                         8-Layer System                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ LAYER 1: MARKET QUALITY ASSESSMENT                             │
│ ├─ MQScore 6D Engine                                           │
│ ├─ Input: MarketData                                           │
│ ├─ Output: market_context (7 dimensions)                       │
│ └─ Gate: MQScore >= 0.5                                        │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ LAYER 2: STRATEGY EXECUTION (20 Strategies)                    │
│ ├─ Event-Driven, Breakout, Microstructure, Detection, etc.    │
│ ├─ Input: MarketData DataFrame                                 │
│ ├─ Output: List[TradingSignal]                                 │
│ └─ Parallel execution with timeout protection                  │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ LAYER 3: META-STRATEGY SELECTION                               │
│ ├─ ML-based strategy weighting                                 │
│ ├─ Input: List[TradingSignal], market_context                  │
│ ├─ Output: strategy_weights, filtered_signals                  │
│ └─ Gate: >= 3 strategies with weight > 0.3                     │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ LAYER 4: SIGNAL AGGREGATION                                    │
│ ├─ Weighted signal combination                                 │
│ ├─ Input: filtered_signals, strategy_weights                   │
│ ├─ Output: aggregated_signal, direction, strength              │
│ └─ Gates: strength >= 0.6, direction != HOLD                   │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ LAYER 5: MODEL GOVERNANCE & DECISION ROUTING                   │
│ ├─ ModelGovernor: Evaluate model performance                   │
│ ├─ DecisionRouter: ML inference for final decision             │
│ ├─ Input: aggregated_signal, market_context, model_weights     │
│ ├─ Output: action, confidence, value_estimate                  │
│ └─ Gate: confidence >= 0.7                                     │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ LAYER 6: RISK MANAGEMENT                                       │
│ ├─ ML Risk Assessment (7 models)                               │
│ ├─ Position Sizing (Kelly Criterion)                           │
│ ├─ Dynamic Position Adjustment                                 │
│ ├─ 7-Layer Risk Validation                                     │
│ ├─ Input: decision, market_context, portfolio                  │
│ ├─ Output: approved/rejected, position_size, stop/take levels  │
│ └─ All 7 validation layers must pass                           │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ LAYER 7: ORDER EXECUTION                                       │
│ ├─ Order creation and submission                               │
│ ├─ Broker API integration                                      │
│ ├─ Fill tracking and position updates                          │
│ ├─ Input: approved decision, position_size                     │
│ └─ Output: Order object, execution status                      │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ LAYER 8: MONITORING & AUDIT                                    │
│ ├─ Performance tracking                                        │
│ ├─ Alert generation                                            │
│ ├─ Audit trail logging                                         │
│ └─ Metrics collection and reporting                            │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Pipeline Execution Flow

**Method:** `TradingPipeline.execute_pipeline()`  
**Location:** Lines 4700-4940

```python
def execute_pipeline(
    self,
    symbol: str,
    market_data: pd.DataFrame,
    portfolio: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute complete 8-layer trading pipeline.
    
    Returns:
        {
            'decision': 'APPROVED' | 'REJECTED' | 'SKIPPED',
            'action': 'BUY' | 'SELL' | 'HOLD',
            'position_size': float,
            'confidence': float,
            'rejection_reason': str or None,
            'layer_outputs': Dict[str, Any]
        }
    """
```

### 1.3 Pipeline Statistics

**Tracked Metrics:**
```python
pipeline_stats = {
    'total_processed': int,        # Total symbols processed
    'approved': int,               # Trades approved
    'layer3_skips': int,           # Skipped at meta-selection
    'layer4_skips': int,           # Skipped at aggregation
    'layer5_skips': int,           # Skipped at decision routing
    'layer6_rejects': int,         # Rejected at risk management
    'duplicate_skips': int,        # Duplicate order prevention
    'approval_rate': float,        # approved / total
    'skip_rate': float,            # skips / total
    'rejection_rate': float        # rejects / total
}
```

---

## 2. Component Integration

### 2.1 Component Interaction Map

```
┌─────────────────────────────────────────────────────────────────┐
│                      NEXUS AI COMPONENTS                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ SecurityManager  │──────┐
│ - Data validation│      │
│ - Authentication │      │
└──────────────────┘      │
                          ▼
┌──────────────────┐  ┌─────────────────────┐
│ DataBuffer       │  │ TradingPipeline     │
│ - Market data    │─▶│ - 8-layer execution │
│ - Memory mgmt    │  │ - Decision making   │
└──────────────────┘  └──────────┬──────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────┐      ┌──────────────────┐    ┌──────────────────┐
│StrategyMgr   │      │ RiskManager      │    │ ExecutionEngine  │
│- 20 strategies│      │- Risk validation │    │- Order creation  │
│- Signal gen  │      │- Position sizing │    │- Broker API      │
└──────────────┘      └──────────────────┘    └──────────────────┘
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────┐
                      │PerformanceMonitor│
                      │- Metrics tracking│
                      │- Alert generation│
                      └──────────────────┘
```

### 2.2 Data Flow Between Components

**Market Data Ingestion:**
```
Exchange → SecurityManager.validate_market_data()
         → DataBuffer.add()
         → TradingPipeline.execute_pipeline()
```

**Signal Generation:**
```
TradingPipeline → StrategyManager.generate_signals()
                → [Strategy1, Strategy2, ..., Strategy20]
                → List[TradingSignal]
```

**Risk Assessment:**
```
TradingPipeline → RiskManager.ml_risk_assessment()
                → RiskManager.evaluate_risk()
                → RiskManager.seven_layer_risk_validation()
                → approved/rejected
```

**Order Execution:**
```
TradingPipeline → ExecutionEngine.create_order()
                → ExecutionEngine.execute_order()
                → RiskManager.update_position()
                → RiskManager.update_pnl()
```

### 2.3 Dependency Graph

```
NexusAI (Main System)
├── ConfigManager
│   └── SystemConfig
├── SecurityManager
├── DataBuffer
│   └── MemoryWatchdog
├── DataCache
├── ModelLoader
│   └── ProductionModelRegistry
├── TradingPipeline
│   ├── MarketQualityLayer1
│   ├── MetaStrategySelector
│   ├── SignalAggregator
│   ├── ModelGovernor
│   └── DecisionRouter
├── StrategyManager
│   └── [20 Strategy Adapters]
├── RiskManager
│   └── [7 ML Risk Models]
├── ExecutionEngine
└── PerformanceMonitor
```

---

## 3. Message Protocols

### 3.1 Internal Message Format

**All internal messages use Python dataclasses for type safety.**

### 3.2 External API Protocols

#### 3.2.1 FIX Protocol Support

**For institutional broker integration:**

```python
# FIX 4.4 New Order Single (MsgType=D)
fix_order = {
    '8': 'FIX.4.4',              # BeginString
    '35': 'D',                   # MsgType (New Order Single)
    '11': 'ORD_000001',          # ClOrdID
    '55': 'SYMBOL',              # Symbol
    '54': '1',                   # Side (1=Buy, 2=Sell)
    '38': '1.5',                 # OrderQty
    '40': '2',                   # OrdType (1=Market, 2=Limit)
    '44': '67250.00',            # Price
    '59': '0',                   # TimeInForce (0=Day, 1=GTC)
    '60': '20251024-16:30:00',   # TransactTime
}
```

**FIX Message Flow:**
```
NEXUS → FIX Engine → Broker
      ← Execution Report (MsgType=8)
      ← Order Cancel Reject (MsgType=9)
```

#### 3.2.2 REST API Protocol

**For modern broker APIs (JSON):**

```json
{
  "method": "POST",
  "endpoint": "/api/v1/orders",
  "headers": {
    "Content-Type": "application/json",
    "Authorization": "Bearer {api_key}",
    "X-Request-ID": "uuid-v4"
  },
  "body": {
    "symbol": "SYMBOL",
    "side": "BUY",
    "type": "LIMIT",
    "quantity": "1.5",
    "price": "67250.00",
    "timeInForce": "GTC",
    "clientOrderId": "ORD_000001"
  }
}
```

**Response:**
```json
{
  "orderId": "12345678",
  "clientOrderId": "ORD_000001",
  "symbol": "SYMBOL",
  "status": "NEW",
  "side": "BUY",
  "type": "LIMIT",
  "quantity": "1.5",
  "price": "67250.00",
  "executedQty": "0.0",
  "timestamp": 1729785000123
}
```

#### 3.2.3 WebSocket Protocol

**For real-time market data:**

```json
{
  "method": "SUBSCRIBE",
  "params": [
    "symbol@trade",
    "symbol@depth20"
  ],
  "id": 1
}
```

**Market Data Stream:**
```json
{
  "e": "trade",
  "E": 1729785000123,
  "s": "SYMBOL",
  "t": 987654321,
  "p": "67250.00",
  "q": "1.5",
  "T": 1729785000120
}
```

### 3.3 Message Versioning

**API Version Strategy:**
- **Current:** v3.0.0
- **Backward Compatibility:** Support v2.x for 6 months
- **Version Header:** `X-API-Version: 3.0.0`
- **Deprecation Notice:** 90 days advance warning

---

## 4. Error Handling Framework

### 4.1 Error Propagation

```
Layer N Error
    │
    ├─ Recoverable? ──Yes──▶ Retry with backoff
    │                        └─ Log warning
    │
    └─ No ──▶ Propagate to caller
              └─ Log error
              └─ Update metrics
              └─ Trigger alert (if critical)
```

### 4.2 Retry Logic

**Exponential Backoff:**
```python
def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> Any:
    """
    Retry function with exponential backoff.
    
    Delay sequence: 1s, 2s, 4s, 8s, ...
    """
    for attempt in range(max_retries):
        try:
            return func()
        except RecoverableError as e:
            if attempt == max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            time.sleep(delay)
    raise MaxRetriesExceeded()
```

### 4.3 Failover Procedures

**ML Model Failover:**
```
Primary ML Model Failed
    │
    ├─ Try Secondary Model
    │   ├─ Success → Continue
    │   └─ Failed → Use Rule-Based Fallback
    │
    └─ Log model failure
        └─ Trigger alert
```

**Broker Connection Failover:**
```
Primary Broker Unavailable
    │
    ├─ Switch to Secondary Broker
    │   ├─ Success → Continue
    │   └─ Failed → Queue orders
    │
    └─ Alert operations team
```

### 4.4 Circuit Breaker Pattern

```python
class CircuitBreaker:
    """Prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60
    ):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.state = 'CLOSED'  # CLOSED | OPEN | HALF_OPEN
        self.last_failure_time = 0
    
    def call(self, func: Callable) -> Any:
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise CircuitBreakerOpen()
        
        try:
            result = func()
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
            raise
```

### 4.5 Idempotency Requirements

**Order Submission:**
- Use `clientOrderId` for deduplication
- Broker must reject duplicate `clientOrderId`
- Store submitted orders in cache (TTL: 24 hours)

**Position Updates:**
- Use sequence numbers for ordering
- Reject out-of-order updates
- Reconcile with broker at EOD

---

## 5. Data Flow Diagrams

### 5.1 Complete Trade Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRADE EXECUTION SEQUENCE                      │
└─────────────────────────────────────────────────────────────────┘

Exchange
   │
   │ Market Data
   ▼
┌──────────────┐
│ Data Ingestion│
│ - Validate   │
│ - Buffer     │
└──────┬───────┘
       │
       │ MarketData
       ▼
┌──────────────┐
│ Layer 1      │
│ MQScore      │──── Gate: MQScore >= 0.5 ────┐
└──────┬───────┘                               │
       │ PASS                                  │ SKIP
       │                                       ▼
       │ market_context                   ┌────────┐
       ▼                                  │ Return │
┌──────────────┐                         │ SKIPPED│
│ Layer 2      │                         └────────┘
│ Strategies   │
└──────┬───────┘
       │
       │ List[TradingSignal]
       ▼
┌──────────────┐
│ Layer 3      │
│ Meta-Select  │──── Gate: >= 3 strategies ───┐
└──────┬───────┘                               │
       │ PASS                                  │ SKIP
       │                                       ▼
       │ filtered_signals                 ┌────────┐
       ▼                                  │ Return │
┌──────────────┐                         │ SKIPPED│
│ Layer 4      │                         └────────┘
│ Aggregation  │──── Gate: strength >= 0.6 ────┐
└──────┬───────┘      Gate: direction != HOLD  │
       │ PASS                                   │ SKIP
       │                                        ▼
       │ aggregated_signal                 ┌────────┐
       ▼                                   │ Return │
┌──────────────┐                          │ SKIPPED│
│ Layer 5      │                          └────────┘
│ Decision     │──── Gate: confidence >= 0.7 ───┐
└──────┬───────┘                                 │
       │ PASS                                    │ SKIP
       │                                         ▼
       │ decision                           ┌────────┐
       ▼                                    │ Return │
┌──────────────┐                           │ SKIPPED│
│ Layer 6      │                           └────────┘
│ Risk Mgmt    │──── 7-Layer Validation ────┐
└──────┬───────┘                             │
       │ APPROVED                            │ REJECTED
       │                                     ▼
       │ position_size                  ┌─────────┐
       ▼                                │ Return  │
┌──────────────┐                       │REJECTED │
│ Layer 7      │                       └─────────┘
│ Execution    │
└──────┬───────┘
       │
       │ Order
       ▼
   Broker API
       │
       │ Fill Confirmation
       ▼
┌──────────────┐
│ Layer 8      │
│ Monitoring   │
└──────────────┘
```

### 5.2 Error Recovery Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ERROR RECOVERY SEQUENCE                       │
└─────────────────────────────────────────────────────────────────┘

Component Error
   │
   ├─ Classify Error
   │  ├─ VALIDATION_FAILED ──▶ Reject request, log warning
   │  ├─ NETWORK_ERROR ──────▶ Retry with backoff (3x)
   │  ├─ TIMEOUT_ERROR ──────▶ Retry once
   │  ├─ MODEL_LOAD_FAILED ──▶ Use fallback, log error
   │  └─ MEMORY_EXHAUSTED ───▶ Trigger cleanup, log critical
   │
   ├─ Log Error
   │  └─ Include: timestamp, component, error_code, details
   │
   ├─ Update Metrics
   │  └─ Increment error counter
   │
   ├─ Check Alert Threshold
   │  ├─ Below threshold ──▶ Continue
   │  └─ Above threshold ──▶ Trigger alert
   │
   └─ Attempt Recovery
      ├─ Recoverable ──▶ Retry/Fallback
      └─ Fatal ──────▶ Escalate to operator
```

### 5.3 Position Reconciliation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                  POSITION RECONCILIATION                         │
└─────────────────────────────────────────────────────────────────┘

Real-Time Updates (Per Fill)
   │
   ├─ Order Fill Event
   │  └─ Update internal position
   │     └─ Calculate P&L
   │        └─ Update risk metrics
   │
   ▼
Intraday Reconciliation (Every 15 min)
   │
   ├─ Query broker positions
   │  └─ Compare with internal positions
   │     ├─ Match ──▶ Continue
   │     └─ Mismatch ──▶ Log discrepancy
   │                     └─ Investigate
   │
   ▼
EOD Reconciliation (Market Close)
   │
   ├─ Fetch broker EOD report
   │  └─ Compare all positions
   │     ├─ Match ──▶ Archive
   │     └─ Mismatch ──▶ Adjust internal
   │                     └─ Alert operations
   │
   └─ Generate EOD report
      └─ Store in database
```

---

## Appendix A: Integration Checklist

### Broker Integration Requirements

- [ ] API credentials configured
- [ ] WebSocket connection established
- [ ] Order submission tested
- [ ] Fill notifications working
- [ ] Position queries functional
- [ ] Error handling implemented
- [ ] Failover tested
- [ ] Rate limits configured
- [ ] Idempotency verified
- [ ] Reconciliation scheduled

### Monitoring Integration

- [ ] Metrics collection enabled
- [ ] Alert thresholds configured
- [ ] Email notifications tested
- [ ] SMS notifications tested
- [ ] PagerDuty integrated
- [ ] Dashboard deployed
- [ ] Log aggregation configured
- [ ] Audit trail enabled

### Compliance Integration

- [ ] Order audit trail enabled
- [ ] Position reporting configured
- [ ] Risk limit monitoring active
- [ ] Regulatory reporting scheduled
- [ ] Data retention policies enforced
- [ ] Access controls implemented
- [ ] Encryption enabled
- [ ] Backup procedures tested

## Appendix B: Performance Benchmarks

| Component | Target Latency | Actual | Throughput |
|-----------|----------------|--------|------------|
| Data Validation | < 1ms | 0.5ms | 10K/s |
| Layer 1 (MQScore) | < 5ms | 3ms | 2K/s |
| Layer 2 (Strategies) | < 20ms | 15ms | 500/s |
| Layer 3-5 (ML) | < 10ms | 8ms | 1K/s |
| Layer 6 (Risk) | < 5ms | 4ms | 2K/s |
| Layer 7 (Execution) | < 50ms | 30ms | 1K/s |
| End-to-End | < 100ms | 60ms | 500/s |
