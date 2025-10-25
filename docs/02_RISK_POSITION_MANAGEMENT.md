# NEXUS AI - Risk & Position Management
## Part 2: Complete Risk Framework

**Version:** 3.0.0  
**Last Updated:** October 24, 2025

---

## 1. Risk Management Framework

### 1.1 Risk Manager Architecture

**Location:** `nexus_ai.py` lines 3586-4370

The Risk Manager implements a 3-tier approach:
1. **Traditional Risk Metrics** (Kelly Criterion, VaR, Sharpe)
2. **ML-Enhanced Assessment** (7 ML models)
3. **7-Layer Validation** (Multi-checkpoint approval)

### 1.2 Risk Configuration

```python
# Position Sizing
max_position_size: 0.1              # 10% per trade
max_position_per_symbol: 0.2        # 20% per symbol
kelly_min_fraction: 0.001           # Min 0.1%
kelly_safety_factor: 0.25           # Use 25% of Kelly

# Loss Limits
max_daily_loss: 0.02                # 2% daily
max_drawdown: 0.15                  # 15% max

# Stop Loss / Take Profit
stop_loss_pct: 0.02                 # 2%
take_profit_pct: 0.04               # 4%
```

### 1.3 Kelly Criterion Position Sizing

**Formula:**
```
kelly_fraction = (p * b - q) / b
position_size = max(min_fraction, kelly_fraction * safety_factor)
position_size = min(position_size, max_position_size)

Where:
  p = signal confidence
  q = 1 - p
  b = take_profit / stop_loss
```

**Example:**
```
Confidence: 0.70, Take Profit: 4%, Stop Loss: 2%
kelly = (0.70 * 2.0 - 0.30) / 2.0 = 0.55
position = 0.55 * 0.25 = 0.1375 → capped at 0.10 (10%)
```

### 1.4 Risk Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| VaR 95% | `max_loss * 1.645` | 95% confidence loss limit |
| Sharpe Ratio | `expected_return / max_loss` | Risk-adjusted return |
| Risk Score | `mean(risk_factors)` | Composite risk (0-1) |

### 1.5 ML-Enhanced Risk Models

**7 Production Models:**
1. Risk Classifier (ONNX) - Market classification
2. Risk Scorer (ONNX) - Risk multiplier calculation
3. Risk Governor (ONNX) - Governance decisions
4. Confidence Calibration (XGBoost) - Confidence adjustment
5. Market Classifier (XGBoost) - Regime detection
6. Regression Model (XGBoost) - Return prediction
7. Gradient Boosting (ONNX) - Ensemble predictions

---

## 2. Position Management

### 2.1 Position Structure

```python
@dataclass
class Position:
    symbol: str
    instrument_type: str           # SPOT | FUTURES | OPTIONS
    quantity: float                # Net position
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    stop_loss_price: float
    take_profit_price: float
    contract_size: float           # For futures/options
    expiry_date: Optional[float]   # For derivatives
```

### 2.2 Position Calculations

**Spot Instruments:**
```python
# Buy: Add to position, weighted average price
new_qty = current_qty + trade_qty
new_avg = (current_qty * current_avg + trade_qty * trade_price) / new_qty

# Sell: Reduce position, realize P&L
realized_pnl = (trade_price - avg_price) * contracts_closed
new_qty = current_qty - trade_qty
```

**Futures Instruments:**
```python
# Account for contract size in P&L
realized_pnl = (trade_price - avg_price) * contracts * contract_size
```

### 2.3 Real-Time vs EOD Reconciliation

| Process | Frequency | Latency | Trigger |
|---------|-----------|---------|---------|
| Real-Time Update | Per fill | < 10ms | Order execution |
| Intraday Reconciliation | Every 15 min | < 1s | Timer |
| EOD Reconciliation | Daily close | < 5s | Market close |
| Broker Reconciliation | Daily | < 30s | Broker report |

### 2.4 Corporate Action Adjustments

**Stock Splits:**
```python
quantity *= split_ratio
avg_entry_price /= split_ratio
stop_loss /= split_ratio
```

**Dividends:**
```python
if quantity > 0:  # Long positions only
    realized_pnl += quantity * dividend_per_share
```

---

## 3. 7-Layer Risk Validation

**Location:** `nexus_ai.py` lines 4231-4336

All layers must pass for trade approval.

### Layer 1: Position Size Limits
- **Check:** `position_size <= max_position_size`
- **Threshold:** 10% of capital
- **Reject:** "Position size exceeds limit"

### Layer 2: Daily Loss Limits
- **Check:** `abs(daily_pnl) / capital < max_daily_loss`
- **Threshold:** 2% of capital
- **Reject:** "Daily loss limit reached"

### Layer 3: Drawdown Limits
- **Check:** `drawdown / capital < max_drawdown`
- **Threshold:** 15% of capital
- **Reject:** "Max drawdown reached"

### Layer 4: Signal Confidence
- **Check:** `confidence >= 0.57`
- **Threshold:** 57%
- **Reject:** "Low signal confidence"

### Layer 5: ML Risk Multiplier
- **Check:** `risk_multiplier >= 0.3`
- **Threshold:** 0.3
- **Reject:** "Low risk multiplier"

### Layer 6: Market Classification
- **Check:** `market_class != 'ADVERSE'`
- **Allowed:** FAVORABLE, NEUTRAL
- **Reject:** "Adverse market conditions"

### Layer 7: Critical Risk Flags
- **Check:** No critical flags
- **Critical:** LOW_MARKET_QUALITY, LOW_LIQUIDITY, NO_MODELS
- **Reject:** "Critical risk flags present"

---

## 4. Alert System

### 4.1 Alert Thresholds

| Alert | Threshold | Severity | Action |
|-------|-----------|----------|--------|
| LOW_CONFIDENCE | < 0.7 | WARNING | Reduce size 50% |
| HIGH_VOLATILITY | > 0.8 | WARNING | Reduce size 30% |
| LOW_LIQUIDITY | < 0.3 | CRITICAL | Reject trade |
| DAILY_LOSS | >= 2% | CRITICAL | Halt trading |
| MAX_DRAWDOWN | >= 15% | CRITICAL | Manual review |

### 4.2 Escalation

```
INFO → Log only
WARNING → Log + Adjust position
ERROR → Log + Reject + Email
CRITICAL → Log + Reject + Email + SMS + PagerDuty
```

---

## 5. Performance Metrics

### 5.1 Risk-Adjusted Returns

```python
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
sortino_ratio = (portfolio_return - risk_free_rate) / downside_std
calmar_ratio = annual_return / max_drawdown
```

### 5.2 Risk Monitoring

**Real-Time Metrics:**
- Current exposure per symbol
- Total portfolio exposure
- Daily P&L
- Current drawdown
- Risk score per position

**Daily Reports:**
- Sharpe ratio
- Max drawdown
- Win rate
- Average win/loss
- Risk-adjusted return

---

## Appendix: Risk Calculation Examples

### Example 1: Kelly Position Sizing
```
Signal: BUY SYMBOL, Confidence: 0.75
Config: TP=4%, SL=2%, Safety=0.25, Max=0.10

kelly = (0.75 * 2.0 - 0.25) / 2.0 = 0.625
position = 0.625 * 0.25 = 0.156
final = min(0.156, 0.10) = 0.10 (10% of capital)
```

### Example 2: 7-Layer Validation
```
Layer 1: position_size=0.08 <= 0.10 ✓ PASS
Layer 2: daily_loss=1.5% < 2% ✓ PASS
Layer 3: drawdown=12% < 15% ✓ PASS
Layer 4: confidence=0.72 >= 0.57 ✓ PASS
Layer 5: risk_multiplier=0.65 >= 0.3 ✓ PASS
Layer 6: market_class=NEUTRAL != ADVERSE ✓ PASS
Layer 7: risk_flags=[] (no critical) ✓ PASS

Result: APPROVED
```

### Example 3: Position Update
```
Initial: 1.0 units @ $66,000
Trade: BUY 0.5 units @ $67,000

new_qty = 1.0 + 0.5 = 1.5 units
new_avg = (1.0 * 66000 + 0.5 * 67000) / 1.5 = $66,333.33
unrealized_pnl = 1.5 * (67000 - 66333.33) = $1,000
```
