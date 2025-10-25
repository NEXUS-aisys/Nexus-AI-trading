# NEXUS AI Trading Pipeline - Data Type Specifications
## Part 1: Complete Data Structure Definitions

**Version:** 3.0.0  
**Last Updated:** October 24, 2025

---

## Table of Contents

1. [Market Data Structures](#1-market-data-structures)
2. [Trading Signal Structures](#2-trading-signal-structures)
3. [Risk & Position Structures](#3-risk--position-structures)
4. [Order & Execution Structures](#4-order--execution-structures)
5. [Error Handling Structures](#5-error-handling-structures)
6. [Validation Rules](#6-validation-rules)

---

## 1. Market Data Structures

### 1.1 MarketData Class

**Location:** `nexus_ai.py` lines 457-507

```python
@dataclass(frozen=True, slots=True)
class MarketData:
    """Immutable market data structure with precision handling."""
    
    # Identification
    symbol: str                    # Trading symbol (e.g., "SYMBOL", "FUTURE", "STOCK")
    
    # Pricing Data
    timestamp: float               # Unix timestamp (seconds, UTC)
    price: Decimal                 # Last trade price (high precision)
    volume: Decimal                # Trade volume
    
    # Order Book
    bid: Decimal                   # Best bid price
    ask: Decimal                   # Best ask price
    bid_size: int                  # Bid quantity (contracts/shares)
    ask_size: int                  # Ask quantity (contracts/shares)
    
    # Classification
    data_type: MarketDataType      # TRADE | QUOTE | BOOK_UPDATE | INDEX | AGGREGATE
    
    # Sequencing & Timing
    exchange_timestamp: float      # Exchange-reported timestamp
    sequence_num: int              # Message sequence number
    
    # Extensibility
    metadata: Dict[str, Any]       # Additional fields
```

### 1.2 Field Specifications

| Field | Type | Format | Range | Constraints |
|-------|------|--------|-------|-------------|
| `symbol` | `str` | Uppercase alphanumeric | 3-20 chars | Non-empty, no special chars except "-" |
| `timestamp` | `float` | Unix epoch seconds | > 0 | Must be within ±60s of current time |
| `price` | `Decimal` | Fixed-point decimal | > 0 | Positive, non-zero, max 8 decimal places |
| `volume` | `Decimal` | Fixed-point decimal | >= 0 | Non-negative, max 8 decimal places |
| `bid` | `Decimal` | Fixed-point decimal | > 0 | Must be < ask |
| `ask` | `Decimal` | Fixed-point decimal | > 0 | Must be > bid |
| `bid_size` | `int` | Integer | >= 0 | Non-negative |
| `ask_size` | `int` | Integer | >= 0 | Non-negative |
| `sequence_num` | `int` | Integer | >= 0 | Monotonically increasing per symbol |

### 1.3 Derived Properties

```python
@property
def mid_price(self) -> Decimal:
    """Mid price: (bid + ask) / 2"""
    return (self.bid + self.ask) / 2

@property
def spread(self) -> Decimal:
    """Bid-ask spread: ask - bid"""
    return self.ask - self.bid

@property
def spread_bps(self) -> Decimal:
    """Spread in basis points: (spread / mid_price) * 10000"""
    if self.mid_price == 0:
        return Decimal(0)
    return (self.spread / self.mid_price) * 10000
```

### 1.4 MarketDataType Enumeration

```python
class MarketDataType(Enum):
    """Market data classification."""
    
    TRADE = "TRADE"              # Individual trade execution
    QUOTE = "QUOTE"              # Top-of-book quote update
    BOOK_UPDATE = "BOOK_UPDATE"  # Order book depth update
    INDEX = "INDEX"              # Index value update
    AGGREGATE = "AGGREGATE"      # Aggregated data (OHLCV bar)
```

### 1.5 Validation Rules

**Implemented in:** `SecurityManager.validate_market_data()` (lines 691-716)

```python
def validate_market_data(data: MarketData) -> bool:
    """
    Validate market data integrity.
    
    Checks:
    1. Symbol is non-empty
    2. Price and volume are positive
    3. Timestamp is within acceptable range (±60s)
    4. Bid < Ask (no crossed market)
    5. All numeric fields are finite (no NaN/Inf)
    """
    # Symbol validation
    if not data.symbol or data.symbol.strip() == "":
        return False
    
    # Price validation
    if data.price <= 0 or data.volume < 0:
        return False
    
    # Timestamp validation
    current_time = time.time()
    if data.timestamp > current_time + 60:  # 60s clock drift tolerance
        return False
    
    # Spread validation
    if data.bid > data.ask:
        return False
    
    return True
```

### 1.6 Sample Payloads

**Valid Market Data:**
```json
{
  "symbol": "SYMBOL",
  "timestamp": 1729785000.123,
  "price": "67234.50",
  "volume": "1.2345",
  "bid": "67234.00",
  "ask": "67235.00",
  "bid_size": 10,
  "ask_size": 15,
  "data_type": "TRADE",
  "exchange_timestamp": 1729785000.120,
  "sequence_num": 123456789,
  "metadata": {
    "exchange": "EXCHANGE",
    "trade_id": "987654321"
  }
}
```

**Invalid Market Data (Crossed Market):**
```json
{
  "symbol": "SYMBOL",
  "timestamp": 1729785000.123,
  "price": "67234.50",
  "volume": "1.2345",
  "bid": "67235.00",  // ERROR: bid > ask
  "ask": "67234.00",
  "bid_size": 10,
  "ask_size": 15,
  "data_type": "TRADE",
  "exchange_timestamp": 1729785000.120,
  "sequence_num": 123456789,
  "metadata": {}
}
```

---

## 2. Trading Signal Structures

### 2.1 TradingSignal Class

**Location:** `nexus_ai.py` lines 509-526

```python
@dataclass
class TradingSignal:
    """Trading signal with confidence scoring and metadata."""
    
    signal_type: SignalType        # Signal direction and strength
    confidence: float              # Confidence score (0.0 to 1.0)
    symbol: str                    # Trading symbol
    timestamp: float               # Signal generation time (Unix epoch)
    strategy: str                  # Strategy name that generated signal
    metadata: Dict[str, Any]       # Strategy-specific data
    
    def __post_init__(self):
        """Validate confidence is in valid range."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(
                f"Confidence must be between 0 and 1, got {self.confidence}"
            )
```

### 2.2 SignalType Enumeration

```python
class SignalType(Enum):
    """Trading signal types with numeric values."""
    
    STRONG_BUY = 2      # High conviction buy (>80% confidence)
    BUY = 1             # Standard buy (60-80% confidence)
    NEUTRAL = 0         # No action / hold
    SELL = -1           # Standard sell (60-80% confidence)
    STRONG_SELL = -2    # High conviction sell (>80% confidence)
```

### 2.3 Signal Mapping Rules

| Signal Type | Numeric Value | Confidence Range | Action |
|-------------|---------------|------------------|--------|
| STRONG_BUY | +2 | 0.80 - 1.00 | Aggressive long entry |
| BUY | +1 | 0.60 - 0.79 | Standard long entry |
| NEUTRAL | 0 | 0.00 - 0.59 | No action / hold |
| SELL | -1 | 0.60 - 0.79 | Standard short entry |
| STRONG_SELL | -2 | 0.80 - 1.00 | Aggressive short entry |

### 2.4 Signal Metadata Schema

```python
signal_metadata = {
    # Strategy-Specific
    "strategy_version": str,           # Strategy version identifier
    "strategy_params": Dict[str, Any], # Parameters used
    
    # Signal Context
    "entry_price": float,              # Suggested entry price
    "stop_loss": float,                # Suggested stop loss
    "take_profit": float,              # Suggested take profit
    "timeframe": str,                  # Timeframe analyzed (e.g., "1m", "5m")
    
    # Technical Indicators
    "indicators": {
        "rsi": float,                  # RSI value
        "macd": float,                 # MACD value
        "volume_profile": Dict,        # Volume profile data
        # ... other indicators
    },
    
    # Risk Metrics
    "risk_reward_ratio": float,        # Expected risk/reward
    "win_probability": float,          # Estimated win probability
    
    # Execution Hints
    "urgency": str,                    # "LOW" | "MEDIUM" | "HIGH"
    "order_type": str,                 # "MARKET" | "LIMIT"
    "time_in_force": str               # "GTC" | "IOC" | "FOK"
}
```

### 2.5 Sample Signal Payloads

**Buy Signal:**
```json
{
  "signal_type": "BUY",
  "confidence": 0.75,
  "symbol": "SYMBOL",
  "timestamp": 1729785000.456,
  "strategy": "LVN-Breakout",
  "metadata": {
    "entry_price": 67250.00,
    "stop_loss": 66900.00,
    "take_profit": 68000.00,
    "timeframe": "5m",
    "risk_reward_ratio": 2.14,
    "urgency": "MEDIUM"
  }
}
```

---

## 3. Risk & Position Structures

### 3.1 RiskMetrics Class

**Location:** `nexus_ai.py` lines 528-551

```python
@dataclass
class RiskMetrics:
    """Risk management metrics for position sizing."""
    
    position_size: float           # Fraction of capital (0.0 to 1.0)
    stop_loss: float               # Stop loss percentage (e.g., 0.02 = 2%)
    take_profit: float             # Take profit percentage (e.g., 0.04 = 4%)
    max_drawdown: float            # Maximum allowed drawdown
    sharpe_ratio: float            # Risk-adjusted return metric
    var_95: float                  # Value at Risk at 95% confidence
    expected_return: float         # Expected return in currency units
    risk_score: float              # Composite risk score (0.0 to 1.0)
    
    def validate(self) -> bool:
        """Validate risk metrics are within acceptable ranges."""
        return all([
            0 <= self.position_size <= 1,
            self.stop_loss > 0,
            self.take_profit > 0,
            0 <= self.risk_score <= 1,
        ])
```

### 3.2 Risk Calculation Formulas

**Kelly Criterion (Position Sizing):**
```
kelly_fraction = (p * b - q) / b

Where:
  p = probability of win (signal confidence)
  q = probability of loss (1 - p)
  b = risk-reward ratio (take_profit / stop_loss)

position_size = max(kelly_min_fraction, kelly_fraction * safety_factor)
position_size = min(position_size, max_position_size)

Default Parameters:
  kelly_min_fraction = 0.001 (0.1% of capital)
  safety_factor = 0.25 (use 25% of Kelly)
  max_position_size = 0.1 (10% of capital)
```

**Value at Risk (VaR 95%):**
```
var_95 = max_loss * z_score_95

Where:
  max_loss = position_size * capital * stop_loss
  z_score_95 = 1.645 (95% confidence level)
```

**Sharpe Ratio:**
```
sharpe_ratio = expected_return / max_loss

Where:
  expected_return = position_size * capital * take_profit * confidence
  max_loss = position_size * capital * stop_loss
```

**Composite Risk Score:**
```
risk_factors = [
    position_size / max_position_size,
    abs(daily_pnl) / (capital * max_daily_loss),
    max_drawdown_reached / max_drawdown,
    1.0 - signal_confidence
]

risk_score = mean([min(factor, 1.0) for factor in risk_factors])
```

### 3.3 Position Structure

```python
@dataclass
class Position:
    """Position tracking for a single symbol."""
    
    # Identification
    symbol: str                    # Trading symbol
    instrument_type: str           # 'SPOT' | 'FUTURES' | 'OPTIONS'
    
    # Position Data
    quantity: float                # Net position (+ = long, - = short)
    avg_entry_price: float         # Average entry price
    current_price: float           # Current market price
    
    # P&L Tracking
    unrealized_pnl: float          # Mark-to-market P&L
    realized_pnl: float            # Closed trade P&L
    
    # Timing
    first_entry_time: float        # First entry timestamp
    last_update_time: float        # Last update timestamp
    num_trades: int                # Number of trades
    
    # Risk Management
    stop_loss_price: float         # Stop loss level
    take_profit_price: float       # Take profit level
    max_position_reached: float    # Peak position size
    
    # Instrument-Specific
    contract_size: float           # Contract multiplier
    expiry_date: Optional[float]   # Expiry timestamp (futures/options)
    point_value: float             # Value per point move
```

---

## 4. Order & Execution Structures

### 4.1 Order Class

**Location:** `nexus_ai.py` lines 5006-5019

```python
@dataclass
class Order:
    """Order representation for execution tracking."""
    
    order_id: str                  # Unique identifier (e.g., "ORD_000001")
    symbol: str                    # Trading symbol
    side: str                      # 'buy' or 'sell'
    quantity: float                # Order quantity
    price: float                   # Order price
    order_type: str                # 'market' | 'limit' | 'stop' | 'stop_limit'
    status: str                    # 'pending' | 'filled' | 'cancelled' | 'rejected'
    timestamp: float               # Order creation time
    metadata: Dict[str, Any]       # Execution details
```

### 4.2 Order Lifecycle

```
┌─────────┐
│ CREATED │
└────┬────┘
     │
     ▼
┌─────────┐     ┌───────────┐
│ PENDING ├────►│  FILLED   │ (Success)
└────┬────┘     └───────────┘
     │
     ├────────►┌───────────┐
     │         │ CANCELLED │ (Manual cancellation)
     │         └───────────┘
     │
     └────────►┌───────────┐
               │ REJECTED  │ (Risk validation failure)
               └───────────┘
```

### 4.3 Order Metadata Schema

```python
order_metadata = {
    # Source Signal
    "signal": TradingSignal,       # Original signal
    "metrics": RiskMetrics,        # Risk metrics
    
    # Execution Details
    "avg_entry_price": float,      # Average fill price
    "last_fill_qty": float,        # Last fill quantity
    "total_filled": float,         # Total filled quantity
    "num_fills": int,              # Number of partial fills
    
    # Risk Levels
    "stop_loss": float,            # Stop loss price
    "take_profit": float,          # Take profit price
    
    # Timing
    "submission_time": float,      # Broker submission time
    "fill_time": float,            # Fill timestamp
    "latency_ms": float,           # Submission to fill latency
    
    # Fees & Costs
    "commission": float,           # Commission paid
    "slippage": float,             # Price slippage
    
    # Broker Details
    "broker_order_id": str,        # Broker's order ID
    "exchange": str,               # Execution exchange
    "routing": str                 # Order routing info
}
```

---

## 5. Error Handling Structures

### 5.1 TradingError Class

**Location:** `nexus_ai.py` lines 325-382

```python
@dataclass
class TradingError:
    """Structured error information with recovery guidance."""
    
    code: ErrorCode                # Standardized error code
    message: str                   # Human-readable message
    component: str                 # Component that raised error
    details: Dict[str, Any]        # Additional context
    timestamp: float               # Error occurrence time
    recoverable: bool              # Can system auto-recover?
    retry_after: Optional[int]     # Seconds to wait before retry
    correlation_id: Optional[str]  # Request correlation ID
```

### 5.2 ErrorCode Enumeration

```python
class ErrorCode(Enum):
    # Input/Validation Errors (4xx equivalent)
    INVALID_INPUT = "INVALID_INPUT"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    
    # System Errors (5xx equivalent)
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    STRATEGY_EXECUTION_FAILED = "STRATEGY_EXECUTION_FAILED"
    RISK_VALIDATION_FAILED = "RISK_VALIDATION_FAILED"
    
    # External Errors
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    
    # Resource Errors
    MEMORY_EXHAUSTED = "MEMORY_EXHAUSTED"
    DISK_FULL = "DISK_FULL"
```

### 5.3 Error Severity Mapping

| Error Code | Severity | Recoverable | Retry Strategy |
|------------|----------|-------------|----------------|
| INVALID_INPUT | WARNING | No | Reject request |
| VALIDATION_FAILED | WARNING | No | Reject request |
| MODEL_LOAD_FAILED | CRITICAL | No | Use fallback |
| STRATEGY_EXECUTION_FAILED | ERROR | Yes | Skip strategy |
| RISK_VALIDATION_FAILED | WARNING | No | Reject trade |
| NETWORK_ERROR | ERROR | Yes | Exponential backoff |
| TIMEOUT_ERROR | WARNING | Yes | Retry once |
| MEMORY_EXHAUSTED | CRITICAL | Yes | Trigger cleanup |
| DISK_FULL | CRITICAL | No | Alert operator |

---

## 6. Validation Rules

### 6.1 Market Data Validation

```python
validation_rules = {
    "symbol": {
        "required": True,
        "type": str,
        "pattern": r"^[A-Z0-9\-]{3,20}$",
        "error": "Symbol must be 3-20 uppercase alphanumeric characters"
    },
    "timestamp": {
        "required": True,
        "type": float,
        "min": 0,
        "max": lambda: time.time() + 60,
        "error": "Timestamp must be within ±60s of current time"
    },
    "price": {
        "required": True,
        "type": Decimal,
        "min": Decimal("0.00000001"),
        "max": Decimal("999999999.99999999"),
        "precision": 8,
        "error": "Price must be positive with max 8 decimal places"
    },
    "bid_ask_spread": {
        "custom": lambda data: data.bid < data.ask,
        "error": "Bid must be less than ask (no crossed market)"
    }
}
```

### 6.2 Trading Signal Validation

```python
signal_validation_rules = {
    "confidence": {
        "required": True,
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "error": "Confidence must be between 0.0 and 1.0"
    },
    "signal_type": {
        "required": True,
        "type": SignalType,
        "allowed_values": [
            SignalType.STRONG_BUY,
            SignalType.BUY,
            SignalType.NEUTRAL,
            SignalType.SELL,
            SignalType.STRONG_SELL
        ],
        "error": "Invalid signal type"
    },
    "symbol": {
        "required": True,
        "type": str,
        "min_length": 3,
        "max_length": 20,
        "error": "Symbol must be 3-20 characters"
    }
}
```

### 6.3 Risk Metrics Validation

```python
risk_validation_rules = {
    "position_size": {
        "required": True,
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "error": "Position size must be between 0.0 and 1.0"
    },
    "stop_loss": {
        "required": True,
        "type": float,
        "min": 0.0001,
        "max": 0.5,
        "error": "Stop loss must be between 0.01% and 50%"
    },
    "risk_score": {
        "required": True,
        "type": float,
        "min": 0.0,
        "max": 1.0,
        "warning_threshold": 0.8,
        "error": "Risk score must be between 0.0 and 1.0"
    }
}
```

---

## Appendix A: Data Retention Policies

| Data Type | Retention Period | Storage Location | Compression |
|-----------|------------------|------------------|-------------|
| Market Data (Raw) | 7 days | Memory buffer | None |
| Market Data (Historical) | 1 year | Database | ZSTD |
| Trading Signals | 90 days | Database | ZSTD |
| Orders | 7 years | Database | ZSTD |
| Positions | Current + 7 years | Database | ZSTD |
| Errors | 30 days | Log files | GZIP |
| Audit Trail | 7 years | Immutable storage | ZSTD |

## Appendix B: Timezone Handling

**System Standard:** UTC (Coordinated Universal Time)

**Conversion Rules:**
- All incoming timestamps converted to UTC on ingestion
- All outgoing timestamps provided in UTC
- Exchange-specific timestamps preserved in metadata
- Clock drift tolerance: ±60 seconds

**Timestamp Formats:**
- Internal: Unix epoch (float, seconds since 1970-01-01 00:00:00 UTC)
- API: ISO 8601 (e.g., "2025-10-24T16:30:00Z")
- Logs: ISO 8601 with milliseconds (e.g., "2025-10-24T16:30:00.123Z")
