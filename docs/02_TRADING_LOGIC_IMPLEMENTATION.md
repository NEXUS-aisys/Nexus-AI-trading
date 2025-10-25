# 2. Trading Logic Implementation

## 2.1 Complete Order Execution Workflow

This workflow details the process from the generation of a trading signal to the final execution confirmation.

| Step | Component | Action | Input | Output | Latency SLA |
|---|---|---|---|---|---|
| **1. Signal Generation** | Strategy Engine | Generate `TradingSignal` | Market Data, Position Data | `TradingSignal` | N/A (Asynchronous) |
| **2. Pre-Trade Validation** | Risk Manager | Check high-level risk limits (e.g., max notional, max position size) | `TradingSignal`, Current Risk Profile | `OrderRequest` (or Rejection) | < 5 ms |
| **3. Data Enrichment** | Data Processor | Map `instrument_id` to broker/exchange symbols, add liquidity data | `OrderRequest`, Instrument Master | Enriched `OrderRequest` | < 2 ms |
| **4. Order Generation** | Order Manager | Convert `OrderRequest` into Broker-specific format (e.g., FIX message) | Enriched `OrderRequest` | Broker `Order` object | < 1 ms |
| **5. Order Submission** | Connectivity Gateway | Send order to execution venue/EMS | Broker `Order` object | `OrderAcknowledge` | < 1 ms (Gateway to Broker) |
| **6. Order Tracking** | Order Manager | Update order status to PENDING | `OrderAcknowledge` | Updated `Order` state | N/A |
| **7. Execution Reporting** | Connectivity Gateway | Receive Execution Reports (Fills/Rejects) | Execution Report (FIX/JSON) | `TradeEvent` | N/A (External Event) |
| **8. Post-Trade Validation** | Trade Validator | Check trade price/quantity against expected values (slippage check) | `TradeEvent`, Original `Order` | Validated `TradeEvent` (or Violation Flag) | < 3 ms |
| **9. Position Update** | Position Manager | Update portfolio position and P&L | Validated `TradeEvent` | Updated `Position` object | < 5 ms |
| **10. Confirmation** | Audit/Alert System | Log trade, send confirmation to upstream systems | Updated `Position`, `TradeEvent` | Audit Log Entry | < 10 ms |

## 2.2 Decision-Making Algorithms

Decision-making is centralized in the Strategy Engine and Pre-Trade Risk Manager.

### Strategy Engine (Signal Generation)

*   **Algorithm**: Time-Series Trend Following / Mean Reversion (Specific model parameters are dynamic and stored in configuration service).
*   **Key Parameters**:
    *   `lookback_period`: (e.g., 200 bars for long-term trend).
    *   `signal_threshold`: (e.g., RSI > 70 for SELL signal).
    *   `confidence_model`: Bayesian probability model calculating $P(Win|Signal)$ (Input: Market Volatility, Spread BPS, Historical Signal Performance).
*   **Edge Case: Volatility Spike**: If VIX (or equivalent measure) exceeds a threshold (e.g., 30), the confidence score is capped at 0.70, and `OrderType` is restricted to LIMIT orders only.
*   **Edge Case: Insufficient Data**: If less than 100 historical bars are available, signal generation is suppressed, and a NEUTRAL signal is emitted.

### Pre-Trade Risk Manager (Order Sizing)

*   **Algorithm**: Dynamic Position Sizing based on Kelly Criterion (Modified) and VaR limits.
*   **Key Parameters**:
    *   `max_leverage`: (e.g., 5.0x).
    *   `max_drawdown_daily`: (e.g., 2.0% of capital).
    *   `position_sizing_factor`: Max fraction of total capital to allocate per signal (e.g., 0.10).
*   **Edge Case: Multiple Signals**: If multiple BUY signals arrive simultaneously for correlated instruments, the total notional exposure is aggregated, and the total order quantity is reduced to comply with the aggregate max notional limit, prioritizing the signal with the highest confidence.

## 2.3 Trade Validation Rules and Severity Levels

Trade validation occurs at two stages: Pre-Trade (Order Request) and Post-Trade (Execution Report).

| Rule ID | Stage | Description | Severity | Action |
|---|---|---|---|---|
| `RTV-001` | Pre-Trade | Max Notional Limit Check: Notional amount of order exceeds account limit. | High | Reject Order Submission, Alert Risk Team. |
| `RTV-002` | Pre-Trade | Price Liveness Check: Last market data quote is older than 500ms. | Medium | Warn, Proceed if confidence > 0.8, else Reject. |
| `RTV-003` | Pre-Trade | Max Order Rate Check: Order submission rate exceeds 10 orders per second. | High | Throttle/Block subsequent orders for 5 seconds. |
| `PT-001` | Post-Trade | Slippage Check: Execution price deviation from limit price (or pre-trade mid-price for Market Order) exceeds 10 BPS. | Medium | Log Warning, Flag for Audit Review, Proceed with Position Update. |
| `PT-002` | Post-Trade | Instrument Restriction: Trade received for an instrument marked as restricted or halted. | High | Reverse/Cancel Trade, Halt Position Update, Immediate Alert. |

## 2.4 Data Enrichment Processes

Data enrichment is primarily performed on the `OrderRequest` before submission (Step 3) and on `TradeEvent` for position accounting.

| Data Field | Source Mapping | Transformation Logic |
|---|---|---|
| **Broker Symbol** (Order) | Instrument Master Database (`instrument_id` $\rightarrow$ `broker_symbol`) | Lookup table based on configured exchange/broker connection. |
| **Contract Multiplier** (Order) | Instrument Master Database (`instrument_id` $\rightarrow$ `multiplier`) | Used to calculate final notional value: $Notional = Quantity \times Price \times Multiplier$. |
| **Current Spread BPS** (Trade Event) | Real-time Market Data Feed | Calculation: $(Ask - Bid) / MidPrice \times 10000$. Added to `TradeEvent.metadata` for slippage analysis. |
| **Account Margin Ratio** (Order) | Account Service API | Fetch current margin usage for the account. Used in Pre-Trade Risk check to prevent margin calls. |
| **Commission Estimate** (Order) | Fee Schedule Service | Look up fixed/percentage fee based on `instrument_type`, `exchange_id`, and `quantity`. Added to Order metadata.