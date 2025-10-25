# 3. Position Management

## 3.1 Calculation Methodologies for Different Instrument Types

The Position Manager maintains a real-time ledger for all instrument types, calculating position attributes using specific methodologies.

### Equities and FX
*   **Methodology**: First-In, First-Out (FIFO) for P&L realization. Position quantity is calculated as the net sum of all realized and unrealized trades (long - short).
*   **Cost Basis**: Calculated as the weighted average price (WAP) of all current open lots.
*   **P&L Calculation**:
    *   *Realized P&L*: $Quantity_{Sold/Bought} \times (Execution Price - WAP_{Closed Lot})$.
    *   *Unrealized P&L*: $Quantity_{Open} \times (Current Market Price - WAP_{Open Lot})$.

### Futures
*   **Methodology**: Marked-to-Market (MTM) daily. Positions are typically offset against the exchange's closing price or marked-to-market multiple times a day.
*   **Cost Basis**: Reset daily based on the previous day's settlement price (for margin purposes).
*   **P&L Calculation**: Daily P&L is calculated based on the difference between the current settlement price and the previous settlement price, multiplied by the contract size and position quantity.

### Options
*   **Methodology**: Black-Scholes or equivalent model used for valuation. Position tracking includes Greeks (Delta, Gamma, Vega, Theta).
*   **Risk Position**: The position for options is tracked in terms of Delta-equivalent shares/contracts for aggregation purposes (see Cross-Instrument Tracking).

## 3.2 Real-Time vs. End-of-Day Reconciliation Processes

| Process | Frequency | Data Sources | Tolerance/Threshold | Action on Discrepancy |
|---|---|---|---|---|
| **Real-Time Position Update** | Per `TradeEvent` | Internal Trade Log, Position Ledger | N/A (Internal Update) | N/A |
| **Intraday Reconciliation** | Every 5 minutes | Internal Position Ledger vs. Broker API (`GetOpenPositions`) | Max 0.05% notional difference or 1 lot (whichever is greater). | Auto-correction if discrepancy is within tolerance, otherwise generate **Medium Severity Alert** and manual investigation required. |
| **End-of-Day Reconciliation (EOD)** | Daily (Post-Market Close) | Internal Position Ledger vs. Settlement Report/General Ledger | Zero Tolerance (Must match GL exactly). | **High Severity Alert**, halt new trading signals for affected accounts until resolved, generate mandatory audit report. |

## 3.3 Cross-Instrument Position Tracking Requirements

Positions are aggregated at multiple levels to monitor consolidated exposure and leverage:

1.  **Instrument Level**: Raw position quantity, WAP, P&L for a single security (e.g., AAPL stock).
2.  **Asset Class Level**: Aggregation of positions within a class (e.g., Total Equity, Total FX).
3.  **Hedged Position Level**: For instruments related via hedging strategies (e.g., S&P 500 Futures vs. SPY ETF).
    *   **Metric**: Net Delta Equivalent Position.
    *   **Calculation**: Sum of raw positions, weighted by their respective Delta/Beta factors relative to a common benchmark.
4.  **Portfolio Level**: Total notional exposure, gross market value, and net leverage across all asset classes.

## 3.4 Position Adjustment Handling for Corporate Actions

Corporate actions require explicit handling to maintain accurate position records and cost basis.

| Corporate Action Type | Data Source | Adjustment Logic | Audit Trail Requirement |
|---|---|---|---|
| **Stock Split (e.g., 2-for-1)** | Corporate Action Feed | Adjust position quantity: $New Quantity = Old Quantity \times Split Ratio$. Adjust WAP: $New WAP = Old WAP / Split Ratio$. | Log original position, action type, adjustment parameters, and final position state with timestamp. |
| **Reverse Split** | Corporate Action Feed | Reverse adjustment of quantity and WAP. Handle fractional shares (usually cash settled). | As above. Note cash settlement details if applicable. |
| **Dividend Payment** | Corporate Action Feed/Custodian | No change to quantity/WAP. Record cash receipt event in the P&L ledger. | Record dividend amount, payment date, and associated position. |
| **Rights Offering/Spin-off** | Corporate Action Feed | Requires creating a new position entry for the new security with an initial cost basis calculated from the fractional allocation of the original security's cost basis. | Full trace of cost basis allocation between original and new positions. |

**Processing Requirements**:
*   Corporate action processing must be idempotent and occur before the market open on the Ex-Date.
*   System must be able to process batch updates from data providers (e.g., Bloomberg, Refinitiv) and reconcile these against custodian reports.