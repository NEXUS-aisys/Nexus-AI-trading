# 4. Risk Management Framework

## 4.1 Mathematical Models for Risk Calculations

### 4.1.1 Value-at-Risk (VaR) Calculation

*   **Model**: Historical Simulation VaR (1-day, 99% confidence level).
*   **Input Parameters**:
    *   `historical_data_window`: 250 trading days of P&L data.
    *   `confidence_level`: 99% (alpha = 0.01).
    *   `position_vector`: Current P&L for each instrument.
    *   `risk_factor_sensitivities`: Pre-calculated sensitivities (e.g., Beta).
*   **Calculation Frequency**: Calculated and stored every 5 minutes (real-time) and End-of-Day (EOD) for reporting.

### 4.1.2 Expected Shortfall (ES) / Conditional VaR (CVaR)

*   **Model**: Calculated as the average loss exceeding the VaR threshold.
*   **Purpose**: Provides a more robust measure of tail risk compared to standard VaR.
*   **Input**: VaR calculation results.
*   **Calculation Frequency**: Daily EOD.

### 4.1.3 Stress Testing

*   **Model**: Scenario-based stress testing.
*   **Scenarios**:
    1.  *Market Crash*: S&P 500 drops 10% in one day (applies correlation matrix).
    2.  *Liquidity Crisis*: Spreads widen by 50% across all instruments.
    3.  *Rate Shock*: Key interest rates shift by $\pm 100$ basis points.
*   **Calculation Frequency**: Weekly (non-production environment) and Monthly (production reporting).

## 4.2 Exposure Monitoring Frequency and Aggregation Levels

Risk metrics are aggregated and monitored continuously.

| Metric | Frequency | Aggregation Level | Purpose |
|---|---|---|---|
| **Net Position** | Real-time (per trade) | Instrument, Sector, Portfolio | Liquidity and concentration monitoring. |
| **VaR** | Every 5 minutes | Portfolio, Asset Class | Regulatory and capital adequacy measurement. |
| **Max Drawdown (MTD)** | Real-time (per P&L update) | Account, Strategy | Performance and strategy violation monitoring. |
| **Notional Exposure (Gross/Net)** | Real-time (per trade) | Portfolio | Leverage management and broker limits. |
| **Liquidity Score** | Hourly | Instrument | Measures time and cost to liquidate position based on depth of book.

## 4.3 Alert Threshold Definitions by Risk Type

Alerts are classified by severity and trigger specific escalation protocols.

### Market Risk Alerts
| Threshold ID | Metric | Level | Trigger | Severity |
|---|---|---|---|---|
| `MR-A01` | Portfolio VaR | Warning | VaR exceeds 80% of mandated daily limit. | Medium |
| `MR-A02` | Portfolio VaR | Breach | VaR exceeds 100% of mandated daily limit. | High |
| `MR-A03` | Single Position Concentration | Warning | Single instrument notional exceeds 15% of Gross Market Value (GMV). | Medium |

### Credit Risk Alerts
| Threshold ID | Metric | Level | Trigger | Severity |
|---|---|---|---|---|
| `CR-A01` | Margin Usage | Warning | Margin Utilization exceeds 90%. | Medium |
| `CR-A02` | Margin Usage | Margin Call | Margin Utilization exceeds 100%. | High |

### Operational Risk Alerts
| Threshold ID | Metric | Level | Trigger | Severity |
|---|---|---|---|---|
| `OR-A01` | System Latency | Warning | Average Order Execution Latency > 200 ms (rolling 1-minute window). | Low |
| `OR-A02` | Connectivity Health | Failure | Loss of heartbeat from primary exchange gateway for > 10 seconds. | High |
| `OR-A03` | Reconciliation Failure | Breach | EOD Position Reconciliation Fails (Non-Zero Tolerance). | High |

## 4.4 Escalation Procedures with Notification Protocols

Escalation is managed via an Alert Management System (AMS) which routes notifications based on severity.

| Severity | Notification Protocol | Required Acknowledgment Time | Escalation Path |
|---|---|---|---|
| **Low** | Email/Chat Bot Message | 60 minutes | Trader Console Log Only. |
| **Medium** | SMS, Email, PagerDuty Alert | 10 minutes | Trader $\rightarrow$ Risk Analyst $\rightarrow$ Head of Trading. |
| **High** | Immediate PagerDuty Alert, Automated Phone Call | 2 minutes | Automated System Shutdown (if applicable) $\rightarrow$ Head of Trading $\rightarrow$ Chief Risk Officer. |

**Automated Mitigation Actions (High Severity)**:
*   **Margin Call**: Automated suspension of new order submissions for affected accounts.
*   **VaR Breach**: Automated reduction of position size for the highest contributing risk factor until VaR is below the 90% threshold.
*   **Connectivity Failure**: Immediate failover to secondary gateway/broker. If failover fails, suspend all trading activities.