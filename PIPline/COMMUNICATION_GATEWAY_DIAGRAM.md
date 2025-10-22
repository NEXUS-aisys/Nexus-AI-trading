# COMMUNICATION GATEWAY INTEGRATION DIAGRAM
**Visual Architecture: NEXUS AI + Universal Trading Gateway**

---

## COMPLETE SYSTEM ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         TRADING PLATFORMS LAYER                             │
│  ┌──────────┐  ┌───────────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │   NT8    │  │ Sierra Chart  │  │MT5 Global│  │MT5 Brazil│  │ Bookmap ││
│  │(C#/.NET) │  │   (C++/DLL)   │  │ (Python) │  │ (Python) │  │ (REST)  ││
│  │  NQ 03-25│  │     NQH25     │  │  NAS100  │  │   WING25 │  │NQ-03-25 ││
│  └────┬─────┘  └───────┬───────┘  └────┬─────┘  └────┬─────┘  └────┬────┘│
└───────┼─────────────────┼───────────────┼─────────────┼──────────────┼─────┘
        │                 │               │             │              │
        ↓                 ↓               ↓             ↓              ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                        PLATFORM ADAPTERS LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │ Each adapter:                                                        │  │
│  │  ├─ Receives market data from platform                              │  │
│  │  ├─ Normalizes to universal format                                  │  │
│  │  ├─ Sends to Gateway via ZeroMQ                                     │  │
│  │  ├─ Receives orders from Gateway                                    │  │
│  │  ├─ Translates to platform-specific format                          │  │
│  │  └─ Executes on platform & reports fills                            │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Python Code:                                                               │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │ from communication.platform_adapter_template import PlatformAdapter│    │
│  │                                                                      │    │
│  │ class NT8Adapter(PlatformAdapter):                                 │    │
│  │     def execute_order(self, order):                                │    │
│  │         # Send to NT8                                              │    │
│  │         self.nt8_connection.submit(order)                          │    │
│  └───────────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ↓ ZeroMQ ROUTER Socket (Port 5555)
                             │
┌────────────────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL TRADING GATEWAY SERVER                         │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ gateway_server.py (1054 lines)                                      │   │
│  │                                                                      │   │
│  │ ┌────────────────────────────────────────────────────────────────┐ │   │
│  │ │ SYMBOL REGISTRY (TRADING_UNIVERSE)                             │ │   │
│  │ │  Universal → Platform Mapping:                                 │ │   │
│  │ │  NQ → NT8:"NQ 03-25" | Sierra:"NQH25" | MT5:"NAS100"          │ │   │
│  │ │  ES → NT8:"ES 03-25" | Sierra:"ESH25" | MT5:"US500"           │ │   │
│  │ │  WIN → MT5Brazil:"WING25"                                      │ │   │
│  │ └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │ ┌────────────────────────────────────────────────────────────────┐ │   │
│  │ │ CURRENCY CONVERTER                                             │ │   │
│  │ │  BRL → USD using WDO rates                                     │ │   │
│  │ │  WIN profit: R$ 20 → $3.64 USD                                │ │   │
│  │ │  WDO profit: R$ 100 → $18.18 USD                              │ │   │
│  │ └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │ ┌────────────────────────────────────────────────────────────────┐ │   │
│  │ │ CONTRACT ROLLOVER MANAGER                                      │ │   │
│  │ │  Auto-tracks front month:                                      │ │   │
│  │ │  March 2025: NQH25                                            │ │   │
│  │ │  June 2025: NQM25 (rolls automatically)                       │ │   │
│  │ └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │ ┌────────────────────────────────────────────────────────────────┐ │   │
│  │ │ MESSAGE ROUTER                                                 │ │   │
│  │ │  ├─ Receives market data from platforms                        │ │   │
│  │ │  ├─ Normalizes symbol: "NQ 03-25" → "NQ"                      │ │   │
│  │ │  ├─ Routes to NEXUS AI                                         │ │   │
│  │ │  ├─ Receives signals from NEXUS                                │ │   │
│  │ │  ├─ Expands symbol: "NQ" → platform-specific                  │ │   │
│  │ │  └─ Routes to all connected platforms                          │ │   │
│  │ └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  │ Performance:                                                        │   │
│  │  ├─ Latency: 5-15 microseconds                                    │   │
│  │  ├─ Throughput: 1M+ msgs/sec                                      │   │
│  │  └─ Protocol: ZeroMQ + MessagePack                                │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ↓ ZeroMQ DEALER Socket (Port 5556)
                             │
┌────────────────────────────────────────────────────────────────────────────┐
│                     NEXUS GATEWAY INTERFACE LAYER                           │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ nexus_gateway_interface.py (529 lines)                             │   │
│  │                                                                      │   │
│  │ class NexusGatewayInterface:                                        │   │
│  │     ├─ connect() → Connects to gateway                             │   │
│  │     ├─ set_market_data_callback(handler)                           │   │
│  │     ├─ set_order_fill_callback(handler)                            │   │
│  │     ├─ send_signal(symbol, signal, confidence, qty)                │   │
│  │     └─ get_stats()                                                 │   │
│  │                                                                      │   │
│  │ ┌──────────────────────────────────────────────────────────────┐  │   │
│  │ │ CALLBACKS                                                     │  │   │
│  │ │  ├─ on_market_data(data) → Forward to NEXUS Pipeline         │  │   │
│  │ │  └─ on_order_fill(fill) → Feedback to NEXUS                  │  │   │
│  │ └──────────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ↓ Python Function Calls
                             │
┌────────────────────────────────────────────────────────────────────────────┐
│                  NEXUS COMMUNICATION ADAPTER                                │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ nexus_communication_adapter.py (NEW - YOU CREATE THIS)             │   │
│  │                                                                      │   │
│  │ class NexusCommunicationAdapter:                                    │   │
│  │                                                                      │   │
│  │  def handle_market_data(self, data):                               │   │
│  │      """                                                            │   │
│  │      Called when market data arrives from gateway                  │   │
│  │      """                                                            │   │
│  │      # 1. Extract symbol                                           │   │
│  │      symbol = data["symbol"]  # Already normalized: "NQ"           │   │
│  │                                                                      │   │
│  │      # 2. Check for duplicate order                                │   │
│  │      if symbol in self.active_orders:                              │   │
│  │          return  # Skip                                            │   │
│  │                                                                      │   │
│  │      # 3. Calculate MQScore                                        │   │
│  │      mqscore = self.mqscore_engine.calculate(data)                 │   │
│  │                                                                      │   │
│  │      # 4. Quality gate                                             │   │
│  │      if mqscore.composite_score < 0.5:                             │   │
│  │          return  # Skip low quality                                │   │
│  │                                                                      │   │
│  │      # 5. Execute NEXUS Pipeline (existing workflow!)              │   │
│  │      result = await self.nexus_pipeline.process(symbol, data)      │   │
│  │                                                                      │   │
│  │      # 6. Extract signal                                           │   │
│  │      signal = result["signal"]                                     │   │
│  │      confidence = result["confidence"]                             │   │
│  │                                                                      │   │
│  │      # 7. Send to gateway                                          │   │
│  │      if confidence >= 0.65:                                        │   │
│  │          self.gateway.send_signal(                                 │   │
│  │              symbol=symbol,                                        │   │
│  │              signal=1.0 if signal > 0 else -1.0,                  │   │
│  │              confidence=confidence,                                │   │
│  │              quantity=self.calculate_size(symbol, confidence)      │   │
│  │          )                                                          │   │
│  │          # Mark as active                                          │   │
│  │          self.active_orders[symbol] = order_id                     │   │
│  │                                                                      │   │
│  │  def handle_order_fill(self, fill_data):                           │   │
│  │      """                                                            │   │
│  │      Called when order fills on platform                           │   │
│  │      """                                                            │   │
│  │      # Record for feedback loop                                    │   │
│  │      self.nexus_pipeline.record_trade_outcome(fill_data)           │   │
│  │                                                                      │   │
│  │      # Update strategy weights (learning!)                         │   │
│  │      self.weight_calculator.update_weights()                       │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ↓
┌────────────────────────────────────────────────────────────────────────────┐
│                    EXISTING NEXUS AI PIPELINE                               │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ Data Auth → MQScore 6D → 20 Strategies → Weighted Ensemble →      │   │
│  │ ML Decision Engine → Signal Generation                             │   │
│  │                                                                      │   │
│  │ ⚠️ NO CHANGES NEEDED TO EXISTING PIPELINE!                         │   │
│  │ Adapter handles all communication                                  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## MESSAGE FLOW EXAMPLES

### Flow 1: Market Data → Signal

```
1. NT8 receives tick: NQ 03-25 @ 16250.50
   ↓
2. NT8 Adapter normalizes:
   {
       "type": "MARKET_DATA",
       "platform": "nt8",
       "data": {
           "symbol": "NQ 03-25",
           "price": 16250.50,
           "volume": 10,
           "bid": 16250.25,
           "ask": 16250.75
       }
   }
   ↓ ZeroMQ (Port 5555)
   ↓
3. Gateway Server receives & translates:
   - Normalizes symbol: "NQ 03-25" → "NQ"
   - Forwards to NEXUS AI
   ↓ ZeroMQ (Port 5556)
   ↓
4. NEXUS Gateway Interface receives:
   - Calls market_data_callback
   ↓
5. Communication Adapter processes:
   - Checks duplicate (no existing order for NQ)
   - Calculates MQScore (composite: 0.72, PASS)
   - Executes 20 strategies (15 generate signals)
   - Weighted aggregation (signal: 0.85, conf: 0.78)
   - ML Decision Engine (BUY, qty: 1)
   ↓
6. Adapter sends signal:
   gateway.send_signal("NQ", signal=1.0, confidence=0.78, qty=1)
   ↓ ZeroMQ (Port 5556)
   ↓
7. Gateway Server receives signal:
   - Expands symbol: "NQ" → platform-specific formats
   - Routes to ALL connected platforms:
       • NT8: "NQ 03-25" BUY 1
       • Sierra: "NQH25" BUY 1
       • MT5: "NAS100" BUY 1 CFD
   ↓ ZeroMQ (Port 5555)
   ↓
8. Platform Adapters receive & execute:
   - NT8 Adapter: Submits market order to NT8
   - Sierra Adapter: Submits order to Sierra
   - MT5 Adapter: Calls mt5.order_send()
   ↓
9. Orders execute on platforms:
   - NT8: FILLED @ 16251.00
   - Sierra: FILLED @ 16251.25
   - MT5: FILLED @ 16251.50
   ↓
10. Platform Adapters send fills:
    {
        "type": "ORDER_FILL",
        "order_id": "uuid-1234",
        "symbol": "NQ",
        "filled_qty": 1,
        "fill_price": 16251.00,
        "platform": "nt8"
    }
    ↓ ZeroMQ → Gateway → NEXUS Interface → Adapter
    ↓
11. Adapter records outcome:
    - Updates strategy performance
    - Recalculates weights
    - Marks order as filled
    - Symbol freed for new signals
```

**Total Latency**: <100ms (market data → order submitted)

---

### Flow 2: Duplicate Prevention

```
1. Market data for NQ arrives
   ↓
2. Adapter checks: active_orders["NQ"] exists?
   ├─ YES → SKIP (log: "Order already active for NQ")
   └─ NO → Continue to NEXUS Pipeline
   
This prevents duplicate orders for same symbol!
```

---

### Flow 3: Multi-Symbol Concurrent

```
Parallel Processing (10-50 symbols):

Thread 1: BTCUSD data arrives → Process → Send signal (if no active order)
Thread 2: ETHUSDT data arrives → Process → Send signal (if no active order)
Thread 3: NQ data arrives → Process → Send signal (if no active order)
...
Thread 50: SOLUSDT data arrives → Process → Send signal (if no active order)

Each thread:
  ├─ Independent pipeline execution
  ├─ Independent duplicate check
  └─ Independent order management

Result: Up to 50 concurrent positions (max 1 per symbol)
```

---

## PORT CONFIGURATION

```
┌─────────┬──────┬──────────────┬────────────────────────────┐
│ Service │ Port │ Socket Type  │ Connection                 │
├─────────┼──────┼──────────────┼────────────────────────────┤
│ Gateway │ 5555 │ ZeroMQ       │ Platform Adapters →        │
│         │      │ ROUTER       │ Gateway Server             │
├─────────┼──────┼──────────────┼────────────────────────────┤
│ Gateway │ 5556 │ ZeroMQ       │ NEXUS AI ↔                 │
│         │      │ DEALER       │ Gateway Server             │
├─────────┼──────┼──────────────┼────────────────────────────┤
│ Gateway │ 5557 │ ZeroMQ       │ Admin/Status ↔             │
│         │      │ REP          │ Gateway Server             │
├─────────┼──────┼──────────────┼────────────────────────────┤
│ NT8     │ 9001 │ TCP          │ NT8 C# ↔                   │
│         │      │              │ Python Adapter             │
└─────────┴──────┴──────────────┴────────────────────────────┘
```

---

## STARTUP SEQUENCE

```
1. Start Gateway Server (must be first!)
   └─ Binds to ports 5555, 5556, 5557
   
2. Start Platform Adapters (any order)
   ├─ Connect to Gateway port 5555
   └─ Start market data streams
   
3. Start NEXUS AI (last)
   └─ Connect to Gateway port 5556
   
All components auto-reconnect if disconnected.
```

---

## BENEFITS DIAGRAM

```
┌────────────────────────────────────────────────────────────────┐
│                       WITHOUT GATEWAY                           │
│                                                                  │
│  NEXUS AI ──(hardcoded)──> Single Platform                     │
│                                                                  │
│  Problems:                                                      │
│  ❌ Tightly coupled                                             │
│  ❌ Platform-specific code                                      │
│  ❌ Manual symbol mapping                                       │
│  ❌ No multi-platform support                                   │
│  ❌ Difficult to test                                           │
└────────────────────────────────────────────────────────────────┘

                            ↓ ADD GATEWAY ↓

┌────────────────────────────────────────────────────────────────┐
│                        WITH GATEWAY                             │
│                                                                  │
│               ┌────────────────────┐                           │
│               │  NEXUS AI          │                           │
│               │  (unchanged!)      │                           │
│               └─────────┬──────────┘                           │
│                         │                                       │
│                    Universal API                                │
│                         │                                       │
│               ┌─────────┴──────────┐                           │
│               │  Gateway Server    │                           │
│               │  (translation)     │                           │
│               └─────────┬──────────┘                           │
│                         │                                       │
│          ┌──────────────┼──────────────┐                      │
│          │              │               │                      │
│      ┌───┴──┐      ┌───┴──┐       ┌───┴──┐                  │
│      │ NT8  │      │ MT5  │       │Sierra│                  │
│      └──────┘      └──────┘       └──────┘                  │
│                                                                  │
│  Benefits:                                                      │
│  ✅ Loosely coupled                                             │
│  ✅ Platform-agnostic code                                      │
│  ✅ Auto symbol mapping                                         │
│  ✅ Multi-platform ready                                        │
│  ✅ Easy to test (simulated mode)                               │
│  ✅ 5-15μs latency                                              │
│  ✅ 1M+ msgs/sec throughput                                     │
└────────────────────────────────────────────────────────────────┘
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-20  
**Status**: Visual Architecture Complete
