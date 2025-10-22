# COMMUNICATION GATEWAY INTEGRATION PLAN
**Complete Integration: NEXUS AI Pipeline + Universal Trading Gateway**

---

## EXECUTIVE SUMMARY

The **communication folder** contains a **Universal Trading Gateway** that enables NEXUS AI to connect to multiple trading platforms (NT8, Sierra Chart, MT5, Bookmap) simultaneously via high-performance ZeroMQ messaging.

**Status**: ✅ Ready for Integration  
**Architecture**: Microservices with ZeroMQ + MessagePack  
**Performance**: 5-15μs latency, 1M+ msgs/sec throughput  
**Multi-Platform**: Single signal → execute on all platforms

---

## WHAT'S IN THE COMMUNICATION FOLDER

### Components Analysis:

| File | Size | Purpose | Status |
|------|------|---------|--------|
| `gateway_server.py` | 38KB | Central routing hub (1054 lines) | ✅ Complete |
| `nexus_gateway_interface.py` | 17KB | NEXUS AI integration adapter (529 lines) | ✅ Complete |
| `platform_adapter_template.py` | 13KB | Template for platform connections (400 lines) | ✅ Complete |
| `complete_system_test.py` | 15KB | Full system test harness (400 lines) | ✅ Complete |
| `main_readme.md` | 11KB | Architecture documentation | ✅ Complete |
| `quick_start_guide.md` | 10KB | Quick start guide | ✅ Complete |

---

## CURRENT PIPELINE vs GATEWAY ARCHITECTURE

### Current NEXUS AI Pipeline (Standalone):
```
Market Data Source (Exchange API)
    ↓
Data Authentication (HMAC-SHA256)
    ↓
MQScore 6D Engine
    ↓
20 Strategies → Weighted Ensemble
    ↓
ML Decision Engine
    ↓
??? HOW TO EXECUTE ORDERS ???
```

**Problem**: No standardized way to connect to trading platforms!

---

### NEW Architecture (With Gateway):
```
┌──────────────────────────────────────────────────────────────────┐
│                     TRADING PLATFORMS                             │
│  NT8 | Sierra Chart | MT5 Global | MT5 Brazil | Bookmap          │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────────────────────┐
│              PLATFORM ADAPTERS (Python/C#)                        │
│  ├─ NT8 Adapter (Python ← → C# bridge)                          │
│  ├─ Sierra Chart Adapter (DTC protocol)                         │
│  ├─ MT5 Adapter (MetaTrader5 Python library)                    │
│  └─ Bookmap Adapter (REST API)                                  │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                  ↓ ZeroMQ (Port 5555)
                        │
┌──────────────────────────────────────────────────────────────────┐
│              UNIVERSAL TRADING GATEWAY                            │
│  ├─ Symbol Normalization (NQ → Platform-specific)               │
│  ├─ Currency Conversion (BRL → USD)                             │
│  ├─ Contract Rollover (Automatic front month)                   │
│  ├─ Order Routing (1 signal → N platforms)                      │
│  ├─ Fill Aggregation (N platforms → 1 response)                 │
│  └─ Statistics & Monitoring                                     │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                  ↓ ZeroMQ (Port 5556)
                        │
┌──────────────────────────────────────────────────────────────────┐
│           NEXUS GATEWAY INTERFACE                                 │
│  ├─ NexusPipelineAdapter (auto execution)                       │
│  ├─ Market data callback handler                                │
│  ├─ Signal sender                                               │
│  └─ Fill receiver                                               │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ↓
┌──────────────────────────────────────────────────────────────────┐
│           NEXUS AI PIPELINE (Existing)                            │
│  Data Auth → MQScore → 20 Strategies → ML Decision → Signal     │
└──────────────────────────────────────────────────────────────────┘
```

---

## INTEGRATION BENEFITS

### 1. **Multi-Platform Execution** 🎯
```
ONE signal from NEXUS AI
    ↓
Gateway routes to ALL connected platforms:
    ├─ NT8: Executes on BTCUSD
    ├─ Sierra Chart: Executes on NQH25
    ├─ MT5: Executes on NAS100
    └─ Bookmap: Visual confirmation
```

### 2. **Universal Symbol Mapping** 📊
```python
# Your NEXUS code ALWAYS uses simple symbols:
nexus.send_signal("NQ", signal=1.0, confidence=0.85)

# Gateway translates automatically:
# - NT8: "NQ 03-25"
# - Sierra: "NQH25"
# - MT5: "NAS100"
# - Bookmap: "NQ-03-25"
```

### 3. **Currency Conversion** 💰
```python
# Brazilian markets (BRL) auto-converted to USD:
WIN profit: 100 points = R$ 20 → $3.64 USD (using WDO rate)
WDO profit: 10 points = R$ 100 → $18.18 USD

# All P&L reported to NEXUS in USD
```

### 4. **Contract Rollover** 📅
```
Gateway automatically tracks front month:
- March 2025: NQH25
- June 2025: NQM25 (rolls automatically)
- September 2025: NQU25

Your code never changes!
```

### 5. **High Performance** ⚡
```
Latency: 5-15 microseconds (ZeroMQ IPC)
Throughput: 1M+ messages/second
Lock-free: Circular buffers
Zero-copy: Direct memory access
```

---

## INTEGRATION PLAN

### PHASE 1: GATEWAY SETUP (Week 1)

#### Step 1.1: Install Dependencies
```bash
pip install pyzmq msgpack-python
```

#### Step 1.2: Start Gateway Server
```bash
cd communication
python gateway_server.py
```

**Expected Output**:
```
2025-10-20 12:00:00 - GatewayServer - INFO - Gateway Server initialized
2025-10-20 12:00:00 - GatewayServer - INFO - Platform socket: port 5555
2025-10-20 12:00:00 - GatewayServer - INFO - NEXUS socket: port 5556
2025-10-20 12:00:00 - GatewayServer - INFO - Gateway Server started ✓
```

#### Step 1.3: Test Gateway (Optional)
```bash
python complete_system_test.py
```

This will run a full end-to-end test with simulated platforms.

---

### PHASE 2: NEXUS INTEGRATION (Week 1-2)

#### Step 2.1: Create Integration Adapter

Create: `nexus_communication_adapter.py`

```python
#!/usr/bin/env python3
"""
NEXUS AI + Communication Gateway Integration
Connects existing NEXUS pipeline to multi-platform gateway
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import sys
import os

# Import NEXUS components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nexus_ai import (
    ProductionSequentialPipeline,
    NexusSecurityLayer,
    MarketData,
    SignalType
)
from MQScore_6D_Engine_v3 import MQScoreEngine
from communication.nexus_gateway_interface import NexusGatewayInterface

logger = logging.getLogger(__name__)


class NexusCommunicationAdapter:
    """
    Integrates NEXUS AI Pipeline with Universal Trading Gateway
    
    Flow:
    1. Gateway sends market data → Adapter
    2. Adapter → NEXUS Pipeline (existing workflow)
    3. NEXUS generates signal → Adapter
    4. Adapter → Gateway → Trading Platforms
    5. Platform fills → Gateway → Adapter → NEXUS (feedback)
    """
    
    def __init__(self,
                 nexus_pipeline: ProductionSequentialPipeline,
                 mqscore_engine: MQScoreEngine,
                 gateway_host: str = "localhost",
                 gateway_port: int = 5556,
                 auto_execute: bool = False):
        """
        Initialize adapter
        
        Args:
            nexus_pipeline: Your existing NEXUS AI pipeline
            mqscore_engine: Your MQScore engine instance
            gateway_host: Gateway server host
            gateway_port: Gateway server port
            auto_execute: Auto-execute signals (True for production)
        """
        self.nexus_pipeline = nexus_pipeline
        self.mqscore_engine = mqscore_engine
        self.auto_execute = auto_execute
        
        # Gateway interface
        self.gateway = NexusGatewayInterface(gateway_host, gateway_port)
        
        # State
        self.active_symbols = set()
        self.active_orders = {}  # symbol → order_id
        self.active_positions = {}  # order_id → position_details
        
        # Statistics
        self.stats = {
            "market_data_received": 0,
            "signals_generated": 0,
            "orders_executed": 0,
            "fills_received": 0
        }
        
        logger.info("NEXUS Communication Adapter initialized")
    
    async def start(self):
        """Start adapter"""
        # Connect to gateway
        if not self.gateway.connect():
            logger.error("Failed to connect to gateway")
            return False
        
        # Set callbacks
        self.gateway.set_market_data_callback(self.handle_market_data)
        self.gateway.set_order_fill_callback(self.handle_order_fill)
        
        logger.info("NEXUS Communication Adapter started ✓")
        logger.info(f"Auto-execute: {self.auto_execute}")
        
        return True
    
    def handle_market_data(self, data: Dict[str, Any]):
        """
        Handle market data from gateway
        
        This is where NEXUS pipeline processes data!
        """
        try:
            symbol = data.get("symbol")
            
            # Track symbol
            self.active_symbols.add(symbol)
            self.stats["market_data_received"] += 1
            
            # ========== EXISTING NEXUS PIPELINE ==========
            
            # 1. Data Authentication (already in NEXUS)
            # (Gateway already provides authenticated data)
            
            # 2. MQScore Calculation
            mqscore_result = self.mqscore_engine.calculate_mqscore_from_tick(data)
            
            # Quality gate
            if mqscore_result.composite_score < 0.5:
                logger.debug(f"Skipping {symbol}: Low MQScore {mqscore_result.composite_score:.2f}")
                return
            
            # 3. Strategy Execution (20 strategies)
            market_data_dict = {
                "symbol": symbol,
                "price": data["price"],
                "volume": data.get("volume", 0),
                "bid": data.get("bid", data["price"]),
                "ask": data.get("ask", data["price"]),
                "timestamp": data["timestamp"],
                "mqscore_composite": mqscore_result.composite_score,
                "mqscore_liquidity": mqscore_result.liquidity,
                "mqscore_volatility": mqscore_result.volatility,
                "mqscore_momentum": mqscore_result.momentum,
                "mqscore_trend": mqscore_result.trend_strength,
                "mqscore_imbalance": mqscore_result.imbalance,
                "mqscore_noise": mqscore_result.noise_level,
            }
            
            # Execute NEXUS pipeline
            pipeline_result = asyncio.run(
                self.nexus_pipeline.process_market_data(symbol, market_data_dict)
            )
            
            if not pipeline_result:
                return
            
            # 4. Check for duplicate order
            if symbol in self.active_orders:
                logger.debug(f"Active order exists for {symbol}, skipping")
                return
            
            # 5. Extract signal
            signal = pipeline_result.get("signal", 0.0)
            confidence = pipeline_result.get("confidence", 0.0)
            
            # Check confidence threshold
            if confidence < 0.65:
                return
            
            # 6. Send signal to gateway
            self.send_trading_signal(symbol, signal, confidence, mqscore_result)
            
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    def send_trading_signal(self,
                           symbol: str,
                           signal: float,
                           confidence: float,
                           mqscore_result):
        """Send trading signal to gateway"""
        
        try:
            # Determine side
            if signal > 0.3:
                side = "BUY"
                signal_value = 1.0
            elif signal < -0.3:
                side = "SELL"
                signal_value = -1.0
            else:
                return  # Neutral, skip
            
            # Calculate quantity (using existing risk management)
            quantity = self.calculate_position_size(symbol, confidence, mqscore_result)
            
            if quantity == 0:
                return
            
            # Send to gateway
            success = self.gateway.send_signal(
                symbol=symbol,
                signal=signal_value,
                confidence=confidence,
                quantity=quantity,
                metadata={
                    "mqscore": mqscore_result.composite_score,
                    "regime": mqscore_result.regime if hasattr(mqscore_result, 'regime') else "unknown",
                    "strategy_count": len(self.nexus_pipeline.strategies)
                }
            )
            
            if success:
                self.stats["signals_generated"] += 1
                logger.info(f"Signal sent: {symbol} {side} qty={quantity} conf={confidence:.2f}")
                
                # Mark as active (prevent duplicates)
                self.active_orders[symbol] = f"PENDING_{symbol}_{int(time.time())}"
            
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
    
    def handle_order_fill(self, fill_data: Dict[str, Any]):
        """
        Handle order fill from gateway
        
        This is the feedback loop!
        """
        try:
            symbol = fill_data.get("symbol")
            order_id = fill_data.get("order_id")
            fill_price = fill_data.get("fill_price")
            filled_qty = fill_data.get("filled_qty")
            status = fill_data.get("status")
            
            self.stats["fills_received"] += 1
            
            # Update active orders
            if order_id in self.active_positions:
                position = self.active_positions[order_id]
                position["fill_price"] = fill_price
                position["filled_qty"] = filled_qty
                position["status"] = status
                
                logger.info(f"Fill: {symbol} @ {fill_price} qty={filled_qty} status={status}")
                
                # Record for strategy feedback
                self.nexus_pipeline.record_trade_outcome(
                    symbol=symbol,
                    entry_price=fill_price,
                    quantity=filled_qty
                )
            
            # If fully filled, register for monitoring
            if status == "FILLED":
                self.active_positions[order_id] = {
                    "symbol": symbol,
                    "entry_price": fill_price,
                    "quantity": filled_qty,
                    "timestamp": time.time()
                }
                self.stats["orders_executed"] += 1
        
        except Exception as e:
            logger.error(f"Error handling fill: {e}")
    
    def calculate_position_size(self, symbol, confidence, mqscore_result) -> int:
        """Calculate position size using NEXUS risk management"""
        # Use existing NEXUS risk management
        # (Integrate with your RiskManager)
        
        # Simple example:
        if confidence > 0.80 and mqscore_result.composite_score > 0.7:
            return 2  # High confidence = 2 contracts
        elif confidence > 0.70:
            return 1  # Normal confidence = 1 contract
        else:
            return 0  # Below threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics"""
        return {
            **self.stats,
            "active_symbols": list(self.active_symbols),
            "active_orders": len(self.active_orders),
            "active_positions": len(self.active_positions),
            "gateway_stats": self.gateway.get_stats()
        }
    
    async def stop(self):
        """Stop adapter"""
        logger.info("Stopping NEXUS Communication Adapter...")
        self.gateway.disconnect()
        logger.info("Stopped ✓")
```

#### Step 2.2: Modify NEXUS AI Main Script

Update your `nexus_ai.py` main entry point:

```python
# Add at the end of nexus_ai.py

async def main_with_gateway():
    """
    NEXUS AI with Communication Gateway Integration
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize existing NEXUS components
    from MQScore_6D_Engine_v3 import MQScoreEngine, MQScoreConfig
    
    # MQScore Engine
    mqscore_config = MQScoreConfig()
    mqscore_engine = MQScoreEngine(config=mqscore_config)
    
    # NEXUS Pipeline (your existing strategies)
    pipeline = ProductionSequentialPipeline(
        strategies=[],  # Will be populated from strategy files
        ml_ensemble=None  # Your ML ensemble
    )
    
    # Load strategies
    from strategy_loader import load_all_strategies
    strategies = load_all_strategies()  # Loads all 20 strategies
    for strategy in strategies:
        pipeline.strategies.append(strategy)
    
    # Communication Adapter
    from nexus_communication_adapter import NexusCommunicationAdapter
    
    adapter = NexusCommunicationAdapter(
        nexus_pipeline=pipeline,
        mqscore_engine=mqscore_engine,
        auto_execute=True  # Set False for paper trading
    )
    
    # Start
    await adapter.start()
    
    # Run forever
    print("NEXUS AI + Gateway running... Press Ctrl+C to stop")
    try:
        while True:
            await asyncio.sleep(1)
            
            # Print stats every 10 seconds
            if int(time.time()) % 10 == 0:
                stats = adapter.get_stats()
                print(f"Stats: Data={stats['market_data_received']} "
                      f"Signals={stats['signals_generated']} "
                      f"Orders={stats['orders_executed']}")
    
    except KeyboardInterrupt:
        await adapter.stop()


if __name__ == "__main__":
    # Choose mode:
    # asyncio.run(main())  # Standalone mode
    asyncio.run(main_with_gateway())  # Gateway mode
```

---

### PHASE 3: PLATFORM ADAPTERS (Week 2-3)

Create adapters for each trading platform.

#### Step 3.1: NT8 Adapter (NinjaTrader 8)

Create: `adapters/nt8_adapter.py`

```python
from communication.platform_adapter_template import PlatformAdapter
import socket
import json

class NT8Adapter(PlatformAdapter):
    def __init__(self):
        super().__init__(platform_name="nt8")
        self.nt8_socket = None
        
    def connect_to_nt8(self):
        """Connect to NT8 C# bridge"""
        self.nt8_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.nt8_socket.connect(("localhost", 9001))  # NT8 listens here
        
    def execute_order(self, order_msg):
        """Send order to NT8"""
        # Convert to NT8 format
        nt8_order = {
            "action": "SubmitOrder",
            "instrument": order_msg["symbol"],
            "quantity": order_msg["quantity"],
            "orderType": "Market",
            "side": order_msg["side"]
        }
        
        # Send to NT8
        self.nt8_socket.send(json.dumps(nt8_order).encode())
```

#### Step 3.2: MT5 Adapter (MetaTrader 5)

Create: `adapters/mt5_adapter.py`

```python
from communication.platform_adapter_template import PlatformAdapter
import MetaTrader5 as mt5

class MT5Adapter(PlatformAdapter):
    def __init__(self, account, password, server):
        super().__init__(platform_name="mt5_global")
        
        # Initialize MT5
        mt5.initialize()
        mt5.login(account, password, server)
        
    def execute_order(self, order_msg):
        """Execute order on MT5"""
        symbol = order_msg["symbol"]
        volume = order_msg["quantity"]
        order_type = mt5.ORDER_TYPE_BUY if order_msg["side"] == "BUY" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "type_filling": mt5.ORDER_FILLING_FOK,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Send fill back to gateway
            self.send_order_fill(
                order_id=order_msg["order_id"],
                filled_qty=volume,
                fill_price=result.price
            )
```

---

### PHASE 4: TESTING (Week 3-4)

#### Test 1: Gateway Only
```bash
python communication/complete_system_test.py
```

#### Test 2: NEXUS + Gateway (Simulated)
```bash
# Terminal 1: Gateway
python communication/gateway_server.py

# Terminal 2: NEXUS with auto_execute=False
python nexus_ai.py --gateway --paper-trade

# Terminal 3: Simulated platform
python communication/platform_adapter_template.py
```

#### Test 3: Real Platform (Paper Trading)
```bash
# Terminal 1: Gateway
python communication/gateway_server.py

# Terminal 2: Real MT5 adapter (paper account)
python adapters/mt5_adapter.py --paper

# Terminal 3: NEXUS
python nexus_ai.py --gateway --auto-execute
```

---

### PHASE 5: PRODUCTION DEPLOYMENT (Week 4+)

```bash
# Start in this order:

# 1. Gateway (always first)
python communication/gateway_server.py &

# 2. Platform adapters
python adapters/nt8_adapter.py &
python adapters/mt5_adapter.py &

# 3. NEXUS AI
python nexus_ai.py --gateway --auto-execute --live
```

---

## INTEGRATION BENEFITS SUMMARY

| Feature | Before Gateway | With Gateway |
|---------|----------------|--------------|
| **Platforms** | 1 (hardcoded) | Multiple (dynamic) |
| **Symbol Format** | Platform-specific | Universal |
| **Currency** | Manual conversion | Auto BRL→USD |
| **Rollover** | Manual tracking | Automatic |
| **Latency** | Variable | 5-15μs guaranteed |
| **Scalability** | Single platform | N platforms |
| **Testing** | Platform required | Simulated mode |

---

## RECOMMENDED NEXT STEPS

1. ✅ **Week 1**: Review communication folder, understand architecture
2. ⬜ **Week 1**: Run `complete_system_test.py` to see it working
3. ⬜ **Week 2**: Create `nexus_communication_adapter.py`
4. ⬜ **Week 2**: Test with simulated adapters
5. ⬜ **Week 3**: Create real platform adapters (MT5 first - easiest)
6. ⬜ **Week 3**: Paper trade with one platform
7. ⬜ **Week 4**: Add more platforms
8. ⬜ **Week 4+**: Go live (start small!)

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-20  
**Status**: Integration Plan Ready for Implementation
