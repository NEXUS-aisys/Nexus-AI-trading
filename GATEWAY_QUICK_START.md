# NEXUS Gateway - Quick Start Guide
## Get Started in 30 Minutes

---

## 🎯 What This Does

Connects your NEXUS AI to multiple trading platforms simultaneously:

```
NEXUS AI (existing code)
    ↓
Gateway (new - routes orders)
    ↓
Multiple Platforms: NT8 | MT5 | Sierra | Bookmap | Quantower
```

**One signal → Execute on all platforms**

---

## ⚡ Quick Setup (Development Mode)

### Step 1: Install Dependencies (5 min)
```bash
cd "C:\Users\Nexus AI\Documents\NEXUS"
pip install pyzmq msgpack-python MetaTrader5 pytest
```

### Step 2: Create Gateway Structure (1 min)
```bash
mkdir Communication
mkdir Communication\adapters
mkdir Communication\config
mkdir Communication\tests
mkdir Communication\utils
```

### Step 3: Priority Files to Create

**Week 1-2: Core Gateway**
```
Communication\
  ├─ gateway_server.py        ← START HERE (most important)
  ├─ nexus_gateway_interface.py
  └─ platform_adapter_template.py
```

**Week 3-4: Integration**
```
nexus_communication_adapter.py  ← Bridge between NEXUS & Gateway
```

**Week 5-6: First Platform**
```
Communication\adapters\
  └─ mt5_adapter.py            ← Easiest to implement
```

---

## 🔥 Priority 1: Gateway Server

Create `Communication/gateway_server.py` with these core components:

```python
#!/usr/bin/env python3
"""
NEXUS Gateway Server
Central routing hub for multi-platform trading
"""

import zmq
import msgpack
import logging
import threading
from datetime import datetime

class GatewayServer:
    """Central communication hub"""
    
    def __init__(self):
        self.context = zmq.Context()
        
        # Port 5555: Platform adapters connect here
        self.platform_socket = self.context.socket(zmq.ROUTER)
        self.platform_socket.bind("tcp://*:5555")
        
        # Port 5556: NEXUS AI connects here
        self.nexus_socket = self.context.socket(zmq.DEALER)
        self.nexus_socket.bind("tcp://*:5556")
        
        # Symbol mapping
        self.symbol_map = {
            "NQ": {
                "nt8": "NQ 03-25",
                "sierra": "NQH25",
                "mt5": "NAS100"
            },
            "ES": {
                "nt8": "ES 03-25",
                "sierra": "ESH25",
                "mt5": "US500"
            }
        }
        
        self.connected_platforms = {}
        self.running = False
        
        logging.info("Gateway Server initialized")
    
    def start(self):
        """Start gateway server"""
        self.running = True
        
        # Thread 1: Handle platform messages
        platform_thread = threading.Thread(target=self.handle_platform_messages)
        platform_thread.start()
        
        # Thread 2: Handle NEXUS messages
        nexus_thread = threading.Thread(target=self.handle_nexus_messages)
        nexus_thread.start()
        
        logging.info("Gateway Server started on ports 5555, 5556")
    
    def handle_platform_messages(self):
        """Receive messages from platform adapters"""
        while self.running:
            try:
                # Receive from platform
                identity, message = self.platform_socket.recv_multipart()
                data = msgpack.unpackb(message, raw=False)
                
                msg_type = data.get("type")
                
                if msg_type == "REGISTER":
                    # Platform connecting
                    platform_name = data["platform"]
                    self.connected_platforms[identity] = platform_name
                    logging.info(f"Platform registered: {platform_name}")
                
                elif msg_type == "MARKET_DATA":
                    # Market data from platform → normalize → send to NEXUS
                    normalized = self.normalize_market_data(data)
                    self.nexus_socket.send(msgpack.packb(normalized))
                
                elif msg_type == "ORDER_FILL":
                    # Fill from platform → send to NEXUS
                    self.nexus_socket.send(msgpack.packb(data))
            
            except Exception as e:
                logging.error(f"Error handling platform message: {e}")
    
    def handle_nexus_messages(self):
        """Receive messages from NEXUS AI"""
        while self.running:
            try:
                # Receive signal from NEXUS
                message = self.nexus_socket.recv()
                data = msgpack.unpackb(message, raw=False)
                
                msg_type = data.get("type")
                
                if msg_type == "SIGNAL":
                    # Trading signal from NEXUS
                    self.route_signal_to_platforms(data)
            
            except Exception as e:
                logging.error(f"Error handling NEXUS message: {e}")
    
    def normalize_market_data(self, data):
        """Normalize platform-specific symbol to universal"""
        platform_symbol = data["data"]["symbol"]
        
        # Reverse lookup: "NQ 03-25" → "NQ"
        for universal, mappings in self.symbol_map.items():
            for platform, platform_sym in mappings.items():
                if platform_sym == platform_symbol:
                    data["data"]["symbol"] = universal
                    return data
        
        return data
    
    def route_signal_to_platforms(self, signal_data):
        """Route signal to all connected platforms"""
        symbol = signal_data["symbol"]
        
        # For each connected platform
        for identity, platform_name in self.connected_platforms.items():
            # Translate symbol
            if symbol in self.symbol_map:
                platform_symbol = self.symbol_map[symbol].get(platform_name)
                
                if platform_symbol:
                    # Create platform-specific order
                    order = {
                        "type": "ORDER",
                        "symbol": platform_symbol,
                        "side": signal_data["side"],
                        "quantity": signal_data["quantity"],
                        "order_id": signal_data["order_id"]
                    }
                    
                    # Send to platform
                    self.platform_socket.send_multipart([
                        identity,
                        msgpack.packb(order)
                    ])
                    
                    logging.info(f"Order routed to {platform_name}: {platform_symbol}")
    
    def stop(self):
        """Stop gateway server"""
        self.running = False
        self.platform_socket.close()
        self.nexus_socket.close()
        self.context.term()
        logging.info("Gateway Server stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    gateway = GatewayServer()
    gateway.start()
    
    try:
        # Run forever
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        gateway.stop()
```

**Test it:**
```bash
python Communication/gateway_server.py
# Should see: "Gateway Server started on ports 5555, 5556"
```

---

## 🔥 Priority 2: Simple Test Adapter

Create `Communication/tests/test_adapter.py` to verify gateway works:

```python
#!/usr/bin/env python3
"""Simple test adapter - simulates a platform"""

import zmq
import msgpack
import time
import random

class TestPlatformAdapter:
    """Simulates MT5 or NT8"""
    
    def __init__(self, platform_name="test_platform"):
        self.platform_name = platform_name
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect("tcp://localhost:5555")
    
    def start(self):
        # Register with gateway
        register_msg = {
            "type": "REGISTER",
            "platform": self.platform_name
        }
        self.socket.send(msgpack.packb(register_msg))
        print(f"{self.platform_name} registered with gateway")
        
        # Send simulated market data
        for i in range(10):
            price = 16250 + random.uniform(-10, 10)
            
            market_data = {
                "type": "MARKET_DATA",
                "platform": self.platform_name,
                "data": {
                    "symbol": "NQ 03-25",  # Platform-specific format
                    "price": price,
                    "volume": 5,
                    "timestamp": time.time()
                }
            }
            
            self.socket.send(msgpack.packb(market_data))
            print(f"Sent: NQ @ {price:.2f}")
            
            time.sleep(1)
        
        # Listen for orders from gateway
        while True:
            if self.socket.poll(1000):  # 1 second timeout
                message = self.socket.recv()
                order = msgpack.unpackb(message, raw=False)
                print(f"Received ORDER: {order}")
                
                # Simulate fill
                if order["type"] == "ORDER":
                    fill = {
                        "type": "ORDER_FILL",
                        "order_id": order["order_id"],
                        "symbol": order["symbol"],
                        "filled_qty": order["quantity"],
                        "fill_price": 16250.50,
                        "status": "FILLED"
                    }
                    self.socket.send(msgpack.packb(fill))
                    print(f"Sent FILL for {order['symbol']}")

if __name__ == "__main__":
    adapter = TestPlatformAdapter("test_mt5")
    adapter.start()
```

**Test the full flow:**
```bash
# Terminal 1
python Communication/gateway_server.py

# Terminal 2
python Communication/tests/test_adapter.py
```

You should see market data flowing through the gateway!

---

## 🔥 Priority 3: NEXUS Integration

Create `nexus_communication_adapter.py` in root directory:

```python
#!/usr/bin/env python3
"""
NEXUS Communication Adapter
Connects NEXUS AI to Gateway Server
"""

import zmq
import msgpack
import time
import logging
from typing import Dict, Any, Callable

class NexusCommunicationAdapter:
    """Bridge between NEXUS AI and Gateway"""
    
    def __init__(self, gateway_host="localhost", gateway_port=5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{gateway_host}:{gateway_port}")
        
        self.market_data_callback = None
        self.order_fill_callback = None
        self.running = False
        
        logging.info("NEXUS Communication Adapter initialized")
    
    def set_market_data_callback(self, callback: Callable):
        """Set function to call when market data arrives"""
        self.market_data_callback = callback
    
    def set_order_fill_callback(self, callback: Callable):
        """Set function to call when order fills"""
        self.order_fill_callback = callback
    
    def start(self):
        """Start listening for messages from gateway"""
        self.running = True
        
        while self.running:
            if self.socket.poll(100):  # 100ms timeout
                message = self.socket.recv()
                data = msgpack.unpackb(message, raw=False)
                
                msg_type = data.get("type")
                
                if msg_type == "MARKET_DATA" and self.market_data_callback:
                    # Market data from gateway
                    self.market_data_callback(data["data"])
                
                elif msg_type == "ORDER_FILL" and self.order_fill_callback:
                    # Order fill from gateway
                    self.order_fill_callback(data)
    
    def send_signal(self, symbol: str, side: str, quantity: int, 
                    confidence: float, metadata: Dict = None):
        """Send trading signal to gateway"""
        
        signal = {
            "type": "SIGNAL",
            "symbol": symbol,  # Use universal symbol (e.g. "NQ")
            "side": side,  # "BUY" or "SELL"
            "quantity": quantity,
            "confidence": confidence,
            "order_id": f"NEXUS_{int(time.time()*1000)}",
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.socket.send(msgpack.packb(signal))
        logging.info(f"Signal sent: {symbol} {side} qty={quantity}")
        
        return signal["order_id"]
    
    def stop(self):
        """Stop adapter"""
        self.running = False
        self.socket.close()
        self.context.term()


# ============================================================================
# INTEGRATION WITH EXISTING NEXUS AI
# ============================================================================

def integrate_with_nexus():
    """
    Example: How to integrate with your existing nexus_ai.py
    """
    
    # Create adapter
    adapter = NexusCommunicationAdapter()
    
    # Define what happens when market data arrives
    def handle_market_data(data):
        print(f"Market Data: {data['symbol']} @ {data['price']}")
        
        # TODO: Feed to your existing NEXUS pipeline
        # result = nexus_pipeline.process(data)
        # 
        # if result["signal"] > 0.3:
        #     adapter.send_signal(
        #         symbol=data["symbol"],
        #         side="BUY",
        #         quantity=1,
        #         confidence=result["confidence"]
        #     )
    
    # Define what happens when order fills
    def handle_fill(fill_data):
        print(f"Order Filled: {fill_data}")
        
        # TODO: Feed back to NEXUS for learning
        # nexus_pipeline.record_trade(fill_data)
    
    # Set callbacks
    adapter.set_market_data_callback(handle_market_data)
    adapter.set_order_fill_callback(handle_fill)
    
    # Start
    adapter.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    integrate_with_nexus()
```

**Test with existing NEXUS:**
```bash
# Terminal 1: Gateway
python Communication/gateway_server.py

# Terminal 2: Test adapter (simulates platform)
python Communication/tests/test_adapter.py

# Terminal 3: NEXUS with adapter
python nexus_communication_adapter.py
```

---

## 🎯 Minimal Working System (MVP)

**Goal**: Get signal flowing end-to-end in 1 week

**You need just 3 files:**

1. ✅ `Communication/gateway_server.py` (core routing)
2. ✅ `Communication/tests/test_adapter.py` (simulates platform)
3. ✅ `nexus_communication_adapter.py` (connects NEXUS)

**Flow:**
```
Test Adapter → Market Data → Gateway → NEXUS Adapter
NEXUS Adapter → Signal → Gateway → Test Adapter → Fill
```

**Success = See this:**
```
[Gateway] Platform registered: test_mt5
[Gateway] Market data received: NQ @ 16250.50
[Gateway] Signal received from NEXUS: BUY NQ
[Gateway] Order routed to test_mt5
[NEXUS] Order filled: NQ @ 16250.50
```

---

## 🚀 Week-by-Week Priorities

### Week 1: Core Infrastructure
- [ ] Create `gateway_server.py` (basic routing)
- [ ] Create `test_adapter.py` (simulated platform)
- [ ] Test gateway can receive and route messages
- [ ] Gateway runs without crashing for 1 hour

### Week 2: NEXUS Integration
- [ ] Create `nexus_communication_adapter.py`
- [ ] Connect to your existing NEXUS AI
- [ ] Market data flows to NEXUS
- [ ] NEXUS generates signals
- [ ] Signals flow back to gateway

### Week 3-4: First Real Platform (MT5)
- [ ] Create `mt5_adapter.py`
- [ ] Connect to real MT5 (paper account)
- [ ] Execute real order on MT5
- [ ] Receive real fill
- [ ] End-to-end latency <100ms

### Week 5-6: Add More Features
- [ ] Symbol normalization (complete mapping)
- [ ] Currency conversion (BRL→USD)
- [ ] Contract rollover logic
- [ ] Duplicate order prevention
- [ ] Better error handling

### Week 7-8: Add More Platforms
- [ ] NT8 adapter
- [ ] Sierra Chart adapter
- [ ] Test multi-platform execution

### Week 9-10: Production Hardening
- [ ] Monitoring & alerting
- [ ] Performance optimization
- [ ] Security audit
- [ ] Documentation

### Week 11-12: Go Live
- [ ] Paper trade for 1 week
- [ ] Small live positions
- [ ] Scale up gradually

---

## 📊 Testing Checklist

After building each component, verify:

### Gateway Server
```bash
# Can start?
python Communication/gateway_server.py
✅ No errors on startup

# Binds to ports?
netstat -an | findstr "5555"
✅ Shows LISTENING

# Logs properly?
tail logs/gateway.log
✅ Shows initialization messages
```

### Test Adapter
```bash
# Can connect to gateway?
python Communication/tests/test_adapter.py
✅ Shows "registered with gateway"

# Sends market data?
✅ Gateway logs show "Market data received"

# Receives orders?
✅ Adapter prints "Received ORDER"
```

### NEXUS Integration
```bash
# Connects to gateway?
python nexus_communication_adapter.py
✅ No connection errors

# Receives market data?
✅ Callback function executes

# Sends signals?
✅ Gateway receives signals

# Receives fills?
✅ Fill callback executes
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Address already in use"
```
Error: Address already in use (port 5555)
```
**Solution:**
```bash
# Windows
netstat -ano | findstr "5555"
taskkill /PID <pid> /F

# Linux
lsof -ti:5555 | xargs kill -9
```

### Issue 2: "Module not found: msgpack"
```
ModuleNotFoundError: No module named 'msgpack'
```
**Solution:**
```bash
pip install msgpack-python
```

### Issue 3: Messages not routing
```
Gateway receives message but NEXUS doesn't get it
```
**Solution:**
- Check NEXUS adapter is connected
- Verify ports match (5556)
- Check socket types (DEALER/ROUTER)
- Add logging to trace message flow

### Issue 4: High latency
```
End-to-end latency >500ms
```
**Solution:**
- Use localhost (not remote connections)
- Remove unnecessary logging
- Use MessagePack (not JSON)
- Check for blocking operations

---

## 🎓 Key Concepts

### ZeroMQ Patterns

**DEALER-ROUTER**: Gateway ↔ Platforms
- Gateway = ROUTER (can talk to many clients)
- Platforms = DEALER (connect to gateway)
- Allows N platforms to connect

**DEALER-DEALER**: Gateway ↔ NEXUS
- Bidirectional communication
- NEXUS sends signals
- Gateway sends market data + fills

### Message Types

```python
# From platform to gateway
REGISTER = {"type": "REGISTER", "platform": "mt5"}
MARKET_DATA = {"type": "MARKET_DATA", "data": {...}}
ORDER_FILL = {"type": "ORDER_FILL", "order_id": "..."}

# From gateway to platform
ORDER = {"type": "ORDER", "symbol": "...", "side": "BUY"}

# From NEXUS to gateway
SIGNAL = {"type": "SIGNAL", "symbol": "NQ", "side": "BUY"}

# From gateway to NEXUS
MARKET_DATA = {"type": "MARKET_DATA", "data": {...}}
ORDER_FILL = {"type": "ORDER_FILL", ...}
```

### Symbol Normalization

```python
# Platform-specific → Universal
"NQ 03-25"  (NT8)     → "NQ"
"NQH25"     (Sierra)  → "NQ"
"NAS100"    (MT5)     → "NQ"

# Universal → Platform-specific
"NQ" → {
    "nt8": "NQ 03-25",
    "sierra": "NQH25",
    "mt5": "NAS100"
}
```

---

## 📞 Need Help?

**Stuck on something?** Check:

1. **Logs**: `logs/gateway.log`, `logs/nexus.log`
2. **Network**: `netstat -an | findstr "5555"`
3. **Processes**: `tasklist | findstr python`
4. **Ports**: Verify nothing else using 5555, 5556

**Common Questions:**

Q: Do I need to modify `nexus_ai.py`?  
A: Minimal changes - just add adapter initialization

Q: Which platform should I start with?  
A: MT5 (easiest - has Python library)

Q: Can I test without real platforms?  
A: Yes! Use `test_adapter.py` to simulate

Q: How do I know it's working?  
A: See market data → signal → order → fill in logs

---

## 🎯 Success Metrics

**Week 1**: Gateway routes 1 message successfully  
**Week 2**: NEXUS receives market data through gateway  
**Week 3**: First real order executed on MT5  
**Week 4**: End-to-end latency <100ms  
**Week 5**: Multi-platform working  

---

## 📚 Next Steps

1. **Read**: `GATEWAY_IMPLEMENTATION_PLAN.md` (full details)
2. **Create**: File structure (folders above)
3. **Build**: `gateway_server.py` (Priority #1)
4. **Test**: With `test_adapter.py`
5. **Integrate**: With your NEXUS AI
6. **Expand**: Add real platforms

**Ready?** Start with `gateway_server.py` - that's the foundation for everything else!

---

**Version**: 1.0  
**Created**: October 22, 2025  
**Time to MVP**: 1-2 weeks  
**Difficulty**: Medium (with templates provided)

