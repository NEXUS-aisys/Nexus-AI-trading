# NEXUS AI Server Integration Plan
## Sierra Chart (C++) & NinjaTrader 8 (C#) via gRPC

**Version:** 1.0.0  
**Date:** October 24, 2025  
**Communication:** gRPC Protocol

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│         TRADING PLATFORMS (Clients)                      │
├──────────────────────────────────────────────────────────┤
│  Sierra Chart (C++)  │  NinjaTrader 8 (C#)              │
│  - Level 1 Data      │  - Level 1 Data (BBO)            │
│  - Level 2 Data      │  - Level 2 Data (Depth)          │
│  - Order Entry       │  - Order Entry                    │
└──────────┬───────────┴──────────┬────────────────────────┘
           │ gRPC                 │ gRPC
           ▼                      ▼
┌──────────────────────────────────────────────────────────┐
│              NEXUS AI gRPC SERVER (Python)               │
│  - MarketDataService  - SignalService                    │
│  - OrderService       - PositionService                  │
│  - RiskService        - HealthService                    │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              NEXUS AI CORE (Python)                      │
│  - 8-Layer Pipeline  - 20 Strategies  - 33 ML Models    │
└──────────────────────────────────────────────────────────┘
```

---

## 2. gRPC Protocol Definitions

**File:** `nexus_trading.proto`

```protobuf
syntax = "proto3";
package nexus.trading;

// ============================================================================
// LEVEL 1 MARKET DATA (Best Bid/Offer)
// ============================================================================
message Level1Data {
    string symbol = 1;
    double timestamp = 2;
    
    // Last Trade
    double last_price = 3;
    double last_size = 4;
    
    // Best Bid/Offer (BBO)
    double bid = 5;
    double ask = 6;
    int32 bid_size = 7;
    int32 ask_size = 8;
    
    // Daily Stats
    double open = 9;
    double high = 10;
    double low = 11;
    double close = 12;
    double volume = 13;
    
    // Additional Level 1 Fields
    double vwap = 14;           // Volume Weighted Average Price
    int64 trade_count = 15;     // Number of trades
    double open_interest = 16;  // For futures
}

// ============================================================================
// LEVEL 2 MARKET DATA (Market Depth / Order Book)
// ============================================================================
message Level2Data {
    string symbol = 1;
    double timestamp = 2;
    
    // Order Book Depth
    repeated PriceLevel bids = 3;  // Bid side (buy orders)
    repeated PriceLevel asks = 4;  // Ask side (sell orders)
    
    // Book Statistics
    int32 bid_depth = 5;           // Number of bid levels
    int32 ask_depth = 6;           // Number of ask levels
    double total_bid_volume = 7;   // Total volume on bid side
    double total_ask_volume = 8;   // Total volume on ask side
    
    // Imbalance Metrics
    double order_imbalance = 9;    // (bid_vol - ask_vol) / (bid_vol + ask_vol)
    double spread_bps = 10;        // Spread in basis points
}

message PriceLevel {
    double price = 1;
    double size = 2;
    int32 num_orders = 3;          // Number of orders at this level
    string exchange = 4;           // Exchange/ECN identifier
}

// Combined Market Data Message
message MarketData {
    string symbol = 1;
    double timestamp = 2;
    
    // Level 1 Data (always included)
    Level1Data level1 = 3;
    
    // Level 2 Data (optional, for strategies that need depth)
    Level2Data level2 = 4;
    
    // Data type indicator
    enum DataType {
        LEVEL1_ONLY = 0;
        LEVEL1_AND_LEVEL2 = 1;
    }
    DataType data_type = 5;
}

// Trading Signal
message TradingSignal {
    enum SignalType {
        BUY = 0;
        SELL = 1;
        NEUTRAL = 2;
    }
    SignalType signal_type = 1;
    double confidence = 2;
    string symbol = 3;
    double position_size = 4;
    double stop_loss = 5;
    double take_profit = 6;
}

// Order
message Order {
    string order_id = 1;
    string symbol = 2;
    string side = 3;  // BUY/SELL
    double quantity = 4;
    double price = 5;
    string status = 6;
}

// Services
service TradingService {
    // Market Data Streaming
    rpc StreamMarketData(stream MarketData) returns (stream TradingSignal);
    rpc StreamLevel1(stream Level1Data) returns (stream TradingSignal);
    rpc StreamLevel2(stream Level2Data) returns (stream TradingSignal);
    
    // Order Management
    rpc SubmitOrder(Order) returns (Order);
    rpc GetPositions(Empty) returns (PositionList);
}
```

**Generate Code:**
```bash
# Python
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. nexus_trading.proto

# C++ (Sierra Chart)
protoc --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=grpc_cpp_plugin nexus_trading.proto

# C# (NinjaTrader 8)
protoc --csharp_out=. --grpc_out=. --plugin=protoc-gen-grpc=Grpc.Tools nexus_trading.proto
```

---

## 3. Python gRPC Server

**File:** `nexus_grpc_server.py`

```python
import grpc
from concurrent import futures
import nexus_trading_pb2_grpc
from nexus_ai import NexusAI

class TradingServicer(nexus_trading_pb2_grpc.TradingServiceServicer):
    def __init__(self):
        self.nexus = NexusAI()
        self.nexus.register_strategies_explicit()
    
    def StreamMarketData(self, request_iterator, context):
        """Receive market data, return signals"""
        for market_data in request_iterator:
            # Process through NEXUS AI
            signal = self.process_data(market_data)
            if signal:
                yield signal
    
    def SubmitOrder(self, request, context):
        """Handle order submission"""
        # Update positions
        return request

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nexus_trading_pb2_grpc.add_TradingServiceServicer_to_server(
        TradingServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

---

## 4. Sierra Chart C++ Client

**File:** `NexusAIClient.cpp`

```cpp
#include <grpcpp/grpcpp.h>
#include "nexus_trading.grpc.pb.h"
#include "sierrachart.h"

class NexusAIClient {
private:
    std::unique_ptr<TradingService::Stub> stub_;
    
public:
    NexusAIClient(std::shared_ptr<grpc::Channel> channel)
        : stub_(TradingService::NewStub(channel)) {}
    
    void StreamMarketData(SCStudyInterfaceRef sc) {
        grpc::ClientContext context;
        
        auto stream = stub_->StreamMarketData(&context);
        
        // Send market data
        MarketData md;
        md.set_symbol(sc.Symbol.GetChars());
        md.set_price(sc.Close[sc.Index]);
        md.set_volume(sc.Volume[sc.Index]);
        md.set_bid(sc.Bid);
        md.set_ask(sc.Ask);
        
        stream->Write(md);
        
        // Receive signals
        TradingSignal signal;
        while (stream->Read(&signal)) {
            HandleSignal(sc, signal);
        }
    }
    
    void HandleSignal(SCStudyInterfaceRef sc, const TradingSignal& signal) {
        if (signal.signal_type() == TradingSignal::BUY) {
            // Submit buy order
            s_SCNewOrder order;
            order.OrderQuantity = signal.position_size();
            order.Price1 = signal.stop_loss();
            order.Price2 = signal.take_profit();
            sc.BuyEntry(order);
        }
    }
};

// Sierra Chart Study Function
SCSFExport scsf_NexusAI(SCStudyInterfaceRef sc) {
    if (sc.SetDefaults) {
        sc.GraphName = "NEXUS AI";
        sc.StudyDescription = "NEXUS AI Trading System";
        sc.AutoLoop = 1;
        return;
    }
    
    static NexusAIClient* client = nullptr;
    
    if (sc.Index == 0) {
        auto channel = grpc::CreateChannel(
            "localhost:50051",
            grpc::InsecureChannelCredentials()
        );
        client = new NexusAIClient(channel);
    }
    
    client->StreamMarketData(sc);
}
```

---

## 5. NinjaTrader 8 C# Client

**File:** `NexusAIIndicator.cs`

```csharp
using System;
using Grpc.Core;
using NinjaTrader.NinjaScript;
using Nexus.Trading;

namespace NinjaTrader.NinjaScript.Indicators
{
    public class NexusAI : Indicator
    {
        private Channel channel;
        private TradingService.TradingServiceClient client;
        
        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "NEXUS AI";
                Description = "NEXUS AI Trading System";
            }
            else if (State == State.Configure)
            {
                channel = new Channel("localhost:50051", 
                    ChannelCredentials.Insecure);
                client = new TradingService.TradingServiceClient(channel);
            }
        }
        
        protected override void OnBarUpdate()
        {
            var marketData = new MarketData
            {
                Symbol = Instrument.FullName,
                Timestamp = Time[0].ToUnixTimeSeconds(),
                Price = Close[0],
                Volume = Volume[0],
                Bid = GetCurrentBid(),
                Ask = GetCurrentAsk()
            };
            
            // Stream to server
            var call = client.StreamMarketData();
            call.RequestStream.WriteAsync(marketData);
            
            // Read signals
            while (await call.ResponseStream.MoveNext())
            {
                var signal = call.ResponseStream.Current;
                HandleSignal(signal);
            }
        }
        
        private void HandleSignal(TradingSignal signal)
        {
            if (signal.SignalType == TradingSignal.Types.SignalType.Buy)
            {
                EnterLong(Convert.ToInt32(signal.PositionSize));
                SetStopLoss(CalculationMode.Price, signal.StopLoss);
                SetProfitTarget(CalculationMode.Price, signal.TakeProfit);
            }
        }
    }
}
```

---

## 6. Market Data Export Requirements

### 6.1 Level 1 Data Export (Required)

**From Sierra Chart:**
- Best Bid/Offer (BBO)
- Last trade price & size
- Daily OHLCV
- VWAP
- Trade count
- Open interest (futures)

**From NinjaTrader 8:**
- Best Bid/Offer
- Last trade
- Session statistics
- Volume profile
- Market depth summary

### 6.2 Level 2 Data Export (Required)

**Order Book Depth:**
- Top 10 bid levels minimum
- Top 10 ask levels minimum
- Price, size, order count per level
- Exchange/ECN identifier

**Calculated Metrics:**
- Order book imbalance
- Bid/Ask volume totals
- Spread in basis points
- Liquidity metrics

### 6.3 Data Update Frequency

| Data Type | Update Rate | Latency Target |
|-----------|-------------|----------------|
| Level 1 (Trades) | Every tick | < 1ms |
| Level 1 (BBO) | Every change | < 1ms |
| Level 2 (Full depth) | Every change | < 5ms |
| Level 2 (Top 5) | Every change | < 2ms |

---

## 7. Deployment Plan

### Phase 1: Development (Week 1-2)
- [ ] Create proto definitions with Level 1 & 2 support
- [ ] Generate code for all platforms
- [ ] Implement Python server with market depth processing
- [ ] Basic testing with simulated Level 2 data

### Phase 2: Sierra Chart Integration (Week 3-4)
- [ ] Implement C++ client with Level 1 export
- [ ] Implement Level 2 order book export
- [ ] Test market data streaming (both levels)
- [ ] Test order execution
- [ ] Performance optimization

### Phase 3: NinjaTrader 8 Integration (Week 5-6)
- [ ] Implement C# indicator with Level 1 export
- [ ] Implement Level 2 market depth export
- [ ] Test market data streaming (both levels)
- [ ] Test order execution
- [ ] Performance optimization

### Phase 4: Production (Week 7-8)
- [ ] Load testing with full Level 2 data
- [ ] Security hardening
- [ ] Monitoring setup
- [ ] Documentation
- [ ] Go-live

---

## 8. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Level 1 Latency | < 1ms | Tick-to-server |
| Level 2 Latency | < 5ms | Full depth update |
| Signal Generation | < 50ms | Server processing |
| Round-trip Latency | < 10ms | Total end-to-end |
| Level 1 Throughput | 10,000 ticks/sec | Per symbol |
| Level 2 Throughput | 1,000 updates/sec | Per symbol |
| Connection uptime | 99.9% | |
| Data loss | 0% | With buffering |

---

## 9. Security

- TLS encryption for production
- API key authentication
- Rate limiting
- Input validation
- Audit logging

---

**Next Steps:**
1. Review proto definitions (Level 1 & 2 support)
2. Set up development environment
3. Implement Python server with order book processing
4. Test Level 2 data export from platforms
5. Begin platform integrations

---

## 10. Level 2 Data Usage in NEXUS AI

### Strategies Using Level 2 Data:

1. **Order Book Imbalance** - Detects buy/sell pressure
2. **Liquidity Absorption** - Identifies large order absorption
3. **Spoofing Detection** - Detects fake orders
4. **Iceberg Detection** - Finds hidden orders
5. **Market Microstructure** - Analyzes order flow

### Level 2 Processing Pipeline:

```
Level 2 Data → Order Book Reconstruction → Imbalance Calculation
                                         ↓
                                    Liquidity Metrics
                                         ↓
                                    Strategy Analysis
                                         ↓
                                    Signal Generation
```

### Benefits of Level 2 Integration:

- **Better Entry/Exit Timing** - See real supply/demand
- **Reduced Slippage** - Identify liquidity pockets
- **Early Warning** - Detect large orders before execution
- **Market Manipulation Detection** - Identify spoofing/layering
- **Improved Risk Management** - Better position sizing based on liquidity
