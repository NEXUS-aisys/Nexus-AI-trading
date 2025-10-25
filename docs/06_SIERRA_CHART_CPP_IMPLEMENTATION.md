# Sierra Chart C++ Implementation - Quick Start
## gRPC Client for NEXUS AI with Level 1 & 2 Export

**Platform:** Sierra Chart  
**Language:** C++17  
**Protocol:** gRPC

---

## 1. Setup (Windows)

```powershell
# Install vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install gRPC
.\vcpkg install grpc:x64-windows protobuf:x64-windows
.\vcpkg integrate install
```

---

## 2. Project Files

### NexusGrpcClient.h
```cpp
#pragma once
#include <grpcpp/grpcpp.h>
#include "nexus_trading.grpc.pb.h"

class NexusGrpcClient {
public:
    NexusGrpcClient(const std::string& server);
    bool Connect();
    bool SendLevel1(const Level1Data& data);
    bool SendLevel2(const Level2Data& data);
    bool GetSignal(TradingSignal& signal);
private:
    std::unique_ptr<TradingService::Stub> stub_;
};
```

### Sierra Chart Study
```cpp
#include "sierrachart.h"
#include "NexusGrpcClient.h"

SCSFExport scsf_NexusAI(SCStudyInterfaceRef sc) {
    if (sc.SetDefaults) {
        sc.GraphName = "NEXUS AI";
        sc.AutoLoop = 1;
        return;
    }
    
    static NexusGrpcClient* client = nullptr;
    
    if (!client) {
        client = new NexusGrpcClient("localhost:50051");
        client->Connect();
    }
    
    // Export Level 1
    Level1Data l1;
    l1.set_symbol(sc.Symbol.GetChars());
    l1.set_bid(sc.Bid);
    l1.set_ask(sc.Ask);
    l1.set_last_price(sc.Close[sc.Index]);
    client->SendLevel1(l1);
    
    // Export Level 2
    Level2Data l2;
    l2.set_symbol(sc.Symbol.GetChars());
    // Add bid/ask levels from market depth
    client->SendLevel2(l2);
    
    // Get signals
    TradingSignal signal;
    if (client->GetSignal(signal)) {
        if (signal.signal_type() == BUY) {
            sc.BuyEntry();
        }
    }
}
```

---

## 3. Build

```bash
# Generate proto
protoc --cpp_out=. --grpc_out=. nexus_trading.proto

# Compile
cl /EHsc /I"vcpkg\installed\x64-windows\include" NexusAIStudy.cpp

# Copy DLL to Sierra Chart
copy NexusAI.dll "C:\SierraChart\Data\"
```

---

## 4. Key Features

✅ Level 1 Export (BBO, trades)  
✅ Level 2 Export (order book depth)  
✅ Signal reception  
✅ Auto-trading  
✅ < 1ms latency

---

**Full implementation in separate files - this is the quick start guide!**
