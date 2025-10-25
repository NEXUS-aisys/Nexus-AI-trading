# NEXUS AI - NinjaTrader 8 Quick Start

## 🚀 **5-Minute Setup**

### 1. Generate C# Files
```powershell
cd "C:\Users\Nexus AI\Documents\NEXUS\ninjatrader"
.\generate_csharp_proto.bat
```

### 2. Install gRPC in NinjaTrader
1. Open NinjaTrader 8
2. Tools → References → Add
3. Click "Manage NuGet Packages"
4. Install:
   - `Grpc.Core`
   - `Google.Protobuf`

### 3. Compile
1. Tools → Compile
2. Fix errors if any
3. Wait for "Compilation successful"

### 4. Start Server
```powershell
cd "C:\Users\Nexus AI\Documents\NEXUS"
python nexus_grpc_server_live.py
```

### 5. Add to Chart
1. Right-click chart → Indicators
2. Find "NEXUS AI"
3. Click Add
4. Configure:
   - Server: `localhost:50051`
   - Auto Trade: `false` (for testing)
5. Click OK

### 6. Verify
Check Output Window (Tools → Output):
```
NEXUS AI: Connected successfully!
NEXUS AI: Streaming started
NEXUS AI: Sent 100 Level 1 updates
```

---

## ✅ **Ready for Monday!**

Both platforms integrated:
- ✅ Sierra Chart (C++)
- ✅ NinjaTrader 8 (C#)

Both connect to same NEXUS AI server with 20 strategies!

---

## 📊 **What Happens Monday**

1. **Market opens**
2. **Data flows**: NT8/Sierra → NEXUS AI
3. **20 strategies analyze** market in real-time
4. **Signals generated** when conditions met
5. **You see signals** in Output Window
6. **Enable auto-trade** when confident

---

## ⚙️ **Key Settings**

| Setting | Recommended | Why |
|---------|-------------|-----|
| Auto Trade | `false` | Test first! |
| Min Confidence | `0.7` | 70%+ only |
| Position Size | `1` | Start small |
| Level 1 Interval | `100ms` | Good balance |
| Level 2 Interval | `500ms` | Not too fast |

---

## 🎯 **Monday Test Plan**

**9:30 AM - Market Open**
1. Start NEXUS AI server
2. Load indicator on chart
3. Watch for connection
4. Monitor data flow

**9:35 AM - 5 minutes in**
1. Check Level 1/2 counts
2. Look for first signals
3. Verify signal quality

**10:00 AM - 30 minutes in**
1. Review signal history
2. Check confidence levels
3. Decide on auto-trade

**If signals look good:**
1. Enable auto-trade
2. Start with 1 contract
3. Monitor closely

---

## 🔥 **Pro Tips**

1. **Test both platforms** - Sierra Chart AND NinjaTrader
2. **Compare signals** - Should be identical
3. **Start manual** - Watch signals before auto-trading
4. **Monitor server** - Check Python logs for errors
5. **Use stops** - Always set stop losses

---

**You're ready! See you Monday! 🚀📈**
