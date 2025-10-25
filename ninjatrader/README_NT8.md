# NEXUS AI - NinjaTrader 8 Integration

Complete integration between NinjaTrader 8 and NEXUS AI trading system via gRPC.

---

## 📋 **Prerequisites**

1. **NinjaTrader 8** (latest version)
2. **NEXUS AI gRPC Server** running
3. **gRPC NuGet packages** for NinjaTrader

---

## 🔧 **Installation**

### Step 1: Install gRPC NuGet Packages

1. Open **NinjaTrader 8**
2. Go to **Tools → References → Add**
3. Click **"Manage NuGet Packages"**
4. Install these packages:
   ```
   Grpc.Core (v2.46.6 or later)
   Google.Protobuf (v3.21.12 or later)
   ```

### Step 2: Generate Protobuf C# Files

Open PowerShell in the NEXUS directory:

```powershell
cd "C:\Users\Nexus AI\Documents\NEXUS\proto"

# Install grpc tools if not already installed
dotnet tool install --global Grpc.Tools

# Generate C# files
protoc --csharp_out=. --grpc_out=. --plugin=protoc-gen-grpc=grpc_csharp_plugin nexus_trading.proto
```

This creates:
- `NexusTrading.cs`
- `NexusTradingGrpc.cs`

### Step 3: Copy Files to NinjaTrader

Copy these files to NinjaTrader's custom folder:

```powershell
# Indicator
copy "NexusAI_NT8.cs" "$env:USERPROFILE\Documents\NinjaTrader 8\bin\Custom\Indicators\"

# Protobuf files
copy "NexusTrading.cs" "$env:USERPROFILE\Documents\NinjaTrader 8\bin\Custom\Indicators\"
copy "NexusTradingGrpc.cs" "$env:USERPROFILE\Documents\NinjaTrader 8\bin\Custom\Indicators\"
```

### Step 4: Compile in NinjaTrader

1. Open **NinjaTrader 8**
2. Go to **Tools → Compile**
3. Fix any errors (usually missing references)
4. Click **Compile**
5. Should see: **"Compilation successful"**

---

## 🚀 **Usage**

### 1. Start NEXUS AI Server

```powershell
cd "C:\Users\Nexus AI\Documents\NEXUS"
python nexus_grpc_server_live.py
```

Wait for:
```
✅ Strategies: 20 loaded
✅ ML Models: 34 loaded
✅ Server started and listening...
```

### 2. Add Indicator to Chart

1. Open a chart in NinjaTrader
2. Right-click chart → **Indicators**
3. Find **"NEXUS AI"** in the list
4. Click **Add**

### 3. Configure Settings

**Connection:**
- Server Address: `localhost`
- Server Port: `50051`

**Data Export:**
- ✅ Enable Level 1
- ✅ Enable Level 2
- Level 1 Interval: `100` ms
- Level 2 Interval: `500` ms

**Trading:**
- Auto Trade: `false` (for testing)
- Position Size: `1`
- Min Confidence: `0.7` (70%)

### 4. Monitor Output

Check **Output Window** (Tools → Output Window):

```
NEXUS AI: Connected successfully!
NEXUS AI: Streaming started
NEXUS AI: Sent 100 Level 1 updates
NEXUS AI: Sent 20 Level 2 updates
NEXUS AI SIGNAL #1:
  Type: BUY
  Confidence: 85.50%
  Symbol: ES 12-24
  Position Size: 1
```

---

## 📊 **Features**

### Data Export
- ✅ **Level 1**: Bid, Ask, Last, Volume, OHLC
- ✅ **Level 2**: 10 levels of market depth
- ✅ **Real-time streaming** to NEXUS AI
- ✅ **Rate limiting** to prevent overload

### Signal Reception
- ✅ **Bidirectional streaming** from NEXUS AI
- ✅ **20 strategies** generating signals
- ✅ **Confidence filtering** (only high-confidence trades)
- ✅ **Symbol matching** (only trades for current instrument)

### Auto-Trading
- ✅ **Market orders** based on signals
- ✅ **Stop loss** from signal
- ✅ **Take profit** from signal
- ✅ **Position sizing** from signal or default

---

## ⚙️ **Configuration Options**

### Connection
| Parameter | Default | Description |
|-----------|---------|-------------|
| Server Address | `localhost` | gRPC server IP/hostname |
| Server Port | `50051` | gRPC server port |

### Data Export
| Parameter | Default | Description |
|-----------|---------|-------------|
| Enable Level 1 | `true` | Send tick data |
| Enable Level 2 | `true` | Send market depth |
| Level 1 Interval | `100` ms | Min time between L1 updates |
| Level 2 Interval | `500` ms | Min time between L2 updates |

### Trading
| Parameter | Default | Description |
|-----------|---------|-------------|
| Auto Trade | `false` | Execute signals automatically |
| Position Size | `1` | Default contracts/shares |
| Min Confidence | `0.7` | Minimum signal confidence (0-1) |

---

## 🔍 **Troubleshooting**

### "Could not load file or assembly 'Grpc.Core'"

**Solution:** Install gRPC NuGet packages (see Step 1)

### "Compilation failed"

**Solution:** 
1. Check all 3 files are in `Documents\NinjaTrader 8\bin\Custom\Indicators\`
2. Verify gRPC packages installed
3. Check Output Window for specific errors

### "Connection failed"

**Solution:**
1. Verify NEXUS AI server is running
2. Check firewall allows port 50051
3. Try `127.0.0.1` instead of `localhost`

### "No signals received"

**Solution:**
1. Check market is open
2. Verify data is flowing (check Level 1/2 counts)
3. Lower `Min Confidence` threshold
4. Check Python server logs for errors

---

## 📈 **Testing (Monday)**

### Test Checklist

1. ✅ **Connection**
   - [ ] Indicator loads without errors
   - [ ] Connects to server
   - [ ] "Connected successfully!" in output

2. ✅ **Data Flow**
   - [ ] Level 1 count increasing
   - [ ] Level 2 count increasing
   - [ ] Python server shows "CLIENT CONNECTED"

3. ✅ **Signal Reception**
   - [ ] Signals appear in output window
   - [ ] Signal details correct (type, confidence, symbol)
   - [ ] Only high-confidence signals shown

4. ✅ **Auto-Trading** (if enabled)
   - [ ] Orders placed automatically
   - [ ] Stop loss set
   - [ ] Take profit set
   - [ ] Position size correct

---

## 🎯 **Next Steps**

1. **Test with paper trading** first
2. **Monitor performance** for a few days
3. **Adjust confidence threshold** based on results
4. **Enable auto-trading** when confident

---

## 📞 **Support**

Check logs:
- **NinjaTrader:** Tools → Output Window
- **Python Server:** Terminal running `nexus_grpc_server_live.py`

---

## ⚠️ **Risk Warning**

**Auto-trading involves significant risk. Test thoroughly with paper trading before using real money.**

- Start with `Auto Trade = false`
- Monitor signals manually
- Verify signal quality
- Use appropriate position sizing
- Set stop losses

---

**NEXUS AI + NinjaTrader 8 = Automated Trading Excellence** 🚀
