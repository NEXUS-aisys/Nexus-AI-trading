/*
 * NEXUS AI - NinjaTrader 8 Integration
 * Real-time market data export to NEXUS AI gRPC server
 * Receives trading signals and executes orders
 * 
 * Installation:
 * 1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Indicators\
 * 2. Compile in NinjaTrader (Tools -> Compile)
 * 3. Add to chart: Right-click chart -> Indicators -> NEXUS AI
 */

#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
using NinjaTrader.Data;
using NinjaTrader.Core.FloatingPoint;

// gRPC
using Grpc.Core;
using Nexus.Trading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class NexusAI : Indicator
    {
        #region Variables
        private Channel grpcChannel;
        private TradingService.TradingServiceClient grpcClient;
        private AsyncDuplexStreamingCall<MarketData, TradingSignal> streamingCall;
        
        private bool isConnected = false;
        private bool isStreaming = false;
        
        // Statistics
        private int level1Count = 0;
        private int level2Count = 0;
        private int signalsReceived = 0;
        private int ordersPlaced = 0;
        
        // Rate limiting
        private DateTime lastLevel1Send = DateTime.MinValue;
        private DateTime lastLevel2Send = DateTime.MinValue;
        
        // Market depth
        private MarketDepth marketDepth;
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Server Address", Description = "gRPC server address", Order = 1, GroupName = "Connection")]
        public string ServerAddress { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Server Port", Description = "gRPC server port", Order = 2, GroupName = "Connection")]
        public int ServerPort { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Level 1", Description = "Send Level 1 market data", Order = 3, GroupName = "Data Export")]
        public bool EnableLevel1 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Level 2", Description = "Send Level 2 market depth", Order = 4, GroupName = "Data Export")]
        public bool EnableLevel2 { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Level 1 Interval (ms)", Description = "Minimum interval between Level 1 updates", Order = 5, GroupName = "Data Export")]
        public int Level1IntervalMs { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Level 2 Interval (ms)", Description = "Minimum interval between Level 2 updates", Order = 6, GroupName = "Data Export")]
        public int Level2IntervalMs { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Auto Trade", Description = "Automatically execute signals", Order = 7, GroupName = "Trading")]
        public bool AutoTrade { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Position Size", Description = "Default position size", Order = 8, GroupName = "Trading")]
        public int PositionSize { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Min Confidence", Description = "Minimum signal confidence (0-1)", Order = 9, GroupName = "Trading")]
        public double MinConfidence { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = @"NEXUS AI - Real-time integration with NEXUS AI trading system";
                Name = "NEXUS AI";
                Calculate = Calculate.OnEachTick;
                IsOverlay = true;
                DisplayInDataBox = true;
                DrawOnPricePanel = true;
                ScaleJustification = ScaleJustification.Right;

                // Default values
                ServerAddress = "localhost";
                ServerPort = 50051;
                EnableLevel1 = true;
                EnableLevel2 = true;
                Level1IntervalMs = 100;
                Level2IntervalMs = 500;
                AutoTrade = false;
                PositionSize = 1;
                MinConfidence = 0.7;
            }
            else if (State == State.Configure)
            {
                // Subscribe to market depth if Level 2 enabled
                if (EnableLevel2)
                {
                    if (BarsArray[0].Instrument.MarketDepth != null)
                    {
                        BarsArray[0].Instrument.MarketDepth.Update += OnMarketDepth;
                    }
                }
            }
            else if (State == State.DataLoaded)
            {
                // Connect to gRPC server
                Task.Run(() => ConnectToServer());
            }
            else if (State == State.Terminated)
            {
                // Cleanup
                DisconnectFromServer();
                
                if (EnableLevel2 && BarsArray[0].Instrument.MarketDepth != null)
                {
                    BarsArray[0].Instrument.MarketDepth.Update -= OnMarketDepth;
                }
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < 1 || !isConnected || !isStreaming)
                return;

            // Send Level 1 data
            if (EnableLevel1)
            {
                var now = DateTime.Now;
                if ((now - lastLevel1Send).TotalMilliseconds >= Level1IntervalMs)
                {
                    SendLevel1Data();
                    lastLevel1Send = now;
                }
            }
        }

        #region gRPC Connection
        private async Task ConnectToServer()
        {
            try
            {
                Print($"NEXUS AI: Connecting to {ServerAddress}:{ServerPort}...");

                grpcChannel = new Channel($"{ServerAddress}:{ServerPort}", ChannelCredentials.Insecure);
                grpcClient = new TradingService.TradingServiceClient(grpcChannel);

                await grpcChannel.ConnectAsync(DateTime.UtcNow.AddSeconds(5));

                if (grpcChannel.State == ChannelState.Ready)
                {
                    isConnected = true;
                    Print("NEXUS AI: Connected successfully!");
                    
                    // Start bidirectional streaming
                    await StartStreaming();
                }
                else
                {
                    Print($"NEXUS AI: Connection failed - State: {grpcChannel.State}");
                }
            }
            catch (Exception ex)
            {
                Print($"NEXUS AI: Connection error - {ex.Message}");
            }
        }

        private async Task StartStreaming()
        {
            try
            {
                streamingCall = grpcClient.StreamMarketData();
                isStreaming = true;
                
                Print("NEXUS AI: Streaming started");

                // Start receiving signals in background
                _ = Task.Run(() => ReceiveSignals());
            }
            catch (Exception ex)
            {
                Print($"NEXUS AI: Streaming error - {ex.Message}");
                isStreaming = false;
            }
        }

        private void DisconnectFromServer()
        {
            try
            {
                isStreaming = false;
                
                if (streamingCall != null)
                {
                    streamingCall.RequestStream.CompleteAsync().Wait(1000);
                    streamingCall.Dispose();
                }

                if (grpcChannel != null)
                {
                    grpcChannel.ShutdownAsync().Wait(2000);
                }

                Print($"NEXUS AI: Disconnected - L1:{level1Count} L2:{level2Count} Signals:{signalsReceived} Orders:{ordersPlaced}");
            }
            catch (Exception ex)
            {
                Print($"NEXUS AI: Disconnect error - {ex.Message}");
            }
        }
        #endregion

        #region Data Export
        private async void SendLevel1Data()
        {
            if (!isStreaming || streamingCall == null)
                return;

            try
            {
                var marketData = new MarketData
                {
                    Symbol = Instrument.FullName,
                    Timestamp = Time[0].ToUnixTimestamp(),
                    DataType = MarketData.Types.DataType.Level1Only,
                    Level1 = new Level1Data
                    {
                        Symbol = Instrument.FullName,
                        Timestamp = Time[0].ToUnixTimestamp(),
                        LastPrice = Close[0],
                        LastSize = Volume[0],
                        Bid = GetCurrentBid(),
                        Ask = GetCurrentAsk(),
                        BidSize = (int)GetCurrentBidSize(),
                        AskSize = (int)GetCurrentAskSize(),
                        Open = Open[0],
                        High = High[0],
                        Low = Low[0],
                        Close = Close[0],
                        Volume = Volume[0]
                    }
                };

                await streamingCall.RequestStream.WriteAsync(marketData);
                level1Count++;

                if (level1Count % 100 == 0)
                {
                    Print($"NEXUS AI: Sent {level1Count} Level 1 updates");
                }
            }
            catch (Exception ex)
            {
                Print($"NEXUS AI: Level 1 send error - {ex.Message}");
            }
        }

        private async void SendLevel2Data()
        {
            if (!isStreaming || streamingCall == null || !EnableLevel2)
                return;

            var now = DateTime.Now;
            if ((now - lastLevel2Send).TotalMilliseconds < Level2IntervalMs)
                return;

            try
            {
                var depth = BarsArray[0].Instrument.MarketDepth;
                if (depth == null)
                    return;

                var level2Data = new Level2Data
                {
                    Symbol = Instrument.FullName,
                    Timestamp = DateTime.Now.ToUnixTimestamp()
                };

                // Add bid levels
                double totalBidVol = 0;
                for (int i = 0; i < Math.Min(10, depth.Bids.Count); i++)
                {
                    var bid = depth.Bids[i];
                    level2Data.Bids.Add(new PriceLevel
                    {
                        Price = bid.Price,
                        Size = bid.Volume
                    });
                    totalBidVol += bid.Volume;
                }

                // Add ask levels
                double totalAskVol = 0;
                for (int i = 0; i < Math.Min(10, depth.Asks.Count); i++)
                {
                    var ask = depth.Asks[i];
                    level2Data.Asks.Add(new PriceLevel
                    {
                        Price = ask.Price,
                        Size = ask.Volume
                    });
                    totalAskVol += ask.Volume;
                }

                // Calculate metrics
                level2Data.BidDepth = level2Data.Bids.Count;
                level2Data.AskDepth = level2Data.Asks.Count;
                level2Data.TotalBidVolume = totalBidVol;
                level2Data.TotalAskVolume = totalAskVol;
                
                // Order imbalance
                double totalVol = totalBidVol + totalAskVol;
                level2Data.OrderImbalance = totalVol > 0 ? (totalBidVol - totalAskVol) / totalVol : 0;

                // Spread in basis points
                if (level2Data.Bids.Count > 0 && level2Data.Asks.Count > 0)
                {
                    double spread = level2Data.Asks[0].Price - level2Data.Bids[0].Price;
                    double midPrice = (level2Data.Asks[0].Price + level2Data.Bids[0].Price) / 2.0;
                    level2Data.SpreadBps = midPrice > 0 ? (spread / midPrice) * 10000.0 : 0;
                }

                var marketData = new MarketData
                {
                    Symbol = Instrument.FullName,
                    Timestamp = DateTime.Now.ToUnixTimestamp(),
                    DataType = MarketData.Types.DataType.Level1AndLevel2,
                    Level1 = new Level1Data
                    {
                        Symbol = Instrument.FullName,
                        Timestamp = Time[0].ToUnixTimestamp(),
                        LastPrice = Close[0],
                        LastSize = Volume[0],
                        Bid = GetCurrentBid(),
                        Ask = GetCurrentAsk(),
                        BidSize = (int)GetCurrentBidSize(),
                        AskSize = (int)GetCurrentAskSize(),
                        Open = Open[0],
                        High = High[0],
                        Low = Low[0],
                        Close = Close[0],
                        Volume = Volume[0]
                    },
                    Level2 = level2Data
                };

                await streamingCall.RequestStream.WriteAsync(marketData);
                level2Count++;
                lastLevel2Send = now;

                if (level2Count % 20 == 0)
                {
                    Print($"NEXUS AI: Sent {level2Count} Level 2 updates");
                }
            }
            catch (Exception ex)
            {
                Print($"NEXUS AI: Level 2 send error - {ex.Message}");
            }
        }

        private void OnMarketDepth(object sender, MarketDepthEventArgs e)
        {
            if (EnableLevel2 && isStreaming)
            {
                SendLevel2Data();
            }
        }
        #endregion

        #region Signal Reception
        private async Task ReceiveSignals()
        {
            try
            {
                while (isStreaming && streamingCall != null)
                {
                    if (await streamingCall.ResponseStream.MoveNext())
                    {
                        var signal = streamingCall.ResponseStream.Current;
                        ProcessSignal(signal);
                    }
                }
            }
            catch (Exception ex)
            {
                Print($"NEXUS AI: Signal reception error - {ex.Message}");
            }
        }

        private void ProcessSignal(TradingSignal signal)
        {
            signalsReceived++;

            Print($"NEXUS AI SIGNAL #{signalsReceived}:");
            Print($"  Type: {signal.SignalType}");
            Print($"  Confidence: {signal.Confidence:P2}");
            Print($"  Symbol: {signal.Symbol}");
            Print($"  Position Size: {signal.PositionSize}");

            // Check if signal meets criteria
            if (signal.Confidence < MinConfidence)
            {
                Print($"  SKIPPED - Confidence {signal.Confidence:P2} < {MinConfidence:P2}");
                return;
            }

            if (signal.Symbol != Instrument.FullName)
            {
                Print($"  SKIPPED - Symbol mismatch");
                return;
            }

            // Execute if auto-trade enabled
            if (AutoTrade)
            {
                ExecuteSignal(signal);
            }
            else
            {
                Print($"  AUTO-TRADE DISABLED - Signal not executed");
            }
        }

        private void ExecuteSignal(TradingSignal signal)
        {
            try
            {
                OrderAction action;
                
                switch (signal.SignalType)
                {
                    case TradingSignal.Types.SignalType.Buy:
                    case TradingSignal.Types.SignalType.StrongBuy:
                        action = OrderAction.Buy;
                        break;
                    
                    case TradingSignal.Types.SignalType.Sell:
                    case TradingSignal.Types.SignalType.StrongSell:
                        action = OrderAction.Sell;
                        break;
                    
                    default:
                        Print($"  NEUTRAL signal - No action");
                        return;
                }

                int quantity = signal.PositionSize > 0 ? (int)signal.PositionSize : PositionSize;

                EnterMarket(0, action, quantity, $"NEXUS_{signal.SignalType}");
                ordersPlaced++;

                Print($"  ORDER PLACED: {action} {quantity} @ Market");
                
                // Set stops if provided
                if (signal.StopLoss > 0)
                {
                    SetStopLoss(CalculationMode.Price, signal.StopLoss);
                    Print($"  Stop Loss: {signal.StopLoss}");
                }
                
                if (signal.TakeProfit > 0)
                {
                    SetProfitTarget(CalculationMode.Price, signal.TakeProfit);
                    Print($"  Take Profit: {signal.TakeProfit}");
                }
            }
            catch (Exception ex)
            {
                Print($"NEXUS AI: Order execution error - {ex.Message}");
            }
        }
        #endregion

        #region Helper Methods
        private double GetCurrentBid()
        {
            return BarsArray[0].Instrument.MarketData.Bid?.Price ?? Close[0];
        }

        private double GetCurrentAsk()
        {
            return BarsArray[0].Instrument.MarketData.Ask?.Price ?? Close[0];
        }

        private double GetCurrentBidSize()
        {
            return BarsArray[0].Instrument.MarketData.Bid?.Volume ?? 0;
        }

        private double GetCurrentAskSize()
        {
            return BarsArray[0].Instrument.MarketData.Ask?.Volume ?? 0;
        }
        #endregion
    }
}

#region Extension Methods
public static class DateTimeExtensions
{
    public static double ToUnixTimestamp(this DateTime dateTime)
    {
        return (dateTime.ToUniversalTime() - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;
    }
}
#endregion
