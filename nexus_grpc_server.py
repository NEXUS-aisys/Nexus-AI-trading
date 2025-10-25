"""
NEXUS AI - gRPC Server
Receives market data from Sierra Chart and NinjaTrader
Sends trading signals back to platforms
"""

import grpc
from concurrent import futures
import time
import sys
import os

# Add proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto'))

import nexus_trading_pb2
import nexus_trading_pb2_grpc


class TradingServiceImpl(nexus_trading_pb2_grpc.TradingServiceServicer):
    """Implementation of TradingService"""
    
    def __init__(self):
        self.level1_count = 0
        self.level2_count = 0
        self.signal_count = 0
        
    def StreamMarketData(self, request_iterator, context):
        """
        Bidirectional streaming - receives market data, sends signals
        """
        print("=" * 60)
        print("CLIENT CONNECTED!")
        print("=" * 60)
        
        try:
            for market_data in request_iterator:
                # Process received data
                symbol = market_data.symbol
                timestamp = market_data.timestamp
                data_type = market_data.data_type
                
                if data_type == nexus_trading_pb2.MarketData.LEVEL1_ONLY:
                    self.level1_count += 1
                    level1 = market_data.level1
                    
                    if self.level1_count % 10 == 0:  # Log every 10th message
                        print(f"\n[LEVEL 1] {symbol}")
                        print(f"  Bid: {level1.bid:.2f} x {level1.bid_size}")
                        print(f"  Ask: {level1.ask:.2f} x {level1.ask_size}")
                        print(f"  Last: {level1.last_price:.2f}")
                        print(f"  Volume: {level1.volume:.0f}")
                        print(f"  VWAP: {level1.vwap:.2f}")
                        print(f"  Total received: {self.level1_count}")
                
                elif data_type == nexus_trading_pb2.MarketData.LEVEL1_AND_LEVEL2:
                    self.level2_count += 1
                    level2 = market_data.level2
                    
                    if self.level2_count % 5 == 0:  # Log every 5th message
                        print(f"\n[LEVEL 2] {symbol}")
                        print(f"  Bid Depth: {level2.bid_depth} levels")
                        print(f"  Ask Depth: {level2.ask_depth} levels")
                        print(f"  Total Bid Vol: {level2.total_bid_volume:.0f}")
                        print(f"  Total Ask Vol: {level2.total_ask_volume:.0f}")
                        print(f"  Imbalance: {level2.order_imbalance:.3f}")
                        print(f"  Spread (bps): {level2.spread_bps:.2f}")
                        
                        if level2.bids:
                            print(f"  Best Bid: {level2.bids[0].price:.2f} x {level2.bids[0].size:.0f}")
                        if level2.asks:
                            print(f"  Best Ask: {level2.asks[0].price:.2f} x {level2.asks[0].size:.0f}")
                        
                        print(f"  Total received: {self.level2_count}")
                
                # Send test signal every 50 messages
                if (self.level1_count + self.level2_count) % 50 == 0:
                    signal = self._generate_test_signal(symbol)
                    print(f"\n>>> SENDING SIGNAL: {signal.signal_type.name} for {symbol}")
                    yield signal
                    
        except Exception as e:
            print(f"\nERROR: {e}")
        finally:
            print("\n" + "=" * 60)
            print("CLIENT DISCONNECTED")
            print(f"Total Level 1 messages: {self.level1_count}")
            print(f"Total Level 2 messages: {self.level2_count}")
            print(f"Total signals sent: {self.signal_count}")
            print("=" * 60)
    
    def _generate_test_signal(self, symbol):
        """Generate a test trading signal"""
        self.signal_count += 1
        
        signal = nexus_trading_pb2.TradingSignal()
        signal.signal_id = f"TEST_{self.signal_count}"
        signal.symbol = symbol
        signal.timestamp = time.time()
        
        # Alternate between BUY and SELL
        if self.signal_count % 2 == 0:
            signal.signal_type = nexus_trading_pb2.TradingSignal.BUY
        else:
            signal.signal_type = nexus_trading_pb2.TradingSignal.SELL
        
        signal.confidence = 0.75
        signal.position_size = 1.0
        signal.entry_price = 0.0  # Will be filled by client
        signal.stop_loss = 0.0
        signal.take_profit = 0.0
        signal.strategy_name = "TEST_STRATEGY"
        signal.timeframe = "1MIN"
        
        return signal


def serve():
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    service = TradingServiceImpl()
    nexus_trading_pb2_grpc.add_TradingServiceServicer_to_server(service, server)
    
    server_address = '0.0.0.0:50051'
    server.add_insecure_port(server_address)
    
    print("=" * 60)
    print("NEXUS AI - gRPC Server")
    print("=" * 60)
    print(f"Server listening on: {server_address}")
    print("Waiting for connections from:")
    print("  - Sierra Chart (C++)")
    print("  - NinjaTrader 8 (C#)")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    server.start()
    
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.stop(0)
        print("Server stopped.")


if __name__ == '__main__':
    serve()
