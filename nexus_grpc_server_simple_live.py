"""
NEXUS AI - Simple Live gRPC Server
Incremental integration with NEXUS AI pipeline
"""

import grpc
from concurrent import futures
import time
import sys
import os
import logging

# Add proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto'))

import nexus_trading_pb2
import nexus_trading_pb2_grpc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NEXUS_Simple_Server")


class SimpleLiveTradingService(nexus_trading_pb2_grpc.TradingServiceServicer):
    """
    Simple live trading service - processes data and generates basic signals
    """
    
    def __init__(self):
        logger.info("="*70)
        logger.info("Initializing NEXUS AI Simple Live Server")
        logger.info("="*70)
        
        self.level1_count = 0
        self.level2_count = 0
        self.signals_sent = 0
        
        # Simple trading logic state
        self.last_price = {}
        self.price_history = {}  # symbol -> list of prices
        self.vwap_data = {}  # symbol -> (total_pv, total_v)
        
        logger.info("✅ Server initialized")
        logger.info("="*70)
        
    def StreamMarketData(self, request_iterator, context):
        """
        Bidirectional streaming - receives market data, sends signals
        """
        client_addr = context.peer()
        logger.info("="*70)
        logger.info(f"✅ CLIENT CONNECTED: {client_addr}")
        logger.info("="*70)
        
        try:
            for market_data in request_iterator:
                symbol = market_data.symbol
                timestamp = market_data.timestamp
                data_type = market_data.data_type
                
                # Process Level 1 data
                if data_type == nexus_trading_pb2.MarketData.LEVEL1_ONLY:
                    self.level1_count += 1
                    level1 = market_data.level1
                    
                    # Store price history
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    self.price_history[symbol].append(level1.last_price)
                    
                    # Keep only last 100 prices
                    if len(self.price_history[symbol]) > 100:
                        self.price_history[symbol] = self.price_history[symbol][-100:]
                    
                    # Update VWAP
                    if symbol not in self.vwap_data:
                        self.vwap_data[symbol] = [0.0, 0.0]  # [pv_sum, v_sum]
                    
                    pv = level1.last_price * level1.last_size
                    self.vwap_data[symbol][0] += pv
                    self.vwap_data[symbol][1] += level1.last_size
                    
                    # Log every 10 messages
                    if self.level1_count % 10 == 0:
                        vwap = self.vwap_data[symbol][0] / self.vwap_data[symbol][1] if self.vwap_data[symbol][1] > 0 else level1.last_price
                        
                        logger.info(f"\n📊 [LEVEL 1] {symbol} (#{self.level1_count})")
                        logger.info(f"   Bid: {level1.bid:.2f} x {level1.bid_size}")
                        logger.info(f"   Ask: {level1.ask:.2f} x {level1.ask_size}")
                        logger.info(f"   Last: {level1.last_price:.2f}")
                        logger.info(f"   Volume: {level1.volume:.0f}")
                        logger.info(f"   VWAP: {vwap:.2f}")
                    
                    # Simple signal generation logic
                    signal = self._generate_signal_from_level1(symbol, level1)
                    if signal:
                        logger.info(f"\n🚀 SIGNAL GENERATED!")
                        logger.info(f"   Symbol: {symbol}")
                        logger.info(f"   Type: {signal.signal_type.name}")
                        logger.info(f"   Confidence: {signal.confidence:.2%}")
                        self.signals_sent += 1
                        yield signal
                
                # Process Level 2 data
                elif data_type == nexus_trading_pb2.MarketData.LEVEL1_AND_LEVEL2:
                    self.level2_count += 1
                    level2 = market_data.level2
                    
                    # Log every 5 messages
                    if self.level2_count % 5 == 0:
                        logger.info(f"\n📖 [LEVEL 2] {symbol} (#{self.level2_count})")
                        logger.info(f"   Bid Depth: {level2.bid_depth} levels")
                        logger.info(f"   Ask Depth: {level2.ask_depth} levels")
                        logger.info(f"   Imbalance: {level2.order_imbalance:.3f}")
                        logger.info(f"   Spread (bps): {level2.spread_bps:.2f}")
                        
                        if level2.bids:
                            logger.info(f"   Best Bid: {level2.bids[0].price:.2f} x {level2.bids[0].size:.0f}")
                        if level2.asks:
                            logger.info(f"   Best Ask: {level2.asks[0].price:.2f} x {level2.asks[0].size:.0f}")
                    
                    # Signal from order book imbalance
                    signal = self._generate_signal_from_level2(symbol, level2)
                    if signal:
                        logger.info(f"\n🚀 ORDER BOOK SIGNAL!")
                        logger.info(f"   Symbol: {symbol}")
                        logger.info(f"   Type: {signal.signal_type.name}")
                        logger.info(f"   Imbalance: {level2.order_imbalance:.3f}")
                        self.signals_sent += 1
                        yield signal
                        
        except Exception as e:
            logger.error(f"Error in StreamMarketData: {e}", exc_info=True)
        
        finally:
            logger.info("\n" + "="*70)
            logger.info(f"❌ CLIENT DISCONNECTED: {client_addr}")
            logger.info(f"   Level 1: {self.level1_count} messages")
            logger.info(f"   Level 2: {self.level2_count} messages")
            logger.info(f"   Signals: {self.signals_sent} sent")
            logger.info("="*70)
    
    def _generate_signal_from_level1(self, symbol, level1):
        """
        Simple VWAP reversion strategy
        """
        if symbol not in self.vwap_data or self.vwap_data[symbol][1] == 0:
            return None
        
        vwap = self.vwap_data[symbol][0] / self.vwap_data[symbol][1]
        current_price = level1.last_price
        
        # Calculate deviation from VWAP
        deviation = (current_price - vwap) / vwap
        
        # Generate signal if deviation > 0.2%
        if abs(deviation) > 0.002:
            signal = nexus_trading_pb2.TradingSignal()
            signal.signal_id = f"VWAP_{symbol}_{int(time.time()*1000)}"
            signal.symbol = symbol
            signal.timestamp = time.time()
            
            if deviation > 0.002:  # Price above VWAP - SELL
                signal.signal_type = nexus_trading_pb2.TradingSignal.SELL
            else:  # Price below VWAP - BUY
                signal.signal_type = nexus_trading_pb2.TradingSignal.BUY
            
            signal.confidence = min(abs(deviation) * 100, 0.95)
            signal.position_size = 1.0
            signal.entry_price = current_price
            signal.stop_loss = 0.0
            signal.take_profit = vwap
            signal.strategy_name = "VWAP_REVERSION"
            signal.timeframe = "1MIN"
            
            return signal
        
        return None
    
    def _generate_signal_from_level2(self, symbol, level2):
        """
        Order book imbalance strategy
        """
        imbalance = level2.order_imbalance
        
        # Strong imbalance threshold: 0.3
        if abs(imbalance) > 0.3:
            signal = nexus_trading_pb2.TradingSignal()
            signal.signal_id = f"OBI_{symbol}_{int(time.time()*1000)}"
            signal.symbol = symbol
            signal.timestamp = time.time()
            
            if imbalance > 0.3:  # More bids - BUY
                signal.signal_type = nexus_trading_pb2.TradingSignal.BUY
            else:  # More asks - SELL
                signal.signal_type = nexus_trading_pb2.TradingSignal.SELL
            
            signal.confidence = min(abs(imbalance), 0.90)
            signal.position_size = 1.0
            signal.entry_price = level2.bids[0].price if level2.bids else 0.0
            signal.strategy_name = "ORDER_BOOK_IMBALANCE"
            signal.timeframe = "1MIN"
            
            return signal
        
        return None


def serve():
    """Start the simple live server"""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    service = SimpleLiveTradingService()
    nexus_trading_pb2_grpc.add_TradingServiceServicer_to_server(service, server)
    
    server_address = '0.0.0.0:50051'
    server.add_insecure_port(server_address)
    
    logger.info("="*70)
    logger.info("🚀 NEXUS AI - Simple Live gRPC Server")
    logger.info("="*70)
    logger.info(f"📡 Server: {server_address}")
    logger.info(f"📊 Strategies: VWAP Reversion + Order Book Imbalance")
    logger.info(f"✅ Ready for live trading!")
    logger.info("="*70)
    logger.info("\nWaiting for connections from:")
    logger.info("  - Sierra Chart (C++)")
    logger.info("  - NinjaTrader 8 (C#)")
    logger.info("="*70)
    logger.info("\n⌨️  Press Ctrl+C to stop\n")
    
    server.start()
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Shutting down server...")
        server.stop(0)
        logger.info("✅ Server stopped.")


if __name__ == '__main__':
    serve()
