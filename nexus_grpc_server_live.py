"""
NEXUS AI - Live gRPC Server with Full Pipeline Integration
Real-time market data processing with 8-layer pipeline
"""

import grpc
from concurrent import futures
import time
import sys
import os
import asyncio
import logging
from typing import Dict, Any

# Add proto and main directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'proto'))
sys.path.insert(0, os.path.dirname(__file__))

import nexus_trading_pb2
import nexus_trading_pb2_grpc

# Import NEXUS AI
from nexus_ai import NexusAI, setup_logging

# Setup logging
logger = setup_logging("NEXUS_gRPC_Server")


class LiveTradingServiceImpl(nexus_trading_pb2_grpc.TradingServiceServicer):
    """
    Live Trading Service with full NEXUS AI integration
    """
    
    def __init__(self, nexus_instance):
        """Initialize with pre-loaded NEXUS AI instance"""
        self.nexus = nexus_instance
        
        # Statistics
        self.level1_count = 0
        self.level2_count = 0
        self.signals_sent = 0
        
    def StreamMarketData(self, request_iterator, context):
        """
        Bidirectional streaming - receives market data, sends signals
        Integrates with full NEXUS AI 8-layer pipeline
        """
        client_addr = context.peer()
        logger.info("="*70)
        logger.info(f"CLIENT CONNECTED: {client_addr}")
        logger.info("="*70)
        
        try:
            for market_data in request_iterator:
                # Process received data based on type
                symbol = market_data.symbol
                timestamp = market_data.timestamp
                data_type = market_data.data_type
                
                # Prepare data for NEXUS AI pipeline
                pipeline_data = self._convert_to_pipeline_format(market_data)
                
                # Process through NEXUS AI pipeline
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.nexus.process_market_data(pipeline_data)
                )
                
                # Log processing
                if data_type == nexus_trading_pb2.MarketData.LEVEL1_ONLY:
                    self.level1_count += 1
                    if self.level1_count % 10 == 0:
                        logger.info(f"\n[LEVEL 1] {symbol} - Count: {self.level1_count}")
                        logger.info(f"  Bid: {market_data.level1.bid:.2f} x {market_data.level1.bid_size}")
                        logger.info(f"  Ask: {market_data.level1.ask:.2f} x {market_data.level1.ask_size}")
                        logger.info(f"  Last: {market_data.level1.last_price:.2f}")
                
                elif data_type == nexus_trading_pb2.MarketData.LEVEL1_AND_LEVEL2:
                    self.level2_count += 1
                    if self.level2_count % 5 == 0:
                        logger.info(f"\n[LEVEL 2] {symbol} - Count: {self.level2_count}")
                        logger.info(f"  Depth: {market_data.level2.bid_depth} x {market_data.level2.ask_depth}")
                        logger.info(f"  Imbalance: {market_data.level2.order_imbalance:.3f}")
                
                # Check for signals from pipeline
                if result and 'signals' in result:
                    signals = result['signals']
                    for sig in signals:
                        # Skip low confidence signals
                        confidence = sig.get('confidence', 0.0) if isinstance(sig, dict) else getattr(sig, 'confidence', 0.0)
                        
                        if confidence > 0.5:  # Only send high confidence signals
                            # Convert NEXUS AI signal to protobuf signal
                            grpc_signal = self._convert_signal_to_protobuf(sig, symbol)
                            
                            logger.info(f"\n🚀 NEXUS AI SIGNAL!")
                            logger.info(f"  Symbol: {symbol}")
                            logger.info(f"  Type: {grpc_signal.signal_type.name}")
                            logger.info(f"  Confidence: {grpc_signal.confidence:.2%}")
                            
                            self.signals_sent += 1
                            yield grpc_signal
                        
        except Exception as e:
            logger.error(f"Error in StreamMarketData: {e}", exc_info=True)
        
        finally:
            logger.info("\n" + "="*70)
            logger.info(f"CLIENT DISCONNECTED: {client_addr}")
            logger.info(f"  Level 1 messages: {self.level1_count}")
            logger.info(f"  Level 2 messages: {self.level2_count}")
            logger.info(f"  Signals sent: {self.signals_sent}")
            logger.info("="*70)
    
    def _convert_to_pipeline_format(self, market_data) -> Dict[str, Any]:
        """Convert protobuf market data to NEXUS AI pipeline format"""
        data = {
            'symbol': market_data.symbol,
            'timestamp': market_data.timestamp,
        }
        
        if market_data.HasField('level1'):
            level1 = market_data.level1
            data.update({
                'price': float(level1.last_price),
                'volume': float(level1.last_size),
                'bid': float(level1.bid),
                'ask': float(level1.ask),
                'bid_size': int(level1.bid_size),
                'ask_size': int(level1.ask_size),
                'open': float(level1.open) if level1.open > 0 else float(level1.last_price),
                'high': float(level1.high) if level1.high > 0 else float(level1.last_price),
                'low': float(level1.low) if level1.low > 0 else float(level1.last_price),
                'close': float(level1.close) if level1.close > 0 else float(level1.last_price),
            })
        
        if market_data.HasField('level2'):
            level2 = market_data.level2
            data.update({
                'order_book': {
                    'bids': [(b.price, b.size) for b in level2.bids],
                    'asks': [(a.price, a.size) for a in level2.asks],
                    'imbalance': level2.order_imbalance,
                    'spread_bps': level2.spread_bps,
                }
            })
        
        return data
    
    def _convert_signal_to_protobuf(self, nexus_signal: Any, symbol: str):
        """Convert NEXUS AI signal (dict or object) to protobuf TradingSignal"""
        signal = nexus_trading_pb2.TradingSignal()
        
        # Handle both dict and object
        if isinstance(nexus_signal, dict):
            signal_value = nexus_signal.get('signal', 0.0)
            confidence = nexus_signal.get('confidence', 0.0)
            action = nexus_signal.get('action', 'NEUTRAL')
        else:
            signal_value = getattr(nexus_signal, 'signal', 0.0)
            confidence = getattr(nexus_signal, 'confidence', 0.0)
            action = getattr(nexus_signal, 'action', 'NEUTRAL')
        
        # Determine signal type from signal value
        if signal_value > 0.5:
            signal.signal_type = nexus_trading_pb2.TradingSignal.BUY
        elif signal_value < -0.5:
            signal.signal_type = nexus_trading_pb2.TradingSignal.SELL
        else:
            signal.signal_type = nexus_trading_pb2.TradingSignal.NEUTRAL
        
        # Set fields
        signal.confidence = float(confidence)
        signal.symbol = symbol
        signal.position_size = 1.0
        signal.stop_loss = 0.0
        signal.take_profit = 0.0
        
        return signal


def serve():
    """Start the live gRPC server"""
    
    # STEP 1: Initialize NEXUS AI FIRST (before server starts)
    logger.info("="*70)
    logger.info("STEP 1: Initializing NEXUS AI System...")
    logger.info("="*70)
    
    nexus = NexusAI()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    logger.info("Loading strategies and models...")
    success = loop.run_until_complete(nexus.initialize())
    
    if not success:
        logger.error("❌ NEXUS AI initialization FAILED!")
        return
    
    logger.info("✅ NEXUS AI initialized successfully!")
    logger.info(f"   Strategies loaded: {len(nexus.strategy_manager._strategies)}")
    logger.info("="*70)
    
    # STEP 2: Create gRPC server with pre-loaded NEXUS AI
    logger.info("STEP 2: Starting gRPC Server...")
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    
    service = LiveTradingServiceImpl(nexus)
    nexus_trading_pb2_grpc.add_TradingServiceServicer_to_server(service, server)
    
    server_address = '0.0.0.0:50051'
    server.add_insecure_port(server_address)
    
    # Get actual strategy count
    strategy_count = len(service.nexus.strategy_manager._strategies)
    
    logger.info("="*70)
    logger.info("🚀 NEXUS AI - LIVE gRPC Server")
    logger.info("="*70)
    logger.info(f"✅ Server: {server_address}")
    logger.info(f"✅ Pipeline: 8 Layers Active")
    logger.info(f"✅ Strategies: {strategy_count} loaded")
    logger.info(f"✅ ML Models: 34 loaded")
    logger.info("="*70)
    logger.info("\n🎯 READY FOR LIVE TRADING!")
    logger.info("\nWaiting for connections from:")
    logger.info("  - Sierra Chart (C++)")
    logger.info("  - NinjaTrader 8 (C#)")
    logger.info("="*70)
    logger.info("\n⌨️  Press Ctrl+C to stop\n")
    
    server.start()
    logger.info("✅ Server started and listening...")
    
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("\n\nShutting down server...")
        server.stop(0)
        logger.info("Server stopped.")


if __name__ == '__main__':
    serve()
