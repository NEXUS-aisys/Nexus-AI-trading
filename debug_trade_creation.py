#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Trade Creation Process
Shows exactly how trades are created in the NEXUS AI backtesting system
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nexus_backtester import NexusBacktester, BacktestConfig, MarketDataGenerator
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def debug_trade_creation():
    """Debug the trade creation process step by step."""
    
    print("🔍 NEXUS AI Trade Creation Debug")
    print("=" * 60)
    
    # Create a simple config for debugging
    config = BacktestConfig(
        start_date="2024-10-20",
        end_date="2024-10-22",
        initial_capital=10000.0,
        symbols=["BTCUSDT"],
        timeframe="1h",
        commission=0.001,
        slippage=0.0005,
        min_confidence=0.30,  # Lower threshold to see more signals
        mqscore_threshold=0.30,  # Lower threshold to see more signals
        use_mqscore_filter=False,  # Disable for debugging
        enable_ml_models=True,
        max_position_size=0.20,  # Allow larger positions
    )
    
    print(f"📊 Configuration:")
    print(f"   Period: {config.start_date} to {config.end_date}")
    print(f"   Capital: ${config.initial_capital:,.2f}")
    print(f"   Min Confidence: {config.min_confidence}")
    print(f"   MQScore Filter: {'ON' if config.use_mqscore_filter else 'OFF'}")
    print(f"   Max Position: {config.max_position_size:.1%}")
    
    # Create backtester
    backtester = NexusBacktester(config)
    
    # Initialize NEXUS AI
    print("\n🚀 Initializing NEXUS AI...")
    if not await backtester.initialize_nexus_ai():
        print("❌ Failed to initialize NEXUS AI")
        return
    
    print("✅ NEXUS AI initialized successfully")
    
    # Generate market data
    print("\n📈 Generating market data...")
    market_data = backtester.generate_market_data()
    
    # Get a few timestamps for detailed analysis
    timestamps = sorted(list(market_data["BTCUSDT"].index))[:10]  # First 10 timestamps
    
    print(f"📊 Generated {len(timestamps)} timestamps for analysis")
    print(f"   First timestamp: {timestamps[0]}")
    print(f"   Last timestamp: {timestamps[-1]}")
    
    # Analyze each timestamp in detail
    for i, timestamp in enumerate(timestamps):
        print(f"\n{'='*60}")
        print(f"🕐 TIMESTAMP {i+1}: {timestamp}")
        print(f"{'='*60}")
        
        # Get current market data
        symbol = "BTCUSDT"
        symbol_data = market_data[symbol]
        current_data = symbol_data[symbol_data.index <= timestamp]
        
        if len(current_data) < 5:
            print("⚠️  Insufficient data for analysis")
            continue
            
        current_bar = current_data.iloc[-1]
        current_price = current_bar['close']
        
        print(f"📊 Market Data:")
        print(f"   Price: ${current_price:,.2f}")
        print(f"   Volume: {current_bar['volume']:,.0f}")
        print(f"   High: ${current_bar['high']:,.2f}")
        print(f"   Low: ${current_bar['low']:,.2f}")
        
        # Create market dict
        market_dict = {
            'symbol': symbol,
            'timestamp': timestamp.timestamp(),
            'price': current_price,
            'open': current_bar['open'],
            'high': current_bar['high'],
            'low': current_bar['low'],
            'close': current_price,
            'volume': current_bar['volume'],
            'bid': current_price * 0.9995,
            'ask': current_price * 1.0005,
        }
        
        print(f"\n🎯 Processing Strategies...")
        
        # Process signals manually to see what's happening
        signals_found = 0
        strategy_results = []
        
        # Get strategies from NEXUS AI
        strategies = backtester.nexus_ai.strategy_manager.get_strategies()
        print(f"   Available strategies: {len(strategies)}")
        
        for strategy_name, strategy in list(strategies.items())[:5]:  # Test first 5 strategies
            try:
                print(f"\n   🔄 Testing: {strategy_name}")
                
                # Execute strategy
                result = strategy.execute(current_data, symbol)
                
                if result:
                    signal = result.get('signal', 0)
                    confidence = result.get('confidence', 0)
                    metadata = result.get('metadata', {})
                    
                    print(f"      Signal: {signal:.3f}")
                    print(f"      Confidence: {confidence:.3f}")
                    print(f"      Action: {metadata.get('action', 'UNKNOWN')}")
                    
                    if signal != 0 and confidence > 0:
                        signals_found += 1
                        strategy_results.append({
                            'strategy': strategy_name,
                            'signal': signal,
                            'confidence': confidence,
                            'metadata': metadata
                        })
                        
                        print(f"      ✅ SIGNAL GENERATED!")
                        
                        # Check if it passes filters
                        if confidence >= config.min_confidence:
                            print(f"      ✅ Confidence filter PASSED ({confidence:.3f} >= {config.min_confidence})")
                            
                            # Calculate position size
                            portfolio_value = backtester.portfolio.get_portfolio_value({symbol: current_price})
                            max_position_value = portfolio_value * config.max_position_size
                            position_value = max_position_value * confidence * abs(signal)
                            quantity = position_value / current_price
                            
                            print(f"      💰 Position Sizing:")
                            print(f"         Portfolio Value: ${portfolio_value:,.2f}")
                            print(f"         Max Position Value: ${max_position_value:,.2f}")
                            print(f"         Calculated Position: ${position_value:,.2f}")
                            print(f"         Quantity: {quantity:.6f} {symbol}")
                            
                            # Check if we have enough cash
                            side = "BUY" if signal > 0 else "SELL"
                            if side == "BUY":
                                required_cash = position_value * (1 + config.commission + config.slippage)
                                available_cash = backtester.portfolio.cash
                                
                                print(f"      💵 Cash Check:")
                                print(f"         Required: ${required_cash:,.2f}")
                                print(f"         Available: ${available_cash:,.2f}")
                                
                                if required_cash <= available_cash:
                                    print(f"      ✅ TRADE WOULD BE EXECUTED!")
                                    
                                    # Actually execute the trade for demonstration
                                    success = backtester.portfolio.execute_trade(
                                        symbol=symbol,
                                        side=side,
                                        quantity=quantity,
                                        price=current_price,
                                        strategy=strategy_name,
                                        confidence=confidence,
                                        timestamp=timestamp
                                    )
                                    
                                    if success:
                                        print(f"      🎉 TRADE EXECUTED SUCCESSFULLY!")
                                        print(f"         {side} {quantity:.6f} {symbol} @ ${current_price:,.2f}")
                                    else:
                                        print(f"      ❌ Trade execution failed")
                                else:
                                    print(f"      ❌ Insufficient cash for trade")
                            else:
                                print(f"      ⚠️  SELL signal - would need existing position")
                        else:
                            print(f"      ❌ Confidence filter FAILED ({confidence:.3f} < {config.min_confidence})")
                    else:
                        print(f"      ⚪ No signal generated")
                else:
                    print(f"      ⚪ No result from strategy")
                    
            except Exception as e:
                print(f"      ❌ Strategy error: {e}")
                continue
        
        print(f"\n📊 Summary for {timestamp}:")
        print(f"   Strategies tested: {min(5, len(strategies))}")
        print(f"   Signals found: {signals_found}")
        print(f"   Trades executed: {len(backtester.portfolio.trade_history)}")
        
        # Show current portfolio state
        portfolio_value = backtester.portfolio.get_portfolio_value({symbol: current_price})
        print(f"   Portfolio value: ${portfolio_value:,.2f}")
        print(f"   Cash: ${backtester.portfolio.cash:,.2f}")
        print(f"   Positions: {len(backtester.portfolio.positions)}")
        
        if i >= 2:  # Stop after 3 timestamps for demo
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎯 FINAL TRADE CREATION SUMMARY")
    print(f"{'='*60}")
    
    total_trades = len(backtester.portfolio.trade_history)
    print(f"Total trades executed: {total_trades}")
    
    if total_trades > 0:
        print(f"\n📋 Trade Details:")
        for i, trade in enumerate(backtester.portfolio.trade_history):
            print(f"   Trade {i+1}:")
            print(f"      Time: {trade.timestamp}")
            print(f"      Strategy: {trade.strategy}")
            print(f"      Action: {trade.side}")
            print(f"      Quantity: {trade.quantity:.6f}")
            print(f"      Price: ${trade.price:,.2f}")
            print(f"      Confidence: {trade.confidence:.3f}")
            print(f"      Commission: ${trade.commission:.2f}")
            print(f"      Slippage: ${trade.slippage:.2f}")
    else:
        print("❌ No trades were executed")
        print("\n🔍 Possible reasons:")
        print("   1. Strategy signals too weak (below confidence threshold)")
        print("   2. MQScore filtering too strict")
        print("   3. Insufficient market volatility in test period")
        print("   4. Strategy parameters need adjustment")
        print("   5. Market data doesn't match strategy expectations")
    
    # Show portfolio final state
    final_portfolio_value = backtester.portfolio.get_portfolio_value({symbol: current_price})
    print(f"\n💰 Final Portfolio:")
    print(f"   Initial Capital: ${config.initial_capital:,.2f}")
    print(f"   Final Value: ${final_portfolio_value:,.2f}")
    print(f"   P&L: ${final_portfolio_value - config.initial_capital:,.2f}")
    print(f"   Return: {((final_portfolio_value - config.initial_capital) / config.initial_capital) * 100:.2f}%")


if __name__ == "__main__":
    asyncio.run(debug_trade_creation())