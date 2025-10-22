#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Trade Creation Demo
Shows the core mechanics of how trades are created
"""

import pandas as pd
import numpy as np
from datetime import datetime
from nexus_backtester import Trade, Position, PortfolioManager, BacktestConfig


def demonstrate_trade_creation():
    """Demonstrate the core trade creation mechanics."""
    
    print("🔍 NEXUS AI Trade Creation Mechanics Demo")
    print("=" * 60)
    
    # Create configuration
    config = BacktestConfig(
        initial_capital=10000.0,
        commission=0.001,  # 0.1%
        slippage=0.0005,   # 0.05%
        max_position_size=0.20  # 20% max position
    )
    
    # Initialize portfolio manager
    portfolio = PortfolioManager(config.initial_capital, config)
    
    print(f"💰 Initial Portfolio:")
    print(f"   Cash: ${portfolio.cash:,.2f}")
    print(f"   Positions: {len(portfolio.positions)}")
    
    # Simulate market data
    symbol = "BTCUSDT"
    current_price = 45250.50
    timestamp = datetime.now()
    
    print(f"\n📊 Market Data:")
    print(f"   Symbol: {symbol}")
    print(f"   Price: ${current_price:,.2f}")
    print(f"   Time: {timestamp}")
    
    # Simulate strategy signals
    signals = [
        {
            'strategy': 'Absorption-Breakout',
            'signal': 0.8,      # Strong buy signal
            'confidence': 0.75,  # High confidence
            'reason': 'Large volume absorption detected at support'
        },
        {
            'strategy': 'Momentum-Breakout', 
            'signal': -0.6,     # Moderate sell signal
            'confidence': 0.65,  # Good confidence
            'reason': 'Momentum divergence detected'
        },
        {
            'strategy': 'Order-Book-Imbalance',
            'signal': 0.4,      # Weak buy signal
            'confidence': 0.45,  # Low confidence (below threshold)
            'reason': 'Minor bid-ask imbalance'
        }
    ]
    
    print(f"\n🎯 Strategy Signals Generated:")
    for i, signal_data in enumerate(signals, 1):
        print(f"   Signal {i}: {signal_data['strategy']}")
        print(f"      Signal Strength: {signal_data['signal']:+.2f}")
        print(f"      Confidence: {signal_data['confidence']:.2f}")
        print(f"      Reason: {signal_data['reason']}")
    
    # Process each signal
    print(f"\n🔄 Processing Signals...")
    
    min_confidence = 0.60  # 60% minimum confidence
    
    for i, signal_data in enumerate(signals, 1):
        print(f"\n--- Processing Signal {i}: {signal_data['strategy']} ---")
        
        signal_strength = signal_data['signal']
        confidence = signal_data['confidence']
        strategy = signal_data['strategy']
        
        # Step 1: Confidence Filter
        print(f"1️⃣ Confidence Filter:")
        print(f"   Required: {min_confidence:.2f}")
        print(f"   Actual: {confidence:.2f}")
        
        if confidence < min_confidence:
            print(f"   ❌ REJECTED - Confidence too low")
            continue
        else:
            print(f"   ✅ PASSED - Confidence sufficient")
        
        # Step 2: Position Sizing
        print(f"2️⃣ Position Sizing:")
        portfolio_value = portfolio.get_portfolio_value({symbol: current_price})
        max_position_value = portfolio_value * config.max_position_size
        
        # Scale by confidence and signal strength
        position_value = max_position_value * confidence * abs(signal_strength)
        quantity = position_value / current_price
        
        print(f"   Portfolio Value: ${portfolio_value:,.2f}")
        print(f"   Max Position (20%): ${max_position_value:,.2f}")
        print(f"   Confidence Scaling: {confidence:.2f}")
        print(f"   Signal Scaling: {abs(signal_strength):.2f}")
        print(f"   Final Position Value: ${position_value:,.2f}")
        print(f"   Quantity: {quantity:.6f} {symbol}")
        
        # Step 3: Cost Calculation
        print(f"3️⃣ Cost Calculation:")
        notional = quantity * current_price
        commission = notional * config.commission
        slippage = notional * config.slippage
        total_cost = commission + slippage
        
        print(f"   Notional Value: ${notional:,.2f}")
        print(f"   Commission (0.1%): ${commission:.2f}")
        print(f"   Slippage (0.05%): ${slippage:.2f}")
        print(f"   Total Costs: ${total_cost:.2f}")
        
        # Step 4: Cash Check & Execution
        print(f"4️⃣ Trade Execution:")
        side = "BUY" if signal_strength > 0 else "SELL"
        
        if side == "BUY":
            required_cash = notional + total_cost
            print(f"   Side: {side}")
            print(f"   Required Cash: ${required_cash:,.2f}")
            print(f"   Available Cash: ${portfolio.cash:,.2f}")
            
            if required_cash <= portfolio.cash:
                print(f"   ✅ EXECUTING TRADE")
                
                # Execute the trade
                success = portfolio.execute_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=current_price,
                    strategy=strategy,
                    confidence=confidence,
                    timestamp=timestamp
                )
                
                if success:
                    print(f"   🎉 TRADE SUCCESSFUL!")
                    print(f"      {side} {quantity:.6f} {symbol} @ ${current_price:,.2f}")
                    
                    # Show updated portfolio
                    new_cash = portfolio.cash
                    new_portfolio_value = portfolio.get_portfolio_value({symbol: current_price})
                    
                    print(f"   📊 Updated Portfolio:")
                    print(f"      Cash: ${new_cash:,.2f}")
                    print(f"      Positions: {len(portfolio.positions)}")
                    print(f"      Total Value: ${new_portfolio_value:,.2f}")
                else:
                    print(f"   ❌ TRADE FAILED")
            else:
                print(f"   ❌ INSUFFICIENT CASH")
        else:
            print(f"   Side: {side}")
            if symbol in portfolio.positions:
                available_quantity = portfolio.positions[symbol].quantity
                print(f"   Available Quantity: {available_quantity:.6f}")
                if quantity <= available_quantity:
                    print(f"   ✅ EXECUTING SELL")
                    
                    success = portfolio.execute_trade(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=current_price,
                        strategy=strategy,
                        confidence=confidence,
                        timestamp=timestamp
                    )
                    
                    if success:
                        print(f"   🎉 SELL SUCCESSFUL!")
                else:
                    print(f"   ❌ INSUFFICIENT POSITION")
            else:
                print(f"   ❌ NO POSITION TO SELL")
    
    # Final Summary
    print(f"\n{'='*60}")
    print(f"📊 FINAL TRADE SUMMARY")
    print(f"{'='*60}")
    
    total_trades = len(portfolio.trade_history)
    print(f"Total Trades Executed: {total_trades}")
    
    if total_trades > 0:
        print(f"\n📋 Trade History:")
        for i, trade in enumerate(portfolio.trade_history, 1):
            print(f"   Trade {i}:")
            print(f"      Strategy: {trade.strategy}")
            print(f"      Action: {trade.side}")
            print(f"      Quantity: {trade.quantity:.6f}")
            print(f"      Price: ${trade.price:,.2f}")
            print(f"      Confidence: {trade.confidence:.2f}")
            print(f"      Commission: ${trade.commission:.2f}")
            print(f"      Slippage: ${trade.slippage:.2f}")
    
    # Portfolio Summary
    final_value = portfolio.get_portfolio_value({symbol: current_price})
    pnl = final_value - config.initial_capital
    pnl_pct = (pnl / config.initial_capital) * 100
    
    print(f"\n💰 Portfolio Summary:")
    print(f"   Initial Capital: ${config.initial_capital:,.2f}")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    print(f"   Cash: ${portfolio.cash:,.2f}")
    print(f"   Positions: {len(portfolio.positions)}")
    
    if portfolio.positions:
        print(f"\n📈 Current Positions:")
        for symbol, position in portfolio.positions.items():
            current_value = position.quantity * current_price
            cost_basis = position.quantity * position.avg_price
            unrealized_pnl = current_value - cost_basis
            
            print(f"   {symbol}:")
            print(f"      Quantity: {position.quantity:.6f}")
            print(f"      Avg Price: ${position.avg_price:,.2f}")
            print(f"      Current Value: ${current_value:,.2f}")
            print(f"      Unrealized P&L: ${unrealized_pnl:+,.2f}")
    
    print(f"\n🎯 Key Trade Creation Factors:")
    print(f"   1. Signal Strength: Determines buy/sell direction")
    print(f"   2. Confidence Level: Must exceed minimum threshold")
    print(f"   3. Position Sizing: Scaled by confidence × signal strength")
    print(f"   4. Risk Management: Limited to max position size")
    print(f"   5. Transaction Costs: Commission + slippage applied")
    print(f"   6. Cash Management: Prevents over-leveraging")


if __name__ == "__main__":
    demonstrate_trade_creation()