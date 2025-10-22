#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI Aggressive Backtest Test
Test with lower thresholds to see trading activity
"""

import asyncio
from nexus_backtester import NexusBacktester, BacktestConfig

async def run_aggressive_test():
    """Run backtest with more aggressive parameters to generate trades."""
    
    print("🚀 NEXUS AI Aggressive Backtest Test")
    print("=" * 50)
    print("Testing with lower thresholds to generate trading activity...")
    
    # More aggressive configuration
    config = BacktestConfig(
        start_date="2024-10-01",
        end_date="2024-10-22",
        initial_capital=50000.0,
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="1h",
        commission=0.001,
        slippage=0.0005,
        min_confidence=0.45,  # Lower threshold
        mqscore_threshold=0.45,  # Lower threshold
        use_mqscore_filter=True,
        enable_ml_models=True,
        max_position_size=0.15,  # Larger positions
        max_daily_loss=0.03,
        max_drawdown=0.20
    )
    
    print(f"📅 Period: {config.start_date} to {config.end_date}")
    print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    print(f"🎯 Min Confidence: {config.min_confidence:.2f} (LOWERED)")
    print(f"🔍 MQScore Threshold: {config.mqscore_threshold:.2f} (LOWERED)")
    print(f"💼 Max Position Size: {config.max_position_size:.1%} (INCREASED)")
    print("\n⏳ Starting aggressive backtest...")
    
    try:
        backtester = NexusBacktester(config)
        results = await backtester.run_backtest()
        
        print("\n" + "=" * 60)
        print("📊 AGGRESSIVE BACKTEST RESULTS")
        print("=" * 60)
        print(f"💹 Total Return: {results.total_return:.2%}")
        print(f"📈 Annualized Return: {results.annualized_return:.2%}")
        print(f"📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
        print(f"🎯 Win Rate: {results.win_rate:.2%}")
        print(f"🔢 Total Trades: {results.total_trades}")
        print(f"💰 Profit Factor: {results.profit_factor:.2f}")
        print("=" * 60)
        
        if results.total_trades > 0:
            print(f"\n✅ SUCCESS: Generated {results.total_trades} trades!")
            print(f"📊 Strategy Activity:")
            for strategy, trades in results.strategy_trades.items():
                if trades > 0:
                    print(f"   {strategy}: {trades} trades")
        else:
            print(f"\n⚠️  Still no trades generated.")
            print(f"💡 Suggestions:")
            print(f"   - Lower confidence threshold further (try 0.30)")
            print(f"   - Lower MQScore threshold (try 0.30)")
            print(f"   - Check strategy signal generation logic")
            print(f"   - Use real historical data instead of simulated")
        
        # Generate report
        backtester.generate_report("nexus_aggressive_test.html")
        print(f"\n📄 Report saved: nexus_aggressive_test.html")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_aggressive_test())