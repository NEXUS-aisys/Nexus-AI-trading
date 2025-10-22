#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI Backtesting Demo
Quick demonstration of the backtesting capabilities
Version: 1.0.0
"""

import asyncio
import sys
from datetime import datetime
from nexus_backtester import NexusBacktester, BacktestConfig


async def run_demo_backtest():
    """Run a quick demo backtest to test the system."""
    
    print("🚀 NEXUS AI Backtesting Demo")
    print("=" * 50)
    print("This demo will run a quick backtest to validate the system.")
    print("Note: This uses simulated market data for demonstration.")
    print("=" * 50)
    
    # Create a simple demo configuration
    config = BacktestConfig(
        start_date="2024-09-01",
        end_date="2024-10-22",
        initial_capital=50000.0,
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="1h",
        commission=0.001,
        slippage=0.0005,
        min_confidence=0.60,
        mqscore_threshold=0.60,
        use_mqscore_filter=True,
        enable_ml_models=True,
        max_position_size=0.10,
        max_daily_loss=0.02,
        max_drawdown=0.15
    )
    
    print(f"📅 Period: {config.start_date} to {config.end_date}")
    print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    print(f"📊 Symbols: {', '.join(config.symbols)}")
    print(f"⏰ Timeframe: {config.timeframe}")
    print(f"🎯 Min Confidence: {config.min_confidence:.2f}")
    print(f"🔍 MQScore Filter: {'Enabled' if config.use_mqscore_filter else 'Disabled'}")
    print(f"🤖 ML Models: {'Enabled' if config.enable_ml_models else 'Disabled'}")
    print("\n⏳ Starting backtest...")
    
    try:
        # Create and run backtester
        backtester = NexusBacktester(config)
        
        start_time = datetime.now()
        results = await backtester.run_backtest()
        end_time = datetime.now()
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 DEMO BACKTEST RESULTS")
        print("=" * 60)
        print(f"⏱️  Execution Time: {end_time - start_time}")
        print(f"💹 Total Return: {results.total_return:.2%}")
        print(f"📈 Annualized Return: {results.annualized_return:.2%}")
        print(f"📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
        print(f"🎯 Win Rate: {results.win_rate:.2%}")
        print(f"🔢 Total Trades: {results.total_trades}")
        print(f"💰 Profit Factor: {results.profit_factor:.2f}")
        
        if results.total_trades > 0:
            print(f"✅ Average Win: ${results.avg_win:.2f}")
            print(f"❌ Average Loss: ${results.avg_loss:.2f}")
        
        print("=" * 60)
        
        # Strategy performance
        if results.strategy_trades:
            print("\n🎯 STRATEGY PERFORMANCE:")
            print("-" * 50)
            for strategy, trades in results.strategy_trades.items():
                if trades > 0:
                    win_rate = results.strategy_win_rates.get(strategy, 0)
                    print(f"{strategy:30} | {trades:3d} trades | {win_rate:.1%} win rate")
        
        # MQScore analysis
        if results.mqscore_stats['trades_with_mqscore'] > 0:
            print(f"\n🔍 MQSCORE ANALYSIS:")
            print("-" * 50)
            print(f"Average MQScore: {results.mqscore_stats['avg_mqscore']:.3f}")
            print(f"MQScore Range: {results.mqscore_stats['min_mqscore']:.3f} - {results.mqscore_stats['max_mqscore']:.3f}")
            print(f"Trades with MQScore: {results.mqscore_stats['trades_with_mqscore']}")
        
        # Generate demo report
        report_filename = f"nexus_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        backtester.generate_report(report_filename)
        
        print(f"\n📄 Demo Report Generated: {report_filename}")
        print("\n✅ Demo backtest completed successfully!")
        
        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        print("-" * 50)
        
        if results.total_return > 0.05:
            print("🟢 Return: Excellent (>5%)")
        elif results.total_return > 0:
            print("🟡 Return: Positive but modest")
        else:
            print("🔴 Return: Negative - needs optimization")
        
        if results.sharpe_ratio > 1.5:
            print("🟢 Sharpe Ratio: Excellent (>1.5)")
        elif results.sharpe_ratio > 1.0:
            print("🟡 Sharpe Ratio: Good (>1.0)")
        elif results.sharpe_ratio > 0.5:
            print("🟡 Sharpe Ratio: Acceptable (>0.5)")
        else:
            print("🔴 Sharpe Ratio: Poor - high volatility relative to returns")
        
        if abs(results.max_drawdown) < 0.05:
            print("🟢 Drawdown: Excellent (<5%)")
        elif abs(results.max_drawdown) < 0.10:
            print("🟡 Drawdown: Acceptable (<10%)")
        else:
            print("🔴 Drawdown: High - risk management needed")
        
        if results.total_trades > 50:
            print("🟢 Activity: Good trading frequency")
        elif results.total_trades > 20:
            print("🟡 Activity: Moderate trading frequency")
        else:
            print("🟡 Activity: Low trading frequency - may need parameter tuning")
        
        print("\n💡 NEXT STEPS:")
        print("-" * 50)
        print("1. Run full backtests with: python run_backtest.py")
        print("2. Try different scenarios: python run_backtest.py --list-scenarios")
        print("3. Optimize parameters: python optimize_parameters.py")
        print("4. Test with real historical data by connecting to data APIs")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Demo backtest failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure NEXUS AI system is properly installed")
        print("2. Check that all 20 strategies are loading correctly")
        print("3. Verify MQScore 6D Engine is available")
        print("4. Run: python nexus_ai.py to test system initialization")
        raise


def main():
    """Main function for demo."""
    try:
        print("Starting NEXUS AI Backtesting Demo...")
        asyncio.run(run_demo_backtest())
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()