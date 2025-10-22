#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI Backtest Runner
Easy-to-use script for running comprehensive backtests
Version: 1.0.0
"""

import asyncio
import json
import argparse
import sys
from datetime import datetime
from nexus_backtester import NexusBacktester, BacktestConfig


def load_config(scenario_name: str = "quick_test") -> BacktestConfig:
    """Load backtest configuration from JSON file."""
    try:
        with open('backtest_config.json', 'r') as f:
            config_data = json.load(f)
        
        if scenario_name not in config_data['backtest_scenarios']:
            print(f"❌ Scenario '{scenario_name}' not found in config")
            print(f"Available scenarios: {list(config_data['backtest_scenarios'].keys())}")
            sys.exit(1)
        
        scenario_config = config_data['backtest_scenarios'][scenario_name]
        
        # Create BacktestConfig object
        config = BacktestConfig(
            start_date=scenario_config['start_date'],
            end_date=scenario_config['end_date'],
            initial_capital=scenario_config['initial_capital'],
            symbols=scenario_config['symbols'],
            timeframe=scenario_config['timeframe'],
            commission=scenario_config['commission'],
            slippage=scenario_config['slippage'],
            min_confidence=scenario_config['min_confidence'],
            mqscore_threshold=scenario_config['mqscore_threshold'],
            use_mqscore_filter=scenario_config['use_mqscore_filter'],
            enable_ml_models=scenario_config['enable_ml_models'],
            max_position_size=scenario_config['max_position_size'],
            max_daily_loss=scenario_config['max_daily_loss'],
            max_drawdown=scenario_config['max_drawdown']
        )
        
        print(f"✅ Loaded configuration for scenario: {scenario_name}")
        return config
        
    except FileNotFoundError:
        print("❌ backtest_config.json not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)


async def run_single_backtest(scenario_name: str):
    """Run a single backtest scenario."""
    print(f"\n🚀 Starting NEXUS AI Backtest - Scenario: {scenario_name}")
    print("=" * 60)
    
    # Load configuration
    config = load_config(scenario_name)
    
    # Display configuration
    print(f"📅 Period: {config.start_date} to {config.end_date}")
    print(f"💰 Initial Capital: ${config.initial_capital:,.2f}")
    print(f"📊 Symbols: {', '.join(config.symbols)}")
    print(f"⏰ Timeframe: {config.timeframe}")
    print(f"🎯 Min Confidence: {config.min_confidence:.2f}")
    print(f"🔍 MQScore Threshold: {config.mqscore_threshold:.2f}")
    print(f"💼 Max Position Size: {config.max_position_size:.1%}")
    print("=" * 60)
    
    # Create and run backtester
    backtester = NexusBacktester(config)
    
    try:
        start_time = datetime.now()
        results = await backtester.run_backtest()
        end_time = datetime.now()
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 BACKTEST RESULTS")
        print("=" * 60)
        print(f"⏱️  Execution Time: {end_time - start_time}")
        print(f"💹 Total Return: {results.total_return:.2%}")
        print(f"📈 Annualized Return: {results.annualized_return:.2%}")
        print(f"📊 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown: {results.max_drawdown:.2%}")
        print(f"🎯 Win Rate: {results.win_rate:.2%}")
        print(f"🔢 Total Trades: {results.total_trades}")
        print(f"💰 Profit Factor: {results.profit_factor:.2f}")
        print(f"⚠️  VaR (95%): {results.var_95:.2%}")
        print(f"📊 Calmar Ratio: {results.calmar_ratio:.2f}")
        print("=" * 60)
        
        # Strategy breakdown
        if results.strategy_trades:
            print("\n🎯 STRATEGY PERFORMANCE:")
            print("-" * 40)
            for strategy, trades in results.strategy_trades.items():
                win_rate = results.strategy_win_rates.get(strategy, 0)
                print(f"{strategy:25} | {trades:3d} trades | {win_rate:.1%} win rate")
        
        # MQScore analysis
        if results.mqscore_stats['trades_with_mqscore'] > 0:
            print(f"\n🔍 MQSCORE ANALYSIS:")
            print("-" * 40)
            print(f"Average MQScore: {results.mqscore_stats['avg_mqscore']:.3f}")
            print(f"MQScore Range: {results.mqscore_stats['min_mqscore']:.3f} - {results.mqscore_stats['max_mqscore']:.3f}")
            print(f"Trades with MQScore: {results.mqscore_stats['trades_with_mqscore']}")
        
        # Generate reports
        report_filename = f"nexus_backtest_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        results_filename = f"nexus_results_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        backtester.generate_report(report_filename)
        
        # Save detailed results
        results_dict = {
            'scenario': scenario_name,
            'config': config.__dict__,
            'execution_time': str(end_time - start_time),
            'performance': {
                'total_return': results.total_return,
                'annualized_return': results.annualized_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'total_trades': results.total_trades,
                'var_95': results.var_95,
                'cvar_95': results.cvar_95,
                'calmar_ratio': results.calmar_ratio,
                'sortino_ratio': results.sortino_ratio
            },
            'strategy_performance': {
                'returns': results.strategy_returns,
                'trades': results.strategy_trades,
                'win_rates': results.strategy_win_rates
            },
            'mqscore_stats': results.mqscore_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"\n📄 Reports Generated:")
        print(f"   📊 HTML Report: {report_filename}")
        print(f"   💾 JSON Results: {results_filename}")
        print("\n✅ Backtest completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Backtest failed: {e}")
        raise


async def run_multiple_backtests(scenarios: list):
    """Run multiple backtest scenarios."""
    print(f"\n🚀 Running Multiple NEXUS AI Backtests")
    print(f"📋 Scenarios: {', '.join(scenarios)}")
    print("=" * 60)
    
    results = {}
    
    for scenario in scenarios:
        try:
            print(f"\n▶️  Running scenario: {scenario}")
            result = await run_single_backtest(scenario)
            results[scenario] = result
            print(f"✅ Completed scenario: {scenario}")
        except Exception as e:
            print(f"❌ Failed scenario {scenario}: {e}")
            results[scenario] = None
    
    # Generate comparison report
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Scenario':<20} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Trades':<8} {'Win Rate':<10}")
    print("-" * 80)
    
    for scenario, result in results.items():
        if result:
            print(f"{scenario:<20} {result.total_return:>8.2%} {result.sharpe_ratio:>7.2f} "
                  f"{result.max_drawdown:>9.2%} {result.total_trades:>7d} {result.win_rate:>9.2%}")
        else:
            print(f"{scenario:<20} {'FAILED':<10}")
    
    print("=" * 80)
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='NEXUS AI Backtesting System')
    parser.add_argument('--scenario', '-s', type=str, default='quick_test',
                       help='Backtest scenario name (default: quick_test)')
    parser.add_argument('--list-scenarios', '-l', action='store_true',
                       help='List available scenarios')
    parser.add_argument('--multiple', '-m', nargs='+',
                       help='Run multiple scenarios')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all available scenarios')
    
    args = parser.parse_args()
    
    # List scenarios
    if args.list_scenarios:
        try:
            with open('backtest_config.json', 'r') as f:
                config_data = json.load(f)
            
            print("\n📋 Available Backtest Scenarios:")
            print("-" * 40)
            for scenario_name, config in config_data['backtest_scenarios'].items():
                print(f"🎯 {scenario_name}")
                print(f"   Period: {config['start_date']} to {config['end_date']}")
                print(f"   Capital: ${config['initial_capital']:,.0f}")
                print(f"   Symbols: {', '.join(config['symbols'])}")
                print(f"   Timeframe: {config['timeframe']}")
                print()
            
            print("📊 Available Strategy Groups:")
            print("-" * 40)
            for group_name, strategies in config_data['strategy_groups'].items():
                print(f"🔧 {group_name}: {len(strategies)} strategies")
            
            return
            
        except Exception as e:
            print(f"❌ Error loading scenarios: {e}")
            return
    
    # Run backtests
    try:
        if args.all:
            # Run all scenarios
            with open('backtest_config.json', 'r') as f:
                config_data = json.load(f)
            scenarios = list(config_data['backtest_scenarios'].keys())
            asyncio.run(run_multiple_backtests(scenarios))
            
        elif args.multiple:
            # Run multiple specified scenarios
            asyncio.run(run_multiple_backtests(args.multiple))
            
        else:
            # Run single scenario
            asyncio.run(run_single_backtest(args.scenario))
            
    except KeyboardInterrupt:
        print("\n⏹️  Backtest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()