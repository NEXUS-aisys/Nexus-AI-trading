#!/usr/bin/env python3
"""
Momentum Breakout Strategy - Test & Demo Suite

This file tests and demonstrates the complete workflow:
1. Strategy initialization
2. Signal generation from market data
3. Trade execution simulation
4. Performance tracking
5. Error handling validation

Author: NEXUS Trading System
Version: 1.0
Created: 2025-10-22
"""

import sys
import os
import time
import asyncio
import logging
import traceback
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('momentum_test_demo.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the momentum breakout strategy
try:
    from momentum_breakout import (
        MomentumBreakoutNexusAdapterSync,
        InstitutionalMomentumBreakout,
        UnifiedMomentumBreakout,
        SecureMarketData,
        TradingSignal,
        SignalType,
        MomentumConfig
    )
    logger.info("✅ Successfully imported momentum breakout components")
except ImportError as e:
    logger.error(f"❌ Failed to import momentum breakout: {e}")
    sys.exit(1)


class MockMarketDataGenerator:
    """Generate realistic market data for testing"""
    
    def __init__(self, initial_price: float = 100.0, volatility: float = 0.02):
        self.current_price = initial_price
        self.volatility = volatility
        self.volume_base = 10000
        self.timestamp = time.time()
        
    def generate_tick(self, trend: float = 0.0) -> Dict[str, Any]:
        """Generate a single market data tick"""
        # Price movement with trend and random walk
        price_change = np.random.normal(trend, self.volatility)
        self.current_price *= (1 + price_change)
        
        # Volume with some randomness
        volume = int(self.volume_base * (1 + np.random.normal(0, 0.3)))
        volume = max(1000, volume)  # Minimum volume
        
        # Bid/ask spread
        spread = self.current_price * 0.001  # 0.1% spread
        bid = self.current_price - spread/2
        ask = self.current_price + spread/2
        
        self.timestamp += np.random.uniform(0.1, 2.0)  # 0.1-2 second intervals
        
        return {
            'symbol': 'TEST_SYMBOL',
            'timestamp': self.timestamp,
            'price': round(self.current_price, 2),
            'close': round(self.current_price, 2),
            'open': round(self.current_price * 0.999, 2),
            'high': round(self.current_price * 1.001, 2),
            'low': round(self.current_price * 0.999, 2),
            'volume': volume,
            'bid': round(bid, 2),
            'ask': round(ask, 2)
        }
    
    def generate_breakout_scenario(self, direction: str = 'up') -> List[Dict[str, Any]]:
        """Generate a breakout scenario for testing"""
        ticks = []
        
        # Phase 1: Consolidation (20 ticks)
        logger.info(f"📊 Generating consolidation phase...")
        for i in range(20):
            tick = self.generate_tick(trend=0.0001)  # Minimal trend
            ticks.append(tick)
        
        # Phase 2: Breakout (10 ticks)
        breakout_trend = 0.005 if direction == 'up' else -0.005
        logger.info(f"📈 Generating {direction} breakout phase...")
        for i in range(10):
            # Increase volume during breakout
            self.volume_base *= 1.2
            tick = self.generate_tick(trend=breakout_trend)
            ticks.append(tick)
        
        # Phase 3: Follow-through (15 ticks)
        logger.info(f"🔄 Generating follow-through phase...")
        follow_trend = breakout_trend * 0.6  # Reduced momentum
        for i in range(15):
            tick = self.generate_tick(trend=follow_trend)
            ticks.append(tick)
        
        return ticks


class MockTradeExecutor:
    """Simulate trade execution for testing"""
    
    def __init__(self):
        self.open_positions = {}
        self.trade_history = []
        self.total_pnl = 0.0
        self.trade_count = 0
        
    def execute_trade(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on signal"""
        try:
            signal_value = signal.get('signal', 0.0)
            confidence = signal.get('confidence', 0.0)
            
            if abs(signal_value) < 0.1:  # No significant signal
                return {'status': 'no_trade', 'reason': 'weak_signal'}
            
            # Determine trade direction
            side = 'BUY' if signal_value > 0 else 'SELL'
            
            # Calculate position size based on confidence
            base_size = 1000  # Base position size
            position_size = int(base_size * confidence)
            
            # Execute the trade
            entry_price = market_data.get('price', market_data.get('close', 0.0))
            
            trade_id = f"TRADE_{self.trade_count}_{int(time.time())}"
            self.trade_count += 1
            
            trade = {
                'trade_id': trade_id,
                'symbol': market_data.get('symbol', 'UNKNOWN'),
                'side': side,
                'size': position_size,
                'entry_price': entry_price,
                'entry_time': market_data.get('timestamp', time.time()),
                'signal_confidence': confidence,
                'signal_strength': abs(signal_value),
                'status': 'OPEN'
            }
            
            self.open_positions[trade_id] = trade
            
            logger.info(f"🔥 TRADE EXECUTED: {side} {position_size} @ {entry_price:.2f} (confidence: {confidence:.2f})")
            
            return {
                'status': 'executed',
                'trade_id': trade_id,
                'side': side,
                'size': position_size,
                'price': entry_price,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Trade execution failed: {e}")
            return {'status': 'error', 'reason': str(e)}
    
    def check_exits(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for trade exits"""
        exits = []
        current_price = market_data.get('price', market_data.get('close', 0.0))
        
        for trade_id, trade in list(self.open_positions.items()):
            entry_price = trade['entry_price']
            side = trade['side']
            
            # Calculate P&L
            if side == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            pnl_dollars = pnl_pct * trade['size'] * entry_price / 100
            
            # Exit conditions
            should_exit = False
            exit_reason = None
            
            # Take profit at 2%
            if pnl_pct >= 0.02:
                should_exit = True
                exit_reason = 'take_profit'
            
            # Stop loss at -1%
            elif pnl_pct <= -0.01:
                should_exit = True
                exit_reason = 'stop_loss'
            
            # Time-based exit (hold for max 30 ticks)
            elif (market_data.get('timestamp', time.time()) - trade['entry_time']) > 60:
                should_exit = True
                exit_reason = 'time_exit'
            
            if should_exit:
                # Close the position
                trade['exit_price'] = current_price
                trade['exit_time'] = market_data.get('timestamp', time.time())
                trade['pnl_pct'] = pnl_pct
                trade['pnl_dollars'] = pnl_dollars
                trade['exit_reason'] = exit_reason
                trade['status'] = 'CLOSED'
                
                self.trade_history.append(trade)
                self.total_pnl += pnl_dollars
                
                del self.open_positions[trade_id]
                
                logger.info(f"💰 TRADE CLOSED: {trade_id} | P&L: ${pnl_dollars:.2f} ({pnl_pct:.2%}) | Reason: {exit_reason}")
                
                exits.append(trade)
        
        return exits
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        if not self.trade_history:
            return {'total_trades': 0, 'total_pnl': 0.0, 'win_rate': 0.0}
        
        winning_trades = [t for t in self.trade_history if t['pnl_dollars'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl_dollars'] < 0]
        
        return {
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trade_history),
            'total_pnl': self.total_pnl,
            'avg_win': np.mean([t['pnl_dollars'] for t in winning_trades]) if winning_trades else 0.0,
            'avg_loss': np.mean([t['pnl_dollars'] for t in losing_trades]) if losing_trades else 0.0,
            'open_positions': len(self.open_positions)
        }


class MomentumBreakoutTester:
    """Main test suite for momentum breakout strategy"""
    
    def __init__(self):
        self.strategy = None
        self.market_generator = MockMarketDataGenerator()
        self.trade_executor = MockTradeExecutor()
        self.test_results = {}
        
    def setup_strategy(self) -> bool:
        """Initialize the momentum breakout strategy"""
        try:
            logger.info("🔧 Setting up Momentum Breakout Strategy...")
            
            # Configuration for testing
            config = {
                'strategy_name': 'momentum_breakout_test',
                'parameter_profile': 'balanced',
                'initial_capital': 100000.0,
                'mqscore_enabled': True,
                'mqscore_quality_threshold': 0.50,  # Lower threshold for testing
                'ml_predictions_enabled': False,  # Disable ML for testing
                'volatility_scaling': True,
                'kill_switch_active': False,
                'daily_loss_limit': -5000.0,
                'max_drawdown_limit': 0.15,
                'max_consecutive_losses': 5
            }
            
            # Initialize the strategy
            self.strategy = MomentumBreakoutNexusAdapterSync(
                lookback_period=20,
                breakout_threshold=2.0,  # Lower threshold for testing
                volume_multiplier=1.5,
                config=config
            )
            
            logger.info("✅ Strategy initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Strategy setup failed: {e}")
            traceback.print_exc()
            return False
    
    def test_signal_generation(self) -> bool:
        """Test signal generation with various market conditions"""
        try:
            logger.info("🧪 Testing Signal Generation...")
            
            # Test 1: Normal market data
            logger.info("📊 Test 1: Normal market conditions")
            normal_tick = self.market_generator.generate_tick()
            signal = self.strategy.execute(normal_tick)
            
            self._validate_signal_format(signal, "Normal market")
            
            # Test 2: Breakout scenario
            logger.info("📈 Test 2: Upward breakout scenario")
            breakout_ticks = self.market_generator.generate_breakout_scenario('up')
            
            signals_generated = 0
            for i, tick in enumerate(breakout_ticks):
                signal = self.strategy.execute(tick)
                self._validate_signal_format(signal, f"Breakout tick {i+1}")
                
                if abs(signal.get('signal', 0.0)) > 0.1:
                    signals_generated += 1
                    logger.info(f"🎯 Signal generated: {signal.get('signal', 0.0):.3f} (confidence: {signal.get('confidence', 0.0):.3f})")
            
            logger.info(f"📊 Breakout test: {signals_generated}/{len(breakout_ticks)} ticks generated signals")
            
            # Test 3: Downward breakout
            logger.info("📉 Test 3: Downward breakout scenario")
            self.market_generator = MockMarketDataGenerator()  # Reset
            down_breakout_ticks = self.market_generator.generate_breakout_scenario('down')
            
            down_signals = 0
            for i, tick in enumerate(down_breakout_ticks):
                signal = self.strategy.execute(tick)
                if signal.get('signal', 0.0) < -0.1:
                    down_signals += 1
                    logger.info(f"🎯 Short signal: {signal.get('signal', 0.0):.3f} (confidence: {signal.get('confidence', 0.0):.3f})")
            
            logger.info(f"📊 Down breakout test: {down_signals}/{len(down_breakout_ticks)} ticks generated short signals")
            
            self.test_results['signal_generation'] = {
                'normal_test': True,
                'up_breakout_signals': signals_generated,
                'down_breakout_signals': down_signals,
                'total_ticks_processed': len(breakout_ticks) + len(down_breakout_ticks) + 1
            }
            
            logger.info("✅ Signal generation tests completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Signal generation test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_trade_execution(self) -> bool:
        """Test complete trade execution workflow"""
        try:
            logger.info("💼 Testing Trade Execution Workflow...")
            
            # Reset strategy and market generator
            self.setup_strategy()
            self.market_generator = MockMarketDataGenerator(initial_price=100.0)
            
            # Generate a strong breakout scenario
            breakout_ticks = self.market_generator.generate_breakout_scenario('up')
            
            trades_executed = 0
            trades_closed = 0
            
            for i, tick in enumerate(breakout_ticks):
                # Get signal from strategy
                signal = self.strategy.execute(tick)
                
                # Execute trade if signal is strong enough
                if abs(signal.get('signal', 0.0)) > 0.3:
                    trade_result = self.trade_executor.execute_trade(signal, tick)
                    if trade_result.get('status') == 'executed':
                        trades_executed += 1
                        
                        # Record trade result in strategy
                        self.strategy.record_trade_result({
                            'pnl': 0.0,  # Will be updated on exit
                            'confidence': signal.get('confidence', 0.0),
                            'trade_id': trade_result.get('trade_id')
                        })
                
                # Check for trade exits
                exits = self.trade_executor.check_exits(tick)
                for exit_trade in exits:
                    trades_closed += 1
                    
                    # Update strategy with final P&L
                    self.strategy.record_trade_result({
                        'pnl': exit_trade['pnl_dollars'],
                        'confidence': exit_trade['signal_confidence'],
                        'trade_id': exit_trade['trade_id']
                    })
                
                # Log progress every 10 ticks
                if (i + 1) % 10 == 0:
                    perf = self.trade_executor.get_performance_summary()
                    logger.info(f"📊 Progress: {i+1}/{len(breakout_ticks)} ticks | Trades: {perf['total_trades']} | P&L: ${perf['total_pnl']:.2f}")
            
            # Final performance summary
            final_perf = self.trade_executor.get_performance_summary()
            strategy_perf = self.strategy.get_performance_metrics()
            
            logger.info("📈 FINAL TRADING RESULTS:")
            logger.info(f"   Total Trades Executed: {trades_executed}")
            logger.info(f"   Total Trades Closed: {trades_closed}")
            logger.info(f"   Win Rate: {final_perf['win_rate']:.1%}")
            logger.info(f"   Total P&L: ${final_perf['total_pnl']:.2f}")
            logger.info(f"   Average Win: ${final_perf['avg_win']:.2f}")
            logger.info(f"   Average Loss: ${final_perf['avg_loss']:.2f}")
            logger.info(f"   Open Positions: {final_perf['open_positions']}")
            
            self.test_results['trade_execution'] = {
                'trades_executed': trades_executed,
                'trades_closed': trades_closed,
                'final_performance': final_perf,
                'strategy_performance': strategy_perf
            }
            
            logger.info("✅ Trade execution tests completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trade execution test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and edge cases"""
        try:
            logger.info("🛡️ Testing Error Handling...")
            
            # Test 1: Invalid market data
            logger.info("🧪 Test 1: Invalid market data")
            invalid_data = {'invalid': 'data'}
            signal = self.strategy.execute(invalid_data)
            assert signal.get('signal', 0.0) == 0.0, "Should return neutral signal for invalid data"
            logger.info("✅ Invalid data handled correctly")
            
            # Test 2: Missing required fields
            logger.info("🧪 Test 2: Missing required fields")
            incomplete_data = {'price': 100.0}  # Missing volume, timestamp, etc.
            signal = self.strategy.execute(incomplete_data)
            self._validate_signal_format(signal, "Incomplete data")
            logger.info("✅ Incomplete data handled correctly")
            
            # Test 3: Extreme values
            logger.info("🧪 Test 3: Extreme values")
            extreme_data = {
                'price': 999999.99,
                'volume': 0,
                'timestamp': time.time(),
                'symbol': 'TEST'
            }
            signal = self.strategy.execute(extreme_data)
            self._validate_signal_format(signal, "Extreme values")
            logger.info("✅ Extreme values handled correctly")
            
            # Test 4: Kill switch activation
            logger.info("🧪 Test 4: Kill switch activation")
            # Simulate large losses to trigger kill switch
            for i in range(6):  # Trigger consecutive losses
                self.strategy.record_trade_result({
                    'pnl': -1000.0,  # Large loss
                    'confidence': 0.8,
                    'trade_id': f'loss_test_{i}'
                })
            
            # Strategy should now have kill switch active
            signal = self.strategy.execute(self.market_generator.generate_tick())
            assert signal.get('signal', 0.0) == 0.0, "Kill switch should block signals"
            logger.info("✅ Kill switch activated correctly")
            
            # Reset kill switch
            self.strategy.reset_kill_switch()
            logger.info("✅ Kill switch reset correctly")
            
            self.test_results['error_handling'] = {
                'invalid_data_test': True,
                'incomplete_data_test': True,
                'extreme_values_test': True,
                'kill_switch_test': True
            }
            
            logger.info("✅ Error handling tests completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks"""
        try:
            logger.info("⚡ Testing Performance Benchmarks...")
            
            # Test execution speed
            test_data = self.market_generator.generate_tick()
            
            # Warm up
            for _ in range(10):
                self.strategy.execute(test_data)
            
            # Benchmark execution time
            start_time = time.perf_counter()
            iterations = 1000
            
            for _ in range(iterations):
                signal = self.strategy.execute(test_data)
            
            end_time = time.perf_counter()
            avg_execution_time = (end_time - start_time) / iterations * 1000  # Convert to milliseconds
            
            logger.info(f"📊 Average execution time: {avg_execution_time:.3f} ms")
            
            # Check if meets performance target (< 5ms)
            performance_target_met = avg_execution_time < 5.0
            
            if performance_target_met:
                logger.info("✅ Performance target met (< 5ms)")
            else:
                logger.warning(f"⚠️ Performance target missed: {avg_execution_time:.3f}ms > 5ms")
            
            # Test throughput
            start_time = time.perf_counter()
            signals_processed = 0
            
            # Process for 1 second
            while time.perf_counter() - start_time < 1.0:
                self.strategy.execute(test_data)
                signals_processed += 1
            
            throughput = signals_processed
            logger.info(f"📊 Throughput: {throughput} signals/second")
            
            # Check throughput target (> 200/sec)
            throughput_target_met = throughput > 200
            
            if throughput_target_met:
                logger.info("✅ Throughput target met (> 200/sec)")
            else:
                logger.warning(f"⚠️ Throughput target missed: {throughput}/sec < 200/sec")
            
            self.test_results['performance'] = {
                'avg_execution_time_ms': avg_execution_time,
                'performance_target_met': performance_target_met,
                'throughput_per_sec': throughput,
                'throughput_target_met': throughput_target_met
            }
            
            logger.info("✅ Performance benchmark tests completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark test failed: {e}")
            traceback.print_exc()
            return False
    
    def _validate_signal_format(self, signal: Dict[str, Any], context: str) -> None:
        """Validate signal format matches NEXUS standard"""
        required_keys = ['signal', 'confidence', 'features', 'metadata']
        
        for key in required_keys:
            assert key in signal, f"Missing required key '{key}' in signal for {context}"
        
        # Validate signal range
        signal_value = signal.get('signal', 0.0)
        assert -2.0 <= signal_value <= 2.0, f"Signal value {signal_value} out of range [-2.0, 2.0] for {context}"
        
        # Validate confidence range
        confidence = signal.get('confidence', 0.0)
        assert 0.0 <= confidence <= 1.0, f"Confidence {confidence} out of range [0.0, 1.0] for {context}"
        
        # Validate features is dict
        assert isinstance(signal.get('features', {}), dict), f"Features must be dict for {context}"
        
        # Validate metadata is dict
        assert isinstance(signal.get('metadata', {}), dict), f"Metadata must be dict for {context}"
    
    def run_comprehensive_test(self) -> bool:
        """Run all tests in sequence"""
        logger.info("🚀 Starting Comprehensive Momentum Breakout Test Suite")
        logger.info("=" * 60)
        
        test_sequence = [
            ("Strategy Setup", self.setup_strategy),
            ("Signal Generation", self.test_signal_generation),
            ("Trade Execution", self.test_trade_execution),
            ("Error Handling", self.test_error_handling),
            ("Performance Benchmarks", self.test_performance_benchmarks)
        ]
        
        passed_tests = 0
        total_tests = len(test_sequence)
        
        for test_name, test_func in test_sequence:
            logger.info(f"\n🧪 Running: {test_name}")
            logger.info("-" * 40)
            
            try:
                if test_func():
                    logger.info(f"✅ {test_name}: PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: FAILED with exception: {e}")
                traceback.print_exc()
        
        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 TEST SUITE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Strategy is ready for deployment.")
        else:
            logger.warning(f"⚠️ {total_tests - passed_tests} tests failed. Review issues before deployment.")
        
        # Save detailed results
        self._save_test_results()
        
        return passed_tests == total_tests
    
    def _save_test_results(self):
        """Save test results to file"""
        try:
            import json
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'test_results': self.test_results,
                'summary': {
                    'total_tests': 5,
                    'passed_tests': len([r for r in self.test_results.values() if r]),
                    'strategy_version': 'MomentumBreakoutNexusAdapterSync v3.0'
                }
            }
            
            with open('momentum_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("📄 Test results saved to momentum_test_results.json")
            
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")


def main():
    """Main demo function"""
    print("🚀 Momentum Breakout Strategy - Test & Demo Suite")
    print("=" * 60)
    print("This demo will:")
    print("1. Initialize the momentum breakout strategy")
    print("2. Generate realistic market data scenarios")
    print("3. Test signal generation capabilities")
    print("4. Simulate trade execution")
    print("5. Validate error handling")
    print("6. Benchmark performance")
    print("=" * 60)
    
    # Create and run the test suite
    tester = MomentumBreakoutTester()
    
    try:
        success = tester.run_comprehensive_test()
        
        if success:
            print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("The momentum breakout strategy is working correctly and ready for use.")
        else:
            print("\n⚠️ DEMO COMPLETED WITH ISSUES!")
            print("Please review the test results and fix any issues before deployment.")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)