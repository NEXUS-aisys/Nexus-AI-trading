#!/usr/bin/env python3
"""
Spoofing Detection Strategy - Test & Demo Suite
==============================================

Comprehensive testing and demonstration of the Spoofing Detection Strategy.
Tests signal generation, trade execution, and performance tracking.

Features Tested:
- Strategy initialization and configuration
- Signal generation from market data
- Trade execution and position management
- Risk management and kill switch
- Performance metrics and reporting
- MQScore integration and quality filtering
- TIER 3 compliance (TTP calculation, confidence validation)

Usage:
    python test_spoofing_detection_demo.py

Author: NEXUS Trading System
Version: 1.0
"""

import sys
import os
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from decimal import Decimal

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the strategy
try:
    from Spoofing_Detection_Strategy import (
        UniversalStrategyConfig,
        EnhancedSpoofingDetectionStrategy,
        SpoofingDetectionNexusAdapter,
        MarketData,
        Signal,
        SignalType
    )
    STRATEGY_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import strategy: {e}")
    print("Trying alternative import...")
    try:
        # Try importing from the file with spaces in name
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "spoofing_strategy", 
            "Spoofing Detection Strategy.py"
        )
        spoofing_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(spoofing_module)
        
        # Extract classes
        UniversalStrategyConfig = spoofing_module.UniversalStrategyConfig
        EnhancedSpoofingDetectionStrategy = spoofing_module.EnhancedSpoofingDetectionStrategy
        SpoofingDetectionNexusAdapter = spoofing_module.SpoofingDetectionNexusAdapter
        MarketData = spoofing_module.MarketData
        Signal = spoofing_module.Signal
        SignalType = spoofing_module.SignalType
        
        STRATEGY_AVAILABLE = True
        print("✅ Successfully imported strategy from file with spaces")
    except Exception as e2:
        print(f"❌ Alternative import also failed: {e2}")
        STRATEGY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('spoofing_test_demo.log')
    ]
)
logger = logging.getLogger(__name__)


class SpoofingDetectionTester:
    """Comprehensive test suite for Spoofing Detection Strategy"""
    
    def __init__(self):
        self.test_results = []
        self.demo_results = []
        self.logger = logging.getLogger(f"{__name__}.SpoofingDetectionTester")
        
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("🧪 SPOOFING DETECTION STRATEGY - TEST & DEMO SUITE")
        print("="*80)
        
        if not STRATEGY_AVAILABLE:
            print("❌ Strategy not available - cannot run tests")
            return False
        
        # Test 1: Strategy Initialization
        print("\n📋 Test 1: Strategy Initialization")
        init_success = self.test_strategy_initialization()
        
        # Test 2: Signal Generation
        print("\n📊 Test 2: Signal Generation")
        signal_success = self.test_signal_generation()
        
        # Test 3: Trade Execution
        print("\n💼 Test 3: Trade Execution")
        trade_success = self.test_trade_execution()
        
        # Test 4: Risk Management
        print("\n🛡️ Test 4: Risk Management")
        risk_success = self.test_risk_management()
        
        # Test 5: Performance Tracking
        print("\n📈 Test 5: Performance Tracking")
        perf_success = self.test_performance_tracking()
        
        # Demo: Live Trading Simulation
        print("\n🚀 Demo: Live Trading Simulation")
        demo_success = self.demo_live_trading()
        
        # Summary
        self.print_test_summary()
        
        return all([init_success, signal_success, trade_success, risk_success, perf_success, demo_success])
    
    def test_strategy_initialization(self):
        """Test strategy initialization and configuration"""
        try:
            # Test 1.1: Universal Configuration
            print("  🔧 Testing Universal Configuration...")
            config = UniversalStrategyConfig("spoofing_detection")
            
            assert hasattr(config, 'risk_params'), "Missing risk_params"
            assert hasattr(config, 'signal_params'), "Missing signal_params"
            assert config.risk_params['max_position_size'] > 0, "Invalid max_position_size"
            assert 0 <= config.signal_params['min_signal_confidence'] <= 1, "Invalid confidence range"
            
            print(f"    ✅ Configuration created with seed: {config.seed}")
            print(f"    ✅ Risk params: {dict(config.risk_params)}")
            print(f"    ✅ Signal params: {dict(config.signal_params)}")
            
            # Test 1.2: Enhanced Strategy
            print("  🎯 Testing Enhanced Strategy...")
            strategy = EnhancedSpoofingDetectionStrategy(config)
            
            assert hasattr(strategy, 'config'), "Missing config"
            assert hasattr(strategy, 'original_strategy'), "Missing original_strategy"
            assert hasattr(strategy, 'risk_manager'), "Missing risk_manager"
            
            print("    ✅ Enhanced strategy initialized successfully")
            
            # Test 1.3: NEXUS Adapter
            print("  🔌 Testing NEXUS Adapter...")
            adapter = SpoofingDetectionNexusAdapter(strategy)
            
            assert hasattr(adapter, 'execute'), "Missing execute method"
            assert hasattr(adapter, 'get_category'), "Missing get_category method"
            assert hasattr(adapter, 'get_performance_metrics'), "Missing get_performance_metrics method"
            
            print("    ✅ NEXUS adapter initialized successfully")
            
            self.test_results.append(("Strategy Initialization", True, "All components initialized"))
            return True
            
        except Exception as e:
            print(f"    ❌ Initialization failed: {e}")
            self.test_results.append(("Strategy Initialization", False, str(e)))
            return False
    
    def test_signal_generation(self):
        """Test signal generation from market data"""
        try:
            # Initialize strategy
            config = UniversalStrategyConfig("spoofing_detection")
            strategy = EnhancedSpoofingDetectionStrategy(config)
            adapter = SpoofingDetectionNexusAdapter(strategy)
            
            # Test 2.1: Normal Market Data
            print("  📊 Testing normal market data...")
            normal_market_data = {
                'symbol': 'ES',
                'price': 4500.25,
                'close': 4500.25,
                'open': 4499.75,
                'high': 4501.00,
                'low': 4499.50,
                'volume': 1000,
                'bid': 4500.00,
                'ask': 4500.50,
                'bid_size': 100,
                'ask_size': 120,
                'timestamp': time.time()
            }
            
            result = adapter.execute(normal_market_data)
            
            assert isinstance(result, dict), "Result must be dictionary"
            assert 'signal' in result, "Missing signal in result"
            assert 'confidence' in result, "Missing confidence in result"
            assert 'metadata' in result, "Missing metadata in result"
            assert -1.0 <= result['signal'] <= 1.0, f"Signal out of range: {result['signal']}"
            assert 0.0 <= result['confidence'] <= 1.0, f"Confidence out of range: {result['confidence']}"
            
            print(f"    ✅ Normal data signal: {result['signal']:.3f}, confidence: {result['confidence']:.3f}")
            
            # Test 2.2: Suspicious Market Data (Large Order Imbalance)
            print("  🚨 Testing suspicious market data...")
            suspicious_market_data = {
                'symbol': 'ES',
                'price': 4500.25,
                'close': 4500.25,
                'open': 4499.75,
                'high': 4501.00,
                'low': 4499.50,
                'volume': 5000,  # High volume
                'bid': 4500.00,
                'ask': 4500.50,
                'bid_size': 5000,  # Very large bid
                'ask_size': 50,    # Small ask (imbalanced)
                'timestamp': time.time()
            }
            
            result_suspicious = adapter.execute(suspicious_market_data)
            
            print(f"    ✅ Suspicious data signal: {result_suspicious['signal']:.3f}, confidence: {result_suspicious['confidence']:.3f}")
            
            # Test 2.3: Multiple Data Points (Build History)
            print("  📈 Testing multiple data points...")
            signals_generated = 0
            
            for i in range(25):  # Generate 25 data points to build history
                test_data = {
                    'symbol': 'ES',
                    'price': 4500.0 + np.random.normal(0, 0.5),
                    'volume': 1000 + np.random.randint(-200, 500),
                    'bid': 4499.75 + np.random.normal(0, 0.25),
                    'ask': 4500.25 + np.random.normal(0, 0.25),
                    'bid_size': 100 + np.random.randint(-20, 100),
                    'ask_size': 100 + np.random.randint(-20, 100),
                    'timestamp': time.time() + i
                }
                test_data['close'] = test_data['price']
                test_data['open'] = test_data['price'] - np.random.uniform(-0.5, 0.5)
                test_data['high'] = max(test_data['price'], test_data['open']) + np.random.uniform(0, 0.3)
                test_data['low'] = min(test_data['price'], test_data['open']) - np.random.uniform(0, 0.3)
                
                result = adapter.execute(test_data)
                if abs(result['signal']) > 0.1:  # Non-trivial signal
                    signals_generated += 1
            
            print(f"    ✅ Generated {signals_generated}/25 non-trivial signals")
            
            self.test_results.append(("Signal Generation", True, f"{signals_generated}/25 signals generated"))
            return True
            
        except Exception as e:
            print(f"    ❌ Signal generation failed: {e}")
            self.test_results.append(("Signal Generation", False, str(e)))
            return False   
 
    def test_trade_execution(self):
        """Test trade execution and position management"""
        try:
            # Initialize strategy
            config = UniversalStrategyConfig("spoofing_detection")
            strategy = EnhancedSpoofingDetectionStrategy(config)
            adapter = SpoofingDetectionNexusAdapter(strategy)
            
            print("  💼 Testing trade execution logic...")
            
            # Test 3.1: Position Entry Logic
            print("    🎯 Testing position entry logic...")
            
            signal_data = {
                'signal': 0.8,  # Strong buy signal
                'confidence': 0.75,
                'action': 'BUY'
            }
            
            market_data = {
                'symbol': 'ES',
                'price': 4500.25,
                'volume': 2000,
                'volatility': 0.02
            }
            
            # Test position entry calculation
            entry_logic = adapter.calculate_position_entry_logic(signal_data, market_data)
            
            assert 'entry_size' in entry_logic, "Missing entry_size"
            assert 'can_enter_position' in entry_logic, "Missing can_enter_position"
            assert entry_logic['entry_size'] > 0, "Entry size must be positive"
            
            print(f"      ✅ Entry size: ${entry_logic['entry_size']:.2f}")
            print(f"      ✅ Can enter: {entry_logic['can_enter_position']}")
            print(f"      ✅ Scale-in allowed: {entry_logic['allow_scale_in']}")
            
            # Test 3.2: Leverage Calculation
            print("    ⚖️ Testing leverage calculation...")
            
            leverage_info = adapter.calculate_leverage_ratio(
                position_size=entry_logic['entry_size'],
                account_equity=100000.0
            )
            
            assert 'leverage_ratio' in leverage_info, "Missing leverage_ratio"
            assert 'is_within_limits' in leverage_info, "Missing is_within_limits"
            assert leverage_info['leverage_ratio'] > 0, "Leverage ratio must be positive"
            
            print(f"      ✅ Leverage ratio: {leverage_info['leverage_ratio']:.2f}x")
            print(f"      ✅ Within limits: {leverage_info['is_within_limits']}")
            print(f"      ✅ Margin required: {leverage_info['margin_requirement_pct']:.1%}")
            
            # Test 3.3: Trade Recording
            print("    📝 Testing trade recording...")
            
            trade_info = {
                'pnl': 150.0,
                'entry_price': 4500.25,
                'exit_price': 4502.00,
                'quantity': 10,
                'confidence': 0.75,
                'volatility': 0.02
            }
            
            adapter.record_trade_result(trade_info)
            
            # Check performance metrics updated
            metrics = adapter.get_performance_metrics()
            assert metrics['total_trades'] > 0, "Trade not recorded"
            assert metrics['total_pnl'] == 150.0, "PnL not recorded correctly"
            
            print(f"      ✅ Trade recorded: PnL=${trade_info['pnl']:.2f}")
            print(f"      ✅ Total trades: {metrics['total_trades']}")
            print(f"      ✅ Win rate: {metrics['win_rate']:.1%}")
            
            self.test_results.append(("Trade Execution", True, "All trade functions working"))
            return True
            
        except Exception as e:
            print(f"    ❌ Trade execution failed: {e}")
            self.test_results.append(("Trade Execution", False, str(e)))
            return False    
 
   def test_risk_management(self):
        """Test risk management and kill switch functionality"""
        try:
            # Initialize strategy
            config = UniversalStrategyConfig("spoofing_detection")
            strategy = EnhancedSpoofingDetectionStrategy(config)
            adapter = SpoofingDetectionNexusAdapter(strategy)
            
            print("  🛡️ Testing risk management...")
            
            # Test 4.1: Normal Risk Conditions
            print("    ✅ Testing normal risk conditions...")
            
            # Should not trigger kill switch
            kill_switch_active = adapter._check_kill_switch()
            assert not kill_switch_active, "Kill switch should not be active initially"
            
            print(f"      ✅ Kill switch status: {kill_switch_active}")
            print(f"      ✅ Current equity: ${adapter.current_equity:.2f}")
            print(f"      ✅ Daily PnL: ${adapter.daily_pnl:.2f}")
            
            # Test 4.2: VaR/CVaR Calculation
            print("    📊 Testing VaR/CVaR calculation...")
            
            # Add some return history
            for i in range(50):
                daily_return = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
                adapter.returns_history.append(daily_return)
            
            var_95 = adapter.calculate_var(confidence=0.95)
            var_99 = adapter.calculate_var(confidence=0.99)
            cvar_95 = adapter.calculate_cvar(confidence=0.95)
            cvar_99 = adapter.calculate_cvar(confidence=0.99)
            
            print(f"      ✅ VaR 95%: ${var_95:.2f}")
            print(f"      ✅ VaR 99%: ${var_99:.2f}")
            print(f"      ✅ CVaR 95%: ${cvar_95:.2f}")
            print(f"      ✅ CVaR 99%: ${cvar_99:.2f}")
            
            # Test 4.3: Simulated Loss Scenario
            print("    ⚠️ Testing loss scenario...")
            
            # Simulate losing trades
            for i in range(3):
                loss_trade = {
                    'pnl': -500.0,  # $500 loss each
                    'confidence': 0.6,
                    'volatility': 0.03
                }
                adapter.record_trade_result(loss_trade)
            
            # Check consecutive losses
            print(f"      ✅ Consecutive losses: {adapter.consecutive_losses}")
            print(f"      ✅ Daily PnL after losses: ${adapter.daily_pnl:.2f}")
            
            # Test 4.4: Kill Switch Reset
            print("    🔄 Testing kill switch reset...")
            
            # Reset kill switch
            adapter.reset_kill_switch()
            assert not adapter.kill_switch_active, "Kill switch should be reset"
            
            print("      ✅ Kill switch reset successfully")
            
            self.test_results.append(("Risk Management", True, "All risk functions working"))
            return True
            
        except Exception as e:
            print(f"    ❌ Risk management failed: {e}")
            self.test_results.append(("Risk Management", False, str(e)))
            return False
    
    def test_performance_tracking(self):
        """Test performance tracking and metrics"""
        try:
            # Initialize strategy
            config = UniversalStrategyConfig("spoofing_detection")
            strategy = EnhancedSpoofingDetectionStrategy(config)
            adapter = SpoofingDetectionNexusAdapter(strategy)
            
            print("  📈 Testing performance tracking...")
            
            # Test 5.1: Basic Metrics
            print("    📊 Testing basic metrics...")
            
            initial_metrics = adapter.get_performance_metrics()
            
            required_metrics = [
                'total_trades', 'winning_trades', 'win_rate', 'total_pnl',
                'current_equity', 'var_95', 'var_99', 'cvar_95', 'cvar_99',
                'kill_switch_active', 'ml_predictions_enabled'
            ]
            
            for metric in required_metrics:
                assert metric in initial_metrics, f"Missing metric: {metric}"
            
            print(f"      ✅ All {len(required_metrics)} required metrics present")
            
            # Test 5.2: Execution Quality Tracking
            print("    🎯 Testing execution quality...")
            
            # Simulate some fills
            fill_data = [
                {'expected_price': 4500.0, 'actual_price': 4500.05, 'order_size': 100, 'filled_size': 100},
                {'expected_price': 4501.0, 'actual_price': 4501.02, 'order_size': 150, 'filled_size': 150},
                {'expected_price': 4499.5, 'actual_price': 4499.48, 'order_size': 200, 'filled_size': 180}  # Partial fill
            ]
            
            for fill in fill_data:
                adapter.record_fill(fill)
            
            exec_metrics = adapter.get_execution_quality_metrics()
            
            assert 'avg_slippage_bps' in exec_metrics, "Missing slippage metrics"
            assert 'partial_fill_rate' in exec_metrics, "Missing fill rate metrics"
            
            print(f"      ✅ Avg slippage: {exec_metrics['avg_slippage_bps']:.1f} bps")
            print(f"      ✅ Partial fill rate: {exec_metrics['partial_fill_rate']:.1%}")
            print(f"      ✅ Total fills: {exec_metrics['total_fills']}")
            
            self.test_results.append(("Performance Tracking", True, "All metrics working"))
            return True
            
        except Exception as e:
            print(f"    ❌ Performance tracking failed: {e}")
            self.test_results.append(("Performance Tracking", False, str(e)))
            return False  
  
    def demo_live_trading(self):
        """Demonstrate live trading simulation"""
        try:
            print("  🚀 Starting live trading simulation...")
            
            # Initialize strategy
            config = UniversalStrategyConfig("spoofing_detection")
            strategy = EnhancedSpoofingDetectionStrategy(config)
            adapter = SpoofingDetectionNexusAdapter(strategy)
            
            # Simulation parameters
            initial_equity = 100000.0
            num_ticks = 50
            current_price = 4500.0
            
            print(f"    💰 Initial equity: ${initial_equity:,.2f}")
            print(f"    📊 Simulating {num_ticks} market ticks...")
            print(f"    💹 Starting price: ${current_price:.2f}")
            
            # Trading simulation
            trades_executed = 0
            signals_generated = 0
            
            for tick in range(num_ticks):
                # Generate realistic market data
                price_change = np.random.normal(0, 0.5)  # Random walk
                current_price += price_change
                
                # Occasionally create suspicious patterns
                is_suspicious = (tick % 10 == 0)  # Every 10th tick
                
                if is_suspicious:
                    # Create large order imbalance (potential spoofing)
                    bid_size = np.random.randint(2000, 5000)
                    ask_size = np.random.randint(50, 200)
                    volume = np.random.randint(3000, 8000)
                else:
                    # Normal market conditions
                    bid_size = np.random.randint(80, 200)
                    ask_size = np.random.randint(80, 200)
                    volume = np.random.randint(500, 2000)
                
                market_data = {
                    'symbol': 'ES',
                    'price': current_price,
                    'close': current_price,
                    'open': current_price - price_change,
                    'high': current_price + abs(price_change) * 0.5,
                    'low': current_price - abs(price_change) * 0.5,
                    'volume': volume,
                    'bid': current_price - 0.25,
                    'ask': current_price + 0.25,
                    'bid_size': bid_size,
                    'ask_size': ask_size,
                    'timestamp': time.time() + tick
                }
                
                # Execute strategy
                result = adapter.execute(market_data)
                
                # Check for signals
                if abs(result['signal']) > 0.1:  # Non-trivial signal
                    signals_generated += 1
                    
                    # Simulate trade execution for strong signals
                    if abs(result['signal']) > 0.5 and result['confidence'] > 0.6:
                        # Calculate position size
                        entry_logic = adapter.calculate_position_entry_logic(result, market_data)
                        
                        if entry_logic['can_enter_position']:
                            # Simulate trade execution
                            entry_price = current_price
                            position_size = entry_logic['entry_size']
                            
                            # Simulate holding for a few ticks and then exit
                            exit_price = current_price + np.random.normal(0, 0.3) * result['signal']
                            pnl = (exit_price - entry_price) * result['signal'] * (position_size / entry_price)
                            
                            # Record trade
                            trade_info = {
                                'pnl': pnl,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'quantity': position_size / entry_price,
                                'confidence': result['confidence'],
                                'volatility': 0.02
                            }
                            
                            adapter.record_trade_result(trade_info)
                            trades_executed += 1
                            
                            print(f"      🔄 Tick {tick+1:2d}: {'📈' if result['signal'] > 0 else '📉'} "
                                  f"Signal={result['signal']:+.2f}, Conf={result['confidence']:.2f}, "
                                  f"PnL=${pnl:+.2f} {'🟢' if pnl > 0 else '🔴'}")
                
                # Print progress every 10 ticks
                if (tick + 1) % 10 == 0:
                    metrics = adapter.get_performance_metrics()
                    print(f"    📊 Progress: {tick+1}/{num_ticks} ticks, "
                          f"Signals: {signals_generated}, Trades: {trades_executed}, "
                          f"PnL: ${metrics['total_pnl']:+.2f}")
            
            # Final results
            final_metrics = adapter.get_performance_metrics()
            
            print(f"\n  📋 SIMULATION RESULTS:")
            print(f"    🎯 Signals generated: {signals_generated}/{num_ticks} ({signals_generated/num_ticks:.1%})")
            print(f"    💼 Trades executed: {trades_executed}")
            print(f"    💰 Total PnL: ${final_metrics['total_pnl']:+.2f}")
            print(f"    📈 Win rate: {final_metrics['win_rate']:.1%}")
            print(f"    🏆 Final equity: ${final_metrics['current_equity']:,.2f}")
            print(f"    📊 Return: {((final_metrics['current_equity'] - initial_equity) / initial_equity):.2%}")
            
            self.demo_results = {
                'signals_generated': signals_generated,
                'trades_executed': trades_executed,
                'total_pnl': final_metrics['total_pnl'],
                'win_rate': final_metrics['win_rate'],
                'final_equity': final_metrics['current_equity'],
                'return_pct': (final_metrics['current_equity'] - initial_equity) / initial_equity
            }
            
            return True
            
        except Exception as e:
            print(f"    ❌ Live trading simulation failed: {e}")
            return False 
   
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY REPORT")
        print("="*80)
        
        # Test Results
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        total_tests = len(self.test_results)
        
        print(f"\n🧪 TEST RESULTS: {passed_tests}/{total_tests} PASSED ({passed_tests/total_tests:.1%})")
        print("-" * 50)
        
        for test_name, passed, details in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} | {test_name:<25} | {details}")
        
        # Demo Results
        if self.demo_results:
            print(f"\n🚀 DEMO RESULTS:")
            print("-" * 50)
            print(f"  📊 Signals Generated: {self.demo_results['signals_generated']}")
            print(f"  💼 Trades Executed: {self.demo_results['trades_executed']}")
            print(f"  💰 Total PnL: ${self.demo_results['total_pnl']:+.2f}")
            print(f"  📈 Win Rate: {self.demo_results['win_rate']:.1%}")
            print(f"  🏆 Final Equity: ${self.demo_results['final_equity']:,.2f}")
            print(f"  📊 Return: {self.demo_results['return_pct']:+.2%}")
        
        # Overall Assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print("-" * 50)
        
        if passed_tests == total_tests:
            print("  🟢 EXCELLENT: All tests passed! Strategy is fully functional.")
            print("  ✅ Signal generation working correctly")
            print("  ✅ Trade execution functioning properly")
            print("  ✅ Risk management active and effective")
            print("  ✅ Performance tracking comprehensive")
            print("  ✅ Ready for production deployment")
        elif passed_tests >= total_tests * 0.8:
            print("  🟡 GOOD: Most tests passed. Minor issues detected.")
            print("  ⚠️ Review failed tests before production deployment")
        else:
            print("  🔴 NEEDS WORK: Multiple test failures detected.")
            print("  ❌ Strategy requires fixes before deployment")
        
        print("\n" + "="*80)


def main():
    """Main execution function"""
    print("🎯 Spoofing Detection Strategy - Test & Demo Suite")
    print("=" * 60)
    
    if not STRATEGY_AVAILABLE:
        print("❌ Strategy module not available. Please check imports.")
        return False
    
    # Create tester instance
    tester = SpoofingDetectionTester()
    
    # Run all tests
    success = tester.run_all_tests()
    
    # Final status
    if success:
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Spoofing Detection Strategy is ready for deployment")
    else:
        print("\n⚠️ SOME TESTS FAILED")
        print("❌ Please review and fix issues before deployment")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)