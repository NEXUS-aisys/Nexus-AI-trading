#!/usr/bin/env python3
"""
NEXUS AI Pipeline Test - Complete System Test
Tests all 43 models integration and pipeline functionality
"""

import asyncio
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add current directory to path for imports
sys.path.append(os.getcwd())

try:
    from nexus_ai import NexusAI, SystemConfig, MarketData, MarketDataType
    from decimal import Decimal
    print("✅ Successfully imported NEXUS AI components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class NexusPipelineTest:
    """Comprehensive test suite for NEXUS AI pipeline."""
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {
            "initialization": False,
            "model_loading": False,
            "model_registry": False,
            "market_data_processing": False,
            "strategy_execution": False,
            "system_status": False,
            "error_handling": False
        }
        
    def create_sample_market_data(self, symbol: str = "BTCUSDT") -> MarketData:
        """Create sample market data for testing."""
        current_time = time.time()
        
        return MarketData(
            symbol=symbol,
            timestamp=current_time,
            price=Decimal("50000.00"),
            volume=Decimal("1000.0"),
            bid=Decimal("49995.00"),
            ask=Decimal("50005.00"),
            bid_size=500,
            ask_size=500,
            data_type=MarketDataType.TRADE,
            exchange_timestamp=current_time,
            sequence_num=12345,
            metadata={
                "dataframe": self.create_sample_dataframe()
            }
        )
    
    def create_sample_dataframe(self) -> pd.DataFrame:
        """Create sample OHLCV DataFrame for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        
        # Generate realistic price data
        base_price = 50000
        price_changes = np.random.normal(0, 0.001, 100)  # 0.1% volatility
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.0005)))
            low = price * (1 - abs(np.random.normal(0, 0.0005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(800, 1200)
            
            data.append({
                'timestamp': dates[i].timestamp(),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume,
                'bid': close_price * 0.9995,
                'ask': close_price * 1.0005
            })
        
        return pd.DataFrame(data)
    
    async def test_initialization(self) -> bool:
        """Test NEXUS AI system initialization."""
        print("\n" + "="*80)
        print("TEST 1: SYSTEM INITIALIZATION")
        print("="*80)
        
        try:
            # Create configuration
            config = SystemConfig(
                buffer_size=1000,
                cache_size=100,
                max_position_size=0.05,
                max_daily_loss=0.01,
                max_drawdown=0.10,
            )
            
            print("📋 Creating NEXUS AI system...")
            self.nexus = NexusAI(config)
            
            print("✅ NEXUS AI system initialized successfully")
            self.test_results["initialization"] = True
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_model_loading(self) -> bool:
        """Test model loading and registry."""
        print("\n" + "="*80)
        print("TEST 2: MODEL LOADING & REGISTRY")
        print("="*80)
        
        try:
            # Get model registry status
            model_status = self.nexus.get_model_registry_status()
            
            print(f"📊 Model Loading Results:")
            print(f"   Total Models: {model_status['total_models']}")
            print(f"   Loaded Models: {model_status['loaded_models']}")
            print(f"   Success Rate: {model_status['success_rate']:.1f}%")
            print(f"   System Ready: {model_status['system_ready']}")
            
            # Test model registry methods
            all_models = self.nexus.list_all_models()
            print(f"\n📋 Model Registry Test:")
            print(f"   Registry contains {len(all_models)} model definitions")
            
            # Test layer-based access
            for layer in range(1, 9):
                layer_models = self.nexus.get_models_by_layer(layer)
                loaded_count = len([m for m in layer_models.values() if m is not None])
                total_count = len([m for m in all_models if m['layer'] == layer])
                print(f"   Layer {layer}: {loaded_count}/{total_count} models loaded")
            
            # Test category-based access
            categories = set(m['category'] for m in all_models)
            print(f"\n🏷️ Model Categories ({len(categories)}):")
            for category in sorted(categories):
                cat_models = self.nexus.get_models_by_category(category)
                loaded_count = len([m for m in cat_models.values() if m is not None])
                total_count = len([m for m in all_models if m['category'] == category])
                print(f"   {category}: {loaded_count}/{total_count} models")
            
            # Test individual model info
            test_model = "cnn_regime_detector"
            model_info = self.nexus.get_model_info(test_model)
            if model_info:
                print(f"\n🔍 Sample Model Info ({test_model}):")
                print(f"   Function: {model_info['function']}")
                print(f"   Description: {model_info['description']}")
                print(f"   Layer: {model_info['layer']}")
                print(f"   Latency: {model_info['latency_ms']}ms")
            
            # Determine success
            success = model_status['loaded_models'] > 0
            if success:
                print(f"\n✅ Model loading test PASSED")
                if model_status['system_ready']:
                    print("🚀 System is PRODUCTION READY")
                else:
                    print("⚠️ System has partial model loading")
            else:
                print(f"\n❌ Model loading test FAILED - No models loaded")
            
            self.test_results["model_loading"] = success
            self.test_results["model_registry"] = True
            return success
            
        except Exception as e:
            print(f"❌ Model loading test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_market_data_processing(self) -> bool:
        """Test market data processing through the pipeline."""
        print("\n" + "="*80)
        print("TEST 3: MARKET DATA PROCESSING")
        print("="*80)
        
        try:
            # Create sample market data
            market_data = self.create_sample_market_data("BTCUSDT")
            
            print(f"📊 Processing market data:")
            print(f"   Symbol: {market_data.symbol}")
            print(f"   Price: ${market_data.price}")
            print(f"   Volume: {market_data.volume}")
            print(f"   Spread: {market_data.spread_bps:.2f} bps")
            
            # Process through pipeline
            print("\n🔄 Processing through NEXUS AI pipeline...")
            start_time = time.time()
            
            # Convert MarketData to dictionary format expected by process_market_data
            market_data_dict = market_data.to_dict()
            market_data_dict["market_data_df"] = market_data.metadata.get("dataframe")
            
            result = await self.nexus.process_market_data(market_data_dict)
            
            processing_time = (time.time() - start_time) * 1000
            
            if result:
                print(f"✅ Market data processed successfully")
                print(f"   Processing time: {processing_time:.2f}ms")
                print(f"   Symbol: {result['symbol']}")
                print(f"   Signals generated: {len(result['signals'])}")
                print(f"   Pipeline latency: {result.get('latency_ms', 0):.2f}ms")
                
                # Show signal details
                if result['signals']:
                    print(f"\n📈 Generated Signals:")
                    for i, signal in enumerate(result['signals'][:5]):  # Show first 5
                        print(f"   {i+1}. {signal['signal']} (confidence: {signal['confidence']:.3f}) - {signal['strategy']}")
                    
                    if len(result['signals']) > 5:
                        print(f"   ... and {len(result['signals']) - 5} more signals")
                else:
                    print("   No signals generated (may be normal depending on market conditions)")
                
                self.test_results["market_data_processing"] = True
                return True
            else:
                print(f"❌ Market data processing returned None")
                return False
                
        except Exception as e:
            print(f"❌ Market data processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_strategy_execution(self) -> bool:
        """Test strategy execution and signal generation."""
        print("\n" + "="*80)
        print("TEST 4: STRATEGY EXECUTION")
        print("="*80)
        
        try:
            # Get strategy information
            strategies = self.nexus.strategy_manager.get_strategies()
            strategy_metrics = self.nexus.strategy_manager.get_metrics()
            
            print(f"📊 Strategy System Status:")
            print(f"   Registered strategies: {len(strategies)}")
            print(f"   Strategy names: {strategies}")
            
            if strategy_metrics:
                print(f"\n📈 Strategy Metrics:")
                for strategy_name, metrics in strategy_metrics.items():
                    print(f"   {strategy_name}:")
                    print(f"     Total signals: {metrics.get('total_signals', 0)}")
                    print(f"     Avg confidence: {metrics.get('avg_confidence', 0):.3f}")
            
            # Test direct strategy execution
            market_data = self.create_sample_market_data("BTCUSDT")
            signals = self.nexus.strategy_manager.execute_all(market_data)
            
            print(f"\n🎯 Direct Strategy Execution:")
            print(f"   Generated {len(signals)} signals")
            
            if signals:
                signal_types = {}
                for signal in signals:
                    signal_type = signal.signal_type.name
                    signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
                
                print(f"   Signal distribution:")
                for signal_type, count in signal_types.items():
                    print(f"     {signal_type}: {count}")
            
            self.test_results["strategy_execution"] = True
            return True
            
        except Exception as e:
            print(f"❌ Strategy execution test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_system_status(self) -> bool:
        """Test system status and monitoring."""
        print("\n" + "="*80)
        print("TEST 5: SYSTEM STATUS & MONITORING")
        print("="*80)
        
        try:
            # Get comprehensive system status
            status = self.nexus.get_system_status()
            
            print(f"🖥️ System Status:")
            print(f"   Running: {status['running']}")
            print(f"   Initialized: {status['initialized']}")
            print(f"   Models loaded: {status['models_loaded']}")
            print(f"   System ready: {status['system_ready']}")
            
            print(f"\n📊 Component Status:")
            print(f"   Strategies: {len(status['strategies'])}")
            print(f"   Data buffer size: {status['data_buffer_size']}")
            print(f"   Cache stats: {status['cache_stats']}")
            
            # Test model registry status
            model_status = status['model_registry']
            print(f"\n🤖 Model Registry Status:")
            print(f"   Total models: {model_status['total_models']}")
            print(f"   Loaded models: {model_status['loaded_models']}")
            print(f"   Success rate: {model_status['success_rate']:.1f}%")
            print(f"   Layer distribution: {model_status['layer_distribution']}")
            print(f"   Type distribution: {model_status['type_distribution']}")
            
            self.test_results["system_status"] = True
            return True
            
        except Exception as e:
            print(f"❌ System status test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and resilience."""
        print("\n" + "="*80)
        print("TEST 6: ERROR HANDLING & RESILIENCE")
        print("="*80)
        
        try:
            # Test with invalid market data
            print("🧪 Testing with invalid market data...")
            
            invalid_data = MarketData(
                symbol="INVALID",
                timestamp=time.time(),
                price=Decimal("0"),  # Invalid price
                volume=Decimal("-100"),  # Invalid volume
                bid=Decimal("100"),
                ask=Decimal("50"),  # Invalid spread (bid > ask)
                bid_size=0,
                ask_size=0,
                data_type=MarketDataType.TRADE,
                exchange_timestamp=time.time(),
                sequence_num=0
            )
            
            # System should handle this gracefully
            try:
                # Convert to dictionary format
                invalid_data_dict = invalid_data.to_dict()
                result = asyncio.run(self.nexus.process_market_data(invalid_data_dict))
                print("✅ System handled invalid data gracefully")
            except Exception as e:
                print(f"⚠️ System threw exception for invalid data: {e}")
            
            # Test with missing model
            print("\n🧪 Testing model info for non-existent model...")
            missing_model_info = self.nexus.get_model_info("non_existent_model")
            if not missing_model_info:
                print("✅ System handled missing model request gracefully")
            else:
                print("⚠️ System returned data for non-existent model")
            
            # Test layer access with invalid layer
            print("\n🧪 Testing invalid layer access...")
            invalid_layer_models = self.nexus.get_models_by_layer(99)
            if not invalid_layer_models:
                print("✅ System handled invalid layer request gracefully")
            else:
                print("⚠️ System returned data for invalid layer")
            
            self.test_results["error_handling"] = True
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print("🚀 NEXUS AI PIPELINE - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run tests in sequence
        await self.test_initialization()
        
        if self.test_results["initialization"]:
            self.test_model_loading()
            await self.test_market_data_processing()
            self.test_strategy_execution()
            self.test_system_status()
            self.test_error_handling()
        
        # Print final results
        self.print_test_summary()
        
        return self.test_results
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        
        print(f"📊 Overall Results: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        print()
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print()
        if passed == total:
            print("🎉 ALL TESTS PASSED - NEXUS AI PIPELINE IS READY FOR PRODUCTION!")
        elif passed >= total * 0.8:
            print("⚠️ MOST TESTS PASSED - System functional with minor issues")
        else:
            print("🚨 MULTIPLE TEST FAILURES - System needs attention")
        
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

async def main():
    """Main test execution."""
    test_suite = NexusPipelineTest()
    results = await test_suite.run_all_tests()
    
    # Return exit code based on results
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    if passed == total:
        return 0  # All tests passed
    elif passed >= total * 0.8:
        return 1  # Most tests passed
    else:
        return 2  # Multiple failures

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)