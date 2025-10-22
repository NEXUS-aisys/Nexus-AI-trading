#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI Parameter Optimization System
Automated parameter tuning for maximum performance
Version: 1.0.0
"""

import asyncio
import json
import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor
import logging

from nexus_backtester import NexusBacktester, BacktestConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Automated parameter optimization for NEXUS AI strategies."""
    
    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
        
    def generate_parameter_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all combinations of parameters to test."""
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def create_config_with_params(self, params: Dict[str, Any]) -> BacktestConfig:
        """Create a new config with modified parameters."""
        config_dict = self.base_config.__dict__.copy()
        config_dict.update(params)
        
        return BacktestConfig(**config_dict)
    
    async def evaluate_parameters(self, params: Dict[str, Any]) -> Tuple[Dict, float, Dict]:
        """Evaluate a single parameter combination."""
        try:
            # Create config with new parameters
            config = self.create_config_with_params(params)
            
            # Run backtest
            backtester = NexusBacktester(config)
            results = await backtester.run_backtest()
            
            # Calculate optimization score (you can customize this)
            score = self.calculate_optimization_score(results)
            
            return params, score, {
                'total_return': results.total_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'total_trades': results.total_trades,
                'profit_factor': results.profit_factor
            }
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {e}")
            return params, -np.inf, {}
    
    def calculate_optimization_score(self, results) -> float:
        """
        Calculate optimization score based on multiple metrics.
        You can customize this function based on your preferences.
        """
        # Multi-objective optimization score
        # Weights can be adjusted based on importance
        weights = {
            'return': 0.3,
            'sharpe': 0.25,
            'drawdown': 0.2,  # Negative weight (lower is better)
            'win_rate': 0.15,
            'trades': 0.1     # Minimum activity requirement
        }
        
        # Normalize metrics (simple approach)
        return_score = min(results.total_return * 2, 1.0)  # Cap at 50% return
        sharpe_score = min(results.sharpe_ratio / 3.0, 1.0)  # Cap at 3.0 Sharpe
        drawdown_score = max(1.0 + results.max_drawdown * 2, 0.0)  # Penalty for drawdown
        win_rate_score = results.win_rate
        trades_score = min(results.total_trades / 100.0, 1.0)  # Minimum 100 trades for full score
        
        # Calculate weighted score
        score = (
            weights['return'] * return_score +
            weights['sharpe'] * sharpe_score +
            weights['drawdown'] * drawdown_score +
            weights['win_rate'] * win_rate_score +
            weights['trades'] * trades_score
        )
        
        return score
    
    async def optimize(self, param_ranges: Dict[str, List], max_concurrent: int = 3) -> Dict:
        """Run parameter optimization."""
        logger.info("Starting parameter optimization...")
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations(param_ranges)
        
        # Limit concurrent executions to avoid overwhelming the system
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(params):
            async with semaphore:
                return await self.evaluate_parameters(params)
        
        # Run evaluations
        tasks = [evaluate_with_semaphore(params) for params in param_combinations]
        
        logger.info(f"Running {len(tasks)} parameter evaluations...")
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            params, score, metrics = await coro
            completed += 1
            
            # Store result
            result = {
                'parameters': params,
                'score': score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                logger.info(f"New best score: {score:.4f} with params: {params}")
            
            # Progress update
            if completed % 5 == 0:
                logger.info(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%)")
        
        logger.info("Parameter optimization completed!")
        
        # Sort results by score
        self.results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'best_parameters': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'total_combinations': len(param_combinations)
        }
    
    def generate_optimization_report(self, output_file: str = "optimization_report.html"):
        """Generate HTML report of optimization results."""
        if not self.results:
            logger.error("No optimization results available")
            return
        
        # Get top 10 results
        top_results = self.results[:10]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NEXUS AI Parameter Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .best {{ background-color: #d4edda; border-color: #c3e6cb; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .rank-1 {{ background-color: #fff3cd; }}
                .rank-2 {{ background-color: #f8f9fa; }}
                .rank-3 {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔧 NEXUS AI Parameter Optimization Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Combinations Tested: {len(self.results)}</p>
            </div>
            
            <div class="section best">
                <h2>🏆 Best Parameters Found</h2>
                <p><strong>Optimization Score:</strong> {self.best_score:.4f}</p>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        for param, value in self.best_params.items():
            html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>📊 Best Performance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        best_metrics = self.results[0]['metrics']
        for metric, value in best_metrics.items():
            if isinstance(value, float):
                if 'rate' in metric or 'return' in metric or 'drawdown' in metric:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            html_content += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{formatted_value}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>🥇 Top 10 Parameter Combinations</h2>
                <table>
                    <tr>
                        <th>Rank</th><th>Score</th><th>Return</th><th>Sharpe</th>
                        <th>Drawdown</th><th>Win Rate</th><th>Trades</th><th>Parameters</th>
                    </tr>
        """
        
        for i, result in enumerate(top_results):
            rank_class = f"rank-{min(i+1, 3)}"
            metrics = result['metrics']
            params_str = ", ".join([f"{k}={v}" for k, v in result['parameters'].items()])
            
            html_content += f"""
                    <tr class="{rank_class}">
                        <td>{i+1}</td>
                        <td>{result['score']:.4f}</td>
                        <td>{metrics.get('total_return', 0):.2%}</td>
                        <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                        <td>{metrics.get('max_drawdown', 0):.2%}</td>
                        <td>{metrics.get('win_rate', 0):.2%}</td>
                        <td>{metrics.get('total_trades', 0)}</td>
                        <td style="font-size: 0.8em;">{params_str}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Optimization report saved to: {output_file}")


async def run_optimization_example():
    """Example optimization run."""
    
    # Base configuration
    base_config = BacktestConfig(
        start_date="2024-08-01",
        end_date="2024-10-22",
        initial_capital=100000.0,
        symbols=["BTCUSDT", "ETHUSDT"],
        timeframe="1h"
    )
    
    # Parameters to optimize
    param_ranges = {
        'min_confidence': [0.55, 0.60, 0.65, 0.70],
        'mqscore_threshold': [0.50, 0.55, 0.60, 0.65],
        'max_position_size': [0.08, 0.10, 0.12, 0.15],
        'commission': [0.0008, 0.001, 0.0012],
        'slippage': [0.0003, 0.0005, 0.0008]
    }
    
    # Run optimization
    optimizer = ParameterOptimizer(base_config)
    results = await optimizer.optimize(param_ranges, max_concurrent=2)
    
    # Display results
    print("\n" + "="*60)
    print("🔧 PARAMETER OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Best Parameters:")
    for param, value in results['best_parameters'].items():
        print(f"  {param}: {value}")
    
    # Generate reports
    optimizer.generate_optimization_report("nexus_optimization_report.html")
    
    # Save results to JSON
    with open('nexus_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Reports Generated:")
    print(f"   📄 HTML Report: nexus_optimization_report.html")
    print(f"   💾 JSON Results: nexus_optimization_results.json")
    print("\n✅ Optimization completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_optimization_example())