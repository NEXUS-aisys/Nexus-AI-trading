#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEXUS AI Deep Backtesting System
Comprehensive backtesting framework for all 20 trading strategies
Version: 1.0.0
"""

import asyncio
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import NEXUS AI system
try:
    from nexus_ai import NexusAI
    HAS_NEXUS_AI = True
except ImportError:
    HAS_NEXUS_AI = False
    print("WARNING: NEXUS AI not available - backtesting disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nexus_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    # Time period
    start_date: str = "2024-01-01"
    end_date: str = "2024-10-22"
    
    # Capital & Risk
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of capital per position
    max_daily_loss: float = 0.02    # 2% daily loss limit
    max_drawdown: float = 0.15      # 15% max drawdown
    
    # Execution
    commission: float = 0.001       # 0.1% commission
    slippage: float = 0.0005       # 0.05% slippage
    min_confidence: float = 0.57    # Minimum signal confidence
    
    # Strategy Selection
    enabled_strategies: List[str] = field(default_factory=lambda: [
        "Event-Driven", "LVN-Breakout", "Absorption-Breakout", "Momentum-Breakout",
        "Market-Microstructure", "Order-Book-Imbalance", "Liquidity-Absorption",
        "Spoofing-Detection", "Iceberg-Detection", "Liquidation-Detection",
        "Liquidity-Traps", "Multi-Timeframe", "Cumulative-Delta", "Delta-Divergence",
        "Open-Drive-Fade", "Profile-Rotation", "VWAP-Reversion", "Stop-Run",
        "Momentum-Ignition", "Volume-Imbalance"
    ])
    
    # Data & Performance
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    timeframe: str = "1h"
    parallel_processing: bool = True
    max_workers: int = 4
    
    # MQScore Integration
    use_mqscore_filter: bool = True
    mqscore_threshold: float = 0.57
    
    # Advanced Features
    enable_ml_models: bool = True
    enable_ensemble_voting: bool = True
    rebalance_frequency: str = "daily"  # daily, hourly, signal-based


@dataclass
class Trade:
    """Individual trade record."""
    
    timestamp: datetime
    symbol: str
    strategy: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    commission: float
    slippage: float
    confidence: float
    mqscore: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position state."""
    
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float
    realized_pnl: float
    strategy: str
    entry_time: datetime
    last_update: datetime


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    
    # Performance Metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    
    # Strategy Performance
    strategy_returns: Dict[str, float]
    strategy_trades: Dict[str, int]
    strategy_win_rates: Dict[str, float]
    
    # Time Series Data
    equity_curve: pd.DataFrame
    drawdown_curve: pd.DataFrame
    trade_history: List[Trade]
    
    # MQScore Analysis
    mqscore_stats: Dict[str, float]
    
    # Risk Metrics
    var_95: float
    cvar_95: float
    calmar_ratio: float
    sortino_ratio: float


class MarketDataGenerator:
    """Generates realistic market data for backtesting."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MarketDataGenerator")
        
    def generate_ohlcv_data(self, symbol: str, start_date: str, end_date: str, 
                           timeframe: str = "1h") -> pd.DataFrame:
        """
        Generate realistic OHLCV data for backtesting.
        In production, this would connect to historical data APIs.
        """
        self.logger.info(f"Generating market data for {symbol} from {start_date} to {end_date}")
        
        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate time index
        if timeframe == "1h":
            freq = "1H"
        elif timeframe == "1d":
            freq = "1D"
        elif timeframe == "15m":
            freq = "15T"
        else:
            freq = "1H"
            
        timestamps = pd.date_range(start=start, end=end, freq=freq)
        
        # Generate realistic price data with trends and volatility
        np.random.seed(42)  # For reproducible results
        
        # Base price levels for different symbols
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 2500,
            "BNBUSDT": 300,
            "ADAUSDT": 0.5,
            "SOLUSDT": 100
        }
        
        base_price = base_prices.get(symbol, 1000)
        
        # Generate price series with realistic characteristics
        n_periods = len(timestamps)
        
        # Trend component
        trend = np.cumsum(np.random.normal(0, 0.001, n_periods))
        
        # Volatility clustering (GARCH-like)
        volatility = np.zeros(n_periods)
        volatility[0] = 0.02
        for i in range(1, n_periods):
            volatility[i] = 0.01 + 0.85 * volatility[i-1] + 0.1 * np.random.normal(0, 0.001)**2
        
        # Price returns
        returns = np.random.normal(trend, volatility)
        
        # Generate prices
        prices = [base_price]
        for i in range(1, n_periods):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC from close price
            volatility_factor = volatility[i] * 0.5
            
            high = price * (1 + np.random.uniform(0, volatility_factor))
            low = price * (1 - np.random.uniform(0, volatility_factor))
            
            if i == 0:
                open_price = price
            else:
                open_price = prices[i-1] * (1 + np.random.normal(0, volatility_factor * 0.3))
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            # Generate volume (higher volume during high volatility)
            base_volume = 1000000
            volume = base_volume * (1 + volatility[i] * 10) * np.random.uniform(0.5, 2.0)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': round(volume, 0)
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Generated {len(df)} data points for {symbol}")
        return df


class PortfolioManager:
    """Manages portfolio state during backtesting."""
    
    def __init__(self, initial_capital: float, config: BacktestConfig):
        self.initial_capital = initial_capital
        self.config = config
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        self.logger = logging.getLogger(f"{__name__}.PortfolioManager")
        
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = position.quantity * current_prices[symbol]
                total_value += position_value
                
        return total_value
    
    def execute_trade(self, symbol: str, side: str, quantity: float, price: float,
                     strategy: str, confidence: float, timestamp: datetime,
                     mqscore: Optional[float] = None) -> bool:
        """Execute a trade and update portfolio."""
        
        # Calculate costs
        notional = quantity * price
        commission = notional * self.config.commission
        slippage = notional * self.config.slippage
        total_cost = commission + slippage
        
        if side == "BUY":
            required_cash = notional + total_cost
            if required_cash > self.cash:
                self.logger.warning(f"Insufficient cash for {symbol} BUY: need {required_cash}, have {self.cash}")
                return False
                
            # Execute buy
            self.cash -= required_cash
            
            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                total_quantity = pos.quantity + quantity
                total_cost_basis = (pos.quantity * pos.avg_price) + notional
                pos.avg_price = total_cost_basis / total_quantity
                pos.quantity = total_quantity
                pos.last_update = timestamp
            else:
                # Create new position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    strategy=strategy,
                    entry_time=timestamp,
                    last_update=timestamp
                )
                
        elif side == "SELL":
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                self.logger.warning(f"Insufficient position for {symbol} SELL")
                return False
                
            # Execute sell
            pos = self.positions[symbol]
            proceeds = notional - total_cost
            self.cash += proceeds
            
            # Calculate realized PnL
            cost_basis = quantity * pos.avg_price
            realized_pnl = proceeds - cost_basis
            pos.realized_pnl += realized_pnl
            
            # Update position
            pos.quantity -= quantity
            pos.last_update = timestamp
            
            # Remove position if fully closed
            if pos.quantity <= 0:
                del self.positions[symbol]
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            strategy=strategy,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            confidence=confidence,
            mqscore=mqscore
        )
        
        self.trade_history.append(trade)
        self.logger.debug(f"Executed {side} {quantity} {symbol} @ {price} via {strategy}")
        
        return True
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float], timestamp: datetime):
        """Update unrealized PnL for all positions."""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_value = position.quantity * current_prices[symbol]
                cost_basis = position.quantity * position.avg_price
                position.unrealized_pnl = current_value - cost_basis
        
        # Record equity
        portfolio_value = self.get_portfolio_value(current_prices)
        self.equity_history.append((timestamp, portfolio_value))


class NexusBacktester:
    """Main backtesting engine for NEXUS AI system."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.NexusBacktester")
        
        # Initialize components
        self.nexus_ai = None
        self.data_generator = MarketDataGenerator(config)
        self.portfolio = PortfolioManager(config.initial_capital, config)
        
        # Results storage
        self.results: Optional[BacktestResults] = None
        
    async def initialize_nexus_ai(self) -> bool:
        """Initialize NEXUS AI system."""
        if not HAS_NEXUS_AI:
            self.logger.error("NEXUS AI not available")
            return False
            
        try:
            self.logger.info("Initializing NEXUS AI system...")
            self.nexus_ai = NexusAI()
            
            # Initialize the system
            await self.nexus_ai.initialize()
            
            # Register strategies
            strategy_count = self.nexus_ai.register_strategies_explicit()
            self.logger.info(f"Initialized NEXUS AI with {strategy_count} strategies")
            
            return strategy_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NEXUS AI: {e}")
            return False
    
    def generate_market_data(self) -> Dict[str, pd.DataFrame]:
        """Generate market data for all symbols."""
        self.logger.info("Generating market data for backtesting...")
        
        market_data = {}
        for symbol in self.config.symbols:
            df = self.data_generator.generate_ohlcv_data(
                symbol=symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                timeframe=self.config.timeframe
            )
            market_data[symbol] = df
            
        return market_data
    
    async def process_signals_for_timestamp(self, timestamp: datetime, 
                                    market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Process all strategy signals for a given timestamp."""
        signals = []
        
        if not self.nexus_ai:
            return signals
            
        for symbol in self.config.symbols:
            try:
                # Get market data up to current timestamp
                symbol_data = market_data[symbol]
                current_data = symbol_data[symbol_data.index <= timestamp]
                
                if len(current_data) < 20:  # Need minimum data for MQScore
                    continue
                
                # Get current price data
                current_bar = current_data.iloc[-1]
                current_price = current_bar['close']
                
                # Create market data dict for NEXUS AI
                market_dict = {
                    'symbol': symbol,
                    'timestamp': timestamp.timestamp(),
                    'price': current_price,
                    'open': current_bar['open'],
                    'high': current_bar['high'],
                    'low': current_bar['low'],
                    'close': current_price,
                    'volume': current_bar['volume'],
                    'bid': current_price * 0.9995,  # Simulate bid/ask
                    'ask': current_price * 1.0005,
                }
                
                # Execute NEXUS AI strategies
                # Use the strategy manager to execute all strategies
                strategy_results = []
                for strategy_name in self.config.enabled_strategies:
                    try:
                        # Get the strategy adapter
                        strategies = self.nexus_ai.strategy_manager.get_strategies()
                        for name, strategy in strategies.items():
                            if strategy_name.lower() in name.lower():
                                # Execute strategy with current data
                                result = strategy.execute(current_data, symbol)
                                if result and result.get('signal', 0) != 0:
                                    strategy_results.append({
                                        'strategy': strategy_name,
                                        'signal': result.get('signal', 0),
                                        'confidence': result.get('confidence', 0),
                                        'metadata': result.get('metadata', {})
                                    })
                                break
                    except Exception as e:
                        self.logger.debug(f"Strategy {strategy_name} error: {e}")
                        continue
                
                # Process strategy results
                for result in strategy_results:
                    signal_strength = result['signal']
                    confidence = result.get('confidence', 0)
                    
                    # Apply filters
                    if confidence >= self.config.min_confidence:
                        # Check MQScore filter if enabled
                        mqscore = result.get('metadata', {}).get('mqscore_quality')
                        
                        if (not self.config.use_mqscore_filter or 
                            not mqscore or 
                            mqscore >= self.config.mqscore_threshold):
                            
                            signals.append({
                                'timestamp': timestamp,
                                'symbol': symbol,
                                'signal': signal_strength,
                                'confidence': confidence,
                                'strategy': result.get('strategy', 'unknown'),
                                'price': current_price,
                                'mqscore': mqscore,
                                'metadata': result.get('metadata', {})
                            })
                            
            except Exception as e:
                self.logger.error(f"Error processing signals for {symbol} at {timestamp}: {e}")
                continue
        
        return signals
    
    def execute_signals(self, signals: List[Dict], timestamp: datetime):
        """Execute trading signals."""
        for signal_data in signals:
            try:
                symbol = signal_data['symbol']
                signal_strength = signal_data['signal']
                confidence = signal_data['confidence']
                price = signal_data['price']
                strategy = signal_data['strategy']
                mqscore = signal_data.get('mqscore')
                
                # Determine position size based on confidence and signal strength
                portfolio_value = self.portfolio.get_portfolio_value({symbol: price})
                max_position_value = portfolio_value * self.config.max_position_size
                
                # Scale position size by confidence
                position_value = max_position_value * confidence * abs(signal_strength)
                quantity = position_value / price
                
                # Determine side
                side = "BUY" if signal_strength > 0 else "SELL"
                
                # Execute trade
                success = self.portfolio.execute_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    strategy=strategy,
                    confidence=confidence,
                    timestamp=timestamp,
                    mqscore=mqscore
                )
                
                if success:
                    self.logger.debug(f"Executed {side} {quantity:.4f} {symbol} @ {price} (confidence: {confidence:.3f})")
                    
            except Exception as e:
                self.logger.error(f"Error executing signal: {e}")
                continue
    
    async def run_backtest(self) -> BacktestResults:
        """Run the complete backtesting process."""
        self.logger.info("Starting NEXUS AI Deep Backtest...")
        
        # Initialize NEXUS AI
        if not await self.initialize_nexus_ai():
            raise RuntimeError("Failed to initialize NEXUS AI system")
        
        # Generate market data
        market_data = self.generate_market_data()
        
        # Get all timestamps (intersection of all symbols)
        all_timestamps = None
        for symbol, df in market_data.items():
            if all_timestamps is None:
                all_timestamps = set(df.index)
            else:
                all_timestamps = all_timestamps.intersection(set(df.index))
        
        timestamps = sorted(list(all_timestamps))
        self.logger.info(f"Backtesting over {len(timestamps)} time periods")
        
        # Run backtest
        for i, timestamp in enumerate(timestamps):
            try:
                # Get current prices for portfolio valuation
                current_prices = {}
                for symbol, df in market_data.items():
                    if timestamp in df.index:
                        current_prices[symbol] = df.loc[timestamp, 'close']
                
                # Update unrealized PnL
                self.portfolio.update_unrealized_pnl(current_prices, timestamp)
                
                # Process signals
                signals = await self.process_signals_for_timestamp(timestamp, market_data)
                
                # Execute signals
                if signals:
                    self.execute_signals(signals, timestamp)
                
                # Progress logging
                if i % 100 == 0:
                    portfolio_value = self.portfolio.get_portfolio_value(current_prices)
                    pnl_pct = ((portfolio_value - self.config.initial_capital) / 
                              self.config.initial_capital) * 100
                    self.logger.info(f"Progress: {i}/{len(timestamps)} ({i/len(timestamps)*100:.1f}%) - "
                                   f"Portfolio: ${portfolio_value:,.2f} ({pnl_pct:+.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Error at timestamp {timestamp}: {e}")
                continue
        
        # Calculate final results
        self.results = self.calculate_results(market_data)
        
        self.logger.info("Backtest completed successfully!")
        return self.results
    
    def calculate_results(self, market_data: Dict[str, pd.DataFrame]) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        self.logger.info("Calculating backtest results...")
        
        # Get final portfolio value
        final_prices = {}
        for symbol, df in market_data.items():
            final_prices[symbol] = df.iloc[-1]['close']
        
        final_portfolio_value = self.portfolio.get_portfolio_value(final_prices)
        
        # Create equity curve
        equity_df = pd.DataFrame(self.portfolio.equity_history, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        total_return = (final_portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # Calculate time-based metrics
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        years = (end_date - start_date).days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        # Calculate drawdown
        equity_df['peak'] = equity_df['equity'].expanding().max()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Calculate Sharpe ratio
        equity_df['returns'] = equity_df['equity'].pct_change()
        sharpe_ratio = (equity_df['returns'].mean() * 252) / (equity_df['returns'].std() * np.sqrt(252))
        
        # Trade statistics
        trades = self.portfolio.trade_history
        total_trades = len(trades)
        
        if total_trades > 0:
            # Calculate trade PnL
            trade_pnls = []
            for trade in trades:
                # Simplified PnL calculation (would need more sophisticated logic for real backtesting)
                if trade.side == "BUY":
                    trade_pnls.append(-trade.commission - trade.slippage)  # Entry cost
                else:
                    trade_pnls.append(-trade.commission - trade.slippage)  # Exit cost
            
            winning_trades = len([pnl for pnl in trade_pnls if pnl > 0])
            losing_trades = len([pnl for pnl in trade_pnls if pnl < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean([pnl for pnl in trade_pnls if pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([pnl for pnl in trade_pnls if pnl < 0]) if losing_trades > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else 0
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Strategy performance
        strategy_stats = defaultdict(lambda: {'trades': 0, 'returns': 0})
        for trade in trades:
            strategy_stats[trade.strategy]['trades'] += 1
        
        strategy_returns = {k: v['returns'] for k, v in strategy_stats.items()}
        strategy_trades = {k: v['trades'] for k, v in strategy_stats.items()}
        strategy_win_rates = {k: 0.5 for k in strategy_stats.keys()}  # Placeholder
        
        # MQScore statistics
        mqscores = [trade.mqscore for trade in trades if trade.mqscore is not None]
        mqscore_stats = {
            'avg_mqscore': np.mean(mqscores) if mqscores else 0,
            'min_mqscore': np.min(mqscores) if mqscores else 0,
            'max_mqscore': np.max(mqscores) if mqscores else 0,
            'trades_with_mqscore': len(mqscores)
        }
        
        # Risk metrics
        returns = equity_df['returns'].dropna()
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0
        
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = (annualized_return / (downside_returns.std() * np.sqrt(252))) if len(downside_returns) > 0 else 0
        
        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            strategy_returns=strategy_returns,
            strategy_trades=strategy_trades,
            strategy_win_rates=strategy_win_rates,
            equity_curve=equity_df,
            drawdown_curve=equity_df[['drawdown']],
            trade_history=trades,
            mqscore_stats=mqscore_stats,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )
    
    def generate_report(self, output_file: str = "nexus_backtest_report.html"):
        """Generate comprehensive HTML report."""
        if not self.results:
            self.logger.error("No results available for report generation")
            return
        
        self.logger.info(f"Generating backtest report: {output_file}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NEXUS AI Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 NEXUS AI Deep Backtest Report</h1>
                <p>Period: {self.config.start_date} to {self.config.end_date}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Performance Summary</h2>
                <div class="metric">
                    <strong>Total Return:</strong> 
                    <span class="{'positive' if self.results.total_return > 0 else 'negative'}">
                        {self.results.total_return:.2%}
                    </span>
                </div>
                <div class="metric">
                    <strong>Annualized Return:</strong> 
                    <span class="{'positive' if self.results.annualized_return > 0 else 'negative'}">
                        {self.results.annualized_return:.2%}
                    </span>
                </div>
                <div class="metric">
                    <strong>Sharpe Ratio:</strong> {self.results.sharpe_ratio:.2f}
                </div>
                <div class="metric">
                    <strong>Max Drawdown:</strong> 
                    <span class="negative">{self.results.max_drawdown:.2%}</span>
                </div>
                <div class="metric">
                    <strong>Win Rate:</strong> {self.results.win_rate:.2%}
                </div>
                <div class="metric">
                    <strong>Profit Factor:</strong> {self.results.profit_factor:.2f}
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Trade Statistics</h2>
                <div class="metric"><strong>Total Trades:</strong> {self.results.total_trades}</div>
                <div class="metric"><strong>Winning Trades:</strong> {self.results.winning_trades}</div>
                <div class="metric"><strong>Losing Trades:</strong> {self.results.losing_trades}</div>
                <div class="metric"><strong>Average Win:</strong> ${self.results.avg_win:.2f}</div>
                <div class="metric"><strong>Average Loss:</strong> ${self.results.avg_loss:.2f}</div>
            </div>
            
            <div class="section">
                <h2>🎯 Strategy Performance</h2>
                <table>
                    <tr><th>Strategy</th><th>Trades</th><th>Win Rate</th></tr>
        """
        
        for strategy, trades in self.results.strategy_trades.items():
            win_rate = self.results.strategy_win_rates.get(strategy, 0)
            html_content += f"<tr><td>{strategy}</td><td>{trades}</td><td>{win_rate:.2%}</td></tr>"
        
        html_content += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>🔍 MQScore Analysis</h2>
                <div class="metric"><strong>Average MQScore:</strong> {self.results.mqscore_stats['avg_mqscore']:.3f}</div>
                <div class="metric"><strong>Min MQScore:</strong> {self.results.mqscore_stats['min_mqscore']:.3f}</div>
                <div class="metric"><strong>Max MQScore:</strong> {self.results.mqscore_stats['max_mqscore']:.3f}</div>
                <div class="metric"><strong>Trades with MQScore:</strong> {self.results.mqscore_stats['trades_with_mqscore']}</div>
            </div>
            
            <div class="section">
                <h2>⚠️ Risk Metrics</h2>
                <div class="metric"><strong>VaR (95%):</strong> {self.results.var_95:.2%}</div>
                <div class="metric"><strong>CVaR (95%):</strong> {self.results.cvar_95:.2%}</div>
                <div class="metric"><strong>Calmar Ratio:</strong> {self.results.calmar_ratio:.2f}</div>
                <div class="metric"><strong>Sortino Ratio:</strong> {self.results.sortino_ratio:.2f}</div>
            </div>
            
            <div class="section">
                <h2>⚙️ Configuration</h2>
                <div class="metric"><strong>Initial Capital:</strong> ${self.config.initial_capital:,.2f}</div>
                <div class="metric"><strong>Symbols:</strong> {', '.join(self.config.symbols)}</div>
                <div class="metric"><strong>Timeframe:</strong> {self.config.timeframe}</div>
                <div class="metric"><strong>Commission:</strong> {self.config.commission:.3%}</div>
                <div class="metric"><strong>Slippage:</strong> {self.config.slippage:.3%}</div>
                <div class="metric"><strong>Min Confidence:</strong> {self.config.min_confidence:.2f}</div>
                <div class="metric"><strong>MQScore Threshold:</strong> {self.config.mqscore_threshold:.2f}</div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Report generated: {output_file}")


# Example usage and main execution
async def main():
    """Main execution function for backtesting."""
    
    # Configure backtest
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-10-22",
        initial_capital=100000.0,
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        timeframe="1h",
        use_mqscore_filter=True,
        enable_ml_models=True
    )
    
    # Create backtester
    backtester = NexusBacktester(config)
    
    try:
        # Run backtest
        results = await backtester.run_backtest()
        
        # Print summary
        print("\n" + "="*60)
        print("🚀 NEXUS AI BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annualized Return: {results.annualized_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Total Trades: {results.total_trades}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print("="*60)
        
        # Generate report
        backtester.generate_report("nexus_backtest_report.html")
        
        # Save results to JSON
        results_dict = {
            'config': config.__dict__,
            'performance': {
                'total_return': results.total_return,
                'annualized_return': results.annualized_return,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown': results.max_drawdown,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'total_trades': results.total_trades
            },
            'strategy_performance': results.strategy_returns,
            'mqscore_stats': results.mqscore_stats
        }
        
        with open('nexus_backtest_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print("✅ Backtest completed successfully!")
        print("📊 Report saved to: nexus_backtest_report.html")
        print("💾 Results saved to: nexus_backtest_results.json")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())