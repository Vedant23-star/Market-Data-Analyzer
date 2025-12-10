from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import pandas as pd
import numpy as np

# Import your existing modules
from models import BacktestResult, PerformanceMetrics
from backtesting import generate_crossover_signals, backtest_strategy
from analytics import (
    calculate_cagr, calculate_mdd, calculate_sharpe_ratio, 
    calculate_sortino_ratio, calculate_annualized_volatility,
    calculate_rsi, calculate_macd, calculate_correlation
)

# -------------------------------------------------------------------
# NEW PYDANTIC MODELS FOR THE NEW ENDPOINTS
# -------------------------------------------------------------------

class MetricsResponse(BaseModel):
    """Response model for individual market metrics"""
    ticker: str
    cagr: float = Field(..., description="Compound Annual Growth Rate")
    mdd: float = Field(..., description="Maximum Drawdown")
    sharpe_ratio: float = Field(..., description="Sharpe Ratio")
    sortino_ratio: float = Field(..., description="Sortino Ratio")
    annualized_volatility: float = Field(..., description="Annualized Volatility")
    current_rsi: float = Field(..., description="Current RSI value")
    data_points: int = Field(..., description="Number of data points analyzed")

class CorrelationResponse(BaseModel):
    """Response model for correlation analysis"""
    correlation: float = Field(..., description="Correlation coefficient between NSE and S&P 500")
    nse_return: float = Field(..., description="Total return for NSE")
    sp500_return: float = Field(..., description="Total return for S&P 500")
    analysis_period: str = Field(..., description="Period of analysis")

class StrategyResponse(BaseModel):
    """Response model for strategy analysis"""
    ticker: str
    strategy_cagr: float
    benchmark_cagr: float
    strategy_mdd: float
    benchmark_mdd: float
    strategy_sharpe: float
    benchmark_sharpe: float
    total_signals: int
    win_rate: Optional[float] = Field(None, description="Percentage of profitable trades")
    recommendation: str = Field(..., description="Strategy recommendation")

# -------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -------------------------------------------------------------------

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional

def load_real_data(ticker: str) -> Optional[pd.DataFrame]:
    """Try to load real market data from Yahoo Finance."""
    try:
        symbol = '^NSEI' if ticker == 'NSE' else '^GSPC'
        df = yf.download(symbol, start='2020-01-01', period='max', progress=False)
        
        if df is None or df.empty:
            return None
        
        df = df.reset_index()
        
        # Handle different column names
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        
        return df[['Date', 'Close']].dropna()
    
    except Exception as e:
        print(f"Failed to load real data: {e}")
        return None

def load_data(ticker: str) -> pd.DataFrame:
    """
    Load market data - tries real data first, falls back to dummy data.
    """
    print(f"\n{'='*60}")
    print(f"Loading data for: {ticker}")
    
    # Try real data
    try:
        symbol = '^NSEI' if ticker == 'NSE' else '^GSPC'
        print(f"Attempting to download {symbol} from Yahoo Finance...")
        
        df = yf.download(symbol, start='2020-01-01', period='max', progress=False, auto_adjust=True)
        
        if df is not None and not df.empty:
            print(f"Downloaded {len(df)} rows")
            
            # CRITICAL FIX: Handle MultiIndex columns from yfinance
            # If columns are MultiIndex, flatten them
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Reset index to get Date as a column
            df = df.reset_index()
            
            # Handle different date column names
            if 'Datetime' in df.columns:
                df = df.rename(columns={'Datetime': 'Date'})
            
            # Ensure we have the required columns
            if 'Date' in df.columns and 'Close' in df.columns:
                # Select only Date and Close, and ensure Close is a simple Series
                df = df[['Date', 'Close']].copy()
                
                # Drop any NaN values
                df = df.dropna()
                
                # Ensure Close is numeric
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df = df.dropna()
                
                if len(df) > 0:
                    print(f"✅ Successfully loaded {len(df)} real data points")
                    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
                    # Fix the print statement - convert to float first
                    print(f"Close range: {float(df['Close'].min()):.2f} to {float(df['Close'].max()):.2f}")
                    print(f"{'='*60}\n")
                    return df
        
        print("⚠️ Real data not available or invalid, using dummy data")
    
    except Exception as e:
        print(f"❌ Error loading real data: {e}")
        import traceback
        print(traceback.format_exc())
    
    # Fallback to dummy data
    print(f"📊 Generating dummy data for {ticker}")
    np.random.seed(42 if ticker == 'NSE' else 123)
    
    dates = pd.date_range('2020-01-01', periods=1250, freq='B')
    if ticker == 'NSE':
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.012, 1250)))
    else:
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0006, 0.01, 1250)))
    
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    print(f"✅ Generated {len(df)} dummy data points")
    print(f"{'='*60}\n")
    return df

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def calculate_win_rate(strategy_df: pd.DataFrame) -> float:
    """Calculate the percentage of profitable trades"""
    trades = strategy_df[strategy_df['Signal'] != 0].copy()
    if len(trades) == 0:
        return 0.0
    
    # Get returns between signals
    trade_returns = []
    for i in range(len(trades) - 1):
        if trades.iloc[i]['Signal'] == 1:  # Buy signal
            entry_price = trades.iloc[i]['Close']
            exit_price = trades.iloc[i + 1]['Close']
            trade_return = (exit_price - entry_price) / entry_price
            trade_returns.append(trade_return)
    
    if len(trade_returns) == 0:
        return 0.0
    
    winning_trades = sum(1 for r in trade_returns if r > 0)
    win_rate = (winning_trades / len(trade_returns)) * 100
    return round(win_rate, 2)

def get_recommendation(strategy_sharpe: float, benchmark_sharpe: float, 
                       strategy_cagr: float, benchmark_cagr: float) -> str:
    """Generate trading recommendation based on performance metrics"""
    if strategy_sharpe > benchmark_sharpe and strategy_cagr > benchmark_cagr * 0.7:
        return "STRONG BUY - Strategy outperforms on risk-adjusted basis"
    elif strategy_sharpe > benchmark_sharpe:
        return "BUY - Better risk-adjusted returns despite lower absolute returns"
    elif strategy_cagr > benchmark_cagr:
        return "HOLD - Higher returns but worse risk profile"
    else:
        return "AVOID - Buy & Hold is superior on all metrics"

def analyze_strategy_api_refactored(org_df, strategy_df):
    """
    Calculates and returns performance metrics as a dictionary for the API.
    """
    # 1. Strategy Data Preparation
    strategy_analysis_df = strategy_df[['Cumulative_Returns']].rename(
        columns={'Cumulative_Returns': 'Close'}
    ).copy()
    
    strategy_analysis_df = strategy_analysis_df.reset_index()
    
    if strategy_analysis_df.columns[0] != 'Date':
        strategy_analysis_df = strategy_analysis_df.rename(
            columns={strategy_analysis_df.columns[0]: 'Date'}
        )
    
    # Safety check
    if strategy_analysis_df.empty or len(strategy_analysis_df) < 2:
        return {
            'cagr_strategy': 0.0, 'cagr_bnh': 0.0,
            'mdd_strategy': 0.0, 'mdd_bnh': 0.0,
            'sharpe_strategy': 0.0, 'sharpe_bnh': 0.0,
            'total_signals': 0
        }
    
    if 'Close' not in strategy_analysis_df.columns:
        return {
            'cagr_strategy': 0.0, 'cagr_bnh': 0.0,
            'mdd_strategy': 0.0, 'mdd_bnh': 0.0,
            'sharpe_strategy': 0.0, 'sharpe_bnh': 0.0,
            'total_signals': 0
        }

    # 2. B&H Data Preparation
    original_analysis_df = org_df[['Date', 'Close']].copy() 
    original_analysis_df = original_analysis_df[
        original_analysis_df['Date'].isin(strategy_analysis_df['Date'])
    ].reset_index(drop=True)

    # 3. Calculate Metrics
    bnh_cagr = calculate_cagr(original_analysis_df.copy())
    bnh_mdd = calculate_mdd(original_analysis_df.copy())
    bnh_sharpe = calculate_sharpe_ratio(original_analysis_df.copy())

    strategy_cagr = calculate_cagr(strategy_analysis_df.copy())
    strategy_mdd = calculate_mdd(strategy_analysis_df.copy())
    strategy_sharpe = calculate_sharpe_ratio(strategy_analysis_df.copy())
    total_signals = int(strategy_df['Signal'].abs().sum())

    return {
        'cagr_strategy': float(strategy_cagr),
        'cagr_bnh': float(bnh_cagr),
        'mdd_strategy': float(strategy_mdd),
        'mdd_bnh': float(bnh_mdd),
        'sharpe_strategy': float(strategy_sharpe),
        'sharpe_bnh': float(bnh_sharpe),
        'total_signals': total_signals,
    }

# -------------------------------------------------------------------
# FASTAPI APPLICATION
# -------------------------------------------------------------------

app = FastAPI(
    title="Market Data Analyzer API",
    description="Comprehensive API for market metrics, correlation analysis, and strategy backtesting",
    version="2.0.0"
)

# -------------------------------------------------------------------
# METRICS ENDPOINTS
# -------------------------------------------------------------------

@app.get("/metrics/nse", response_model=MetricsResponse)
async def get_nse_metrics():
    """
    Get comprehensive metrics for NSE (National Stock Exchange).
    Returns CAGR, MDD, Sharpe Ratio, Sortino Ratio, Volatility, and current RSI.
    """
    try:
        df = load_data('NSE')
        df_for_analysis = df.set_index('Date')
        
        # Calculate all metrics
        cagr = calculate_cagr(df.copy())
        mdd = calculate_mdd(df.copy())
        sharpe = calculate_sharpe_ratio(df.copy())
        sortino = calculate_sortino_ratio(df_for_analysis.copy())
        volatility = calculate_annualized_volatility(df_for_analysis.copy())
        rsi = calculate_rsi(df_for_analysis.copy())
        current_rsi = float(rsi.iloc[-1])
        
        return MetricsResponse(
            ticker="NSE",
            cagr=float(cagr),
            mdd=float(mdd),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            annualized_volatility=float(volatility),
            current_rsi=current_rsi,
            data_points=len(df)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating NSE metrics: {str(e)}")

@app.get("/metrics/sp500", response_model=MetricsResponse)
async def get_sp500_metrics():
    """
    Get comprehensive metrics for S&P 500.
    Returns CAGR, MDD, Sharpe Ratio, Sortino Ratio, Volatility, and current RSI.
    """
    try:
        df = load_data('SP500')
        df_for_analysis = df.set_index('Date')
        
        # Calculate all metrics
        cagr = calculate_cagr(df.copy())
        mdd = calculate_mdd(df.copy())
        sharpe = calculate_sharpe_ratio(df.copy())
        sortino = calculate_sortino_ratio(df_for_analysis.copy())
        volatility = calculate_annualized_volatility(df_for_analysis.copy())
        rsi = calculate_rsi(df_for_analysis.copy())
        current_rsi = float(rsi.iloc[-1])
        
        return MetricsResponse(
            ticker="SP500",
            cagr=float(cagr),
            mdd=float(mdd),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            annualized_volatility=float(volatility),
            current_rsi=current_rsi,
            data_points=len(df)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating SP500 metrics: {str(e)}")

# -------------------------------------------------------------------
# CORRELATION ENDPOINT
# -------------------------------------------------------------------

@app.get("/metrics/correlation", response_model=CorrelationResponse)
async def get_correlation():
    """
    Calculate the correlation between NSE and S&P 500.
    Returns correlation coefficient and individual returns.
    """
    try:
        # Load both datasets
        nse_df = load_data('NSE')
        sp500_df = load_data('SP500')
        
        # Ensure both have the same date range
        common_dates = set(nse_df['Date']) & set(sp500_df['Date'])
        nse_df = nse_df[nse_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
        sp500_df = sp500_df[sp500_df['Date'].isin(common_dates)].sort_values('Date').reset_index(drop=True)
        
        # Calculate correlation
        nse_df_indexed = nse_df.set_index('Date')
        sp500_df_indexed = sp500_df.set_index('Date')
        
        correlation = calculate_correlation(nse_df_indexed, sp500_df_indexed, window=None)
        
        # Calculate total returns
        nse_return = (nse_df['Close'].iloc[-1] / nse_df['Close'].iloc[0] - 1) * 100
        sp500_return = (sp500_df['Close'].iloc[-1] / sp500_df['Close'].iloc[0] - 1) * 100
        
        # Analysis period
        start_date = nse_df['Date'].iloc[0].strftime('%Y-%m-%d')
        end_date = nse_df['Date'].iloc[-1].strftime('%Y-%m-%d')
        period = f"{start_date} to {end_date}"
        
        return CorrelationResponse(
            correlation=round(float(correlation), 4),
            nse_return=round(float(nse_return), 2),
            sp500_return=round(float(sp500_return), 2),
            analysis_period=period
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating correlation: {str(e)}")

# -------------------------------------------------------------------
# STRATEGY ENDPOINTS
# -------------------------------------------------------------------

@app.get("/strategy/nse", response_model=StrategyResponse)
async def get_nse_strategy(fast_window: int = 20, slow_window: int = 50):
    """
    Run SMA Crossover strategy backtest for NSE.
    Returns comprehensive strategy performance vs buy & hold.
    """
    try:
        org_df = load_data('NSE')
        
        # Run strategy
        strategy_df_with_signals = generate_crossover_signals(org_df.copy(), fast_window, slow_window)
        final_strategy_df = backtest_strategy(strategy_df_with_signals)
        
        # Calculate metrics
        metrics_dict = analyze_strategy_api_refactored(org_df, final_strategy_df)
        
        # Calculate win rate
        win_rate = calculate_win_rate(final_strategy_df)
        
        # Generate recommendation
        recommendation = get_recommendation(
            metrics_dict['sharpe_strategy'],
            metrics_dict['sharpe_bnh'],
            metrics_dict['cagr_strategy'],
            metrics_dict['cagr_bnh']
        )
        
        return StrategyResponse(
            ticker="NSE",
            strategy_cagr=metrics_dict['cagr_strategy'],
            benchmark_cagr=metrics_dict['cagr_bnh'],
            strategy_mdd=metrics_dict['mdd_strategy'],
            benchmark_mdd=metrics_dict['mdd_bnh'],
            strategy_sharpe=metrics_dict['sharpe_strategy'],
            benchmark_sharpe=metrics_dict['sharpe_bnh'],
            total_signals=metrics_dict['total_signals'],
            win_rate=win_rate,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running NSE strategy: {str(e)}")

@app.get("/strategy/sp500", response_model=StrategyResponse)
async def get_sp500_strategy(fast_window: int = 20, slow_window: int = 50):
    """
    Run SMA Crossover strategy backtest for S&P 500.
    Returns comprehensive strategy performance vs buy & hold.
    """
    try:
        org_df = load_data('SP500')
        
        # Run strategy
        strategy_df_with_signals = generate_crossover_signals(org_df.copy(), fast_window, slow_window)
        final_strategy_df = backtest_strategy(strategy_df_with_signals)
        
        # Calculate metrics
        metrics_dict = analyze_strategy_api_refactored(org_df, final_strategy_df)
        
        # Calculate win rate
        win_rate = calculate_win_rate(final_strategy_df)
        
        # Generate recommendation
        recommendation = get_recommendation(
            metrics_dict['sharpe_strategy'],
            metrics_dict['sharpe_bnh'],
            metrics_dict['cagr_strategy'],
            metrics_dict['cagr_bnh']
        )
        
        return StrategyResponse(
            ticker="SP500",
            strategy_cagr=metrics_dict['cagr_strategy'],
            benchmark_cagr=metrics_dict['cagr_bnh'],
            strategy_mdd=metrics_dict['mdd_strategy'],
            benchmark_mdd=metrics_dict['mdd_bnh'],
            strategy_sharpe=metrics_dict['sharpe_strategy'],
            benchmark_sharpe=metrics_dict['sharpe_bnh'],
            total_signals=metrics_dict['total_signals'],
            win_rate=win_rate,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running SP500 strategy: {str(e)}")

# -------------------------------------------------------------------
# ROOT ENDPOINT
# -------------------------------------------------------------------

@app.get("/")
async def root():
    """API information and available endpoints"""
    return {
        "message": "Market Data Analyzer API",
        "version": "2.0.0",
        "endpoints": {
            "metrics": [
                "/metrics/nse - Get NSE market metrics",
                "/metrics/sp500 - Get S&P 500 market metrics",
                "/metrics/correlation - Get NSE vs SP500 correlation"
            ],
            "strategies": [
                "/strategy/nse?fast_window=20&slow_window=50 - NSE strategy backtest",
                "/strategy/sp500?fast_window=20&slow_window=50 - SP500 strategy backtest"
            ],
            "legacy": [
                "/backtest/crossover - Original crossover backtest endpoint"
            ]
        },
        "documentation": "/docs"
    }

# -------------------------------------------------------------------
# KEEP YOUR ORIGINAL ENDPOINT
# -------------------------------------------------------------------

@app.get("/backtest/crossover", response_model=BacktestResult)
async def run_sma_crossover_backtest(
    ticker: str = 'NSE',
    fast_window: int = 20,
    slow_window: int = 50
):
    """
    Original backtest endpoint - runs the SMA Crossover backtest.
    """
    try:
        org_df = load_data(ticker)
        strategy_df_with_signals = generate_crossover_signals(org_df.copy(), fast_window, slow_window)
        final_strategy_df = backtest_strategy(strategy_df_with_signals)
        metrics_dict = analyze_strategy_api_refactored(org_df, final_strategy_df)
        
        metrics = PerformanceMetrics(**metrics_dict)
        return BacktestResult(
            metrics=metrics,
            plot_status="Metrics calculated successfully."
        )
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{error_details}")


# @app.get("/test/data")
# async def test_data_loading():
#     """Test endpoint to check data loading"""
#     try:
#         nse_df = load_data('NSE')
#         sp500_df = load_data('SP500')
        
#         return {
#             "nse": {
#                 "rows": len(nse_df),
#                 "columns": nse_df.columns.tolist(),
#                 "dtypes": {col: str(dtype) for col, dtype in nse_df.dtypes.items()},
#                 "first_date": str(nse_df['Date'].iloc[0]),
#                 "last_date": str(nse_df['Date'].iloc[-1]),
#                 "first_close": float(nse_df['Close'].iloc[0]),
#                 "last_close": float(nse_df['Close'].iloc[-1])
#             },
#             "sp500": {
#                 "rows": len(sp500_df),
#                 "columns": sp500_df.columns.tolist(),
#                 "dtypes": {col: str(dtype) for col, dtype in sp500_df.dtypes.items()},
#                 "first_date": str(sp500_df['Date'].iloc[0]),
#                 "last_date": str(sp500_df['Date'].iloc[-1]),
#                 "first_close": float(sp500_df['Close'].iloc[0]),
#                 "last_close": float(sp500_df['Close'].iloc[-1])
#             }
#         }
#     except Exception as e:
#         import traceback
#         return {
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }

# @app.get("/test/analytics")
# async def test_analytics():
#     """Test each analytics function individually"""
#     results = {}
    
#     try:
#         # Create simple test data
#         dates = pd.date_range('2023-01-01', periods=100, freq='D')
#         prices = 100 + np.cumsum(np.random.randn(100))
#         test_df = pd.DataFrame({'Date': dates, 'Close': prices})
        
#         print("Testing analytics functions...")
        
#         # Test CAGR
#         try:
#             cagr = calculate_cagr(test_df.copy())
#             results['cagr'] = {'status': 'success', 'value': float(cagr)}
#         except Exception as e:
#             results['cagr'] = {'status': 'failed', 'error': str(e)}
        
#         # Test MDD
#         try:
#             mdd = calculate_mdd(test_df.copy())
#             results['mdd'] = {'status': 'success', 'value': float(mdd)}
#         except Exception as e:
#             results['mdd'] = {'status': 'failed', 'error': str(e)}
        
#         # Test Sharpe
#         try:
#             sharpe = calculate_sharpe_ratio(test_df.copy())
#             results['sharpe'] = {'status': 'success', 'value': float(sharpe)}
#         except Exception as e:
#             results['sharpe'] = {'status': 'failed', 'error': str(e)}
        
#         # Test Sortino (needs Date as index)
#         try:
#             test_df_indexed = test_df.set_index('Date')
#             sortino = calculate_sortino_ratio(test_df_indexed.copy())
#             results['sortino'] = {'status': 'success', 'value': float(sortino)}
#         except Exception as e:
#             results['sortino'] = {'status': 'failed', 'error': str(e)}
        
#         # Test Volatility
#         try:
#             test_df_indexed = test_df.set_index('Date')
#             vol = calculate_annualized_volatility(test_df_indexed.copy())
#             results['volatility'] = {'status': 'success', 'value': float(vol)}
#         except Exception as e:
#             results['volatility'] = {'status': 'failed', 'error': str(e)}
        
#         # Test RSI
#         try:
#             test_df_indexed = test_df.set_index('Date')
#             rsi = calculate_rsi(test_df_indexed.copy())
#             results['rsi'] = {'status': 'success', 'value': float(rsi.iloc[-1])}
#         except Exception as e:
#             results['rsi'] = {'status': 'failed', 'error': str(e)}
        
#         return results
    
#     except Exception as e:
#         import traceback
#         return {
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }