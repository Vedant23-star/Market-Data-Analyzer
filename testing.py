from data_loader import load_data

# # Load
nifty = load_data("C:/Users/Lenovo/OneDrive/Desktop/Project/Market Data Analyzer/data/nifty.csv")
# sp500 = load_data("C:/Users/Lenovo/OneDrive/Desktop/Project/Market Data Analyzer/data/sp500.csv")

# print("NIFTY:\n", nifty.head())
# print("\nS&P500:\n", sp500.head())

# # Align
# nifty_aligned, sp500_aligned = align_dates(nifty, sp500)

# print("\nAligned shapes:", nifty_aligned.shape, sp500_aligned.shape)
# print("\nAligned index start:", nifty_aligned.index[0])
# print("Aligned index end:", nifty_aligned.index[-1])

# from analytics import calculate_cagr
# print(calculate_cagr(nifty)) 
# from analytics import calculate_mdd
# print(calculate_mdd(nifty))
# from analytics import calculate_annualized_volatility
# print(calculate_annualized_volatility(nifty))
import pandas as pd

# Import all functions from your two modules
from analytics import (
    calculate_cagr, calculate_mdd, calculate_sharpe_ratio, 
    calculate_annualized_volatility, calculate_sortino_ratio
) 
from backtesting import (
    generate_crossover_signals, 
    backtest_strategy, 
    analyze_strategy
)

# Since the SMA Crossover uses rolling windows and then drops NaNs, 
# it's best to work on a copy of the data.
trading_df = nifty.copy()

## --- 2. Generate Signals ---

print("\n--- Starting SMA Crossover Analysis ---")

# Apply the signal generation function. 
# It adds 'SMA_50', 'SMA_200', and 'Signal' columns to trading_df and drops the initial NaNs.
# We keep a copy of the original DF (before dropping the initial 200 rows) for the B&H benchmark
original_df_for_bnh = nifty.copy()
strategy_df_with_signals = generate_crossover_signals(trading_df, fast_window=50, slow_window=200)

print(f"Signals generated successfully for a total of {len(strategy_df_with_signals)} periods.")


## --- 3. Run Backtest Simulation ---

# Apply the backtesting function. 
# This calculates 'Strategy_Return' and 'Cumulative_Returns' based on the signals.
final_strategy_df = backtest_strategy(strategy_df_with_signals)

print("Backtesting simulation completed.")


## --- 4. Analyze and Display Results ---

# The analyze_strategy function takes:
# 1. The original DataFrame (for Buy & Hold benchmark)
# 2. The strategy DataFrame (for strategy metrics)

# NOTE: Since the strategy drops the initial 200 rows, 
# we pass the *same trimmed data* to the B&H benchmark so the comparison periods match exactly.
bnh_benchmark_df = original_df_for_bnh.loc[final_strategy_df.index]

analysis_result_df = analyze_strategy(bnh_benchmark_df, final_strategy_df)

print("\n--- Strategy Analysis Complete ---")

# def debug_signals(df, fast_window=50, slow_window=200):
#     df = df.copy() 
#     df[f'SMA_{fast_window}'] = df["Close"].rolling(window=fast_window).mean()
#     df[f'SMA_{slow_window}'] = df["Close"].rolling(window=slow_window).mean()
#     df.dropna(inplace=True) 
    
#     # 1. Base Crossover Condition (Fast > Slow TODAY AND Fast <= Slow YESTERDAY)
#     df['Crossover_Base'] = (df[f'SMA_{fast_window}'] > df[f'SMA_{slow_window}']) & \
#                           (df[f'SMA_{fast_window}'].shift(1) <= df[f'SMA_{slow_window}'].shift(1))
    
#     # 2. Trend Filter Condition
#     df['Filter_Condition'] = (df['Close'] > df[f'SMA_{slow_window}'])
    
#     # 3. Combined BUY Signal
#     df['Buy_Signal_Combined'] = df['Crossover_Base'] & df['Filter_Condition']
    
#     return df

# debug_df = debug_signals(nifty) 
# print(debug_df[debug_df['Crossover_Base'] == True])
# print(debug_df[debug_df['Buy_Signal_Combined'] == True])