from analytics import calculate_cagr, calculate_mdd, calculate_sharpe_ratio

def generate_crossover_signals(df, fast_window=50, slow_window=200):
    """
    Generates trading signals (1=BUY, -1=SELL) based on the SMA Crossover.
    """
    df = df.copy()
    
    # Ensure Date is preserved
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
    
    # Calculate SMAs
    df[f'SMA_{fast_window}'] = df["Close"].rolling(window=fast_window).mean()
    df[f'SMA_{slow_window}'] = df["Close"].rolling(window=slow_window).mean()
    
    # Drop NaNs but keep Date column
    df = df.dropna().reset_index(drop=True)
    
    # Initialize Signal column
    df['Signal'] = 0
    
    # BUY Signal (Golden Cross + Trend Filter)
    df.loc[
        (df[f'SMA_{fast_window}'] > df[f'SMA_{slow_window}']) & 
        (df[f'SMA_{fast_window}'].shift(1) <= df[f'SMA_{slow_window}'].shift(1)) &
        (df['Close'] > df[f'SMA_{slow_window}']),
        'Signal'
    ] = 1

    # SELL Signal (Death Cross)
    df.loc[
        (df[f'SMA_{fast_window}'] < df[f'SMA_{slow_window}']) & 
        (df[f'SMA_{fast_window}'].shift(1) >= df[f'SMA_{slow_window}'].shift(1)), 
        'Signal'
    ] = -1

    return df

def backtest_strategy(df):
    """
    Simulates trading based on the 'Signal' column and calculates performance metrics.
    """
    
    # Ensure Date is set as index for this function
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    # 1. Calculate Daily Simple Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Use signal from previous day
    df['Position'] = df['Signal'].replace(to_replace=-1, value=0).ffill().fillna(0)
    df['Strategy_Position'] = df['Position'].shift(1).fillna(0)
    df['Strategy_Return'] = df['Position'] * df['Daily_Return']
    
    # Drop NaNs
    df = df.dropna(subset=['Strategy_Return'])
    
    # Calculate Cumulative Strategy Performance
    df['Cumulative_Returns'] = (1 + df['Strategy_Return']).cumprod()
    
    return df  # Returns with Date as index

def analyze_strategy(org_df, strategy_df):
    
    # 1. Prepare Strategy Equity Curve for Analytics
    strategy_analysis_df = strategy_df[['Cumulative_Returns']].rename(columns={'Cumulative_Returns': 'Close'}).copy()
    
    # --- ROBUST INDEX RESET for Strategy DF ---
    # Check if 'Date' already exists as a column
    if 'Date' in strategy_analysis_df.columns:
        # Drop the Date column first, then reset index to create it fresh
        strategy_analysis_df = strategy_analysis_df.drop(columns=['Date'])
    
    # Now safe to reset index
    strategy_analysis_df = strategy_analysis_df.reset_index()
    strategy_analysis_df = strategy_analysis_df.rename(columns={strategy_analysis_df.columns[0]: 'Date'})
    # ------------------------------------------

    # 2. Prepare Buy & Hold Data (Original Price Data) for Analytics
    original_analysis_df = org_df.copy()
    
    # === FIX: ELIMINATE THE DUPLICATE COLUMN BEFORE RESETTING THE INDEX ===
    if 'Date' in original_analysis_df.columns:
        # Drop the regular 'Date' column, keeping the 'Date' index
        original_analysis_df = original_analysis_df.drop(columns=['Date'])
    # ======================================================================
    
    # --- ROBUST INDEX RESET for B&H DF (Now safe to reset index) ---
    original_analysis_df = original_analysis_df.reset_index()
    original_analysis_df = original_analysis_df.rename(columns={original_analysis_df.columns[0]: 'Date'})
    # ------------------------------------------------------------
    
    # Calculate Metrics (The rest of the function remains the same and is correct)
    
    # B&H Metrics...
    bnh_cagr = calculate_cagr(original_analysis_df)
    bnh_mdd = calculate_mdd(original_analysis_df)
    bnh_sharpe = calculate_sharpe_ratio(original_analysis_df)

    # Strategy Metrics...
    strategy_cagr = calculate_cagr(strategy_analysis_df)
    strategy_mdd = calculate_mdd(strategy_analysis_df)
    strategy_sharpe = calculate_sharpe_ratio(strategy_analysis_df)

    # Display Results...
    print("\n## 📈 Strategy vs. Buy & Hold Performance\n")
    print("------------------------------------------------------------------")
    print(f"{'Metric':<30}{'Strategy (SMA Crossover)':<25}{'Benchmark (Buy & Hold)':<25}")
    print("------------------------------------------------------------------")
    print(f"{'Annualized Return (CAGR)':<30}{strategy_cagr * 100:<25.2f}%{bnh_cagr * 100:<25.2f}%")
    print(f"{'Maximum Drawdown (MDD)':<30}{strategy_mdd * 100:<25.2f}%{bnh_mdd * 100:<25.2f}%")
    print(f"{'Sharpe Ratio (SR)':<30}{strategy_sharpe:<25.2f}{bnh_sharpe:<25.2f}")
    print(f"{'Total Signals Generated':<30}{strategy_df['Signal'].abs().sum():<25.0f}")
    print("------------------------------------------------------------------")
    
    return strategy_analysis_df

#The results show a classic limitation of simple trend-following strategies: they are excellent at cutting risk but terrible in sideways markets.
# use a filter to only trade when the market is clearly trending