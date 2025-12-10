# #BASIC METRICS

# def calculate_cagr(df):
#     df = df.reset_index()
#     if len(df) < 2: return 0.0
#     starting_price = df['Close'].iloc[0]
#     ending_price = df['Close'].iloc[-1]

#     # Calculate number of years
#     start_date = df['Date'].iloc[0]
#     end_date = df['Date'].iloc[-1]
#     years = (end_date - start_date).days / 365.25

#     cagr = (ending_price / starting_price) ** (1 / years) - 1
#     cagr = round(cagr, 2)
#     return cagr

# def calculate_mdd(df):
# #While you could use the Low price to calculate the maximum potential drawdown within a 
# #single trading day, the standard financial definition of Maximum Drawdown (MDD) usually relies
# # on Closing Prices.
#     if len(df) < 2: return 0.0

#     if 'Date' in df.columns:
#         df = df.set_index('Date')
    
#     # If the index setting failed, or if the data is just unstable:
#     if df.empty or 'Close' not in df.columns:
#         return 0.0
#     peak_price = df['Close'].expanding().max()
#     drawdown_ratio = (df["Close"]/peak_price)
#     mdd_ratio = drawdown_ratio.min()-1
    
#     return round(mdd_ratio,2)

# import numpy as np

# def calculate_annualized_volatility(df):
#     # Step 1: Calculate Logarithmic Returns
#     # Note: .shift(1) aligns the current day's price with the previous day's price
#     log_returns = np.log(df['Close'] / df['Close'].shift(1))
    
#     # Step 2: Calculate Daily Volatility (Standard Deviation of returns)
#     # We use .std() on the returns, and we use .iloc[1:] to skip the first NaN value
#     sigma_daily = log_returns.iloc[1:].std()
    
#     # Step 3: Annualize (Multiply by the square root of 252 trading days)
#     sigma_annual = sigma_daily * np.sqrt(252)
#     sigma_annual = round(sigma_annual, 2)
#     return sigma_annual

# def calculate_sharpe_ratio(df):
#     #The Sharpe Ratio (or Sharpe Index) measures the excess return generated per unit of risk taken.
#     #S = (Rp -Rf)/annualized volatility, where Rp = CAGR, Rf= return on a risk free investment
#     #Lets use the Rf as 4.18% for USA and 6.61% for India 
#     RF_rate = 0.0539 #avg of us and india
#     if len(df) < 2: return 0.0

#     df = df.set_index('Date')
    
#     # If the index setting failed, or if the data is just unstable:
#     if df.empty: return 0.0
#     cagr_df_input = df[['Close']].reset_index()
#     cagr = calculate_cagr(cagr_df_input)
#     annual_volatility = calculate_annualized_volatility(df)
#     if annual_volatility ==0:
#         return 0.0
#     sharpe_ratio = (cagr - RF_rate)/annual_volatility
#     sharpe_ratio = round(sharpe_ratio, 2)
#     return sharpe_ratio

# def calculate_sortino_ratio(df):
# #The Sortino ratio measures an investment's return compared to its bad risk (downside volatility)
# #,not all risk like the Sharpe ratio, showing how much return you get for the losses you tolerate
# # S = (Rp -Rf)/Downside Deviation
# #Now lets calculate the downside devistion i.e standard deviation of only the returns that fall
# #below a specified Minimum Acceptable Return (MAR), which we usually set to the Risk-Free Rate 
#     MAR = 0.0539 #same as RF_rate
#     daily_return = MAR /252 # We convert the annual MAR to a daily equivalent
# # Filter the Downside Returns :returns that are less than the Daily Rf.
#     log_returns = np.log(df['Close'] / df['Close'].shift(1)).iloc[1:]
#     downside_return = log_returns[log_returns < daily_return]
#     downside_deviation = downside_return.std()
# #Annualize downside deviation
#     sigma_down_dev = downside_deviation*np.sqrt(252) 
#     if sigma_down_dev == 0:
#         return 0.0
# #finally sortino ratio
#     cagr = calculate_cagr(df)
#     s_ratio =(cagr - MAR)/sigma_down_dev
#     s_ratio = round(s_ratio,2)
#     return s_ratio    

# # TREND INDICATORS

# def calculate_sma(df,window):
# #A Simple Moving Average is the average of a stock's closing prices over a specific number of 
# #periods (window). As each new period passes, the oldest price is dropped, and the newest price
# #is added, making the average "move" with the price action.
#     sma = df["Close"].rolling(window).mean()
#     return sma

# def calculate_ema(df,window):
# #The EMA is an improvement over the SMA because it gives more weight to recent prices, 
# # making it more responsive to new information and quicker to signal a change in trend
# #The smoothing factor(K) determines how much weight is given to the most recent price.
#     ema = df['Close'].ewm(span=window, adjust=False).mean()
#     return ema

# def calculate_macd(df):
# # MACD represents the convergence and divergence of two trends (fast and slow) to determine momentum.
#     fast_ema = calculate_ema(df,12)
#     slow_ema = calculate_ema(df,26)

#     macd_line = fast_ema - slow_ema
# #The Signal Line is the EMA of the MACD Line (using the signal_period, default 9)    
#     signal_line = macd_line.ewm(span = 9, adjust=False).mean()
# # Calculate the MACD Histogram
#     macd_histogram = macd_line - signal_line
#     return macd_line, signal_line, macd_histogram

# def calculate_rsi(df, period = 14):
# #The Relative Strength Index is a momentum oscillator used to measure the speed and change of price movements.
#     delta = df['Close'].diff(1) #price change from previous day
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#     avg_gain = gain.ewm(span = period, min_periods=period-1,adjust=False).mean()
#     avg_loss = loss.ewm(span = period, min_periods=period-1,adjust=False).mean()

#     rs = avg_gain/avg_loss #relative strength
#     rsi = 100 - (100 / (1 + rs)) # index
#     return rsi.round(2)#RSI > 70: Overbought (Potential Sell Signal)
#                        #RSI < 30: Oversold (Potential Buy Signal)

# # CORRELATION

# def calculate_correlation(df1,df2,window):
#     returns1 = df1['Close'].pct_change().dropna()
#     returns2 = df2['Close'].pct_change().dropna()
# #.pct_change() method calculates simple returns. simple return = (day_n price/day_n-1) -1
#     if window:
#         correlation = df1['Close'].rolling(window).corr(df2['Close'])
#     correlation = returns1.corr(returns2) #.corr() calculates Pearson correlation
#     return correlation #A correlation close to 0 is highly desirable for diversification, as it means losses in one asset are not likely to be matched by losses in the other.
import numpy as np
import pandas as pd

# BASIC METRICS

def calculate_cagr(df):
    df = df.reset_index(drop=True)
    if len(df) < 2: 
        return 0.0
    
    starting_price = df['Close'].iloc[0]
    ending_price = df['Close'].iloc[-1]

    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]
    years = (end_date - start_date).days / 365.25

    if years == 0:
        return 0.0
    
    cagr = (ending_price / starting_price) ** (1 / years) - 1
    return round(float(cagr), 2)

def calculate_mdd(df):
    if len(df) < 2: 
        return 0.0
    
    df = df.copy()
    
    # Ensure Date is the index
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    if df.empty: 
        return 0.0
        
    peak_price = df['Close'].expanding().max()
    drawdown_ratio = (df["Close"] / peak_price)
    mdd_ratio = drawdown_ratio.min() - 1
    
    return round(float(mdd_ratio), 2)

def calculate_annualized_volatility(df):
    """Calculate annualized volatility - expects Date as index."""
    if len(df) < 2:
        return 0.0
    
    df = df.copy()
    
    # Calculate log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    log_returns = log_returns.dropna()
    
    if len(log_returns) == 0:
        return 0.0
    
    # Calculate daily volatility
    sigma_daily = log_returns.std()
    
    # Check if result is valid
    if pd.isna(sigma_daily) or sigma_daily == 0:
        return 0.0
    
    # Annualize
    sigma_annual = sigma_daily * np.sqrt(252)
    return round(float(sigma_annual), 2)

def calculate_sharpe_ratio(df):
    """Calculate Sharpe Ratio - expects Date as column."""
    RF_rate = 0.0539
    
    if len(df) < 2: 
        return 0.0

    df = df.copy()
    
    # For CAGR and MDD, we need Date as column
    df_for_cagr = df.copy()
    if 'Date' not in df_for_cagr.columns:
        df_for_cagr = df_for_cagr.reset_index()
    
    # For volatility, we need Date as index
    df_for_vol = df.copy()
    if 'Date' in df_for_vol.columns:
        df_for_vol = df_for_vol.set_index('Date')
    
    cagr = calculate_cagr(df_for_cagr)
    annual_volatility = calculate_annualized_volatility(df_for_vol)
    
    if annual_volatility == 0:
        return 0.0
    
    sharpe_ratio = (cagr - RF_rate) / annual_volatility
    return round(float(sharpe_ratio), 2)

def calculate_sortino_ratio(df):
    """Calculate Sortino Ratio - expects Date as index."""
    MAR = 0.0539
    daily_return = MAR / 252
    
    if len(df) < 2:
        return 0.0
    
    df = df.copy()
    
    # Calculate log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    log_returns = log_returns.dropna()
    
    if len(log_returns) == 0:
        return 0.0
    
    # Filter downside returns - THIS IS WHERE THE ERROR HAPPENS
    # We need to be explicit about the comparison
    downside_mask = log_returns < daily_return
    downside_return = log_returns[downside_mask]
    
    if len(downside_return) == 0:
        return 0.0
    
    downside_deviation = downside_return.std()
    
    if pd.isna(downside_deviation) or downside_deviation == 0:
        return 0.0
    
    sigma_down_dev = downside_deviation * np.sqrt(252)
    
    if sigma_down_dev == 0:
        return 0.0
    
    # Calculate CAGR - needs Date as column
    cagr_df = df[['Close']].reset_index()
    cagr = calculate_cagr(cagr_df)
    
    s_ratio = (cagr - MAR) / sigma_down_dev
    return round(float(s_ratio), 2)

# TREND INDICATORS

def calculate_sma(df, window):
    sma = df["Close"].rolling(window).mean()
    return sma

def calculate_ema(df, window):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    return ema

def calculate_macd(df):
    fast_ema = calculate_ema(df, 12)
    slow_ema = calculate_ema(df, 26)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_rsi(df, period=14):
    """Calculate RSI - expects Date as index."""
    if len(df) < period + 1:
        return pd.Series([50.0] * len(df), index=df.index)
    
    delta = df['Close'].diff(1)
    
    # Explicitly handle the comparison
    gain_mask = delta > 0
    loss_mask = delta < 0
    
    gain = delta.where(gain_mask, 0)
    loss = -delta.where(loss_mask, 0)
    
    avg_gain = gain.ewm(span=period, min_periods=period-1, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period-1, adjust=False).mean()

    # Avoid division by zero
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    
    # Fill NaN with 50
    rsi = rsi.fillna(50)
    
    return rsi.round(2)

# CORRELATION

def calculate_correlation(df1, df2, window=None):
    """Calculate correlation between two dataframes."""
    returns1 = df1['Close'].pct_change().dropna()
    returns2 = df2['Close'].pct_change().dropna()
    
    # Align the series
    aligned = pd.concat([returns1, returns2], axis=1, join='inner')
    aligned.columns = ['returns1', 'returns2']
    
    if len(aligned) < 2:
        return 0.0
    
    if window:
        correlation = aligned['returns1'].rolling(window).corr(aligned['returns2'])
        return float(correlation.iloc[-1]) if len(correlation) > 0 else 0.0
    
    correlation = aligned['returns1'].corr(aligned['returns2'])
    return round(float(correlation), 4)