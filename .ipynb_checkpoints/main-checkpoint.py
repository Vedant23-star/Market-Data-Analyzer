import yfinance as yf

nifty = yf.Ticker("^NSEI")
nifty_data = nifty.history(period="10y")
# print(nifty_data.tail())
# print(nifty_data.head())

sp500 = yf.Ticker("^NSEI")
sp500_data = sp500.history(period="10y")
print(sp500_data.tail())