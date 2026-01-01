import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. SIMULATE 60 DAYS OF BITCOIN PRICE DATA
# ============================================================================

def simulate_bitcoin_prices(initial_price=45000, days=60, volatility=0.03, drift=0.0005):
    """
    Simulate Bitcoin prices using Geometric Brownian Motion (GBM).
    
    Parameters:
    - initial_price: Starting BTC price
    - days: Number of trading days to simulate
    - volatility: Daily volatility (standard deviation)
    - drift: Daily expected return
    """
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(days)]
    prices = [initial_price]
    
    for _ in range(days - 1):
        # GBM: dP = μ*P*dt + σ*P*dW
        daily_return = np.random.normal(drift, volatility)
        next_price = prices[-1] * (1 + daily_return)
        prices.append(next_price)
    
    return dates, prices


# Generate price data
dates, prices = simulate_bitcoin_prices()

# Create DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Close': prices
})

# ============================================================================
# 2. CALCULATE MOVING AVERAGES
# ============================================================================

df['MA7'] = df['Close'].rolling(window=7).mean()
df['MA30'] = df['Close'].rolling(window=30).mean()

# ============================================================================
# 3. IMPLEMENT GOLDEN CROSS TRADING ALGORITHM
# ============================================================================

def golden_cross_signals(df):
    """
    Golden Cross Strategy:
    - BUY: when 7-day MA crosses above 30-day MA
    - SELL: when 7-day MA crosses below 30-day MA
    """
    df['Signal'] = 0
    df['Position'] = 0
    
    for i in range(1, len(df)):
        ma7 = df.loc[i, 'MA7']
        ma30 = df.loc[i, 'MA30']
        
        # Skip if moving averages are not yet calculated
        if pd.isna(ma7) or pd.isna(ma30):
            continue
        
        prev_ma7 = df.loc[i-1, 'MA7']
        prev_ma30 = df.loc[i-1, 'MA30']
        
        if pd.isna(prev_ma7) or pd.isna(prev_ma30):
            continue
        
        # Golden Cross (BUY signal)
        if prev_ma7 <= prev_ma30 and ma7 > ma30:
            df.loc[i, 'Signal'] = 1  # BUY
        
        # Death Cross (SELL signal)
        elif prev_ma7 >= prev_ma30 and ma7 < ma30:
            df.loc[i, 'Signal'] = -1  # SELL
    
    return df


df = golden_cross_signals(df)

# ============================================================================
# 4. TRACK TRADES AND PORTFOLIO PERFORMANCE
# ============================================================================

class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.btc_holdings = 0
        self.trades = []
        self.portfolio_values = []
    
    def execute_trade(self, date, price, signal):
        """Execute a buy or sell trade based on signal."""
        if signal == 1:  # BUY
            if self.cash > 0:
                # Use all available cash to buy BTC
                btc_to_buy = self.cash / price
                self.btc_holdings += btc_to_buy
                self.cash = 0
                self.trades.append({
                    'Date': date,
                    'Type': 'BUY',
                    'Price': price,
                    'Amount_BTC': btc_to_buy,
                    'Total_USD': btc_to_buy * price,
                    'BTC_Holdings': self.btc_holdings,
                    'Cash': self.cash
                })
        
        elif signal == -1:  # SELL
            if self.btc_holdings > 0:
                # Sell all BTC holdings
                usd_from_sale = self.btc_holdings * price
                self.cash += usd_from_sale
                self.trades.append({
                    'Date': date,
                    'Type': 'SELL',
                    'Price': price,
                    'Amount_BTC': self.btc_holdings,
                    'Total_USD': usd_from_sale,
                    'BTC_Holdings': 0,
                    'Cash': self.cash
                })
                self.btc_holdings = 0
    
    def calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value in USD."""
        return self.cash + (self.btc_holdings * current_price)
    
    def record_daily_value(self, current_price):
        """Record the daily portfolio value."""
        portfolio_value = self.calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)


# Execute the trading algorithm
portfolio = Portfolio(initial_cash=100000)

for idx, row in df.iterrows():
    signal = row['Signal']
    if signal != 0:
        portfolio.execute_trade(row['Date'], row['Close'], signal)
    
    # Record portfolio value daily
    portfolio.record_daily_value(row['Close'])

# ============================================================================
# 5. PRINT RESULTS
# ============================================================================

print("=" * 100)
print("BITCOIN GOLDEN CROSS TRADING ALGORITHM - 60 DAY BACKTEST")
print("=" * 100)
print()

# Print price data with moving averages (sample)
print("PRICE DATA & MOVING AVERAGES (First 35 days):")
print("-" * 100)
display_cols = ['Date', 'Close', 'MA7', 'MA30']
display_df = df[display_cols].head(35).copy()
display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
display_df['Close'] = display_df['Close'].apply(lambda x: f"${x:,.2f}")
display_df['MA7'] = display_df['MA7'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
display_df['MA30'] = display_df['MA30'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
print(display_df.to_string(index=False))
print()

# Print trade ledger
print("=" * 100)
print("TRADE LEDGER:")
print("-" * 100)
if portfolio.trades:
    trades_df = pd.DataFrame(portfolio.trades)
    trades_df['Date'] = trades_df['Date'].dt.strftime('%Y-%m-%d')
    trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:,.2f}")
    trades_df['Amount_BTC'] = trades_df['Amount_BTC'].apply(lambda x: f"{x:.6f}")
    trades_df['Total_USD'] = trades_df['Total_USD'].apply(lambda x: f"${x:,.2f}")
    trades_df['BTC_Holdings'] = trades_df['BTC_Holdings'].apply(lambda x: f"{x:.6f}")
    trades_df['Cash'] = trades_df['Cash'].apply(lambda x: f"${x:,.2f}")
    print(trades_df.to_string(index=False))
else:
    print("No trades executed.")
print()

# Print final portfolio performance
print("=" * 100)
print("FINAL PORTFOLIO PERFORMANCE:")
print("-" * 100)

final_price = df.iloc[-1]['Close']
final_portfolio_value = portfolio.calculate_portfolio_value(final_price)
total_return = ((final_portfolio_value - portfolio.initial_cash) / portfolio.initial_cash) * 100
btc_return = ((final_price - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100

print(f"Initial Capital:        ${portfolio.initial_cash:,.2f}")
print(f"Final Portfolio Value:  ${final_portfolio_value:,.2f}")
print(f"Total Profit/Loss:      ${final_portfolio_value - portfolio.initial_cash:,.2f}")
print(f"Total Return:           {total_return:+.2f}%")
print()
print(f"Starting BTC Price:     ${df.iloc[0]['Close']:,.2f}")
print(f"Ending BTC Price:       ${final_price:,.2f}")
print(f"BTC Price Change:       {btc_return:+.2f}%")
print()
print(f"BTC Currently Held:     {portfolio.btc_holdings:.6f} BTC")
print(f"Cash Remaining:         ${portfolio.cash:,.2f}")
print(f"Number of Trades:       {len(portfolio.trades)}")
print()

# Compare with Buy & Hold strategy
initial_btc_amount = portfolio.initial_cash / df.iloc[0]['Close']
buy_hold_value = initial_btc_amount * final_price
buy_hold_return = ((buy_hold_value - portfolio.initial_cash) / portfolio.initial_cash) * 100

print("COMPARISON WITH BUY & HOLD STRATEGY:")
print("-" * 100)
print(f"Buy & Hold Value:       ${buy_hold_value:,.2f}")
print(f"Buy & Hold Return:      {buy_hold_return:+.2f}%")
print(f"Strategy vs Buy & Hold: {total_return - buy_hold_return:+.2f}%")
print("=" * 100)
