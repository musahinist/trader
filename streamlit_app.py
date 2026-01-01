import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Bitcoin Trading Algorithm", layout="wide", initial_sidebar_state="expanded")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simulate_bitcoin_prices(initial_price=45000, days=60, volatility=0.03, drift=0.0005):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(days)]
    prices = [initial_price]
    
    for _ in range(days - 1):
        daily_return = np.random.normal(drift, volatility)
        next_price = prices[-1] * (1 + daily_return)
        prices.append(max(next_price, 100))  # Ensure positive prices
    
    return dates, prices


def golden_cross_signals(df):
    """Apply Golden Cross trading signals."""
    df = df.copy()
    df['Signal'] = 0
    
    for i in range(1, len(df)):
        ma7 = df.loc[i, 'MA7']
        ma30 = df.loc[i, 'MA30']
        
        if pd.isna(ma7) or pd.isna(ma30):
            continue
        
        prev_ma7 = df.loc[i-1, 'MA7']
        prev_ma30 = df.loc[i-1, 'MA30']
        
        if pd.isna(prev_ma7) or pd.isna(prev_ma30):
            continue
        
        # Golden Cross (BUY signal)
        if prev_ma7 <= prev_ma30 and ma7 > ma30:
            df.loc[i, 'Signal'] = 1
        
        # Death Cross (SELL signal)
        elif prev_ma7 >= prev_ma30 and ma7 < ma30:
            df.loc[i, 'Signal'] = -1
    
    return df


class Portfolio:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.btc_holdings = 0
        self.trades = []
        self.portfolio_values = []
    
    def execute_trade(self, date, price, signal):
        """Execute a trade based on signal."""
        if signal == 1:  # BUY
            if self.cash > 0:
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
        """Calculate total portfolio value."""
        return self.cash + (self.btc_holdings * current_price)
    
    def record_daily_value(self, current_price):
        """Record portfolio value."""
        portfolio_value = self.calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)


# ============================================================================
# STREAMLIT UI
# ============================================================================

# Header
st.title("🚀 Bitcoin Golden Cross Trading Algorithm")
st.markdown("---")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    initial_price = st.number_input("Initial BTC Price ($)", value=45000, min_value=1000)
    days = st.slider("Days to Simulate", 30, 120, 60)
    volatility = st.slider("Daily Volatility (%)", 1, 10, 3) / 100
    drift = st.slider("Daily Drift (%)", -1.0, 1.0, 0.05) / 100
    initial_cash = st.number_input("Initial Capital ($)", value=100000, min_value=1000)
    
    st.markdown("---")
    
    if st.button("🔄 Run Simulation", use_container_width=True):
        st.session_state.run_simulation = True
    
    st.markdown("---")
    st.markdown("""
    ### Trading Strategy
    **Golden Cross Algorithm:**
    - 🟢 **BUY**: 7-day MA crosses above 30-day MA
    - 🔴 **SELL**: 7-day MA crosses below 30-day MA
    """)


# Run simulation
if st.session_state.get('run_simulation', True):
    # Generate data
    dates, prices = simulate_bitcoin_prices(initial_price, days, volatility, drift)
    
    df = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    
    # Calculate moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    
    # Apply signals
    df = golden_cross_signals(df)
    
    # Execute trades
    portfolio = Portfolio(initial_cash)
    
    for idx, row in df.iterrows():
        signal = row['Signal']
        if signal != 0:
            portfolio.execute_trade(row['Date'], row['Close'], signal)
        portfolio.record_daily_value(row['Close'])
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    final_price = df.iloc[-1]['Close']
    final_portfolio_value = portfolio.calculate_portfolio_value(final_price)
    total_return = ((final_portfolio_value - portfolio.initial_cash) / portfolio.initial_cash) * 100
    
    with col1:
        st.metric("Final Portfolio Value", f"${final_portfolio_value:,.0f}", 
                  f"${final_portfolio_value - portfolio.initial_cash:+,.0f}")
    
    with col2:
        st.metric("Return %", f"{total_return:+.2f}%", 
                  f"${final_portfolio_value - portfolio.initial_cash:+,.0f}")
    
    with col3:
        st.metric("Number of Trades", len(portfolio.trades))
    
    with col4:
        st.metric("BTC Holdings", f"{portfolio.btc_holdings:.6f} BTC", 
                  f"${portfolio.cash:,.0f} Cash")
    
    st.markdown("---")
    
    # Price Chart with Moving Averages
    st.subheader("📈 Bitcoin Price & Moving Averages")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Date'], df['Close'], label='BTC Price', linewidth=2, color='#1f77b4', alpha=0.8)
    ax.plot(df['Date'], df['MA7'], label='7-day MA', linewidth=2, color='#ff7f0e', alpha=0.8)
    ax.plot(df['Date'], df['MA30'], label='30-day MA', linewidth=2, color='#2ca02c', alpha=0.8)
    
    # Mark buy/sell points
    buy_signals = df[df['Signal'] == 1]
    sell_signals = df[df['Signal'] == -1]
    
    ax.scatter(buy_signals['Date'], buy_signals['Close'], color='green', marker='^', 
               s=200, label='Buy Signal', zorder=5, alpha=0.8)
    ax.scatter(sell_signals['Date'], sell_signals['Close'], color='red', marker='v', 
               s=200, label='Sell Signal', zorder=5, alpha=0.8)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Price ($)', fontsize=11)
    ax.set_title('Bitcoin Price with Golden Cross Trading Signals', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Portfolio Value Chart
    st.subheader("💰 Portfolio Value Over Time")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['Date'][:len(portfolio.portfolio_values)], portfolio.portfolio_values, 
            linewidth=2.5, color='#9467bd', label='Portfolio Value')
    ax.axhline(y=portfolio.initial_cash, color='gray', linestyle='--', 
               linewidth=1.5, label='Initial Capital', alpha=0.7)
    
    ax.fill_between(df['Date'][:len(portfolio.portfolio_values)], portfolio.initial_cash, 
                    portfolio.portfolio_values, alpha=0.2, color='#9467bd')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Value ($)', fontsize=11)
    ax.set_title('Portfolio Value Over Time', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Trade Ledger
    st.subheader("📋 Trade Ledger")
    
    if portfolio.trades:
        trades_df = pd.DataFrame(portfolio.trades)
        trades_df['Date'] = trades_df['Date'].dt.strftime('%Y-%m-%d')
        trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:,.2f}")
        trades_df['Amount_BTC'] = trades_df['Amount_BTC'].apply(lambda x: f"{x:.6f}")
        trades_df['Total_USD'] = trades_df['Total_USD'].apply(lambda x: f"${x:,.2f}")
        trades_df['BTC_Holdings'] = trades_df['BTC_Holdings'].apply(lambda x: f"{x:.6f}")
        trades_df['Cash'] = trades_df['Cash'].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No trades executed during the simulation period.")
    
    st.markdown("---")
    
    # Performance Comparison
    st.subheader("🎯 Strategy Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    initial_btc_amount = portfolio.initial_cash / df.iloc[0]['Close']
    buy_hold_value = initial_btc_amount * final_price
    buy_hold_return = ((buy_hold_value - portfolio.initial_cash) / portfolio.initial_cash) * 100
    btc_price_change = ((final_price - df.iloc[0]['Close']) / df.iloc[0]['Close']) * 100
    
    with col1:
        st.metric("Golden Cross Return", f"{total_return:+.2f}%", 
                  f"${final_portfolio_value - portfolio.initial_cash:+,.0f}")
    
    with col2:
        st.metric("Buy & Hold Return", f"{buy_hold_return:+.2f}%", 
                  f"${buy_hold_value - portfolio.initial_cash:+,.0f}")
    
    with col3:
        outperformance = total_return - buy_hold_return
        color = "green" if outperformance > 0 else "red"
        st.metric("Strategy Outperformance", f"{outperformance:+.2f}%", 
                  f"BTC Change: {btc_price_change:+.2f}%")
    
    st.markdown("---")
    
    # Detailed Statistics
    with st.expander("📊 Detailed Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trading Strategy Stats")
            st.write(f"**Initial Capital:** ${portfolio.initial_cash:,.2f}")
            st.write(f"**Final Portfolio Value:** ${final_portfolio_value:,.2f}")
            st.write(f"**Total Return:** {total_return:+.2f}%")
            st.write(f"**Number of Trades:** {len(portfolio.trades)}")
            st.write(f"**BTC Holdings:** {portfolio.btc_holdings:.6f} BTC")
            st.write(f"**Cash Remaining:** ${portfolio.cash:,.2f}")
        
        with col2:
            st.markdown("### Buy & Hold Comparison")
            st.write(f"**Initial Price:** ${df.iloc[0]['Close']:,.2f}")
            st.write(f"**Final Price:** ${final_price:,.2f}")
            st.write(f"**Price Change:** {btc_price_change:+.2f}%")
            st.write(f"**Buy & Hold Value:** ${buy_hold_value:,.2f}")
            st.write(f"**Buy & Hold Return:** {buy_hold_return:+.2f}%")
            st.write(f"**Outperformance:** {outperformance:+.2f}%")
    
    st.markdown("---")
    st.markdown("*Created with Streamlit • Bitcoin Trading Algorithm*")
